# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real four-op A3 harness and independent signed-INT4 mathematical reference.

This harness checks routing/selection/numerics and times the complete A/F chain.
It does not claim to prove the cross-device memory model or native hot-path
parity. The latter require the separate litmus/profiling gates in the design.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

# Standalone torchrun scripts need the source checkout on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from afd_plugin.compat.npu.mega_moe_abi import (  # noqa: E402
    MegaMoeStaticConfig,
)

DEFAULT_SEED = 3721
DEFAULT_LAYER_IDS = (2, 4, 7)
DEFAULT_LAYER_ORDER = (7, 2, 7, 4)


def parse_arguments(description: str, *, performance: bool) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--iterations", type=int, default=100 if performance else 4)
    parser.add_argument("--warmup", type=int, default=20 if performance else 0)
    parser.add_argument("--rtol", type=float, default=0.04)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--relative-l2", type=float, default=0.05)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    if any(
        not math.isfinite(v) or v < 0 for v in (args.rtol, args.atol, args.relative_l2)
    ):
        parser.error("precision tolerances must be finite and nonnegative")
    return args


class MegaMoeHarness:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
        self.tokens_per_tp = tuple(manifest.get("tokens_per_tp", [17, 0, 33, 19]))
        self.layer_ids = tuple(manifest.get("layer_ids", DEFAULT_LAYER_IDS))
        self.layer_order = tuple(manifest.get("layer_order", DEFAULT_LAYER_ORDER))
        self.mask_stride = int(manifest.get("mask_stride", 13))
        self.hotspot = bool(manifest.get("hotspot", True))
        self.swiglu_limit = float(manifest.get("swiglu_limit", 0.0))
        self.seed = int(manifest.get("seed", DEFAULT_SEED))
        if not self.tokens_per_tp or not self.layer_ids:
            raise ValueError("tokens_per_tp and layer_ids must be nonempty")
        if not math.isfinite(self.swiglu_limit) or self.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be finite and nonnegative")
        self.base_config = MegaMoeStaticConfig(
            session_epoch=1,
            physical_rank=0,
            attn_dp_size=int(manifest.get("attn_dp_size", 2)),
            attn_tp_size=len(self.tokens_per_tp),
            ffn_ep_size=int(manifest.get("ffn_ep_size", 8)),
            local_expert_count=int(manifest.get("local_expert_count", 2)),
            hidden_size=int(manifest.get("hidden_size", 1024)),
            intermediate_size=int(manifest.get("intermediate_size", 512)),
            topk=int(manifest.get("topk", 2)),
            source_token_capacity=max(1, max(self.tokens_per_tp)),
            recv_route_capacity=int(manifest.get("recv_route_capacity", 32)),
            layer_count=len(self.layer_ids),
            layer_id_capacity=max(self.layer_ids) + 1,
        )
        if len(set(self.layer_ids)) != len(self.layer_ids) or not self.layer_order:
            raise ValueError("layer_ids must be unique and layer_order nonempty")
        if any(layer not in self.layer_ids for layer in self.layer_order):
            raise ValueError("layer_order refers to an absent layer")
        if min(self.tokens_per_tp) < 0 or self.mask_stride < 0:
            raise ValueError("token lengths and mask_stride must be nonnegative")
        self.sequence = 0

    def describe(self) -> dict:
        return {
            "kind": "A3 W4A8 complete four-op chain",
            "config": asdict(self.base_config),
            "window": asdict(self.base_config.layout),
            "ffn_scratch_bytes": self.base_config.ffn_scratch_bytes,
            "tokens_per_tp": self.tokens_per_tp,
            "layer_ids": self.layer_ids,
            "layer_order": self.layer_order,
            "seed": self.seed,
            "mask_stride": self.mask_stride,
            "hotspot": self.hotspot,
            "swiglu_limit": self.swiglu_limit,
            "device_validation": False,
        }

    def start(self) -> None:
        # Lazy runtime loading keeps --dry-run usable on a CPU-only machine.
        import torch
        import torch.distributed as dist
        import torch_npu

        from afd_plugin.compat.npu.cam_async_megamoe import MegaMoeContext
        from afd_plugin.v1.worker.npu.async_cam_megamoe import MegaMoeLayeredExecutor

        self.torch, self.dist, self.torch_npu = torch, dist, torch_npu
        self.rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        if int(os.environ["WORLD_SIZE"]) != self.base_config.world_size:
            raise ValueError(
                "torchrun world size differs from the manifest A/F topology"
            )
        torch.npu.set_device(local_rank)
        torch.npu.config.allow_internal_format = True
        dist.init_process_group("hccl", timeout=timedelta(minutes=5))
        self.device = torch.device("npu", local_rank)
        group = dist.group.WORLD
        group_name = group._get_backend(self.device).get_hccl_comm_name(self.rank)
        epoch = [
            (int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)) or 1
            if self.rank == 0
            else None
        ]
        dist.broadcast_object_list(epoch, src=0)
        values = asdict(self.base_config)
        values.update(session_epoch=epoch[0], physical_rank=self.rank)
        self.config = MegaMoeStaticConfig(**values)
        self.context = MegaMoeContext(self.config, str(group_name), self.device)
        self.executor = None
        if not self.config.is_attention:
            self.executor = MegaMoeLayeredExecutor(self._layers(), self.config)
            # Deliberately choose different first DPs on different receivers.
            self.context.state[4] = (
                self.rank - self.config.attn_rank_count
            ) % self.config.attn_dp_size
        dist.barrier()

    def _signed_weight(self, layer: int, expert: int, which: int):
        torch, cfg = self.torch, self.config
        k, n = (
            (cfg.hidden_size, 2 * cfg.intermediate_size)
            if which == 1
            else (cfg.intermediate_size, cfg.hidden_size)
        )
        generator = torch.Generator().manual_seed(
            self.seed + layer * 1009 + expert * 17 + which
        )
        values = torch.randint(-7, 8, (k, n), dtype=torch.int32, generator=generator)
        scale = (
            torch.full((n,), 0.01 * (self.layer_ids.index(layer) + 1)).half().float()
        )
        return values, scale

    def _layers(self):
        from afd_plugin.model_executor.npu.async_cam_w4a8 import W4A8LayerWeights

        torch, cfg = self.torch, self.config
        first = (self.rank - cfg.attn_rank_count) * cfg.local_expert_count
        layers = []
        for layer in self.layer_ids:
            packed, encoded, compensation = [], [], []
            for which in (1, 2):
                source = [
                    self._signed_weight(layer, first + e, which)
                    for e in range(cfg.local_expert_count)
                ]
                values = torch.stack([entry[0] for entry in source])
                scales = torch.stack([entry[1] for entry in source])
                pairs = values.to(torch.int8).reshape(-1, 2)
                byte_values = ((pairs[:, 1] << 4) | (pairs[:, 0] & 15)).reshape(
                    values.shape[0], values.shape[1], values.shape[2] // 2
                )
                nz = self.torch_npu.npu_format_cast(byte_values.to(self.device), 29)
                packed.append(nz.view(torch.int32).contiguous())
                encoded.append(scales.view(torch.int32).to(torch.int64).to(self.device))
                compensation.append(
                    (8 * (values.float() * scales[:, None]).sum(1)).to(self.device)
                )
            layers.append(
                W4A8LayerWeights(
                    layer,
                    packed[0],
                    packed[1],
                    encoded[0],
                    encoded[1][:, None, :],
                    compensation[0],
                    compensation[1],
                    True,
                    self.swiglu_limit,
                    1.0,
                )
            )
        return layers

    def inputs(self, iteration: int):
        torch, cfg = self.torch, self.config
        rows = self.tokens_per_tp[self.rank % cfg.attn_tp_size]
        generator = torch.Generator().manual_seed(
            self.seed + self.rank * 97 + iteration
        )
        x = (torch.randn((rows, cfg.hidden_size), generator=generator) * 0.2).bfloat16()
        tokens = torch.arange(rows, dtype=torch.int32)
        if self.hotspot:
            ids = (
                tokens[:, None] * 0 + torch.arange(cfg.topk)[None, :] + 1
            ) % cfg.num_experts
        else:
            ids = (
                tokens[:, None] + torch.arange(cfg.topk)[None, :] + self.rank
            ) % cfg.num_experts
        weights = torch.full((rows, cfg.topk), 1 / cfg.topk, dtype=torch.float32)
        mask = torch.ones(rows, dtype=torch.bool)
        if self.mask_stride:
            mask[:: self.mask_stride] = False
        return x, ids.to(torch.int32), weights, mask

    def golden(self, inputs, layer: int):
        torch, cfg = self.torch, self.config
        x, ids, weights, mask = inputs
        result = torch.zeros_like(x, dtype=torch.float32)
        if x.shape[0] == 0:
            return result
        maximum = x.float().abs().amax(-1, keepdim=True)
        multiplier = torch.where(
            maximum > 0, 127 / maximum, torch.full_like(maximum, 127)
        )
        activation = (x.float() * multiplier).round().clamp(-127, 127) / multiplier
        for expert in range(cfg.num_experts):
            positions = ((ids == expert) & mask[:, None]).nonzero()
            if not positions.numel():
                continue
            rows, slots = positions[:, 0], positions[:, 1]
            w1, s1 = self._signed_weight(layer, expert, 1)
            w2, s2 = self._signed_weight(layer, expert, 2)
            gate, up = (activation[rows] @ (w1.float() * s1)).chunk(2, -1)
            if self.swiglu_limit > 0:
                gate = gate.clamp(max=self.swiglu_limit)
                up = up.clamp(-self.swiglu_limit, self.swiglu_limit)
            activated = torch.nn.functional.silu(gate) * up
            maximum = activated.abs().amax(-1, keepdim=True)
            scale = torch.where(
                maximum > 0, maximum / 127, torch.ones_like(maximum) / 127
            )
            quantized = (activated / scale).round().clamp(-127, 127) * scale
            routed = (quantized @ (w2.float() * s2)).bfloat16().float()
            result.index_add_(0, rows, routed * weights[rows, slots, None])
        return result.bfloat16().float()

    def validate_attention(self, inputs, layer: int, output) -> dict:
        torch, cfg = self.torch, self.config
        actual, expected = output.cpu().float(), self.golden(inputs, layer)
        failures = []
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            failures.append("nonfinite output/reference")
        error = actual - expected
        norm = float(expected.norm())
        relative = float(error.norm()) / norm if norm else float(error.norm())
        try:
            torch.testing.assert_close(
                actual, expected, rtol=self.args.rtol, atol=self.args.atol
            )
        except AssertionError as exc:
            failures.append(str(exc))
        if relative > self.args.relative_l2:
            failures.append(f"relative L2 {relative} exceeds {self.args.relative_l2}")
        x, ids, _, mask = inputs
        observed = self.context.source_counts.cpu()
        row_map = self.context.route_index.cpu()
        for tile in range(cfg.layout.num_chunks):
            begin = tile * cfg.layout.chunk_tokens
            end = min(begin + cfg.layout.chunk_tokens, x.shape[0])
            chunk_ids, chunk_mask = ids[begin:end], mask[begin:end]
            counts = torch.bincount(
                chunk_ids[chunk_mask].reshape(-1).to(torch.int64),
                minlength=cfg.num_experts,
            )
            if not torch.equal(observed[tile, : cfg.num_experts].long(), counts):
                failures.append(f"tile {tile}: expert counts differ")
            mapping = row_map[begin * cfg.topk : end * cfg.topk].reshape(-1, cfg.topk)
            if (mapping[~chunk_mask] != -1).any():
                failures.append(f"tile {tile}: masked route index is not -1")
            valid_map = mapping[chunk_mask].flatten().long()
            if not torch.equal(
                valid_map.sort().values, torch.arange(int(counts.sum()))
            ):
                failures.append(f"tile {tile}: routes omitted or duplicated")
        if actual[~mask].count_nonzero():
            failures.append("masked output is not zero")
        return {
            "failures": failures,
            "relative_l2": relative,
            "max_abs": float(error.abs().max()) if error.numel() else 0.0,
        }

    def run_iteration(self, iteration: int, *, validate: bool) -> dict:
        torch, cfg = self.torch, self.config
        layer = self.layer_order[iteration % len(self.layer_order)]
        inputs = self.inputs(iteration) if cfg.is_attention else None
        if inputs is not None:
            x, ids, weights, mask = [tensor.to(self.device) for tensor in inputs]
            output = torch.empty_like(x)
        # Measurement-only boundary, not part of the protocol under test.
        self.dist.barrier()
        start, end = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        start.record()
        if cfg.is_attention:
            self.sequence += 1
            metadata = torch.tensor(
                cfg.make_task_meta(
                    self.sequence,
                    self.rank // cfg.attn_tp_size,
                    layer,
                    iteration % 2,
                    x.shape[0],
                ),
                dtype=torch.int64,
                device=self.device,
            )
            self.context.dispatch_send(x, ids, mask, metadata)
            self.context.combine_recv(weights, mask, metadata, output)
        else:
            for _ in range(cfg.attn_dp_size):
                self.context.schedule()
                executor = self.executor
                self.context.ffn(
                    executor.layer_id_to_slot,
                    executor.w13,
                    executor.w2,
                    executor.weight_scale13,
                    executor.weight_scale2,
                    executor.compensation13,
                    executor.compensation2,
                    executor.layer_params,
                )
        end.record()
        end.synchronize()
        report = {
            "rank": self.rank,
            "role": "A" if cfg.is_attention else "F",
            "elapsed_ms": start.elapsed_time(end),
            "failures": [],
            "allocated_peak_bytes": torch.npu.max_memory_allocated(),
        }
        self.context.check_health()
        if validate and cfg.is_attention:
            report.update(self.validate_attention(inputs, layer, output))
        return report

    def gather(self, report: dict) -> list[dict]:
        reports = [None] * self.config.world_size
        self.dist.all_gather_object(reports, report)
        return reports

    def close(self) -> None:
        # No idle Scheduler remains queued by this harness.
        self.context.synchronize()
        self.dist.barrier()
        self.dist.destroy_process_group()


def run(args: argparse.Namespace, *, performance: bool) -> int:
    harness = MegaMoeHarness(args)
    report = harness.describe()
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    harness.start()
    # Always validate the exact synthetic weights/routes before timing.
    preflight = harness.gather(harness.run_iteration(0, validate=True))
    if any(rank["failures"] for rank in preflight):
        report.update(status="failed", preflight=preflight)
        harness.close()
        if harness.rank == 0:
            print(json.dumps(report, indent=2))
        return 1
    for iteration in range(1, args.warmup + 1):
        harness.run_iteration(iteration, validate=False)
    rounds = []
    for iteration in range(args.iterations):
        local = harness.run_iteration(
            args.warmup + iteration + 1, validate=not performance
        )
        rounds.append(harness.gather(local))
    failed = any(rank["failures"] for ranks in rounds for rank in ranks)
    latencies = sorted(max(rank["elapsed_ms"] for rank in ranks) for ranks in rounds)
    report.update(
        status="failed" if failed else "passed",
        device_validation=True,
        preflight=preflight,
        rounds=rounds,
        median_ms=latencies[len(latencies) // 2],
        p95_ms=latencies[min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)],
        timing_scope="full A/F four-op chain including receiver readiness wait",
        allocated_peak_bytes_per_rank=[
            max(ranks[rank]["allocated_peak_bytes"] for ranks in rounds)
            for rank in range(harness.config.world_size)
        ],
    )
    harness.close()
    if harness.rank == 0:
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
    return 1 if failed else 0
