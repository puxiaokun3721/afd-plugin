# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real idle Scheduler STOP/ABORT checks after one successful four-op task.

Use a fresh torchrun domain for each mode. This checks the control/terminal
contract; it does not establish that the device was already polling when the
control DMA was issued, or replace the memory-publication litmus in the design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a3_camasync_megamoe_common import MegaMoeHarness


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stop", "abort"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.rtol, args.atol, args.relative_l2 = 0.04, 0.05, 0.05
    harness = MegaMoeHarness(args)
    report = harness.describe()
    report["control_mode"] = args.mode
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    harness.start()
    preflight = harness.gather(harness.run_iteration(0, validate=True))
    if any(rank["failures"] for rank in preflight):
        harness.close()
        if harness.rank == 0:
            print(json.dumps({"status": "failed", "preflight": preflight}, indent=2))
        return 1
    failures = []
    if not harness.config.is_attention:
        context, executor = harness.context, harness.executor
        context.schedule()
        context.ffn(
            executor.layer_id_to_slot,
            executor.w13,
            executor.w2,
            executor.weight_scale13,
            executor.weight_scale2,
            executor.compensation13,
            executor.compensation2,
            executor.layer_params,
        )
        if args.mode == "stop":
            context.request_stop()
        else:
            context.abort()
        error = None
        try:
            context.synchronize()
        except RuntimeError as exc:
            error = str(exc)
        work = context.work_desc.cpu().tolist()
        completion = context.completion.cpu().tolist()
        # These are public wire values, not executor implementation details.
        expected_kind = 2 if args.mode == "stop" else 3
        if work[2] != expected_kind or completion[1] != expected_kind:
            failures.append(
                f"unexpected terminal work/completion: {work}, {completion}"
            )
        if args.mode == "abort":
            if not error or "error_code=" not in error or not work[3]:
                failures.append(f"ABORT did not surface protocol failure: {error}")
        elif error:
            failures.append(error)
    ranks = harness.gather({"rank": harness.rank, "failures": failures})
    # synchronize() above drained the failed F stream before this teardown.
    harness.dist.barrier()
    harness.dist.destroy_process_group()
    failed = any(rank["failures"] for rank in ranks)
    report.update(
        status="failed" if failed else "passed", ranks=ranks, device_validation=True
    )
    if harness.rank == 0:
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
