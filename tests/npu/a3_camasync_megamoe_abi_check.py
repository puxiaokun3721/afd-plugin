# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Compare the compiled host ABI probe with Python allocation contracts."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

from a3_camasync_megamoe_common import MegaMoeStaticConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = []
    for tokens, capacity, hidden in itertools.product(
        (1, 1023, 1024, 1025, 8192), (1, 8, 128, 257, 4096), (1024, 7168)
    ):
        config = MegaMoeStaticConfig(
            1, 0, 2, 4, 8, 2, hidden, 512, 2, tokens, capacity, 3, 8
        )
        window = config.layout
        expected = (
            window.chunk_tokens,
            window.num_chunks,
            window.padded_tokens,
            window.expert_stride,
            window.x_base,
            window.scale_base,
            window.d_base,
            window.done_base,
            window.inbox_stride,
            window.tail_bytes,
            window.required_bytes,
            config.send_scratch_bytes,
            config.ffn_native_scratch_bytes,
            config.ffn_scratch_bytes,
            config.max_recv_rounds,
        )
        cases.append((config.config_values, expected))
    probe = subprocess.run(
        [str(args.probe.resolve())],
        input=("\n".join(" ".join(map(str, c[0])) for c in cases) + "\n").encode(),
        capture_output=True,
        check=True,
        timeout=30,
    )
    results = probe.stdout.decode("utf-8").splitlines()
    if len(results) != len(cases):
        raise AssertionError(f"probe returned {len(results)} of {len(cases)} cases")
    for result, (config_values, expected) in zip(results, cases, strict=True):
        actual = tuple(map(int, result.split()))
        if actual != expected:
            raise AssertionError(
                f"ABI mismatch for {config_values}: C++={actual}, Python={expected}"
            )
    report = {
        "kind": "CPU C++/Python ABI contract",
        "cases": len(cases),
        "passed": True,
        "device_validation": False,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
