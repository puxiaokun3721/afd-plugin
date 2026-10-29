# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Benchmark the A3 four-op chain after a matching numerical preflight."""

from a3_camasync_megamoe_common import parse_arguments, run

if __name__ == "__main__":
    raise SystemExit(run(parse_arguments(__doc__, performance=True), performance=True))
