# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Run real A3 four-op W4A8 precision, routing, masked-tail and layer checks."""

from a3_camasync_megamoe_common import parse_arguments, run

if __name__ == "__main__":
    raise SystemExit(
        run(parse_arguments(__doc__, performance=False), performance=False)
    )
