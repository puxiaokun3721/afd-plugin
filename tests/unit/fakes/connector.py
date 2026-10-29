# SPDX-License-Identifier: Apache-2.0
"""Runtime endpoints for host connector error-propagation seam tests."""


class MegaMoeRuntimeProbe:
    """Expose an already-reported fatal; do not emulate device computation."""

    def __init__(self, error: str | None = None) -> None:
        self.error = error

    def check_health(self) -> None:
        if self.error is not None:
            raise RuntimeError(self.error)

    def synchronize(self) -> None:
        self.check_health()

    def abort(self, error_code: int) -> None:
        # These tests observe the connector/model result contract, not device
        # visibility. Control-copy qualification belongs to the NPU tests.
        pass
