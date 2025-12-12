"""Simple CLI progress helper reused by the console scripts."""

from __future__ import annotations

import sys


class CLIProgress:
    """Prints lightweight progress updates for console helpers."""

    def __init__(self, name: str, total_steps: int | None = None) -> None:
        self.name = name
        self.total_steps = total_steps or 0
        self.completed = 0

    def step(self, message: str) -> None:
        """Advance the progress and emit the updated status line."""

        self.completed += 1
        percent = ""
        bar = ""
        if self.total_steps:
            percent_value = int(min(100, (self.completed / self.total_steps) * 100))
            percent = f"{percent_value:3d}%"
            filled = int((self.completed / self.total_steps) * 20)
            bar = "[" + "=" * filled + " " * max(0, 20 - filled) + "]"
        status = f"{self.name:12s} {bar} {percent} {message}"
        print(status.strip(), file=sys.stderr)

    def set_total(self, total_steps: int) -> None:
        self.total_steps = total_steps
