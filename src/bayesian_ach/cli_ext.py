"""Backward-compatible command dispatcher with design and replay support."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from bayesian_ach.cli import main as legacy_main
from bayesian_ach.design_cli import main as design_main
from bayesian_ach.replay_cli import main as replay_main


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "design-benchmark":
        return design_main(arguments[1:])
    if arguments and arguments[0] == "replay-benchmark":
        return replay_main(arguments[1:])
    return legacy_main(arguments)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
