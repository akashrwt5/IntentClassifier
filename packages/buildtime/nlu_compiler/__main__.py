"""CLI: python -m nlu_compiler <bundle_dir> [--format 3.0]

Exit code 0 = no errors (warnings allowed), 1 = errors, 2 = usage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .diagnostics import Severity
from .validator import validate_bundle_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nlu_compiler", description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--format", default="3.0")
    args = parser.parse_args(argv)

    if not args.bundle_dir.is_dir():
        print(f"not a directory: {args.bundle_dir}", file=sys.stderr)
        return 2

    diags = validate_bundle_dir(args.bundle_dir, args.format)
    for d in diags:
        print(d)
    n_err = sum(1 for d in diags if d.severity is Severity.ERROR)
    n_warn = len(diags) - n_err
    print(f"\n{n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
