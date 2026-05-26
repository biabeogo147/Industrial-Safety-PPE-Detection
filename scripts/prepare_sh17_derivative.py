"""Prepare SH17-derived canonical objects and relation manifests."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_project

bootstrap_project()

from site_safety_monitor.data.sh17_prepare import DEFAULT_OUTPUT_ROOT, DEFAULT_SH17_ROOT, export_sh17_derivative


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sh17-root", default=str(DEFAULT_SH17_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_paths = export_sh17_derivative(
        sh17_root=args.sh17_root,
        output_root=args.output_root,
        limit=args.limit,
    )
    print(json.dumps({name: str(path) for name, path in output_paths.items()}, indent=2))


if __name__ == "__main__":
    main()
