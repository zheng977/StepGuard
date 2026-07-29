from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evals.reporting import (
    collect_result_summary_rows,
    print_static_result_table,
    write_result_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index AgentGuard eval result summaries.")
    parser.add_argument(
        "output_root",
        nargs="?",
        default="results",
        help="Result root to scan. Defaults to results/.",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Only scan one level below output_root instead of recursive results/**.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write index files without printing the comparison table.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    rows = collect_result_summary_rows(output_root, recursive=not args.flat)
    if not rows:
        print(f"No results_summary.json files found under {output_root}")
        return 1

    written = write_result_index(output_root, rows)
    if not args.quiet:
        print_static_result_table(rows)
    if written is not None:
        csv_path, md_path = written
        print(f"Indexed {len(rows)} runs")
        print(f"  CSV: {csv_path}")
        print(f"  MD:  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
