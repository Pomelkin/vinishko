"""Command-line entry point for the AI sommelier evaluation package."""

from __future__ import annotations

import argparse

from ai_somelier.eval.common import EvaluationError
from ai_somelier.eval import metrics
from ai_somelier.eval import scaffold


def main() -> int:
    """Dispatch scaffold and metrics subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scaffold_parser = subparsers.add_parser("scaffold", help="create a draft evaluation")
    scaffold.configure_parser(scaffold_parser)
    scaffold_parser.set_defaults(handler=scaffold.run_cli)
    metrics_parser = subparsers.add_parser("metrics", help="validate and score an evaluation")
    metrics.configure_parser(metrics_parser)
    metrics_parser.set_defaults(handler=metrics.run_cli)
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (EvaluationError, OSError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
