"""Run the Vineliner pipeline on one area.

    python src/run_pipeline.py configs/bordeaux.toml
    python src/run_pipeline.py configs/bordeaux.toml --blocks 0 3 --name trial
    python src/run_pipeline.py configs/bordeaux.toml --plan

Config files after the first are merged on top in order, over configs/default.toml.
Outputs go to runs/<timestamp>[-name]/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import tools  # noqa: F401  (installed with `pip install -e .`)
except ModuleNotFoundError:  # running from a checkout without installing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.config import ConfigError, load_config, merge
from tools.pipeline import plan_run, run_pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=Path, help="per-area config file(s), merged over configs/default.toml")
    parser.add_argument("--blocks", nargs="+", type=int, help="process only these block ids")
    parser.add_argument("--name", help="suffix for the run directory name")
    parser.add_argument("--plan", action="store_true", help="print centre and hypothesis counts and exit without searching")
    parser.add_argument("--batch-size", type=int, default=64, help="centres scored per batch (memory vs speed)")
    args = parser.parse_args(argv)

    config = load_config(*args.configs)
    if args.blocks is not None:
        config = merge(config, {"vineyards": {"blocks": args.blocks}})
    try:
        if args.plan:
            print(json.dumps(plan_run(config), indent=2))
        else:
            run_pipeline(config, name=args.name, batch_size=args.batch_size)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
