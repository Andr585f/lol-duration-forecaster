"""Command line: `python -m forecaster.cli demo | evaluate --csv FILE | serve`."""
from __future__ import annotations

import argparse
import json
import os

from . import evaluation, nowcast
from .data import from_oracles_elixir
from .synthetic import generate_games


def _load(csv: str | None):
    if csv:
        return from_oracles_elixir(csv)
    return generate_games()


def cmd_evaluate(args) -> dict:
    games = _load(args.csv)
    report = evaluation.full_report(games, n_boot=args.boot)
    res = evaluation.walk_forward(games)
    report["nowcast"] = nowcast.evaluate(res)
    print(json.dumps(report, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="forecaster")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="walk-forward evaluation with bootstrap CIs")
    ev.add_argument("--csv", help="Oracle's Elixir CSV; omit to use synthetic data")
    ev.add_argument("--boot", type=int, default=1000)
    ev.add_argument("--out", help="write JSON report here")
    sub.add_parser("demo", help="evaluate on synthetic data and save reports/demo_evaluation.json")
    sv = sub.add_parser("serve", help="run the local web app")
    sv.add_argument("--csv")
    sv.add_argument("--port", type=int, default=5050)
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        args.csv, args.boot, args.out = None, 1000, "reports/demo_evaluation.json"
        cmd_evaluate(args)
    elif args.cmd == "evaluate":
        cmd_evaluate(args)
    else:
        from .app import create_app
        create_app(csv=args.csv).run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
