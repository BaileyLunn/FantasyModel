"""CLI entry: python -m fantasy_model train|evaluate|predict ..."""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fantasy_model", description="NFL fantasy points prediction")
    parser.add_argument("--config", default=None, help="Path to YAML/JSON config")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Prefer data/sample fixtures (offline smoke)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Train model with time-based split")
    sub.add_parser("evaluate", help="Evaluate saved model")

    p = sub.add_parser("predict", help="Predict a season/week")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--player", type=str, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        from fantasy_model.train import train_model

        metrics = train_model(config_path=args.config, prefer_sample=args.sample)
        printable = {k: v for k, v in metrics.items() if k != "feature_cols"}
        print(json.dumps(printable, indent=2))
        return 0

    if args.command == "evaluate":
        from fantasy_model.evaluate import evaluate_model

        report = evaluate_model(config_path=args.config, prefer_sample=args.sample)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "predict":
        from fantasy_model.predict import predict_week

        df = predict_week(
            season=args.season,
            week=args.week,
            player=args.player,
            prefer_sample=args.sample,
            config_path=args.config,
        )
        if df.empty:
            print(json.dumps({"error": "no rows matched", "season": args.season, "week": args.week}))
            return 1
        print(df.head(50).to_string(index=False))
        print(f"\nrows={len(df)}")
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
