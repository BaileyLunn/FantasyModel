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
    parser.add_argument(
        "--scoring",
        default=None,
        help="Scoring profile: half_ppr (config default) | ppr | standard. Selects the scoring map, "
        "model artifact (models/fantasy_hgb_{half,ppr,std}.joblib) and output file suffixes.",
    )
    parser.add_argument(
        "--per-position",
        default=None,
        help="Override model.strategy for train: none|pooled (one model), all (a model per position), "
        "or a list like QB,TE (hybrid: those positions get their own model, the rest use the pooled one).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train model with time-based split")
    t.add_argument("--test-season", type=int, default=None, help="Override model.test_season (validation holdout)")
    t.add_argument("--refit-full", dest="refit_full", action="store_true", default=None)
    t.add_argument("--no-refit-full", dest="refit_full", action="store_false")
    t.add_argument("--model-out", type=str, default=None)
    t.add_argument("--no-reports", action="store_true", help="Do not overwrite reports/train_metrics.json etc.")
    e = sub.add_parser("evaluate", help="Evaluate saved model (holdout season, or --season/--weeks slice)")
    e.add_argument("--season", type=int, default=None, help="Score this season's played games instead of test_season")
    e.add_argument("--weeks", type=int, nargs="*", default=None)
    e.add_argument("--model", type=str, default=None, help="Model artifact path (default models/fantasy_hgb_<profile>.joblib)")
    e.add_argument("--out", type=str, default=None, help="Write JSON report here")

    p = sub.add_parser("predict", help="Predict a season/week")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--player", type=str, default=None)

    pj = sub.add_parser("project", help="Forward projections for an upcoming week (roster + schedule)")
    pj.add_argument("--season", type=int, required=True)
    pj.add_argument("--week", type=int, required=True)
    pj.add_argument("--model", type=str, default=None)
    pj.add_argument("--no-refresh-roster", action="store_true")
    pj.add_argument("--out", type=str, default=None)
    pj.add_argument("--no-drivers", action="store_true", help="Skip per-row top-feature drivers")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    from fantasy_model.config import apply_per_position_override, load_config, set_scoring_profile

    cfg = apply_per_position_override(set_scoring_profile(load_config(args.config), args.scoring), args.per_position)

    if args.command == "train":
        from fantasy_model.train import train_model

        metrics = train_model(
            cfg=cfg,
            prefer_sample=args.sample,
            test_season=args.test_season,
            refit_full=args.refit_full,
            model_out=args.model_out,
            write_reports=not args.no_reports,
        )
        printable = {k: v for k, v in metrics.items() if k != "feature_cols"}
        print(json.dumps(printable, indent=2))
        return 0

    if args.command == "evaluate":
        from fantasy_model.evaluate import evaluate_model, evaluate_slice

        if args.season is not None:
            report = evaluate_slice(
                args.season, args.weeks, cfg=cfg, model_path=args.model, out_path=args.out
            )
        else:
            report = evaluate_model(cfg=cfg, prefer_sample=args.sample, model_path=args.model)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "predict":
        from fantasy_model.predict import predict_week

        df = predict_week(
            season=args.season,
            week=args.week,
            player=args.player,
            prefer_sample=args.sample,
            cfg=cfg,
        )
        if df.empty:
            print(json.dumps({"error": "no rows matched", "season": args.season, "week": args.week}))
            return 1
        print(df.head(50).to_string(index=False))
        print(f"\nrows={len(df)}")
        return 0

    if args.command == "project":
        from fantasy_model.project import project_week

        df = project_week(
            args.season, args.week, cfg=cfg, model_path=args.model,
            refresh_roster=not args.no_refresh_roster, out_path=args.out, drivers=not args.no_drivers,
        )
        print(df.head(40).to_string(index=False))
        print(f"\nrows={len(df)} -> {df.attrs.get('out_path')}")
        return 0

    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
