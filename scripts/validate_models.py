#!/usr/bin/env python3
"""Standard validation for one or more scoring profiles (writes reports/validation_<tag>.json).

For each profile:
  * 2025 holdout: train seasons 2016–2024, score 2025 REG (all weeks).
  * 2026 in-season: train seasons 2016–2025, score 2026 weeks present in the panel.
Optional ``--ablate-usage`` also runs both with the depth/snap usage features disabled.
Artifacts are written to models/validation/ so production models are not touched.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fantasy_model.config import load_config, profile_tag, set_scoring_profile
from fantasy_model.evaluate import evaluate_slice
from fantasy_model.train import train_model


def run(cfg: dict, label: str, out_dir: Path) -> dict:
    tag = profile_tag(cfg)
    res: dict = {}
    m = train_model(cfg=cfg, test_season=2025, refit_full=False,
                    model_out=str(out_dir / f"{label}_{tag}_thru2024.joblib"), write_reports=False)
    res["holdout_2025"] = {k: m.get(k) for k in ("n_train", "n_test", "mae", "rmse", "r2", "by_position", "interval_holdout_coverage", "stats_rows_only", "zero_stat_rows_in_test")}
    p = out_dir / f"{label}_{tag}_thru2025.joblib"
    m2 = train_model(cfg=cfg, test_season=2026, refit_full=False, model_out=str(p), write_reports=False)
    rep = evaluate_slice(2026, None, cfg=cfg, model_path=str(p))
    res["in_season_2026"] = {"n_train": m2["n_train"], "weeks": rep["weeks"], **rep["overall"],
                             "by_position": rep["by_position"], "stats_rows_only": rep.get("stats_rows_only"), "by_week": {w: {k: v for k, v in d.items() if k != "by_position"} for w, d in rep["by_week"].items()}}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", nargs="+", default=["half_ppr", "ppr"])
    ap.add_argument("--ablate-usage", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--panel-note", default=None)
    ap.add_argument("--suffix", default="", help="append to report file name")
    a = ap.parse_args(argv)
    base = load_config(a.config)
    out_dir = Path(base["paths"]["models_dir"]) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for prof in a.scoring:
        cfg = set_scoring_profile(base, prof)
        result = {"scoring_profile": prof, "full": run(cfg, "full", out_dir)}
        if a.ablate_usage and a.panel_note:
            result["panel_note"] = a.panel_note
        if a.ablate_usage:
            c2 = copy.deepcopy(cfg)
            c2.setdefault("features", {})["usage_enabled"] = False
            result["no_usage"] = run(c2, "nousage", out_dir)
        path = Path(base["paths"]["reports_dir"]) / f"validation_{profile_tag(cfg)}{a.suffix}.json"
        path.write_text(json.dumps(result, indent=2))
        for k, v in result.items():
            if not isinstance(v, dict):
                continue
            for sl, d in v.items():
                c = d.get("stats_rows_only") or {}
                print(prof, k, sl, "all:", {m: d.get(m) for m in ("mae", "rmse", "r2")},
                      "stats_rows:", {m: c.get(m) for m in ("n", "mae", "rmse", "r2")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
