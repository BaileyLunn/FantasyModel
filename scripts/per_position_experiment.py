#!/usr/bin/env python3
"""Pooled vs per-position models (pre-registered protocol; writes reports/per_position_<tag>.json).

Variants
  pooled   current production model: one HGB over QB/RB/WR/TE with position_code as a feature.
  separate one HGB per position (same features minus position_code). Hyper-parameters tuned per position
           on PRE-HOLDOUT data only: time-based CV, folds validate 2021, 2022, 2023 (train seasons < fold),
           metric = mean RMSE (all rows) of that position; single seed (--tune-seed).
  hybrid   per position, pooled or separate, whichever has the lower mean RMSE (over --seeds) when trained
           on seasons <= 2023 and validated on 2024. Locked before any 2025/2026 row is scored.
  posfeat  (optional d) separate per-position models that also see position-specific EWMA features
           (rush share / rush yds, target share, air-yards share, receiving air yards, pass attempts,
           pass air yards; features.position_specific_enabled), tuned the same way.

Checks (every variant, every seed in --seeds; sample weights = training.recency_half_life_seasons)
  2025 holdout     train seasons < 2025, score 2025 REG.
  2026 in-season   train seasons < 2026, score 2026 weeks in the panel.
  roster backtest  same thru-2025 model, every ACT roster QB/RB/WR/TE of 2026 weeks 1..N (Out/IR -> 0,
                   active no-stat players = 0), built exactly like project_week.

Decision rule (per position; PPR primary, half-PPR must not contradict):
  adopt a per-position model for P only if, vs pooled, on P's rows
    * 2025 holdout: mean RMSE (all rows) lower by more than seed noise (2 x sd of the per-seed paired
      difference), and MAE not higher;
    * 2026 in-season and roster backtest: mean RMSE not higher by more than seed noise.
  The primary candidate is the hybrid's locked choice (separate where it chose separate). posfeat is
  adopted where it passes and either nothing else qualified for P or it also beats separate on 2025.
  Otherwise P stays pooled.
  "adopted" = the resulting per-position mix (pooled / separate / posfeat per P), re-scored as its own
  variant with --reuse-tuning (same locked params and seeds, so the other variants reproduce exactly).

  python scripts/per_position_experiment.py --scoring ppr half_ppr --seeds 0 1 2 3 4
  python scripts/per_position_experiment.py --scoring ppr half_ppr --reuse-tuning   # + "adopted" variant
No production artifacts are touched.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from fantasy_model.config import load_config, profile_tag, scoring_dict, set_scoring_profile
from fantasy_model.data import load_player_games
from fantasy_model.features.pipeline import POSITION_SPECIFIC_COLUMNS, build_feature_matrix
from fantasy_model.model import POSITION_CODES, POSITIONS, make_model, position_cfg
from fantasy_model.project import actual_points, projection_features, ruled_out_mask
from fantasy_model.train import fit_weighted, impute
from fantasy_model.weights import recency_weights, training_recency

GRID = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.06],
    "min_samples_leaf": [25, 60, 150],
    "max_iter": [300, 600],
}
if __import__("os").environ.get("FM_PP_SMOKE"):   # tiny grid for a quick end-to-end check
    GRID = {"max_depth": [4], "learning_rate": [0.06], "min_samples_leaf": [25], "max_iter": [300]}
TUNE_FOLDS = [2021, 2022, 2023]
INNER_VAL = 2024


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return {"n": 0}
    return {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
            "rmse": float(mean_squared_error(y, p) ** 0.5),
            "r2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else None, "bias": float(np.mean(p - y))}


class Data:
    """Feature matrix for one profile (+ roster-backtest rows)."""

    def __init__(self, cfg, panel, roster_weeks):
        scoring = scoring_dict(cfg)
        raw_dir = Path(cfg["paths"]["raw_dir"])
        X_all, y, self.fcols_all = build_feature_matrix(panel, scoring, cfg)
        ps = set(POSITION_SPECIFIC_COLUMNS)
        self.base_cols = [i for i, c in enumerate(self.fcols_all) if c not in ps]           # pooled
        self.sep_cols = [i for i, c in enumerate(self.fcols_all) if c not in ps and c != "position_code"]
        self.posf_cols = [i for i, c in enumerate(self.fcols_all) if c != "position_code"]
        self.X = X_all[self.fcols_all].to_numpy(dtype=float)
        self.y = y.to_numpy(dtype=float)
        self.season = panel["season"].to_numpy()
        self.week = panel["week"].to_numpy()
        self.pos = panel["position"].astype(str).str.upper().to_numpy()
        self.lab = ~np.isnan(self.y)
        self.zs = pd.to_numeric(panel["zero_stat_appearance"], errors="coerce").fillna(0).to_numpy() == 1
        Xf, af, ro, pf = [], [], [], []
        for w in roster_weeks:
            _, out, feats = projection_features(2026, w, cfg, refresh_roster=False, hist=panel)
            act, played = actual_points(out, raw_dir, 2026, w, scoring)
            keep = played.to_numpy() & act.notna().to_numpy()
            Xf.append(feats[self.fcols_all].to_numpy(dtype=float)[keep])
            af.append(act.to_numpy()[keep])
            ro.append(ruled_out_mask(out).to_numpy()[keep])
            pf.append(out["position"].astype(str).str.upper().to_numpy()[keep])
        self.rX, self.ry, self.rout, self.rpos = map(np.concatenate, (Xf, af, ro, pf))

    def idx(self, lo_excl=None, eq=None):
        if eq is not None:
            return np.where(self.lab & (self.season == eq))[0]
        return np.where(self.lab & (self.season < lo_excl))[0]


def fit_pooled(cfg, D, tr, seed):
    c = copy.deepcopy(cfg)
    c["model"]["random_state"] = int(seed)
    Xtr = D.X[tr][:, D.base_cols]
    model, med, _ = fit_weighted(c, Xtr, D.y[tr], D.season[tr], D.week[tr])
    return lambda X: model.predict(impute(X[:, D.base_cols], med))


def fit_position(cfg, D, tr, p, params, cols, seed, med_full):
    """One position's model, trained exactly like fit_strategy (global medians of the training rows)."""
    c = copy.deepcopy(cfg)
    c["model"] = {**c["model"], **params, "random_state": int(seed)}
    rows = tr[D.pos[tr] == p]
    hl, wps = training_recency(cfg)
    w = recency_weights(D.season[rows], D.week[rows], hl, wps)
    m = make_model(c)
    m.fit(impute(D.X[rows], med_full)[:, cols], D.y[rows], sample_weight=w)
    return m


def train_medians(D, tr):
    med = np.nanmedian(D.X[tr], axis=0)
    return np.where(np.isnan(med), 0.0, med)


def fit_separate(cfg, D, tr, params_by_pos, cols, seed):
    med = train_medians(D, tr)
    models = {p: fit_position(cfg, D, tr, p, params_by_pos[p], cols, seed, med) for p in POSITIONS}

    def pred(X, pos):
        out = np.full(len(X), np.nan)
        Xi = impute(X, med)
        for p, m in models.items():
            k = pos == p
            if k.any():
                out[k] = m.predict(Xi[k][:, cols])
        return out
    return pred


def tune(cfg, D, cols, seed, log):
    """Per-position grid search on folds TUNE_FOLDS (train < fold); returns best params + table."""
    keys = list(GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    folds = []
    for f in TUNE_FOLDS:
        tr, va = D.idx(lo_excl=f), D.idx(eq=f)
        folds.append((tr, va, train_medians(D, tr)))
    best, table = {}, {}
    for p in POSITIONS:
        scores = []
        for params in combos:
            r = []
            for tr, va, med in folds:
                m = fit_position(cfg, D, tr, p, params, cols, seed, med)
                vp = va[D.pos[va] == p]
                r.append(mean_squared_error(D.y[vp], m.predict(impute(D.X[vp], med)[:, cols])) ** 0.5)
            scores.append((float(np.mean(r)), params))
        scores.sort(key=lambda t: t[0])
        best[p] = scores[0][1]
        table[p] = [{"cv_rmse": round(s, 4), **pr} for s, pr in scores[:5]]
        # the pooled hyper-parameters as a per-position model, for reference
        ref = {k: cfg["model"][k] for k in keys}
        table[p + "_pooled_params_cv_rmse"] = next((round(s, 4) for s, pr in scores if pr == ref), None)
        log(f"  tuned {p}: {best[p]} cv_rmse={scores[0][0]:.4f} (pooled params {table[p + '_pooled_params_cv_rmse']})")
    return best, table


def summarize(per_seed: list[dict]) -> dict:
    """mean/sd over seeds of nested metric dicts."""
    def rec(ds):
        if isinstance(ds[0], dict):
            return {k: rec([d[k] for d in ds]) for k in ds[0]}
        if ds[0] is None or isinstance(ds[0], int):
            return ds[0]
        a = np.array(ds, float)
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}
    return rec(per_seed)


def check_metrics(y, p, pos, zs=None) -> dict:
    out = {"ALL": metrics(y, p)}
    for q in POSITIONS:
        out[q] = metrics(y[pos == q], p[pos == q])
    if zs is not None:
        k = ~zs
        out["stats_rows"] = {"ALL": metrics(y[k], p[k]), **{q: metrics(y[k & (pos == q)], p[k & (pos == q)]) for q in POSITIONS}}
    return out


def paired_bootstrap(y, p_a, p_b, n=2000, seed=0) -> dict:
    """RMSE(b) - RMSE(a) with a player-game bootstrap 90% interval (sampling noise, not seed noise)."""
    rng = np.random.default_rng(seed)
    ea, eb = (p_a - y) ** 2, (p_b - y) ** 2
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(np.sqrt(eb[i].mean()) - np.sqrt(ea[i].mean()))
    return {"delta_rmse": float(np.sqrt(eb.mean()) - np.sqrt(ea.mean())),
            "ci90": [float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))]}


def run_profile(base, panel, prof, seeds, tune_seed, roster_weeks, do_posfeat, log, prev=None) -> dict:
    cfg = copy.deepcopy(set_scoring_profile(base, prof))
    cfg["features"]["position_specific_enabled"] = True   # built once; pooled/separate ignore those columns
    t0 = time.time()
    D = Data(cfg, panel, roster_weeks)
    log(f"[{prof}] features {len(D.fcols_all)} cols, roster rows {len(D.ry)} ({time.time() - t0:.0f}s)")

    res: dict = {"scoring_profile": prof, "seeds": seeds, "tune_seed": tune_seed, "grid": GRID,
                 "tune_folds": TUNE_FOLDS, "inner_validation": INNER_VAL,
                 "pooled_params": {k: cfg["model"][k] for k in GRID}}
    t0 = time.time()
    if prev is not None:   # locked params / hybrid choice / decision from an earlier full run
        for k in ("tuning_separate", "params_separate", "tuning_posfeat", "params_posfeat",
                  "inner_2024_rmse", "hybrid_choice"):
            if k in prev:
                res[k] = prev[k]
        sep_params, pf_params = prev["params_separate"], prev.get("params_posfeat")
        do_posfeat = pf_params is not None
        adopted = {p: prev["decision"][p]["adopt"] for p in POSITIONS}
        res["adopted_from_decision"] = adopted
        log(f"[{prof}] reusing tuning/hybrid/decision from previous run; adopted = {adopted}")
    else:
        adopted = None
        sep_params, res["tuning_separate"] = tune(cfg, D, D.sep_cols, tune_seed, log)
        res["params_separate"] = sep_params
        if do_posfeat:
            pf_params, res["tuning_posfeat"] = tune(cfg, D, D.posf_cols, tune_seed, log)
            res["params_posfeat"] = pf_params
        log(f"[{prof}] tuning done ({time.time() - t0:.0f}s)")

    # ---- hybrid selection: train <= 2023, validate 2024 (5 seeds)
    if prev is None:
        choice = inner_selection(cfg, D, seeds, sep_params, pf_params if do_posfeat else None, res, log, prof)
    else:
        choice = res["hybrid_choice"]
    return final_checks(cfg, D, seeds, sep_params, pf_params, do_posfeat, choice, adopted, res, log, prof)


def inner_selection(cfg, D, seeds, sep_params, pf_params, res, log, prof):
    do_posfeat = pf_params is not None
    tr, va = D.idx(lo_excl=INNER_VAL), D.idx(eq=INNER_VAL)
    inner = {v: {p: [] for p in POSITIONS} for v in ("pooled", "separate", "posfeat")}
    for s in seeds:
        pp = fit_pooled(cfg, D, tr, s)(D.X[va])
        ps = fit_separate(cfg, D, tr, sep_params, D.sep_cols, s)(D.X[va], D.pos[va])
        preds = {"pooled": pp, "separate": ps}
        if do_posfeat:
            preds["posfeat"] = fit_separate(cfg, D, tr, pf_params, D.posf_cols, s)(D.X[va], D.pos[va])
        for v, pr in preds.items():
            for p in POSITIONS:
                k = D.pos[va] == p
                inner[v][p].append(mean_squared_error(D.y[va][k], pr[k]) ** 0.5)
    choice = {p: ("separate" if np.mean(inner["separate"][p]) < np.mean(inner["pooled"][p]) else "pooled") for p in POSITIONS}
    res["inner_2024_rmse"] = {v: {p: {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1))} for p, x in d.items() if x}
                              for v, d in inner.items() if any(d.values())}
    res["hybrid_choice"] = choice
    log(f"[{prof}] hybrid (locked on 2024): {choice}  inner RMSE pooled/sep: " + ", ".join(
        f"{p} {np.mean(inner['pooled'][p]):.3f}/{np.mean(inner['separate'][p]):.3f}" for p in POSITIONS))
    return choice


def final_checks(cfg, D, seeds, sep_params, pf_params, do_posfeat, choice, adopted, res, log, prof):
    variants = ["pooled", "separate", "hybrid"] + (["posfeat"] if do_posfeat else []) + (["adopted"] if adopted else [])
    per_seed = {v: [] for v in variants}
    seed_preds = {v: {"h25": [], "s26": [], "roster": []} for v in variants}
    tr25, te25 = D.idx(lo_excl=2025), D.idx(eq=2025)
    tr26, te26 = D.idx(lo_excl=2026), D.idx(eq=2026)
    ro = D.rout
    for s in seeds:
        t1 = time.time()
        P = {}
        for label, tr_, te_ in (("h25", tr25, te25), ("s26", tr26, te26)):
            pool = fit_pooled(cfg, D, tr_, s)
            sep = fit_separate(cfg, D, tr_, sep_params, D.sep_cols, s)
            fits = {"pooled": lambda X, pos, f=pool: f(X), "separate": sep}
            if do_posfeat:
                fits["posfeat"] = fit_separate(cfg, D, tr_, pf_params, D.posf_cols, s)
            for v, f in fits.items():
                P.setdefault(v, {})[label] = f(D.X[te_], D.pos[te_])
                if label == "s26":
                    P[v]["roster"] = np.where(ro, 0.0, f(D.rX, D.rpos))
        # hybrid = per-position composition of pooled and separate predictions (identical to the router)
        P["hybrid"] = {}
        for label, pos in (("h25", D.pos[te25]), ("s26", D.pos[te26]), ("roster", D.rpos)):
            h = P["pooled"][label].copy()
            for p in POSITIONS:
                if choice[p] == "separate":
                    h[pos == p] = P["separate"][label][pos == p]
            P["hybrid"][label] = h
            if adopted:
                a_ = P["pooled"][label].copy()
                for p in POSITIONS:
                    if adopted[p] != "pooled":
                        a_[pos == p] = P[adopted[p]][label][pos == p]
                P.setdefault("adopted", {})[label] = a_
        for v in variants:
            per_seed[v].append({
                "holdout_2025": check_metrics(D.y[te25], P[v]["h25"], D.pos[te25], D.zs[te25]),
                "in_season_2026": check_metrics(D.y[te26], P[v]["s26"], D.pos[te26], D.zs[te26]),
                "roster_backtest_2026": check_metrics(D.ry, P[v]["roster"], D.rpos),
            })
            for k in seed_preds[v]:
                seed_preds[v][k].append(P[v][k])
        log(f"[{prof}] seed {s} done ({time.time() - t1:.0f}s): " + "  ".join(
            f"{v} 2025 {per_seed[v][-1]['holdout_2025']['ALL']['rmse']:.3f}" for v in variants))

    res["checks"] = {v: summarize(per_seed[v]) for v in variants}
    res["per_seed"] = per_seed
    # paired differences vs pooled (per seed) -> mean / sd
    diffs = {}
    for v in variants[1:]:
        diffs[v] = {}
        for chk in ("holdout_2025", "in_season_2026", "roster_backtest_2026"):
            diffs[v][chk] = {}
            for grp in ("ALL",) + POSITIONS:
                d = {}
                for m in ("mae", "rmse", "r2", "bias"):
                    a = np.array([per_seed[v][i][chk][grp][m] - per_seed["pooled"][i][chk][grp][m] for i in range(len(seeds))])
                    d[m] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}
                diffs[v][chk][grp] = d
    res["diff_vs_pooled"] = diffs
    # sampling noise (bootstrap on seed-averaged predictions)
    boot = {}
    ys = {"holdout_2025": (D.y[te25], D.pos[te25], "h25"), "in_season_2026": (D.y[te26], D.pos[te26], "s26"),
          "roster_backtest_2026": (D.ry, D.rpos, "roster")}
    for v in variants[1:]:
        boot[v] = {}
        for chk, (y, pos, key) in ys.items():
            pa, pb = np.mean(seed_preds["pooled"][key], axis=0), np.mean(seed_preds[v][key], axis=0)
            boot[v][chk] = {g: paired_bootstrap(y[k], pa[k], pb[k]) for g, k in
                            [("ALL", np.ones(len(y), bool))] + [(p, pos == p) for p in POSITIONS]}
    res["bootstrap_vs_pooled"] = boot
    res["decision"] = decide(res, [v for v in variants if v != "adopted"])
    return res


def decide(res, variants) -> dict:
    d = res["diff_vs_pooled"]
    out = {}
    for p in POSITIONS:
        verdicts = {}
        for v in variants[1:]:
            h = d[v]["holdout_2025"][p]
            s26, rb = d[v]["in_season_2026"][p], d[v]["roster_backtest_2026"][p]
            noise = lambda x: 2 * x["rmse"]["sd"]
            beats = h["rmse"]["mean"] < 0 and -h["rmse"]["mean"] > noise(h) and h["mae"]["mean"] <= 0
            ok26 = s26["rmse"]["mean"] <= noise(s26)
            okrb = rb["rmse"]["mean"] <= noise(rb)
            verdicts[v] = {"beats_2025": bool(beats), "ok_2026": bool(ok26), "ok_roster": bool(okrb),
                           "passes": bool(beats and ok26 and okrb),
                           "d_rmse_2025": round(h["rmse"]["mean"], 4), "noise_2025": round(noise(h), 4),
                           "d_rmse_2026": round(s26["rmse"]["mean"], 4), "noise_2026": round(noise(s26), 4),
                           "d_rmse_roster": round(rb["rmse"]["mean"], 4), "noise_roster": round(noise(rb), 4)}
        hy = res["hybrid_choice"][p]
        adopt = "pooled"
        if hy == "separate" and verdicts["separate"]["passes"]:
            adopt = "separate"
        if "posfeat" in verdicts and verdicts["posfeat"]["passes"] and (
                adopt == "pooled" or d["posfeat"]["holdout_2025"][p]["rmse"]["mean"] < d["separate"]["holdout_2025"][p]["rmse"]["mean"]):
            adopt = "posfeat"
        out[p] = {"hybrid_choice": hy, "verdicts": verdicts, "adopt": adopt}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scoring", nargs="+", default=["ppr", "half_ppr"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tune-seed", type=int, default=42)
    ap.add_argument("--roster-weeks", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--no-posfeat", action="store_true")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--reuse-tuning", action="store_true",
                    help="load locked params / hybrid choice / decision from reports/per_position_<tag>.json, "
                         "skip tuning, and add the 'adopted' variant")
    a = ap.parse_args(argv)
    log = lambda s: print(s, flush=True)
    base = load_config()
    panel, _ = load_player_games(base)
    panel = panel.reset_index(drop=True)
    for prof in a.scoring:
        out = Path(base["paths"]["reports_dir"]) / f"per_position_{profile_tag(set_scoring_profile(base, prof))}{a.suffix}.json"
        prev = json.loads(out.read_text()) if a.reuse_tuning else None
        if prev is not None and prev.get("seeds") != a.seeds:
            raise SystemExit(f"--reuse-tuning: seeds {a.seeds} differ from the previous run {prev.get('seeds')}")
        res = run_profile(base, panel, prof, a.seeds, a.tune_seed, a.roster_weeks, not a.no_posfeat, log, prev)
        out.write_text(json.dumps(res, indent=1))
        log(f"wrote {out}")
        for p, dd in res["decision"].items():
            log(f"  {prof} {p}: adopt={dd['adopt']} hybrid={dd['hybrid_choice']} " + json.dumps(dd["verdicts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def render_tables(res: dict, variants=("pooled", "separate", "hybrid", "posfeat", "adopted")) -> str:
    """Markdown tables (mean over seeds, RMSE ± seed sd) for one profile's results."""
    vs = [v for v in variants if v in res["checks"]]
    names = {"holdout_2025": "2025 holdout (train <2025)", "in_season_2026": "2026 wk1-2 (train <2026)",
             "roster_backtest_2026": "Roster backtest 2026 wk1-2 (train <2026, all ACT players)"}
    lines = []
    for chk, title in names.items():
        lines += [f"**{title}** - MAE / RMSE ± seed sd / R² / bias (all rows)", "",
                  "| group | n | " + " | ".join(vs) + " |", "|---|---|" + "---|" * len(vs)]
        for g in ("ALL",) + POSITIONS:
            cells = []
            for v in vs:
                m = res["checks"][v][chk][g]
                cells.append(f"{m['mae']['mean']:.3f} / {m['rmse']['mean']:.3f} ± {m['rmse']['sd']:.3f} / "
                             f"{m['r2']['mean']:.3f} / {m['bias']['mean']:+.2f}")
            lines.append(f"| {g} | {res['checks']['pooled'][chk][g]['n']} | " + " | ".join(cells) + " |")
        if chk != "roster_backtest_2026":
            st = res["checks"]["pooled"][chk]["stats_rows"]["ALL"]["n"]
            cells = []
            for v in vs:
                m = res["checks"][v][chk]["stats_rows"]["ALL"]
                cells.append(f"{m['mae']['mean']:.3f} / {m['rmse']['mean']:.3f} ± {m['rmse']['sd']:.3f} / {m['r2']['mean']:.3f} / {m['bias']['mean']:+.2f}")
            lines.append(f"| ALL, stats rows only | {st} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["**RMSE difference vs pooled** (mean of per-seed paired differences ± sd; negative = better). "
              "Bootstrap 90% CI (player-game resampling of seed-averaged predictions) in brackets.", "",
              "| variant | group | 2025 | 2026 | roster |", "|---|---|---|---|---|"]
    for v in vs[1:]:
        for g in ("ALL",) + POSITIONS:
            cells = []
            for chk in names:
                d = res["diff_vs_pooled"][v][chk][g]["rmse"]
                b = res["bootstrap_vs_pooled"][v][chk][g]["ci90"]
                cells.append(f"{d['mean']:+.3f} ± {d['sd']:.3f} [{b[0]:+.3f}, {b[1]:+.3f}]")
            lines.append(f"| {v} | {g} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
