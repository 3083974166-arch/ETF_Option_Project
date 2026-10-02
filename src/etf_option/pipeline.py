from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from .cleaning import Filters, prepare, infer_forward, enrich
from .research import atm_term, constant_maturity, features, forecast_study, purged_split
from .backtest import Config, run
from .plots import make_plots
from .data import save_json


def read_inputs(root):
    root = Path(root)
    frames = {}
    for name in ["options", "contracts", "underlying", "rates"]:
        path = root/f"{name}.csv"
        frames[name] = pd.read_csv(path, dtype={"contract_id": str})
    for name in ["options", "contracts", "underlying", "rates"]:
        d = frames[name]
        if "source" not in d or d.source.isna().any():
            raise ValueError(f"{name}: missing source provenance")
    for name in ["options", "underlying"]:
        d = frames[name]
        if "is_synthetic" not in d or not d.is_synthetic.astype(str).str.lower().eq("false").all():
            raise ValueError("Formal pipeline accepts only explicitly labelled real data. Synthetic fixtures stay in tests.")
    if frames["options"].empty:
        raise ValueError("Empty option panel")
    return frames


def analyze(frames, filters):
    p, rejected = prepare(frames["options"], frames["contracts"], frames["underlying"], frames["rates"], filters)
    forward = infer_forward(p)
    enriched, invalid = enrich(p, forward)
    if enriched.empty:
        raise ValueError("No valid IV rows: inspect mapping, liquidity and parity pairs")
    term = atm_term(enriched)
    if term.empty:
        raise ValueError("No bracketed ATM observations")
    atm = constant_maturity(term)
    feat = features(frames["underlying"], atm)
    if feat.atm_iv.notna().sum() < 10:
        raise ValueError("Insufficient 30-day bracketed ATM history")
    return enriched, forward, pd.concat([rejected, invalid], ignore_index=True), term, feat


def pipeline(input_dir, output_dir, split_date, config=Config(), filters=Filters(), charts=True):
    inp, out = Path(input_dir), Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output run directory must be empty; preserve previous results")
    frames = read_inputs(inp)
    out.mkdir(parents=True, exist_ok=True)
    # Also label test orchestration truthfully if a unit test injects frames.
    # read_inputs rejects these in every production CLI path.
    synthetic = frames["options"].is_synthetic.astype(str).str.lower().eq("true").any()
    execution = "CLOSE_PROXY_COST_SCENARIO" if config.price_mode == "proxy" else "QUOTED_RESEARCH_SIMULATION"
    manifest = {"status": "running", "data_kind": "SYNTHETIC_TEST_ONLY" if synthetic else "REAL_SOURCE_UNVERIFIED",
                "execution": "SYNTHETIC_TEST_ONLY: "+execution if synthetic else execution,
                "config": asdict(config), "filters": asdict(filters), "split_date": split_date,
                "python": platform.python_version(),
                "versions": {k: importlib.metadata.version(k) for k in ["numpy", "pandas", "scipy", "statsmodels"]},
                "inputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inp.glob("*.csv")}}
    save_json(out/"manifest.json", manifest)
    try:
        p, forward, rejected, term, feat = analyze(frames, filters)
        for name, df in [("iv_panel", p), ("forwards", forward), ("rejections", rejected), ("atm_term", term), ("features", feat)]:
            df.to_csv(out/f"{name}.csv", index=False)
        cut = pd.Timestamp(split_date)
        if not feat.date.min() < cut < feat.date.max():
            raise ValueError("Split must be inside sample")
        # End IS one observation before OOS. Each segment begins flat and ends flat.
        is_end = feat.loc[feat.date < cut, "date"].max()
        train_ledger, train_metrics = run(p, feat, config, end=is_end)
        oos_ledger, oos_metrics = run(p, feat, config, start=cut)
        train_ledger.to_csv(out/"ledger_is.csv", index=False)
        oos_ledger.to_csv(out/"ledger_oos.csv", index=False)
        study, predictions = forecast_study(feat, split_date)
        predictions.to_csv(out/"oos_forecasts.csv", index=False)
        save_json(out/"forecast_study.json", study)
        save_json(out/"performance.json", {"is": train_metrics, "oos": oos_metrics,
                  "label": manifest["execution"], "note": "Hypothetical historical simulation; execution and borrow assumptions are not verified."})
        sensitivity = []
        variants = [("base", config, filters),
                    ("cost_x2", replace(config, option_fee=2*config.option_fee, option_slip=2*config.option_slip,
                     option_halfspread_proxy=2*config.option_halfspread_proxy, etf_cost_bps=2*config.etf_cost_bps,
                     borrow_rate=2*config.borrow_rate, funding_rate=2*config.funding_rate), filters),
                    ("hedge_2d", replace(config, hedge_every=2), filters),
                    ("hedge_5d", replace(config, hedge_every=5), filters),
                    ("threshold_0", replace(config, threshold=0), filters),
                    ("threshold_0.02", replace(config, threshold=.02), filters),
                    ("hold_5d", replace(config, max_hold=5), filters),
                    ("volume_50", config, replace(filters, min_volume=50)),
                    ("spread_025", config, replace(filters, max_relative_spread=.25))]
        for name, cfg, flt in variants:
            try:
                pp, ff = p, feat
                if flt != filters:
                    pp, _, _, _, ff = analyze(frames, flt)
                # Quoted half-spread also doubles in transaction-cost stress.
                if name == "cost_x2":
                    pp = pp.copy()
                    half = (pp.ask-pp.bid)/2
                    pp["bid"], pp["ask"] = pp.mark-2*half, pp.mark+2*half
                _, met = run(pp, ff, cfg, start=cut)
                sensitivity.append({"scenario": name, "status": "ok", **met})
            except ValueError as exc:
                sensitivity.append({"scenario": name, "status": "failed", "reason": str(exc)})
        pd.DataFrame(sensitivity).to_csv(out/"robustness_oos.csv", index=False)
        if charts:
            make_plots(p, term, feat, oos_ledger, out/"figures", manifest["execution"])
        manifest["status"] = "complete"
        manifest["rows"] = {"valid": len(p), "rejected": len(rejected), "dates": p.date.nunique()}
    except Exception as exc:
        manifest["status"], manifest["error"] = "failed", str(exc)
        save_json(out/"manifest.json", manifest)
        raise
    save_json(out/"manifest.json", manifest)
    return manifest
