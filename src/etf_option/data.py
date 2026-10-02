"""Public-source acquisition with raw retention, bounded requests, provenance.

Historical mappings MUST be supplied independently. Current listed contracts
cannot reconstruct a survivorship-free historical universe.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import pandas as pd
import numpy as np


def save_json(path, obj):
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite_json(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(finite_json(obj), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def bounded_ak(function, kwargs, output, timeout=35):
    """Isolated child gives even AKShare calls without timeout a hard deadline."""
    result = subprocess.run([sys.executable, "-m", "etf_option.data", function,
                             json.dumps(kwargs), str(output)], capture_output=True,
                            text=True, timeout=timeout, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr[-1600:])
    return pd.read_csv(output, dtype={"期权代码": str})


def probe(outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("sina_option_history", "option_sse_daily_sina", {"symbol": "10003889"}),
            ("sina_current_months", "option_sse_list_sina", {"symbol": "50ETF", "exchange": "null"}),
            ("etf_daily", "fund_etf_hist_em", {"symbol": "510050", "period": "daily", "start_date": "20250101", "end_date": "20250930", "adjust": ""})]
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "real_market_conclusions": False, "checks": []}
    for label, function, kwargs in jobs:
        path = out/(label+".csv")
        entry = {"name": label, "function": function, "parameters": kwargs}
        try:
            frame = bounded_ak(function, kwargs, path)
            entry.update(status="success" if len(frame) else "empty", rows=len(frame), columns=list(frame.columns),
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except (RuntimeError, subprocess.TimeoutExpired, pd.errors.EmptyDataError) as exc:
            entry.update(status="failed", error=str(exc))
        report["checks"].append(entry)
        save_json(out/"availability.json", report)
    return report


def download_history(contracts_path, outdir):
    """Download real prices for a verified user-supplied point-in-time manifest."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    contracts = pd.read_csv(contracts_path, dtype={"contract_id": str})
    required = {"contract_id", "cp", "K", "expiry", "multiplier", "adjusted", "valid_from", "valid_to", "source"}
    if not required.issubset(contracts.columns) or contracts.empty:
        raise ValueError("Supply nonempty contracts.csv matching docs/数据规范.md")
    results, audit = [], []
    for cid in contracts.contract_id.unique():
        if not str(cid).isdigit():
            raise ValueError("Expected numeric exchange contract IDs")
        path = out/f"sina_{cid}.csv"
        try:
            d = bounded_ak("option_sse_daily_sina", {"symbol": cid}, path)
            d = d.rename(columns={"日期": "date", "收盘": "close", "成交量": "volume"})
            if not {"date", "close", "volume"}.issubset(d) or d.empty:
                raise ValueError("Empty history or changed upstream schema")
            d = d[["date", "close", "volume"]].copy()
            d["contract_id"] = cid
            d["bid"], d["ask"], d["open_interest"] = np.nan, np.nan, np.nan
            d["source"] = "AKShare/Sina daily; no historical quote or OI"
            d["is_synthetic"] = False
            results.append(d)
            audit.append({"contract_id": cid, "status": "success", "rows": len(d), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        except Exception as exc:
            audit.append({"contract_id": cid, "status": "failed", "error": str(exc)})
    save_json(out/"download_audit.json", {"retrieved_at": datetime.now(timezone.utc).isoformat(), "contracts": audit})
    if any(x["status"] != "success" for x in audit):
        raise RuntimeError("Partial acquisition: audit saved; no research panel emitted")
    pd.concat(results, ignore_index=True).to_csv(out/"options.csv", index=False)


def download_underlying(start, end, dividends_path, outdir):
    """Unadjusted tradable close + explicitly supplied cash distribution events.

    An empty dividend file asserts the requested interval has NO distributions;
    that assertion must be verified from fund announcements by the researcher.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    d = bounded_ak("fund_etf_hist_em", {"symbol": "510050", "period": "daily",
                   "start_date": start, "end_date": end, "adjust": ""}, out/"etf_unadjusted.csv")
    d = d.rename(columns={"日期": "date", "收盘": "S"})[["date", "S"]].copy()
    d["date"] = pd.to_datetime(d.date)
    d = d.sort_values("date")
    events = pd.read_csv(dividends_path)
    if not {"date", "cash_dividend", "source"}.issubset(events.columns):
        raise ValueError("Dividend CSV requires date,cash_dividend,source; one event per ex-date")
    events["date"] = pd.to_datetime(events.date)
    if events.date.duplicated().any() or (events.cash_dividend < 0).any() or events.source.isna().any():
        raise ValueError("Invalid dividend events")
    if not set(events.date).issubset(set(d.date)):
        raise ValueError("Dividend date not in downloaded sample")
    d = d.merge(events[["date", "cash_dividend"]], on="date", how="left", validate="one_to_one")
    d["cash_dividend"] = d.cash_dividend.fillna(0)
    growth = ((d.S+d.cash_dividend)/d.S.shift()).fillna(1)
    d["total_return_close"] = d.S.iloc[0]*growth.cumprod()
    d["source"], d["is_synthetic"] = "AKShare/Eastmoney + supplied dividend events", False
    d.to_csv(out/"underlying.csv", index=False)
    save_json(out/"underlying_audit.json", {"retrieved_at": datetime.now(timezone.utc).isoformat(),
              "start": start, "end": end, "rows": len(d), "dividend_events": len(events),
              "dividend_file_sha256": hashlib.sha256(Path(dividends_path).read_bytes()).hexdigest(),
              "raw_sha256": hashlib.sha256((out/"etf_unadjusted.csv").read_bytes()).hexdigest()})


def official_validate(contracts_path, official_path, output):
    """Compare normalized SSE daily contract export against acquisition mapping.

    Official CSV format: asof,contract_id,cp,K,expiry,multiplier. Effective mapping
    is checked per asof; official upload/hash is retained by the caller.
    """
    c = pd.read_csv(contracts_path, dtype={"contract_id": str})
    official = pd.read_csv(official_path, dtype={"contract_id": str})
    for col in ["valid_from", "valid_to"]:
        c[col] = pd.to_datetime(c[col])
    official["asof"] = pd.to_datetime(official["asof"])
    records = []
    for row in official.to_dict("records"):
        match = c[(c.contract_id == row["contract_id"]) & (c.valid_from <= row["asof"]) & (c.valid_to >= row["asof"])]
        bad = []
        if len(match) != 1:
            bad.append("mapping_missing_or_ambiguous")
        else:
            a = match.iloc[0]
            for field in ["K", "multiplier"]:
                if not np.isclose(float(a[field]), float(row[field]), rtol=0, atol=1e-8):
                    bad.append(field)
            if a.cp != row["cp"]:
                bad.append("cp")
            if pd.Timestamp(a.expiry) != pd.Timestamp(row["expiry"]):
                bad.append("expiry")
        records.append({"asof": row["asof"], "contract_id": row["contract_id"], "mismatches": ";".join(bad)})
    result = pd.DataFrame(records)
    result.to_csv(output, index=False)
    return result


if __name__ == "__main__":
    import akshare as ak
    function, kwargs, output = sys.argv[1:]
    allowed = {"option_sse_daily_sina", "option_sse_list_sina", "fund_etf_hist_em"}
    if function not in allowed:
        raise ValueError("Function not permitted")
    frame = getattr(ak, function)(**json.loads(kwargs))
    if isinstance(frame, list):
        frame = pd.DataFrame({"contract_month": frame})
    frame.to_csv(output, index=False, encoding="utf-8-sig")
