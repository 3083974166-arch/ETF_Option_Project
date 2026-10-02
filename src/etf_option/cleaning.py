"""Point-in-time contract mapping and auditable row rejection."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .pricing import bounds, implied_vol, greeks


@dataclass(frozen=True)
class Filters:
    min_days: int = 7
    min_volume: int = 10
    min_oi: int = 0  # Missing OI is permitted only when this filter is disabled.
    max_relative_spread: float = .5
    min_price: float = .0001
    max_abs_log_moneyness: float = .25


def map_contracts(quotes, contracts):
    q, c = quotes.copy(), contracts.rename(columns={"source": "contract_source"}).copy()
    for col in ["date"]:
        q[col] = pd.to_datetime(q[col])
    for col in ["valid_from", "valid_to", "expiry"]:
        c[col] = pd.to_datetime(c[col])
    if q.duplicated(["date", "contract_id"]).any():
        raise ValueError("Duplicate date/contract quote; resolve source priority explicitly")
    q["_row"] = range(len(q))
    m = q.merge(c, on="contract_id", how="left", validate="many_to_many")
    m = m[(m.date >= m.valid_from) & (m.date <= m.valid_to)]
    if m._row.duplicated().any() or len(m) != len(q):
        raise ValueError("Missing or overlapping effective contract mapping")
    return m.drop(columns="_row")


def prepare(quotes, contracts, underlying, rates, filters=Filters()):
    p = map_contracts(quotes, contracts)
    u = underlying.rename(columns={"source": "underlying_source", "is_synthetic": "underlying_is_synthetic"}).copy()
    r = rates.rename(columns={"source": "rate_source"}).copy()
    u["date"], r["date"] = pd.to_datetime(u.date), pd.to_datetime(r.date)
    r["available_date"] = pd.to_datetime(r.available_date)
    if u.date.duplicated().any() or r.date.duplicated().any():
        raise ValueError("Duplicate underlying/rate date")
    if r[["date", "available_date"]].isna().any().any() or (r.available_date > r.date).any():
        raise ValueError("Rate unavailable at valuation date")
    p = p.merge(u, on="date", how="left", validate="many_to_one")
    p = p.merge(r, on="date", how="left", validate="many_to_one")
    for col in ["S", "K", "r", "multiplier", "close", "bid", "ask", "volume", "open_interest"]:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    adjusted = p.adjusted.astype(str).str.lower()
    if not adjusted.isin(["true", "false"]).all():
        raise ValueError("adjusted must be True/False")
    p["adjusted"] = adjusted.eq("true")
    p["days"] = (p.expiry-p.date).dt.days
    p["T"] = p.days/365.0
    p["moneyness"] = p.S/p.K
    p["log_moneyness"] = np.log(p.K/p.S)
    p["has_quote"] = p.bid.notna() & p.ask.notna()
    p["mark"] = np.where(p.has_quote, (p.bid+p.ask)/2, p.close)
    p["reason"] = ""
    def reject(mask, label):
        p.loc[mask, "reason"] += label+";"
    reject(p[["S", "K", "T", "r", "mark", "volume", "multiplier"]].isna().any(axis=1), "missing")
    reject(~np.isfinite(p[["S", "K", "T", "r", "mark", "volume", "multiplier"]]).all(axis=1), "nonfinite")
    reject((p.S <= 0) | (p.K <= 0) | (p.multiplier <= 0), "invalid_contract")
    reject(~p.cp.isin(["C", "P"]) | p.adjusted.astype(bool), "unsupported_contract")
    reject(p.days < filters.min_days, "near_expiry")
    reject(p.volume < filters.min_volume, "volume")
    if filters.min_oi > 0:
        reject(p.open_interest.isna() | (p.open_interest < filters.min_oi), "oi")
    reject(p.mark < filters.min_price, "price")
    reject(p.bid.isna() != p.ask.isna(), "partial_quote")
    reject(p.has_quote & ((p.bid <= 0) | (p.ask < p.bid) |
           ((p.ask-p.bid)/p.mark > filters.max_relative_spread)), "spread")
    reject(p.log_moneyness.abs() > filters.max_abs_log_moneyness, "moneyness")
    return p[p.reason == ""].copy(), p[p.reason != ""].copy()


def infer_forward(panel, min_pairs=2, max_dispersion=.015):
    """Median parity forward across synchronized C/P pairs, by date/expiry.

    Matched units and strikes only; no q=0 fallback. Dispersion is a diagnostic,
    not an assertion of static-arbitrage-free surfaces.
    """
    records = []
    keys = ["date", "expiry", "K", "multiplier"]
    if panel.duplicated(keys+["cp"]).any():
        raise ValueError("Ambiguous same-strike contracts; choose one canonical standard series")
    cp = panel[panel.cp == "C"].merge(panel[panel.cp == "P"], on=keys, suffixes=("_c", "_p"))
    cp = cp[np.isclose(cp.S_c, cp.S_p) & np.isclose(cp.r_c, cp.r_p)]
    cp["F_pair"] = cp.K + np.exp(cp.r_c*cp.T_c)*(cp.mark_c-cp.mark_p)
    for (date, expiry), g in cp.groupby(["date", "expiry"]):
        F = g.F_pair.median()
        dispersion = (g.F_pair.max()-g.F_pair.min())/F if F > 0 else np.inf
        if len(g) < min_pairs or F <= 0 or dispersion > max_dispersion:
            continue
        a = g.iloc[0]
        records.append(dict(date=date, expiry=expiry, F=F,
                            q=a.r_c-np.log(F/a.S_c)/a.T_c,
                            parity_pairs=len(g), parity_dispersion=dispersion))
    return pd.DataFrame(records, columns=["date", "expiry", "F", "q", "parity_pairs", "parity_dispersion"])


def enrich(panel, forwards):
    p = panel.merge(forwards, on=["date", "expiry"], how="left", validate="many_to_one")
    rows, rejected = [], []
    for rec in p.to_dict("records"):
        try:
            if not np.isfinite(rec["q"]):
                raise ValueError("no_parity_forward")
            args = {k: rec[k] for k in ["S", "K", "T", "r", "q", "cp"]}
            low, high = bounds(**args)
            if not low < rec["mark"] < high:
                raise ValueError("no_arbitrage_or_zero_time_value")
            iv = implied_vol(rec["mark"], **args)
            if iv < .005 or iv > 3:
                raise ValueError("extreme_iv")
            rec.update(iv=iv, log_forward_moneyness=np.log(rec["K"]/rec["F"]))
            rec.update(greeks(sigma=iv, **args))
            rows.append(rec)
        except (ValueError, RuntimeError) as exc:
            rec["reason"] = str(exc)
            rejected.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(rejected)

