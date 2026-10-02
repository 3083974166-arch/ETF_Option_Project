"""Conservative daily event simulation with next-close execution.

Only long straddles, avoiding uncovered option margin. ETF short hedges require
explicit borrow availability. Real physical delivery is avoided by forced exit
before expiry. A missing held quote is a fatal error, never a zero return.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    capital: float = 100000.0
    threshold: float = .01
    max_contracts: int = 1
    max_premium_fraction: float = .15
    max_hedge_notional_fraction: float = .5
    max_hold: int = 10
    hedge_every: int = 1
    exit_days: int = 14
    option_fee: float = 2.0
    option_slip: float = .0001
    option_halfspread_proxy: float = .001
    etf_cost_bps: float = 2.0
    borrow_rate: float = .03
    funding_rate: float = .03
    allow_short_etf: bool = False
    price_mode: str = "quotes"  # proxy mode is explicitly a cost scenario


def metrics(ledger, capital):
    if ledger.empty:
        return {"status": "no_dates"}
    equity = ledger.equity
    returns = equity.div(equity.shift(fill_value=capital))-1
    peaks = equity.cummax().clip(lower=capital)
    active = ledger.net_pnl.abs() > 1e-12
    years = max((ledger.date.iloc[-1]-ledger.date.iloc[0]).days/365, 1/252)
    return dict(total_return=equity.iloc[-1]/capital-1,
                annualized_return=(equity.iloc[-1]/capital)**(1/years)-1,
                sharpe=float(np.sqrt(252)*returns.mean()/returns.std(ddof=1)) if returns.std(ddof=1)>0 else None,
                max_drawdown=float((equity/peaks-1).min()),
                active_day_win_rate=float((ledger.loc[active, "net_pnl"]>0).mean()) if active.any() else None,
                turnover_gross_notional=float(ledger.turnover.sum()/capital),
                total_cost=float(ledger.cost.sum()), days=len(ledger),
                entries=int(ledger.entries.sum()), final_equity=float(equity.iloc[-1]))


def run(panel, feature_frame, config=Config(), start=None, end=None):
    c = config
    if c.hedge_every < 1 or c.max_contracts < 1 or c.capital <= 0 or c.price_mode not in ("quotes", "proxy"):
        raise ValueError("Invalid backtest configuration")
    if any(v < 0 for v in [c.option_fee, c.option_slip, c.option_halfspread_proxy, c.etf_cost_bps, c.borrow_rate, c.funding_rate]):
        raise ValueError("Negative costs")
    p = panel.copy()
    f = feature_frame.sort_values("date").copy()
    # Decision made yesterday; today's close/Greeks are not used for the target hedge.
    f["signal"] = f.iv_minus_past_rv.shift(1)
    f["signal_date"] = f.date.shift(1)
    all_days = list(f.date)
    dates = [d for d in all_days if (start is None or d >= pd.Timestamp(start)) and (end is None or d <= pd.Timestamp(end))]
    if not dates:
        return pd.DataFrame(), {"status": "no_dates"}
    fmap = f.set_index("date")
    quotes = {d: g.set_index("contract_id", verify_integrity=True) for d, g in p.groupby("date")}
    held, previous_marks, hedge, cash, prev_S, prev_date = {}, {}, 0.0, c.capital, None, None
    age, halted = 0, False
    rows = []
    def halfspread(row):
        if bool(row.has_quote):
            return (row.ask-row.bid)/2
        if c.price_mode == "proxy":
            return c.option_halfspread_proxy
        raise ValueError("Historical bid/ask missing: use explicit --proxy-costs scenario or supply quotes")
    for date in dates:
        today = quotes.get(date, pd.DataFrame())
        ft = fmap.loc[date]
        S = float(ft.S)
        if not np.isfinite(S) or S <= 0:
            raise ValueError("Missing ETF mark")
        elapsed = (date-prev_date).days/365 if prev_date is not None else 0
        dividend = float(ft.cash_dividend)
        hedge_pnl = hedge*(S-prev_S+dividend) if prev_S is not None else 0.0
        option_pnl = 0.0
        for cid, qty in held.items():
            if cid not in today.index:
                raise ValueError(f"Missing held contract {cid} at {date}; incomplete history")
            option_pnl += qty*today.loc[cid, "multiplier"]*(today.loc[cid, "mark"]-previous_marks[cid])
        borrow = max(-hedge, 0)*(prev_S or S)*c.borrow_rate*elapsed
        funding = max(-cash, 0)*c.funding_rate*elapsed
        cost, turnover, entries = borrow+funding, 0.0, 0
        cash += (hedge*dividend)-borrow-funding
        age += bool(held)
        exit_now = bool(held) and (age >= c.max_hold or date == dates[-1] or halted or
                     min(today.loc[k, "days"] for k in held) <= c.exit_days)
        if exit_now:
            for cid, qty in held.items():
                a = today.loc[cid]
                fee = abs(qty)*(c.option_fee+a.multiplier*(halfspread(a)+c.option_slip))
                cash += qty*a.multiplier*a.mark-fee
                cost += fee
                turnover += abs(qty)*a.multiplier*a.mark
            held = {}
        prev_day = ft.signal_date
        yesterday = quotes.get(prev_day, pd.DataFrame())
        # Pair selection is also based on yesterday, not today's winning contracts.
        if not held and not exit_now and not halted and date != dates[-1] and np.isfinite(ft.signal) and ft.signal < -c.threshold:
            if not yesterday.empty:
                candidates = yesterday[(yesterday.days >= 25) & (yesterday.days <= 60)].reset_index()
                calls = candidates[candidates.cp == "C"]
                puts = candidates[candidates.cp == "P"]
                pairs = calls.merge(puts, on=["expiry", "K", "multiplier"], suffixes=("_c", "_p"))
                if not pairs.empty:
                    pairs["distance"] = abs(np.log(pairs.K/pairs.F_c))
                    pair = pairs.sort_values(["expiry", "distance"]).iloc[0]
                    ids = [pair.contract_id_c, pair.contract_id_p]
                    if all(k in today.index for k in ids):
                        qty = c.max_contracts
                        premium = sum(today.loc[k, "mark"]*today.loc[k, "multiplier"]*qty for k in ids)
                        target = -sum(yesterday.loc[k, "delta"]*yesterday.loc[k, "multiplier"]*qty for k in ids)
                        target = np.round(target/100)*100  # ETF board lot
                        equity_before = cash+hedge*S
                        eligible = ((target >= 0 or c.allow_short_etf) and
                                    premium <= max(equity_before, 0)*c.max_premium_fraction and
                                    abs(target*S) <= max(equity_before, 0)*c.max_hedge_notional_fraction and
                                    premium+abs(target*S) <= max(equity_before, 0)*.8)
                        if eligible:
                            for cid in ids:
                                a = today.loc[cid]
                                fee = qty*(c.option_fee+a.multiplier*(halfspread(a)+c.option_slip))
                                cash -= qty*a.multiplier*a.mark+fee
                                cost += fee
                                turnover += qty*a.multiplier*a.mark
                                held[cid] = qty
                            age, entries = 0, 1
        target = hedge
        if not held:
            target = 0.0
        elif entries or age % c.hedge_every == 0:
            if all(k in yesterday.index for k in held):
                target = -sum(qty*yesterday.loc[k, "delta"]*yesterday.loc[k, "multiplier"] for k, qty in held.items())
                target = np.round(target/100)*100
                current_option_value = sum(qty*today.loc[k, "multiplier"]*today.loc[k, "mark"] for k, qty in held.items())
                current_equity = cash+hedge*S+current_option_value
                cap = max(current_equity, 0)*c.max_hedge_notional_fraction/S
                target = np.clip(target, -np.floor(cap/100)*100, np.floor(cap/100)*100)
                if target < 0 and not c.allow_short_etf:
                    target = 0.0
        change = target-hedge
        hedge_fee = abs(change)*S*c.etf_cost_bps/1e4
        cash -= change*S+hedge_fee
        cost += hedge_fee
        turnover += abs(change)*S
        hedge = target
        option_value = sum(qty*today.loc[k, "multiplier"]*today.loc[k, "mark"] for k, qty in held.items())
        equity = cash+hedge*S+option_value
        if equity <= 0:
            raise ValueError("Insolvent portfolio; results invalid")
        if equity < c.capital*.7:
            halted = True  # liquidate at the next available close
        previous_marks = {k: today.loc[k, "mark"] for k in held}
        delta = hedge+sum(qty*today.loc[k, "multiplier"]*today.loc[k, "delta"] for k, qty in held.items())
        rows.append(dict(date=date, signal_date=prev_day, option_pnl=option_pnl, hedge_pnl=hedge_pnl,
                         cost=cost, net_pnl=option_pnl+hedge_pnl-cost, equity=equity,
                         cash=cash, hedge_shares=hedge, residual_delta=delta, entries=entries,
                         option_contracts=sum(held.values()), turnover=turnover))
        prev_S, prev_date = S, date
    ledger = pd.DataFrame(rows)
    changes = ledger.equity.diff().fillna(ledger.equity.iloc[0]-c.capital)
    if not np.allclose(changes, ledger.net_pnl, atol=1e-7):
        raise AssertionError("P&L attribution does not reconcile")
    return ledger, metrics(ledger, c.capital)
