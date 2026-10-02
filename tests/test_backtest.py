from dataclasses import replace
from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
from etf_option.backtest import run, Config


def cfg(**kwargs):
    return Config(allow_short_etf=True, **kwargs)


def test_accounting_and_execution(analyzed):
    panel, _, _, _, f = analyzed
    ledger, met = run(panel, f, cfg())
    assert met["entries"] > 0
    assert np.allclose(ledger.equity.diff().fillna(ledger.equity.iloc[0]-100000), ledger.net_pnl)
    assert (ledger.cost >= 0).all()
    assert ledger.iloc[-1].option_contracts == 0
    assert ledger.iloc[-1].hedge_shares == 0
    assert (ledger.signal_date.dropna() < ledger.loc[ledger.signal_date.notna(), "date"]).all()
    assert (ledger.option_contracts <= 2).all()


def test_future_labels_never_change_trades(analyzed):
    p, _, _, _, f = analyzed
    changed = f.copy()
    changed["rv_future"] = 9876
    changed["ex_post_iv_minus_rv"] = -9876
    a, _ = run(p, f, cfg())
    b, _ = run(p, changed, cfg())
    pd.testing.assert_frame_equal(a, b)


def test_future_quotes_never_change_past_backtest(analyzed):
    p, _, _, _, f = analyzed
    cutoff = f.date.iloc[60]
    changed = p.copy()
    changed.loc[changed.date > cutoff, ["mark", "bid", "ask"]] *= 1.1
    a, _ = run(p, f, cfg())
    b, _ = run(changed, f, cfg())
    pd.testing.assert_frame_equal(a[a.date <= cutoff], b[b.date <= cutoff])


def test_missing_held_mark_fails(analyzed):
    p, _, _, _, f = analyzed
    a, _ = run(p, f, cfg())
    entry = a.loc[a.entries > 0, "date"].iloc[0]
    next_date = f.loc[f.date > entry, "date"].iloc[0]
    with pytest.raises(ValueError, match="Missing held contract"):
        run(p[p.date != next_date], f, cfg())


def test_quote_requirement_and_proxy_mode(analyzed):
    p, _, _, _, f = analyzed
    p = p.copy()
    p["has_quote"] = False
    with pytest.raises(ValueError, match="bid/ask"):
        run(p, f, cfg())
    ledger, _ = run(p, f, cfg(price_mode="proxy"))
    assert ledger.entries.sum() > 0


def test_higher_cost_reduces_equity(analyzed):
    p, _, _, _, f = analyzed
    a, _ = run(p, f, cfg())
    b, _ = run(p, f, cfg(option_fee=4, option_slip=.0002, etf_cost_bps=4, borrow_rate=.06))
    assert b.cost.sum() > a.cost.sum()
    assert b.equity.iloc[-1] < a.equity.iloc[-1]


def test_oos_starts_flat_and_no_borrow_default(analyzed):
    p, _, _, _, f = analyzed
    ledger, _ = run(p, f, Config(), start=f.date.iloc[65])
    assert (ledger.hedge_shares >= 0).all()
    assert ledger.equity.iloc[0] == pytest.approx(100000-ledger.cost.iloc[0])


def test_no_signal_no_trade(analyzed):
    p, _, _, _, f = analyzed
    f = f.copy()
    f["iv_minus_past_rv"] = np.nan
    ledger, met = run(p, f, cfg())
    assert (ledger.equity == 100000).all()
    assert met["entries"] == 0
    assert met["sharpe"] is None


def test_dividend_and_short_hedge_accounting():
    # Ex-dividend ETF price fall is offset by paid dividend. A short seller
    # owes the distribution; forgetting it would manufacture hedge profit.
    dates = pd.bdate_range("2024-01-02", periods=4)
    f = pd.DataFrame({"date": dates, "S": [2.5, 2.5, 2.4, 2.4],
                      "cash_dividend": [0., 0., .1, 0.], "iv_minus_past_rv": -.1})
    rows = []
    for date in dates:
        for cp, delta in [("C", .6), ("P", -.2)]:
            rows.append(dict(date=date, contract_id=cp, cp=cp, expiry=pd.Timestamp(dates[0].to_pydatetime()+timedelta(days=40)),
                             days=40-(date-dates[0]).days, K=2.5, F=2.5, multiplier=10000,
                             mark=.05, bid=.05, ask=.05, has_quote=True, delta=delta))
    p = pd.DataFrame(rows)
    config = cfg(option_fee=0, option_slip=0, etf_cost_bps=0, borrow_rate=0, funding_rate=0)
    ledger, met = run(p, f, config)
    assert ledger.hedge_shares.iloc[1] == -4000
    assert abs(ledger.hedge_pnl.iloc[2]) < 1e-8
    assert np.allclose(ledger.equity, config.capital)
    assert met["entries"] == 1


def test_forced_exit_before_expiry():
    dates = pd.bdate_range("2024-01-02", periods=14)
    expiry = pd.Timestamp(dates[0].to_pydatetime()+timedelta(days=29))
    f = pd.DataFrame({"date": dates, "S": 2.5, "cash_dividend": 0., "iv_minus_past_rv": -.1})
    rows = []
    for date in dates:
        for cp, delta in [("C", .5), ("P", -.5)]:
            rows.append(dict(date=date, contract_id=cp, cp=cp, expiry=expiry,
                             days=(expiry-date).days, K=2.5, F=2.5, multiplier=10000,
                             mark=.05, bid=.049, ask=.051, has_quote=True, delta=delta))
    ledger, _ = run(pd.DataFrame(rows), f, cfg(max_hold=100))
    mask = (expiry-ledger.date).dt.days <= 14
    assert mask.any()
    assert (ledger.loc[mask, "option_contracts"] == 0).all()

