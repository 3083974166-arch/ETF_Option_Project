"""Small deterministic synthetic fixture ONLY for tests. Not market evidence.

Weekdays are NOT an exchange calendar; maturities/prices are invented, IDs start
with TEST. This factory is deliberately outside the installable package.
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from etf_option.pricing import price


def market_fixture(n=100, seed=71):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    S = 2.5*np.exp(np.cumsum(rng.normal(0, .25/np.sqrt(252), n)))
    u = pd.DataFrame({"date": dates, "S": S, "total_return_close": S,
                      "cash_dividend": 0., "source": "SYNTHETIC_TEST_ONLY", "is_synthetic": True})
    r = pd.DataFrame({"date": dates, "available_date": dates, "r": .02, "source": "SYNTHETIC_TEST_ONLY"})
    expiries = pd.date_range(dates[0].to_pydatetime()+timedelta(days=35),
                            dates[-1].to_pydatetime()+timedelta(days=100), freq="28D")
    contracts, quotes = [], []
    for expiry in expiries:
        for K in np.round(np.arange(1.9, 3.01, .1), 2):
            for cp in ["C", "P"]:
                cid = f"TEST_{expiry:%Y%m%d}_{K}_{cp}"
                contracts.append(dict(contract_id=cid, cp=cp, K=K, expiry=expiry, multiplier=10000,
                                      adjusted=False, valid_from=dates[0], valid_to=expiry, source="SYNTHETIC_TEST_ONLY"))
                for i, date in enumerate(dates):
                    days = (expiry-date).days
                    if not 1 <= days <= 100:
                        continue
                    # Same vol for paired calls/puts preserves parity exactly.
                    sigma = .15+.02*(K/S[i]-1)**2+.015*days/365
                    value = price(S[i], K, days/365, .02, sigma, .012, cp)
                    half = min(.0003, value*.08)
                    quotes.append(dict(date=date, contract_id=cid, close=value, bid=value-half, ask=value+half,
                                       volume=200, open_interest=1000, source="SYNTHETIC_TEST_ONLY", is_synthetic=True))
    return dict(options=pd.DataFrame(quotes), contracts=pd.DataFrame(contracts), underlying=u, rates=r)

