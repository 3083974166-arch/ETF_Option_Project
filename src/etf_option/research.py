"""Past-only features, forward labels, purged chronological evaluation."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def atm_term(panel):
    p = panel.copy()
    # OTM quotes generally suffer less intrinsic-value cancellation.
    p = p[((p.cp == "C") & (p.K >= p.F)) | ((p.cp == "P") & (p.K < p.F))]
    out = []
    for (date, expiry), g in p.groupby(["date", "expiry"]):
        g = g.sort_values("log_forward_moneyness")
        x, y = g.log_forward_moneyness.to_numpy(), g.iv.to_numpy()
        if len(g) < 3 or not x.min() <= 0 <= x.max():
            continue  # no extrapolation
        a = g.iloc[0]
        out.append(dict(date=date, expiry=expiry, T=a["T"],
                        atm_iv=np.interp(0, x, y),
                        skew=np.polyfit(x, y, 1)[0], F=a.F, q=a.q))
    return pd.DataFrame(out)


def constant_maturity(term, target_days=30):
    target = target_days/365
    rows = []
    for date, g in term.groupby("date"):
        g = g.sort_values("T")
        times = g["T"].to_numpy()
        if len(g) < 2 or not times.min() <= target <= times.max():
            continue
        var = np.interp(target, times, g.atm_iv.to_numpy()**2*times)
        rows.append(dict(date=date, atm_iv=np.sqrt(var/target), target_days=target_days))
    return pd.DataFrame(rows, columns=["date", "atm_iv", "target_days"])


def features(underlying, atm, window=20):
    u = underlying.copy().sort_values("date")
    u["date"] = pd.to_datetime(u.date)
    if u.date.duplicated().any() or (u.S <= 0).any():
        raise ValueError("Invalid underlying series")
    # total_return_close avoids treating distributions as economic return losses.
    tr = u.total_return_close
    ret = np.log(tr/tr.shift())
    u["rv_past"] = np.sqrt(ret.pow(2).rolling(window).mean()*252)
    u["rv_future"] = np.sqrt(ret.pow(2).rolling(window).mean().shift(-window)*252)
    u["label_end"] = u.date.shift(-window)
    u = u.merge(atm, on="date", how="left", validate="one_to_one")
    u["iv_minus_past_rv"] = u.atm_iv-u.rv_past
    u["ex_post_iv_minus_rv"] = u.atm_iv-u.rv_future
    return u


def purged_split(frame, split_date):
    cut = pd.Timestamp(split_date)
    train = frame[(frame.date < cut) & (frame.label_end < cut)].copy()
    test = frame[frame.date >= cut].copy()
    return train, test


def forecast_study(frame, split_date, horizon=20):
    cols = ["atm_iv", "rv_past", "rv_future", "label_end"]
    d = frame.dropna(subset=cols)
    train, test = purged_split(d, split_date)
    if len(train) < 25 or len(test) < 10:
        return {"status": "insufficient_sample", "train_n": len(train), "test_n": len(test)}, pd.DataFrame()
    X = sm.add_constant(train[["atm_iv", "rv_past"]], has_constant="add")
    fit = sm.OLS(train.rv_future, X).fit(cov_type="HAC", cov_kwds={"maxlags": horizon-1})
    pred = fit.predict(sm.add_constant(test[["atm_iv", "rv_past"]], has_constant="add"))
    detail = test[["date", "label_end", "rv_future", "rv_past"]].copy()
    detail["forecast"] = pred
    baseline = float(np.mean((test.rv_future-test.rv_past)**2))
    mse = float(np.mean((test.rv_future-pred)**2))
    return {"status": "ok", "train_n": len(train), "test_n": len(test),
            "coefficients": fit.params.to_dict(), "hac_pvalues": fit.pvalues.to_dict(),
            "training_correlations": train[["atm_iv", "iv_minus_past_rv", "rv_future"]].corr().loc["rv_future", ["atm_iv", "iv_minus_past_rv"]].to_dict(),
            "equivalent_vrp_regression": {"iv_minus_past_rv_coefficient": float(fit.params["atm_iv"]),
                                          "past_rv_coefficient": float(fit.params["atm_iv"]+fit.params["rv_past"])},
            "oos_mse": mse, "past_rv_baseline_mse": baseline,
            "oos_r2_vs_past_rv": 1-mse/baseline if baseline > 0 else None,
            "horizon_note": "30 calendar-day IV vs 20 trading-day RV: approximate horizon match"}, detail
