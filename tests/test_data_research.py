import numpy as np
import pandas as pd
import pytest
from datetime import timedelta
from etf_option.cleaning import prepare, Filters, map_contracts, infer_forward
from etf_option.research import features, purged_split, constant_maturity
from etf_option.pipeline import read_inputs
from etf_option.data import official_validate


def test_forward_carry(analyzed):
    _, forward, _, _, _ = analyzed
    assert np.allclose(forward.q, .012, atol=1e-10)
    assert (forward.parity_pairs >= 2).all()


def test_iv_recovery_panel(analyzed):
    p = analyzed[0]
    expected = .15+.02*(p.K/p.S-1)**2+.015*p.days/365
    assert np.allclose(p.iv, expected, atol=3e-6)
    assert not p.is_synthetic.eq(False).any()


def test_mapping_duplicate_and_missing(raw):
    q, c = raw["options"], raw["contracts"]
    with pytest.raises(ValueError, match="Duplicate"):
        map_contracts(pd.concat([q, q.iloc[:1]]), c)
    with pytest.raises(ValueError, match="mapping"):
        map_contracts(q, c.iloc[1:])
    with pytest.raises(ValueError, match="overlapping"):
        map_contracts(q, pd.concat([c, c.iloc[:1]]))


def test_filter_audit(raw):
    q = raw["options"].copy()
    q.loc[0, "volume"] = 0
    q.loc[1, ["bid", "ask"]] = [2, 1]
    q.loc[2, "close"] = np.nan
    q.loc[2, ["bid", "ask"]] = np.nan
    valid, rejected = prepare(q, raw["contracts"], raw["underlying"], raw["rates"])
    assert rejected.reason.str.contains("volume").any()
    assert rejected.reason.str.contains("spread").any()
    assert rejected.reason.str.contains("missing").any()
    assert len(valid)+len(rejected) == len(q)


def test_unavailable_rate_rejected(raw):
    r = raw["rates"].copy()
    r["available_date"] = [d.to_pydatetime()+timedelta(days=1) for d in r.available_date]
    with pytest.raises(ValueError, match="unavailable"):
        prepare(raw["options"], raw["contracts"], raw["underlying"], r)
    r = raw["rates"].copy()
    r.loc[0, "available_date"] = pd.NaT
    with pytest.raises(ValueError, match="unavailable"):
        prepare(raw["options"], raw["contracts"], raw["underlying"], r)


def test_future_perturbation_cannot_change_features(raw, analyzed):
    original = analyzed[-1]
    atm = original[["date", "atm_iv"]]
    cutoff = raw["underlying"].date.iloc[55]
    changed = raw["underlying"].copy()
    changed.loc[changed.date > cutoff, "total_return_close"] *= 1.8
    after = features(changed, atm)
    for col in ["rv_past", "iv_minus_past_rv", "atm_iv"]:
        np.testing.assert_allclose(after.loc[after.date <= cutoff, col], original.loc[original.date <= cutoff, col], equal_nan=True)
    assert not np.allclose(after.rv_future, original.rv_future, equal_nan=True)


def test_future_rv_alignment_and_purge(raw, analyzed):
    f = analyzed[-1]
    i, h = 25, 20
    closes = raw["underlying"].total_return_close
    increments = np.log(closes/closes.shift()).iloc[i+1:i+h+1]
    assert f.rv_future.iloc[i] == pytest.approx(np.sqrt(np.mean(increments**2)*252))
    assert f.rv_future.tail(h).isna().all()
    cut = f.date.iloc[65]
    train, test = purged_split(f, cut)
    assert (train.label_end < cut).all()
    assert (test.date >= cut).all()
    assert set(train.date).isdisjoint(set(test.date))


def test_no_term_extrapolation():
    term = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]*2, "T": [.2,.4], "atm_iv": [.2,.3]})
    assert constant_maturity(term).empty


def test_synthetic_formal_pipeline_rejected(raw, tmp_path):
    for name, frame in raw.items():
        frame.to_csv(tmp_path/f"{name}.csv", index=False)
    with pytest.raises(ValueError, match="Synthetic"):
        read_inputs(tmp_path)


def test_official_contract_comparison(raw, tmp_path):
    c = raw["contracts"].iloc[:3].copy()
    c.to_csv(tmp_path/"contracts.csv", index=False)
    o = c[["contract_id", "cp", "K", "expiry", "multiplier"]].copy()
    o["asof"] = c.valid_from
    o.iloc[1, o.columns.get_loc("multiplier")] = 9999
    o.to_csv(tmp_path/"official.csv", index=False)
    result = official_validate(tmp_path/"contracts.csv", tmp_path/"official.csv", tmp_path/"audit.csv")
    assert result.mismatches.tolist() == ["", "multiplier", ""]

