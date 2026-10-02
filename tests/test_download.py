import pandas as pd
import pytest
import json
from etf_option import data


def test_partial_acquisition_never_emits_complete_panel(raw, tmp_path, monkeypatch):
    c = raw["contracts"].iloc[:2].copy()
    c["contract_id"] = ["10000001", "10000002"]
    c.to_csv(tmp_path/"contracts.csv", index=False)
    def fake(function, kwargs, output, timeout=35):
        if kwargs["symbol"] == "10000002":
            raise RuntimeError("test upstream failure")
        df = pd.DataFrame({"日期": ["2024-01-02"], "收盘": [.1], "成交量": [100]})
        df.to_csv(output, index=False)
        return df
    monkeypatch.setattr(data, "bounded_ak", fake)
    with pytest.raises(RuntimeError, match="Partial"):
        data.download_history(tmp_path/"contracts.csv", tmp_path/"download")
    assert (tmp_path/"download/download_audit.json").exists()
    assert not (tmp_path/"download/options.csv").exists()


def test_underlying_distribution_normalization(tmp_path, monkeypatch):
    def fake(function, kwargs, output, timeout=35):
        df = pd.DataFrame({"日期": ["2024-01-02", "2024-01-03"], "收盘": [2.5, 2.4]})
        df.to_csv(output, index=False)
        return df
    monkeypatch.setattr(data, "bounded_ak", fake)
    dividends = tmp_path/"dividends.csv"
    pd.DataFrame({"date": ["2024-01-03"], "cash_dividend": [.1], "source": ["SYNTHETIC_TEST_ONLY"]}).to_csv(dividends, index=False)
    data.download_underlying("20240102", "20240103", dividends, tmp_path/"download")
    result = pd.read_csv(tmp_path/"download/underlying.csv")
    assert result.total_return_close.iloc[0] == pytest.approx(result.total_return_close.iloc[1])
    assert result.S.iloc[1] == 2.4


def test_undefined_statistics_serialize_as_null(tmp_path):
    data.save_json(tmp_path/"stats.json", {"sharpe": float("nan"), "values": [float("inf"), .2]})
    result = json.loads((tmp_path/"stats.json").read_text(encoding="utf-8"))
    assert result == {"sharpe": None, "values": [None, .2]}
