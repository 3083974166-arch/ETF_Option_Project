import json
from pathlib import Path
import pandas as pd
from etf_option.plots import make_plots
from etf_option.backtest import run, Config
from etf_option.research import forecast_study


def test_pipeline_components_and_figures(analyzed, tmp_path):
    p, forward, rejected, term, feat = analyzed
    ledger, metrics = run(p, feat, Config(allow_short_etf=True))
    study, pred = forecast_study(feat, str(feat.date.iloc[65].date()))
    assert study["status"] in ["ok", "insufficient_sample"]
    make_plots(p, term, feat, ledger, tmp_path, "SYNTHETIC UNIT TEST - NO MARKET CONCLUSION")
    assert len(list(tmp_path.glob("*.png"))) == 5
    assert all(x.stat().st_size > 1000 for x in tmp_path.glob("*.png"))


def test_notebook_schema():
    import nbformat
    for path in Path("notebooks").glob("*.ipynb"):
        nbformat.validate(nbformat.read(path, as_version=4))


def test_full_pipeline_test_only(raw, tmp_path, monkeypatch):
    from importlib import import_module
    module = import_module("etf_option.pipeline")
    # The real CLI has no synthetic bypass. Only this unit-test monkeypatch
    # exercises orchestration with invented data in pytest's temporary directory.
    monkeypatch.setattr(module, "read_inputs", lambda root: raw)
    out = tmp_path/"test_run"
    result = module.pipeline(tmp_path, out, "2024-04-02", Config(allow_short_etf=True), charts=False)
    assert result["status"] == "complete"
    assert result["data_kind"] == "SYNTHETIC_TEST_ONLY"
    robustness = pd.read_csv(out/"robustness_oos.csv")
    assert len(robustness) == 9
    assert robustness.status.eq("ok").all()
    assert (out/"performance.json").exists()
