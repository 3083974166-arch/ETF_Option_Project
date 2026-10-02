"""Descriptive interpolation; no claim of arbitrage-free surface calibration."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


def make_plots(panel, term, features, ledger, outdir, label):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    def save(fig, filename):
        fig.suptitle(label, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(out/filename, dpi=160)
        plt.close(fig)
    date = panel.date.max()
    p = panel[panel.date == date].copy()
    p = p[((p.cp == "C") & (p.K >= p.F)) | ((p.cp == "P") & (p.K < p.F))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for expiry, g in p.groupby("expiry"):
        g = g.sort_values("log_forward_moneyness")
        axes[0].plot(g.log_forward_moneyness, g.iv, ".-", label=str(expiry.date()))
    axes[0].set(xlabel="log(K/F)", ylabel="IV (annual)", title="OTM smile / skew")
    axes[0].legend(fontsize=7)
    t = term[term.date == date].sort_values("T")
    axes[1].plot(t["T"]*365, t.atm_iv, "o-")
    axes[1].set(xlabel="Calendar days", ylabel="ATM IV", title="Term structure")
    save(fig, "01_smile_term.png")
    if len(p) >= 6 and p["T"].nunique() > 1:
        x = np.linspace(p.log_forward_moneyness.min(), p.log_forward_moneyness.max(), 45)
        y = np.linspace(p["T"].min()*365, p["T"].max()*365, 35)
        X, Y = np.meshgrid(x, y)
        Z = griddata((p.log_forward_moneyness, p["T"]*365), p.iv, (X, Y), method="linear")
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.colorbar(ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis"), ax=ax, label="IV")
        ax.set(xlabel="log(K/F)", ylabel="Calendar days", title="Linear interpolation; blank outside hull")
        save(fig, "02_surface_2d.png")
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=.85)
        ax.set(xlabel="log(K/F)", ylabel="Calendar days", zlabel="IV")
        save(fig, "03_surface_3d.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    for col in ["atm_iv", "rv_past", "rv_future"]:
        axes[0].plot(features.date, features[col], label=col)
    axes[0].legend()
    axes[0].set_title("Future RV is a label, never a trading input")
    axes[1].plot(features.date, features.iv_minus_past_rv)
    axes[1].axhline(0, color="gray", lw=1)
    axes[1].set_ylabel("IV minus past RV")
    save(fig, "04_iv_rv.png")
    if not ledger.empty:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        axes[0].plot(ledger.date, ledger.equity)
        axes[0].set_ylabel("Equity CNY")
        for col in ["option_pnl", "hedge_pnl", "cost"]:
            values = ledger[col].cumsum() * (-1 if col == "cost" else 1)
            axes[1].plot(ledger.date, values, label=col)
        axes[1].legend()
        save(fig, "05_pnl_attribution.png")
