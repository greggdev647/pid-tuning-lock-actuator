"""Run every experiment in this project and write tables and figures to results/.

    python run_experiments.py            # full run (about a minute)
    python run_experiments.py --quick    # smaller optimiser budget and fewer plants

Steps
1. Ziegler-Nichols gains from the plant's ultimate gain and period.
2. Gains from a global optimiser (differential evolution) minimising ITAE + an overshoot penalty.
3. Nominal-plant comparison of all controllers (step + load disturbance).
4. Robustness check on 200 randomly perturbed plants (K, tau_m, tau_e each +/-30%).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pidlab import Plant, Scenario, optimize_gains, simulate, ziegler_nichols
from pidlab.robustness import evaluate_controller, sample_plants, summarise

OUT = Path(__file__).parent / "results"

# --- plot styling -------------------------------------------------------------
# Neutral ink and surface colours; one fixed colour per controller in every figure.
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLOR = {
    "ZN classic": "#2a78d6",
    "ZN no-overshoot": "#eb6834",
    "Optimised (DE)": "#1baf7a",
    "ZN classic, no anti-windup": "#eda100",
}
SHORT = {
    "ZN classic": "ZN\nclassic",
    "ZN no-overshoot": "ZN\nno-overshoot",
    "Optimised (DE)": "Optimised\n(DE)",
    "ZN classic, no anti-windup": "ZN classic,\nno anti-windup",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": BASELINE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "lines.linewidth": 1.7,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK,
    }
)


def _title(ax, text):
    ax.set_title(text, loc="left", fontsize=11, color=INK, fontweight="bold")


def plot_step(results: dict, names: list[str], scenario: Scenario, path: Path, title: str):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    r = scenario.r_step
    a1.axhspan(0.98 * r, 1.02 * r, color=GRID, alpha=0.7, lw=0)
    a1.axhline(r, color=MUTED, lw=1, ls="--")
    for n in names:
        res = results[n]
        a1.plot(res.t, res.y, color=COLOR[n], label=n)
        a2.plot(res.t, res.u, color=COLOR[n], label=n)
    for a in (a1, a2):
        a.axvline(scenario.t_dist, color=MUTED, lw=1, ls=":")
    a1.text(scenario.t_dist + 0.05, 5, "load disturbance", color=INK2, fontsize=9)
    a1.text(scenario.t_end - 0.05, r + 1.5, f"setpoint {r:g}° (shaded: ±2%)", color=INK2, fontsize=9, ha="right", va="bottom")
    for v in (-scenario.u_max, scenario.u_max):
        a2.axhline(v, color=MUTED, lw=1, ls="--")
    a2.text(scenario.t_end - 0.05, scenario.u_max + 0.6, f"supply limit ±{scenario.u_max:g} V", color=INK2, fontsize=9, ha="right", va="bottom")
    a1.set_ylabel("Bolt angle (deg)")
    a2.set_ylabel("Motor voltage (V)")
    a2.set_xlabel("Time (s)")
    a2.set_ylim(-scenario.u_max * 1.35, scenario.u_max * 1.35)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK, fontweight="bold")
    a1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=len(names), borderaxespad=0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_robustness(summary: pd.DataFrame, runs: pd.DataFrame, order: list[str], path: Path, n_plants: int):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))

    x = np.arange(len(order))
    vals = [summary.loc[n, "success_rate_pct"] for n in order]
    a1.bar(x, vals, color=[COLOR[n] for n in order], width=0.6, edgecolor="none")
    for xi, v in zip(x, vals):
        a1.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", color=INK, fontsize=10)
    a1.set_xticks(x, [SHORT[n] for n in order], fontsize=8.5)
    a1.set_ylim(0, 112)
    a1.set_ylabel("Runs that settle with overshoot ≤ 10% (%)")
    a1.grid(axis="x", visible=False)
    _title(a1, "Success rate on perturbed plants")

    data = [runs.loc[runs["controller"] == n, "itae"].to_numpy() for n in order]
    bp = a2.boxplot(data, positions=x, widths=0.5, patch_artist=True, showfliers=True, whis=(5, 95))
    for patch, n in zip(bp["boxes"], order):
        patch.set(facecolor=COLOR[n], alpha=0.55, edgecolor=COLOR[n], linewidth=1.4)
    for med in bp["medians"]:
        med.set(color=INK, linewidth=1.6)
    for k in ("whiskers", "caps"):
        for line in bp[k]:
            line.set(color=MUTED, linewidth=1.1)
    for fl in bp["fliers"]:
        fl.set(marker="o", markersize=3, markerfacecolor=MUTED, markeredgecolor=SURFACE, alpha=0.7)
    a2.set_yscale("log")
    a2.set_xticks(x, [SHORT[n] for n in order], fontsize=8.5)
    a2.set_ylabel("ITAE (deg·s², log scale; lower is better)")
    a2.grid(axis="x", visible=False)
    _title(a2, "Tracking and recovery cost")

    fig.suptitle(
        f"Controllers tuned on the nominal plant, tested on {n_plants} plants with K, τm, τe each varied by ±30%",
        x=0.01, ha="left", fontsize=9.5, color=INK2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller optimiser budget and fewer plants")
    ap.add_argument("--workers", type=int, default=2, help="processes for the optimiser (result is identical for any value)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    plant, scenario = Plant(), Scenario()
    ku, pu = plant.ultimate_gain(), plant.ultimate_period()
    print(f"Plant: K={plant.K}, tau_m={plant.tau_m}, tau_e={plant.tau_e}  ->  Ku={ku:.3f} V/deg, Pu={pu:.4f} s")

    # 1. Ziegler-Nichols
    zn_classic = ziegler_nichols(ku, pu, "classic")
    zn_no_os = ziegler_nichols(ku, pu, "no_overshoot")

    # 2. Optimiser
    t0 = time.time()
    maxiter, popsize, n_plants = (8, 8, 40) if args.quick else (40, 15, 200)
    opt_gains, opt = optimize_gains(plant, scenario, maxiter=maxiter, popsize=popsize, workers=args.workers)
    print(f"Optimiser: {opt.nfev} cost evaluations in {time.time() - t0:.0f} s, best cost {opt.fun:.3f}, gains {opt_gains}")

    controllers = {
        "ZN classic": (zn_classic, True),
        "ZN no-overshoot": (zn_no_os, True),
        "Optimised (DE)": (opt_gains, True),
        "ZN classic, no anti-windup": (zn_classic, False),
    }

    pd.DataFrame(
        [
            {"controller": n, "anti_windup": aw, "Kp_V_per_deg": g.kp, "Ki_V_per_deg_s": g.ki, "Kd_V_s_per_deg": g.kd}
            for n, (g, aw) in controllers.items()
        ]
    ).to_csv(OUT / "gains.csv", index=False)

    # 3. Nominal comparison
    results = {n: simulate(plant, g, scenario, anti_windup=aw) for n, (g, aw) in controllers.items()}
    nominal = pd.DataFrame({n: r.metrics().as_dict() for n, r in results.items()}).T
    nominal.index.name = "controller"
    nominal.to_csv(OUT / "nominal_metrics.csv", float_format="%.4g")
    print("\nNominal plant:\n", nominal.round(3).to_string())

    plot_step(
        results,
        ["ZN classic", "ZN no-overshoot", "Optimised (DE)"],
        scenario,
        OUT / "step_response.png",
        "Bolt angle after a 90° lock command, with a load disturbance at 2.5 s",
    )
    plot_step(
        results,
        ["ZN classic", "ZN classic, no anti-windup"],
        scenario,
        OUT / "windup.png",
        "Same gains with and without anti-windup",
    )

    # 4. Robustness
    plants = sample_plants(plant, n=n_plants, spread=0.30, seed=0)
    runs = pd.concat(
        [evaluate_controller(n, g, plants, scenario, anti_windup=aw) for n, (g, aw) in controllers.items()],
        ignore_index=True,
    )
    summary = summarise(runs, os_limit_pct=10.0)
    runs.to_csv(OUT / "robustness_runs.csv", index=False, float_format="%.5g")
    summary.to_csv(OUT / "robustness_summary.csv", float_format="%.4g")
    print(f"\nRobustness over {n_plants} perturbed plants:\n", summary.round(2).to_string())

    plot_robustness(summary, runs, list(controllers), OUT / "robustness.png", n_plants)
    print(f"\nWrote tables and figures to {OUT}")


if __name__ == "__main__":
    main()
