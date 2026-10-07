"""Robustness check: how do tuned controllers behave when the plant is not the nominal one?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .controller import Gains
from .plant import Plant
from .simulate import Scenario, simulate


def sample_plants(nominal: Plant, n: int = 200, spread: float = 0.30, seed: int = 0) -> list[Plant]:
    """Plants with K, tau_m and tau_e each scaled by Uniform(1 - spread, 1 + spread)."""
    rng = np.random.default_rng(seed)
    scales = rng.uniform(1.0 - spread, 1.0 + spread, size=(n, 3))
    return [
        Plant(K=nominal.K * a, tau_m=nominal.tau_m * b, tau_e=nominal.tau_e * c)
        for a, b, c in scales
    ]


def evaluate_controller(
    name: str,
    gains: Gains,
    plants: list[Plant],
    scenario: Scenario,
    anti_windup: bool = True,
) -> pd.DataFrame:
    """Simulate one controller on every plant; one row of metrics per plant."""
    rows = []
    for i, p in enumerate(plants):
        m = simulate(p, gains, scenario, anti_windup=anti_windup).metrics()
        rows.append({"controller": name, "plant_id": i, "K": p.K, "tau_m": p.tau_m, "tau_e": p.tau_e, **m.as_dict()})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, os_limit_pct: float = 10.0, settle_limit_s: float = None) -> pd.DataFrame:
    """Per-controller summary. A run counts as a success if it settles before the
    disturbance with overshoot <= ``os_limit_pct``."""
    if settle_limit_s is None:
        settle_limit_s = float("inf")
    ok = (df["overshoot_pct"] <= os_limit_pct) & df["settling_time"].notna() & (df["settling_time"] <= settle_limit_s)
    g = df.assign(success=ok).groupby("controller", sort=False)
    return pd.DataFrame(
        {
            "success_rate_pct": g["success"].mean() * 100.0,
            "median_itae": g["itae"].median(),
            "p90_itae": g["itae"].quantile(0.9),
            "median_overshoot_pct": g["overshoot_pct"].median(),
            "p90_overshoot_pct": g["overshoot_pct"].quantile(0.9),
            "median_settling_s": g["settling_time"].median(),
            "never_settled_pct": g["settling_time"].apply(lambda s: s.isna().mean() * 100.0),
            "median_dist_dev_deg": g["dist_peak_dev"].median(),
        }
    )
