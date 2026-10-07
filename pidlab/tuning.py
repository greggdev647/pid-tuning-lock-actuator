"""Two ways of choosing PID gains: Ziegler-Nichols rules and numerical optimisation."""

from __future__ import annotations

from functools import partial

import numpy as np
from scipy.optimize import differential_evolution

from .controller import Gains
from .plant import Plant
from .simulate import Scenario, simulate


def ziegler_nichols(ku: float, pu: float, variant: str = "classic") -> Gains:
    """Closed-loop Ziegler-Nichols PID rules.

    ku, pu   ultimate gain and ultimate period of the plant under P-only control.
    variant  "classic"       Kp = 0.6 Ku, Ti = Pu/2,  Td = Pu/8
             "no_overshoot"  Kp = 0.2 Ku, Ti = Pu/2,  Td = Pu/3
    """
    if variant == "classic":
        kp, ti, td = 0.6 * ku, 0.5 * pu, 0.125 * pu
    elif variant == "no_overshoot":
        kp, ti, td = 0.2 * ku, 0.5 * pu, pu / 3.0
    else:
        raise ValueError(f"unknown Ziegler-Nichols variant: {variant!r}")
    return Gains(kp=kp, ki=kp / ti, kd=kp * td)


# Search box for the optimiser: Kp in V/deg, Ki in V/(deg*s), Kd in V*s/deg.
DEFAULT_BOUNDS = [(0.05, 10.0), (0.0, 20.0), (0.0, 1.0)]


def tuning_cost(
    x: np.ndarray,
    plant: Plant,
    scenario: Scenario,
    os_limit_pct: float = 5.0,
    os_weight: float = 50.0,
    anti_windup: bool = True,
) -> float:
    """Cost = ITAE over the whole run + a penalty for overshoot above ``os_limit_pct``.

    ITAE weights late errors heavily, so it rewards fast settling and fast
    recovery from the disturbance. The overshoot penalty reflects that the
    bolt should not slam past its end stop.
    """
    res = simulate(plant, Gains(*map(float, x)), scenario, anti_windup=anti_windup)
    m = res.metrics()
    return m.itae + os_weight * max(0.0, m.overshoot_pct - os_limit_pct)


def optimize_gains(
    plant: Plant,
    scenario: Scenario = Scenario(),
    bounds=DEFAULT_BOUNDS,
    seed: int = 42,
    maxiter: int = 40,
    popsize: int = 15,
    workers: int = 1,
    **cost_kwargs,
):
    """Global search (differential evolution) for the gains that minimise ``tuning_cost``.

    Returns (Gains, scipy OptimizeResult). ``updating='deferred'`` makes the
    result identical for any number of workers.
    """
    func = partial(tuning_cost, plant=plant, scenario=scenario, **cost_kwargs)
    out = differential_evolution(
        func,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        polish=False,
        updating="deferred",
        workers=workers,
    )
    return Gains(*map(float, out.x)), out
