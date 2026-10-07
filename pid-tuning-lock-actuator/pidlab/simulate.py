"""Closed-loop simulation of the lock actuator under PID control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import PID, Gains
from .metrics import Metrics, compute_metrics
from .plant import Plant


@dataclass(frozen=True)
class Scenario:
    """One test run: a setpoint step at t = 0, then a constant load disturbance."""

    r_step: float = 90.0      # deg, bolt travel from "unlocked" to "locked"
    t_end: float = 5.0        # s
    t_dist: float = 2.5       # s, when the load disturbance starts
    d_volts: float = -2.0     # V, disturbance expressed as an equivalent voltage at the plant input
    dt: float = 0.002         # s, sample time
    u_max: float = 12.0       # V, motor supply limit (symmetric)


@dataclass
class Result:
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    r: float
    scenario: Scenario

    def metrics(self, **kwargs) -> Metrics:
        return compute_metrics(self.t, self.y, self.u, self.r, self.scenario.t_dist, **kwargs)


def simulate(
    plant: Plant,
    gains: Gains,
    scenario: Scenario = Scenario(),
    anti_windup: bool = True,
    tf: float = 0.005,
) -> Result:
    dt = scenario.dt
    n = int(round(scenario.t_end / dt))
    k_dist = int(round(scenario.t_dist / dt))

    Ad, Bd, cvec = plant.discretize(dt)
    pid = PID(gains, dt, u_min=-scenario.u_max, u_max=scenario.u_max, tf=tf, anti_windup=anti_windup)

    x = np.zeros(Ad.shape[0])
    t = np.arange(n) * dt
    y = np.empty(n)
    u = np.empty(n)

    for k in range(n):
        y_k = float(cvec @ x)
        u_k = pid.step(scenario.r_step, y_k)
        d_k = scenario.d_volts if k >= k_dist else 0.0
        x = Ad @ x + Bd * (u_k + d_k)
        y[k] = y_k
        u[k] = u_k

    return Result(t=t, y=y, u=u, r=scenario.r_step, scenario=scenario)
