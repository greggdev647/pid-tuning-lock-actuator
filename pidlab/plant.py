"""Plant model for a motorised door-lock bolt.

The actuator is a DC motor driving the bolt, modelled as

    G(s) = K / ( s (tau_m s + 1)(tau_e s + 1) )

with input = motor voltage (V) and output = bolt angle (deg).

* K       steady-state speed gain, in (deg/s) per volt
* tau_m   mechanical time constant (s)
* tau_e   electrical (armature) time constant (s)

The parameter values used in this project are illustrative. They were chosen
to give a plausible lock actuator, not measured from hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Plant:
    K: float = 20.0
    tau_m: float = 0.15
    tau_e: float = 0.01

    def state_space(self):
        """Continuous-time state-space realisation (A, B, C, D)."""
        den = np.polymul([1.0, 0.0], np.polymul([self.tau_m, 1.0], [self.tau_e, 1.0]))
        return signal.tf2ss([self.K], den)

    def discretize(self, dt: float):
        """Zero-order-hold discretisation. Returns (Ad, Bd, Cd) with Bd, Cd as 1-D arrays."""
        A, B, C, D = self.state_space()
        Ad, Bd, Cd, _, _ = signal.cont2discrete((A, B, C, D), dt, method="zoh")
        return Ad, Bd.ravel(), Cd.ravel()

    # --- closed-form ultimate gain / period (for Ziegler-Nichols) -------------
    #
    # With proportional control only, the characteristic equation is
    #   tau_m tau_e s^3 + (tau_m + tau_e) s^2 + s + K Kp = 0.
    # The Routh-Hurwitz condition gives the stability limit Ku, and the
    # oscillation frequency at that limit is sqrt(1 / (tau_m tau_e)).

    def ultimate_gain(self) -> float:
        """Proportional gain Ku (V/deg) at which the loop oscillates steadily."""
        return (self.tau_m + self.tau_e) / (self.tau_m * self.tau_e * self.K)

    def ultimate_period(self) -> float:
        """Period Pu (s) of the sustained oscillation at Kp = Ku."""
        omega_u = 1.0 / np.sqrt(self.tau_m * self.tau_e)
        return float(2.0 * np.pi / omega_u)
