"""Discrete-time PID controller with output saturation and anti-windup."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gains:
    """PID gains in parallel form.

    kp  V/deg
    ki  V/(deg*s)
    kd  V*s/deg
    """

    kp: float
    ki: float
    kd: float


class PID:
    """Parallel-form PID.

    * The derivative acts on the measurement (not the error), so a setpoint
      step does not cause a derivative kick.
    * The derivative is low-pass filtered with time constant ``tf``.
    * The output is clipped to [u_min, u_max] (the motor supply limit).
    * With ``anti_windup=True`` the integrator is frozen while the output is
      saturated and the error would push it further into saturation
      (conditional integration / clamping).
    """

    def __init__(
        self,
        gains: Gains,
        dt: float,
        u_min: float = -12.0,
        u_max: float = 12.0,
        tf: float = 0.005,
        anti_windup: bool = True,
    ):
        self.g = gains
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.tf = tf
        self.anti_windup = anti_windup
        self.reset()

    def reset(self) -> None:
        self._integral = 0.0
        self._y_prev = None
        self._y_rate = 0.0

    def step(self, r: float, y: float) -> float:
        e = r - y

        # Filtered derivative of the measurement.
        if self._y_prev is None:
            self._y_prev = y
        self._y_rate = (self.tf * self._y_rate + (y - self._y_prev)) / (self.tf + self.dt)
        self._y_prev = y

        p_term = self.g.kp * e
        d_term = -self.g.kd * self._y_rate
        i_candidate = self._integral + self.g.ki * e * self.dt

        u_unsat = p_term + i_candidate + d_term
        u = min(max(u_unsat, self.u_min), self.u_max)

        if self.anti_windup:
            winding_up = (u_unsat - u) * e > 0.0
            if not winding_up:
                self._integral = i_candidate
        else:
            self._integral = i_candidate

        return u
