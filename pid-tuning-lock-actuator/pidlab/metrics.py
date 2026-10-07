"""Step-response and disturbance-rejection metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _trapz(f: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral (written out so it works on any NumPy version)."""
    return float(np.sum((f[1:] + f[:-1]) * np.diff(x)) / 2.0)


@dataclass(frozen=True)
class Metrics:
    overshoot_pct: float        # % above the setpoint, before the disturbance
    rise_time: float            # s, 10% -> 90% of the setpoint
    settling_time: float        # s, last entry into the +/-2% band (NaN if never settles)
    steady_state_error: float   # deg, mean error over the last 0.2 s before the disturbance
    iae: float                  # deg*s, integral of |error| over the whole run
    itae: float                 # deg*s^2, integral of t*|error| over the whole run
    peak_voltage: float         # V, largest |control voltage|
    dist_peak_dev: float        # deg, largest |error| after the disturbance
    dist_recovery_time: float   # s, time after the disturbance until the error stays in band

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    t: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    r: float,
    t_dist: float,
    band_pct: float = 2.0,
    dist_band_pct: float = 0.5,
) -> Metrics:
    """Compute metrics for a positive step of size ``r`` followed by a disturbance at ``t_dist``."""
    t, y, u = np.asarray(t, float), np.asarray(y, float), np.asarray(u, float)
    e = r - y
    nan = float("nan")

    pre = t < t_dist - 1e-9
    post = ~pre
    tp, yp = t[pre], y[pre]

    overshoot = max(0.0, (yp.max() - r) / abs(r) * 100.0)

    def first_reach(level: float) -> float:
        idx = np.nonzero(yp >= level)[0]
        return float(tp[idx[0]]) if idx.size else nan

    rise = first_reach(0.9 * r) - first_reach(0.1 * r)

    band = band_pct / 100.0 * abs(r)
    outside = np.nonzero(np.abs(yp - r) > band)[0]
    if outside.size == 0:
        settling = float(tp[0])
    elif outside[-1] == len(yp) - 1:
        settling = nan
    else:
        settling = float(tp[outside[-1] + 1])

    window = tp >= tp[-1] - 0.2
    sse = abs(float(np.mean(yp[window])) - r)

    if post.any():
        td, ed = t[post], np.abs(e[post])
        dist_dev = float(ed.max())
        dband = dist_band_pct / 100.0 * abs(r)
        out_d = np.nonzero(ed > dband)[0]
        if out_d.size == 0:
            recovery = 0.0
        elif out_d[-1] == len(ed) - 1:
            recovery = nan
        else:
            recovery = float(td[out_d[-1] + 1] - t_dist)
    else:
        dist_dev, recovery = nan, nan

    return Metrics(
        overshoot_pct=float(overshoot),
        rise_time=float(rise),
        settling_time=settling,
        steady_state_error=sse,
        iae=_trapz(np.abs(e), t),
        itae=_trapz(t * np.abs(e), t),
        peak_voltage=float(np.max(np.abs(u))),
        dist_peak_dev=dist_dev,
        dist_recovery_time=recovery,
    )
