import numpy as np
import pytest

from pidlab import (
    Gains,
    Plant,
    Scenario,
    compute_metrics,
    optimize_gains,
    simulate,
    ziegler_nichols,
)


def _linear_scenario(t_end=12.0):
    """No saturation and no disturbance: a purely linear loop."""
    return Scenario(r_step=1.0, t_end=t_end, t_dist=t_end + 1.0, d_volts=0.0, dt=0.001, u_max=1e9)


def test_ultimate_gain_marks_the_stability_limit():
    """P-only control decays below Ku and grows above it (checks the closed-form Ku)."""
    plant = Plant()
    ku = plant.ultimate_gain()
    sc = _linear_scenario()

    def envelope(kp):
        res = simulate(plant, Gains(kp, 0.0, 0.0), sc, anti_windup=False)
        dev = np.abs(res.y - 1.0)
        early = dev[(res.t >= 2) & (res.t < 4)].max()
        late = dev[res.t >= 10].max()
        return early, late

    early, late = envelope(0.8 * ku)
    assert late < early, "loop should be stable below Ku"

    early, late = envelope(1.2 * ku)
    assert late > early, "loop should be unstable above Ku"


def test_oscillation_period_at_ku_matches_pu():
    plant = Plant()
    sc = _linear_scenario(t_end=3.0)
    res = simulate(plant, Gains(plant.ultimate_gain(), 0.0, 0.0), sc, anti_windup=False)
    y = res.y - 1.0
    # upward zero crossings after the initial transient
    idx = np.nonzero((y[:-1] < 0) & (y[1:] >= 0) & (res.t[1:] > 1.0))[0]
    periods = np.diff(res.t[idx])
    assert len(periods) >= 3
    assert np.median(periods) == pytest.approx(plant.ultimate_period(), rel=0.05)


def test_ziegler_nichols_formulas():
    g = ziegler_nichols(ku=5.0, pu=0.4, variant="classic")
    assert g.kp == pytest.approx(3.0)
    assert g.ki == pytest.approx(3.0 / 0.2)       # Kp / Ti, Ti = Pu / 2
    assert g.kd == pytest.approx(3.0 * 0.05)      # Kp * Td, Td = Pu / 8
    with pytest.raises(ValueError):
        ziegler_nichols(5.0, 0.4, variant="nope")


def test_output_never_exceeds_supply_limit():
    plant = Plant()
    g = ziegler_nichols(plant.ultimate_gain(), plant.ultimate_period())
    res = simulate(plant, g, Scenario())
    assert np.max(np.abs(res.u)) <= Scenario().u_max + 1e-9


def test_anti_windup_reduces_overshoot():
    plant = Plant()
    g = ziegler_nichols(plant.ultimate_gain(), plant.ultimate_period())
    with_aw = simulate(plant, g, Scenario(), anti_windup=True).metrics()
    without = simulate(plant, g, Scenario(), anti_windup=False).metrics()
    assert with_aw.overshoot_pct < without.overshoot_pct
    assert with_aw.steady_state_error < 0.1


def test_integral_action_rejects_constant_disturbance():
    plant = Plant()
    g = ziegler_nichols(plant.ultimate_gain(), plant.ultimate_period())
    res = simulate(plant, g, Scenario(), anti_windup=True)
    final_error = abs(res.y[-1] - res.r)
    assert final_error < 0.05

    # Without integral action the disturbance leaves a permanent offset.
    p_only = simulate(plant, Gains(g.kp, 0.0, g.kd), Scenario(), anti_windup=True)
    assert abs(p_only.y[-1] - p_only.r) > 0.3


def test_metrics_on_a_first_order_response():
    tau = 0.2
    t = np.arange(0, 3.0, 0.001)
    y = 1.0 - np.exp(-t / tau)
    u = np.ones_like(t)
    m = compute_metrics(t, y, u, r=1.0, t_dist=3.5)
    assert m.overshoot_pct == pytest.approx(0.0, abs=1e-9)
    assert m.rise_time == pytest.approx(2.197 * tau, abs=0.004)       # 10-90% rise time
    assert m.settling_time == pytest.approx(3.912 * tau, abs=0.004)   # 2% settling time
    assert np.isnan(m.dist_peak_dev)                                  # no disturbance phase in this signal


def test_metrics_flag_a_response_that_never_settles():
    t = np.arange(0, 2.0, 0.001)
    y = 0.5 * np.ones_like(t)
    m = compute_metrics(t, y, np.zeros_like(t), r=1.0, t_dist=3.0)
    assert np.isnan(m.settling_time)
    assert np.isnan(m.rise_time)


def test_optimizer_returns_gains_inside_bounds_and_beats_a_poor_guess():
    from pidlab.tuning import DEFAULT_BOUNDS, tuning_cost

    plant, sc = Plant(), Scenario()
    g, out = optimize_gains(plant, sc, maxiter=4, popsize=6)
    for value, (lo, hi) in zip((g.kp, g.ki, g.kd), DEFAULT_BOUNDS):
        assert lo <= value <= hi
    poor = tuning_cost(np.array([0.1, 0.0, 0.0]), plant, sc)
    assert out.fun < poor
