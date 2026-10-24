"""Back-rotated RQS must work below the full likelihood parameter-count floor."""

from collections import defaultdict
import warnings

import numpy as np
import pytest
from scipy.linalg import cho_solve

import randomized_sensing as R
from randomized_sensing._estimator import finish_moment
from randomized_sensing.readout import _timed_moment


@pytest.fixture(scope="module")
def pool():
    e = R.Experiment((1e6, 1.5e6), 100e3, clock_s=2.5e-9,
                     readout_bias=.12, visibility=.8, t2_s=200e-6)
    return R.build_pool(e, duration_s=4e-6, pool_size=512, start_time_s=123e-9)


def grouped(plan, counts):
    totals, positive = defaultdict(int), defaultdict(int)
    for row in plan.rows:
        key = (row["basis"], row["sequence_index"], row["outcome_sign"])
        totals[key] += 1
        positive[key] += counts.positive_counts[row["row_id"]]
    rows, values = [], {}
    for (basis, index, sign), shots in totals.items():
        key = f"{basis}_{index}"
        rows.append(dict(row_id=key, sequence_index=index, basis=basis, shots=shots, outcome_sign=sign))
        values[key] = positive[(basis, index, sign)]
    return R.AcquisitionPlan(plan.pool, tuple(rows), plan.seed), values


def test_reference_frame_and_commensurate_schedule_match_phase_locked(pool):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        plan = R.plan_acquisition(pool, n_shots=2000, field_mode="free_running", shot_period_s=6e-6)
    counts = R.simulate(plan, [.12, -.08, .1, .15])
    locked, positives = grouped(plan, counts)
    expected = R.decode(locked, positives)
    actual = R.decode(plan, counts)
    np.testing.assert_allclose(actual.amplitudes, expected.amplitudes, rtol=0, atol=1e-12)
    # All starts equal is a mathematical identity check only. Actual acquisition
    # validation continues to reject overlapping shots.
    a, floor, mu = _timed_moment(pool, plan.rows, counts.positive_counts,
                                 np.full(len(plan.rows), pool.start_time_s))
    np.testing.assert_allclose(a, expected.amplitudes, rtol=0, atol=1e-12)
    assert floor == expected.guard_floor and mu == pytest.approx(expected.x_mean)
    with pytest.raises(ValueError, match="overlapping"):
        R.decode(plan, R.TimedCounts(counts.positive_counts,
                 {r["row_id"]: pool.start_time_s for r in plan.rows}))


def test_whiten_before_rotate_matches_direct_formula_and_order_matters(pool):
    plan = R.plan_acquisition(pool, n_shots=120, field_mode="free_running", shot_period_s=5.2345e-6)
    counts = R.simulate(plan, [.12, -.08, .1, .15])
    times = np.array([counts.start_times_s[r["row_id"]] for r in plan.rows])
    back, wrong, x = [], [], []
    for row, t in zip(plan.rows, times):
        value = 2*counts.positive_counts[row["row_id"]]-1-pool.experiment.readout_bias
        if row["basis"] == "X":
            x.append(value)
        else:
            z = pool.Z[row["sequence_index"]]
            back.append(row["outcome_sign"]*value*pool.rotate_vectors(cho_solve(pool.chol, z), t))
            wrong.append(row["outcome_sign"]*value*cho_solve(pool.chol, pool.rotate_vectors(z, t)))
    expected, _ = finish_moment(pool, np.mean(back, axis=0), np.mean(x))
    actual = R.decode(plan, counts)
    np.testing.assert_allclose(actual.amplitudes, expected, rtol=0, atol=1e-12)
    assert np.linalg.norm(np.mean(back, axis=0)-np.mean(wrong, axis=0)) > 1e-4


def test_fifty_tones_below_six_n_and_two_shot_minimum():
    f = 10e6+(np.arange(50)-24.5)*25e3
    p = R.build_pool(R.Experiment(tuple(f), 28e3, clock_s=2.5e-9),
                     duration_s=40e-6, pool_size=1024)
    for shots in (2, 200):
        plan = R.plan_acquisition(p, n_shots=shots, field_mode="free_running", shot_period_s=41.2345e-6)
        counts = R.simulate(plan, np.zeros(100))
        basic = R.decode(plan, counts)
        refined = R.decode(plan, counts, method="TIMESTAMPED-ML")
        assert basic.method == "RQS" and not basic.ml_applied
        assert np.isfinite(basic.amplitudes).all()
        assert refined.amplitudes == basic.amplitudes and not refined.ml_applied
        assert "3*2N" in refined.refinement_skipped_reason


def test_noise_free_random_schedule_and_continuous_raw_outcomes(pool):
    plan = R.plan_acquisition(pool, n_shots=40000, field_mode="free_running", shot_period_s=5.2345e-6)
    a = np.array([.12, -.08, .1, .15])
    starts = np.array([r["start_time_s"] for r in plan.rows])+np.random.default_rng(7).uniform(0, 10e-9, len(plan.rows))
    population = R.population_estimate(plan, a, start_times_s=starts)
    locked = R.population_estimate(pool, a)
    assert population.method == "RQS-population" and not population.ml_applied
    assert max(abs(np.array(population.amplitudes)-a)) < .015
    assert max(abs(np.array(locked.amplitudes)-a)) < .015
    counts = R.simulate(plan, a, start_times_s=starts)
    values = [2*counts.positive_counts[r["row_id"]]-1 for r in plan.rows]
    continuous = R.TimedOutcomes(values, starts)
    assert R.decode(plan, continuous) == R.decode(plan, counts)
    values[:2] = [.54, -.1]
    assert np.isfinite(R.decode(plan, R.TimedOutcomes(values, starts)).amplitudes).all()
    values[0] = 1.01
    with pytest.raises(ValueError, match="finite numbers"):
        R.decode(plan, R.TimedOutcomes(values, starts))


def test_rotation_helper_accepts_arbitrary_vectors_and_preserves_norm(pool):
    v = np.random.default_rng(1).normal(size=(7, 4))
    times = np.linspace(0, .1, 7)
    actual = pool.rotate_vectors(v, times)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), np.linalg.norm(v, axis=1), atol=1e-12)
    np.testing.assert_array_equal(pool.rotate_vectors(v, np.full(7, pool.start_time_s)), v)
    with pytest.raises(ValueError, match="matching"):
        pool.rotate_vectors(v, times[:2])


def test_missing_measured_timestamps_never_fall_back_to_planned_times(pool):
    plan = R.plan_acquisition(pool, n_shots=4, field_mode="free_running", shot_period_s=5.2345e-6)
    counts = R.simulate(plan, np.zeros(4))
    for outcomes in (R.TimedCounts(counts.positive_counts, None), R.TimedOutcomes([.54]*4, None)):
        with pytest.raises(ValueError, match="actual start timestamps"):
            R.decode(plan, outcomes)


def test_actual_likelihood_failure_is_not_silently_replaced_by_moments(pool, monkeypatch):
    plan = R.plan_acquisition(pool, n_shots=120, field_mode="free_running", shot_period_s=5.2345e-6)
    counts = R.simulate(plan, np.zeros(4))

    def failed_fit(*args, **kwargs):
        raise RuntimeError("Timestamped likelihood fit did not converge")

    monkeypatch.setattr("randomized_sensing.readout.fit_timestamped", failed_fit)
    with pytest.raises(RuntimeError, match="did not converge"):
        R.decode(plan, counts, method="TIMESTAMPED-ML")


def test_fifty_tone_population_error_is_comparable_between_field_modes():
    f = 10e6+(np.arange(50)-24.5)*25e3
    p = R.build_pool(R.Experiment(tuple(f), 28e3, clock_s=2.5e-9), duration_s=120e-6)
    a = np.random.default_rng(13).normal(size=100)
    a *= .65/np.linalg.norm(a)
    plan = R.plan_acquisition(p, n_shots=60000, field_mode="free_running", shot_period_s=121.2345e-6)
    locked = R.population_estimate(p, a)
    free = R.population_estimate(plan, a)
    e_locked = max(abs(a-locked.amplitudes))
    e_free = max(abs(a-free.amplitudes))
    assert e_locked < .015 and e_free < .015
    assert e_free < 2*e_locked+.002


def test_parallel_table_preparation_keeps_identical_phase_locked_results():
    f = 10e6+(np.arange(20)-9.5)*25e3
    p = R.build_pool(R.Experiment(tuple(f), 28e3, clock_s=2.5e-9), duration_s=80e-6, pool_size=512)
    a = np.random.default_rng(3).normal(size=40)
    a *= .65/np.linalg.norm(a)
    before = R.population_estimate(p, a)
    expected = p._H.copy()
    del p._H
    p._map_workers = 2
    after = R.population_estimate(p, a)
    np.testing.assert_array_equal(p._H, expected)
    assert before == after
