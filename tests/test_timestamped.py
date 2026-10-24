from dataclasses import replace
import csv
import json

import numpy as np
import pytest

import randomized_sensing as R
from randomized_sensing._timestamped import objective, fit as legacy_fit


@pytest.fixture(scope="module")
def pool():
    e = R.Experiment((.813456e6, 1.28391e6), 100e3, t2_s=200e-6,
                     visibility=.8, readout_bias=.12, min_gap_s=20.1e-9,
                     edge_guard_s=30.1e-9, clock_s=7e-9, overhead_s=1e-6)
    return R.build_pool(e, duration_s=7e-6, pool_size=128, start_time_s=123e-9, seed=4)


def test_rotated_responses_match_literal_pulse_integrals_at_actual_starts(pool):
    indices = np.array([0, 1, 7, 63, 64, 65, 127])
    starts = np.array([0., 5e-9, 3.71e-6, .001234567, .08, .17, .314159265])
    rotated = pool.responses_at(indices, starts)
    f = np.array(pool.experiment.frequencies_hz, np.longdouble)
    pi = np.arccos(np.longdouble(-1))
    for index, start, actual in zip(indices, starts, rotated):
        q = pool.sequence(int(index))
        edges = np.array([0., *q.switch_times_s, pool.duration_s], np.longdouble)
        widths = np.diff(edges)[:, None]
        midpoints = np.longdouble(start)+(edges[:-1]+edges[1:])[:, None]/2
        signs = q.initial_sign*(-1.)**np.arange(len(edges)-1)
        integral = signs@(widths*np.sinc(widths*f)*np.exp(2j*pi*midpoints*f))
        z = 2*pi*pool.experiment.coupling_hz*integral
        # On float64-only platforms, direct integrals at ~1e6 radians lose
        # precision when adding sub-microsecond intervals to long start times.
        np.testing.assert_allclose(actual, np.r_[z.real, z.imag], rtol=0, atol=2e-9)
        np.testing.assert_array_equal(pool.responses_at(int(index), float(start)), actual)
        assert R.audit(pool, int(index))["timing_valid"]
    with pytest.raises(ValueError, match="matching"):
        pool.responses_at(indices, starts[:-1])


def test_running_schedule_keeps_individual_shots_and_validates_timing(pool):
    plan = R.plan_acquisition(pool, n_shots=100, field_mode="free_running", shot_period_s=10.037e-6, seed=5)
    assert len(plan.rows) == plan.total_shots == 100
    assert all(r["shots"] == 1 for r in plan.rows)
    assert plan.rows == R.plan_acquisition(pool, n_shots=100, field_mode="free_running", shot_period_s=10.037e-6, seed=5).rows
    assert [r["basis"] for r in plan.rows] != sorted(r["basis"] for r in plan.rows)
    assert plan.scheduled_span_s == pytest.approx(99*10.037e-6+pool.duration_s)
    assert plan.active_time_s == pytest.approx(100*pool.duration_s)
    with pytest.raises(ValueError, match="overhead"):
        R.plan_acquisition(pool, n_shots=100, field_mode="free_running", shot_period_s=pool.duration_s)
    with pytest.raises(ValueError, match="overlapping"):
        R.plan_acquisition(pool, n_shots=100, field_mode="free_running", start_times_s=np.zeros(100))
    with pytest.raises(ValueError, match="not both"):
        R.plan_acquisition(pool, n_shots=100, field_mode="free_running", start_times_s=np.arange(100)*1e-5, shot_period_s=1e-5)


def test_actual_timestamps_control_inference_and_posthoc_queries(pool):
    plan = R.plan_acquisition(pool, n_shots=8000, x_fraction=.25, field_mode="free_running", shot_period_s=10.037e-6, seed=5)
    starts = np.array([r["start_time_s"] for r in plan.rows])+127e-9
    a = np.array([.12, -.08, .1, .15])
    counts = R.simulate(plan, a, start_times_s=starts, seed=42)
    identity = plan.plan_id
    result = R.decode(plan, counts, method="TIMESTAMPED-ML")
    assert result.method == "TIMESTAMPED-ML" and result.fit_converged
    assert result.guard_floor is not None and result.values == ()
    assert np.max(abs(np.array(result.amplitudes)-a)) < .04
    assert abs(result.fitted_contrast-pool.experiment.contrast(pool.duration_s)) < .04
    wrong = R.TimedCounts(counts.positive_counts, {r["row_id"]: r["start_time_s"] for r in plan.rows})
    assert np.max(abs(np.array(R.decode(plan, wrong, method="TIMESTAMPED-ML").amplitudes)-a)) > .1
    queries = dict(cosine_weights=((1., .2), (.1, -.3)), sine_weights=((.2, -.3), (.4, .1)))
    queried = R.decode(plan, counts, method="TIMESTAMPED-ML", **queries)
    assert queried.amplitudes == result.amplitudes
    assert queried.values == R.evaluate_observables(result.amplitudes, **queries)
    assert plan.plan_id == identity
    # Exact outcome probabilities, not a true-amplitude initialization in the decoder.
    population = R.population_estimate(plan, a, start_times_s=counts.start_times_s,
                                      method="TIMESTAMPED-ML", **queries)
    np.testing.assert_allclose(population.amplitudes, a, atol=2e-5)
    with pytest.raises(ValueError, match="TIMESTAMPED-ML"):
        R.decode(plan, counts, method="RQS-ML")
    with pytest.raises(ValueError, match="actual start"):
        R.decode(plan, counts.positive_counts)


@pytest.mark.parametrize("t2_s", [None, 200e-6])
def test_both_decoders_recover_independent_physical_pulse_data(pool, t2_s):
    # Generate physical outcomes without the simulator, stored response table,
    # response rotation helper, or the decoder's outcome_sign metadata.
    e = replace(pool.experiment, t2_s=t2_s)
    p = R.Pool(e, pool.duration_s, pool.start_time_s, pool.design_clock_s,
               pool.sequences, pool.Z.copy(), dict(pool.settings))
    plan = R.plan_acquisition(p, n_shots=8000, x_fraction=.25,
                              field_mode="free_running", shot_period_s=10.037e-6)
    starts = np.array([r["start_time_s"] for r in plan.rows])+127e-9
    starts += np.random.default_rng(31).uniform(0, 10e-9, len(starts))
    truth = np.array([.12, -.08, .1, .15])
    f = np.asarray(e.frequencies_hz, np.longdouble)
    pi = np.arccos(np.longdouble(-1))
    raw, parities = [], set()
    for row, start in zip(plan.rows, starts):
        switches = p.sequence(row["sequence_index"]).switch_times_s
        edges = np.array([0., *switches, p.duration_s], np.longdouble)
        widths = np.diff(edges)[:, None]
        midpoints = np.longdouble(start)+(edges[:-1]+edges[1:])[:, None]/2
        # Physical controls always begin at +1, including negative virtual entries.
        modulation = (-1.)**np.arange(len(edges)-1)
        integral = modulation@(widths*np.sinc(widths*f)*np.exp(2j*pi*midpoints*f))
        phase = float(2*pi*e.coupling_hz*(integral.real@truth[:p.N]+integral.imag@truth[p.N:]))
        parity = (-1)**len(switches)
        parities.add(parity)
        response = parity*np.sin(phase) if row["basis"] == "Y" else np.cos(phase)
        raw.append(e.readout_bias+e.visibility*np.exp(-p.duration_s/t2_s if t2_s else 0)*response)
    assert parities == {-1, 1}
    assert {r["sequence_index"] < len(p.sequences) for r in plan.rows} == {False, True}
    outcomes = R.TimedOutcomes(raw, starts)
    moment = R.decode(plan, outcomes)
    refined = R.decode(plan, outcomes, method="TIMESTAMPED-ML")
    assert np.max(abs(np.array(moment.amplitudes)-truth)) < .015
    assert refined.ml_applied and refined.fit_converged
    np.testing.assert_allclose(refined.amplitudes, truth, atol=2e-5, rtol=0)
    assert refined.fitted_contrast == pytest.approx(e.contrast(p.duration_s), abs=2e-5)
    # The fourth start must retain the likelihood quality of the legacy three.
    positive = {r["row_id"]: (1+v)/2 for r, v in zip(plan.rows, raw)}
    legacy, contrast, _, _ = legacy_fit(p, plan.rows, positive, starts)
    z = p.responses_at(np.array([r["sequence_index"] for r in plan.rows]), starts)
    evaluate = objective(z, np.array([r["basis"] == "Y" for r in plan.rows]),
                         np.array([r["outcome_sign"] for r in plan.rows]),
                         np.ones(len(plan.rows)), np.array([(1+v)/2 for v in raw]), e.readout_bias)
    assert evaluate(np.r_[refined.amplitudes, refined.fitted_contrast])[0] <= evaluate(np.r_[legacy, contrast])[0]+1e-10


def test_timestamped_csv_roundtrip_and_missing_or_invalid_timestamps(pool, tmp_path):
    plan = R.plan_acquisition(pool, n_shots=500, field_mode="free_running", shot_period_s=10.037e-6, seed=5)
    output = R.export_bundle(plan, tmp_path/"running")
    saved = R.load_bundle(output)
    assert saved.field_mode == "free_running" and saved.plan_id == plan.plan_id
    starts = np.array([r["start_time_s"] for r in plan.rows])+31e-9
    counts = R.simulate(plan, np.array([.1, .05, -.1, .2]), start_times_s=starts, seed=3)
    measured = output/"measured.csv"
    R.write_counts(measured, plan, counts)
    restored = R.read_counts(measured, saved)
    assert restored == counts
    assert R.decode(saved, restored) == R.decode(plan, counts)
    assert "cosine_weights" not in json.loads((output/"bundle.json").read_text())["experiment"]
    with (output/"acquisition.csv").open() as stream:
        assert "planned_start_time_s" in csv.DictReader(stream).fieldnames
    with measured.open() as stream:
        reader = csv.DictReader(stream); columns = reader.fieldnames; rows = list(reader)
    assert rows[0]["start_time_s"] != str(plan.rows[0]["start_time_s"])
    for invalid in ("", "nan", "inf", "-1"):
        rows[0]["start_time_s"] = invalid
        with measured.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
        with pytest.raises(ValueError, match="start_time|start timestamp"):
            R.read_counts(measured, saved)
    with pytest.raises(ValueError, match="integer"):
        R.read_counts(output/"counts_template.csv", saved)


def test_binomial_likelihood_gradient_includes_physical_parity_and_bias(pool):
    plan = R.plan_acquisition(pool, n_shots=31, field_mode="free_running", shot_period_s=10.037e-6, seed=5)
    rows = plan.rows
    z = pool.responses_at(np.array([r["sequence_index"] for r in rows]), np.array([r["start_time_s"] for r in rows]))
    y = np.array([r["basis"] == "Y" for r in rows]); signs = np.array([r["outcome_sign"] for r in rows])
    plus = np.random.default_rng(2).integers(2, size=len(rows)).astype(float)
    evaluate = objective(z, y, signs, np.ones(len(rows)), plus, pool.experiment.readout_bias)
    x = np.array([.1, -.2, .15, .05, .7])
    _, gradient = evaluate(x)
    finite = np.empty_like(x)
    for i in range(len(x)):
        delta = np.zeros_like(x); delta[i] = 1e-6
        finite[i] = (evaluate(x+delta)[0]-evaluate(x-delta)[0])/2e-6
    np.testing.assert_allclose(gradient, finite, rtol=1e-6, atol=1e-9)


def test_degenerate_timestamped_design_returns_moments_without_refinement():
    pool = R.build_pool(R.Experiment((1e6, 2e6), 100e3, clock_s=2.5e-9),
                        duration_s=1e-6, pool_size=32, seed=4)
    plan = R.plan_acquisition(pool, n_shots=64, field_mode="free_running", start_times_s=np.arange(64)*2e-6)
    q = pool.sequence(0)
    rows = tuple(dict(r, sequence_index=0, outcome_sign=q.initial_sign*(-1)**len(q.switch_times_s) if r["basis"] == "Y" else 1) for r in plan.rows)
    plan = replace(plan, rows=rows)
    counts = R.simulate(plan, np.zeros(4), seed=1)
    result = R.decode(plan, counts, method="TIMESTAMPED-ML")
    moment = R.decode(plan, counts)
    assert not result.ml_applied and result.fit_converged is None
    assert "span all" in result.refinement_skipped_reason
    assert result.amplitudes == moment.amplitudes
