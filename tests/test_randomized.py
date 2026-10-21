from dataclasses import replace
import csv

import numpy as np
import pytest

import randomized_sensing as R

QUERIES = dict(cosine_weights=((1., .2), (.1, -.3), (1.2, -.4)),
               sine_weights=((.2, -.3), (.4, .1), (1., -.1)))


@pytest.fixture(scope="module")
def pool():
    e = R.Experiment((.813456e6, 1.28391e6), 100e3, t2_s=200e-6,
                     visibility=.8, readout_bias=.12, min_gap_s=20.1e-9,
                     edge_guard_s=30.1e-9, clock_s=7e-9, overhead_s=1e-6)
    return R.build_pool(e, duration_s=7e-6, pool_size=512, max_pulses=32, start_time_s=123e-9, seed=4)


def test_arbitrary_frequencies_literal_integrals_and_physical_readout(pool):
    a = np.array([.2, -.1, .3, .1])
    seen_parities = set()
    f = np.array(pool.experiment.frequencies_hz)
    for i in range(len(pool.Z)):
        q = pool.sequence(i)
        check = R.audit(pool, i)
        assert check["timing_valid"] and check["pulse_count"] <= 32
        assert check["minimum_interval_s"] >= 35e-9*(1-1e-12)
        assert check["response_error"] < 1e-12
        edges = np.r_[0., q.switch_times_s, pool.duration_s]
        z = np.sum((-1.)**np.arange(len(edges)-1)[:, None]
                   * np.diff(np.exp(2j*np.pi*(edges[:, None]+pool.start_time_s)*f), axis=0)
                   / (2j*np.pi*f), axis=0)*2*np.pi*pool.experiment.coupling_hz
        canonical = np.r_[z.real, z.imag]
        np.testing.assert_allclose(q.initial_sign*canonical, pool.Z[i], atol=1e-12)
        # Independently propagate |+X> through free evolution and actual X(pi) pulses.
        state = np.array([1., 1.], complex)/np.sqrt(2)
        for j, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            integ = (np.exp(2j*np.pi*(right+pool.start_time_s)*f)-np.exp(2j*np.pi*(left+pool.start_time_s)*f))/(2j*np.pi*f)
            phase = 2*np.pi*pool.experiment.coupling_hz*np.r_[integ.real, integ.imag]@a
            state *= np.exp(np.array([-1j, 1j])*phase/2)
            if j < len(q.switch_times_s):
                state = -1j*state[::-1]
        physical_y = 2*np.imag(np.conj(state[0])*state[1])
        assert physical_y == pytest.approx(check["y_outcome_sign"]*np.sin(pool.Z[i]@a), abs=1e-12)
        seen_parities.add(len(q.switch_times_s) % 2)
    assert seen_parities == {0, 1}
    np.testing.assert_array_equal(pool.Z[len(pool.sequences):], -pool.Z[:len(pool.sequences)])


def test_reproducibility_and_upward_timing_rounding(pool):
    p = R.build_pool(pool.experiment, duration_s=7e-6, pool_size=512, max_pulses=32, start_time_s=123e-9, seed=4)
    assert p.sequences == pool.sequences
    np.testing.assert_array_equal(p.Z, pool.Z)
    a = R.plan_acquisition(pool, n_shots=100, seed=5)
    b = R.plan_acquisition(pool, n_shots=100, seed=5)
    assert a.plan_id == b.plan_id and a.rows == b.rows
    assert a.active_time_s == pytest.approx(100*7e-6)
    assert a.time_with_overhead_s == pytest.approx(100*8e-6)


def test_raw_bias_parity_counts_and_shared_queries(pool):
    plan = R.plan_acquisition(pool, n_shots=50000, x_fraction=.25, seed=19)
    a = np.array([.12, -.08, .1, .15])
    counts = R.simulate(plan, a, seed=42)
    results = [R.decode(plan, counts, method=m, **QUERIES) for m in ("RQS", "RQS-ML")]
    for result in results:
        assert np.max(abs(np.array(result.amplitudes)-a)) < .08
        assert result.values[2] == pytest.approx(result.values[0]+2*result.values[1], abs=1e-14)
        assert not result.floor_active
    assert results[1].ml_applied
    # Low X means expose the guard rather than claiming an accuracy certificate.
    for row in plan.rows:
        if row["basis"] == "X":
            counts[row["row_id"]] = 0
    assert R.decode(plan, counts).floor_active
    with pytest.raises(ValueError, match="every"):
        R.decode(plan, {})
    with pytest.raises(ValueError, match="amplitude_bound"):
        R.simulate(plan, np.ones(4))


def test_portable_export_counts_and_integrity(pool, tmp_path):
    plan = R.plan_acquisition(pool, n_shots=400, seed=5)
    output = R.export_bundle(plan, tmp_path/"experiment")
    saved = R.load_bundle(output)
    assert saved.plan_id == plan.plan_id
    counts = R.simulate(plan, np.zeros(4), seed=18)
    R.write_counts(output/"measured.csv", plan, counts)
    assert R.decode(saved, R.read_counts(output/"measured.csv", saved)) == R.decode(plan, counts)
    with pytest.raises(ValueError, match="integer"):
        R.read_counts(output/"counts_template.csv", saved)
    with pytest.raises(ValueError, match="separate"):
        R.write_counts(output/"pulses.csv", plan, counts)
    with (output/"acquisition.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert sum(int(r["shots"]) for r in rows) == plan.total_shots
    assert all(int(r["outcome_sign"]) in (-1, 1) for r in rows)
    (output/"intervals.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        R.load_bundle(output)


def test_invalid_inputs_and_dc(pool):
    with pytest.raises(ValueError, match="even"):
        R.build_pool(pool.experiment, duration_s=7e-6, pool_size=31)
    with pytest.raises(ValueError, match="whole number"):
        R.build_pool(pool.experiment, duration_s=7.001e-6, pool_size=32)
    with pytest.raises(ValueError, match="positive frequencies"):
        R.build_pool(R.Experiment((0.,), 100e3), duration_s=1e-6, pool_size=32)
    with pytest.raises(ValueError, match="Too few|singular"):
        R.build_pool(pool.experiment, duration_s=7e-6, pool_size=32, max_pulses=0)
    with pytest.raises(ValueError, match="integer"):
        R.plan_acquisition(pool, n_shots=1.5)
    plan = R.plan_acquisition(pool, n_shots=50)
    row = dict(plan.rows[0], outcome_sign=0)
    with pytest.raises(ValueError, match="outcome sign"):
        R.decode(replace(plan, rows=(row, *plan.rows[1:])), {})


@pytest.mark.parametrize("method", ["RQS", "RQS-ML"])
def test_queries_selected_after_acquisition_reuse_identical_counts(pool, method):
    plan = R.plan_acquisition(pool, n_shots=2000, seed=7)
    identity = plan.plan_id
    amplitudes = np.array([.1, -.2, .05, .15])
    counts = R.simulate(plan, amplitudes, seed=8)
    original_counts = counts.copy()
    reconstruction = R.decode(plan, counts, method=method)
    assert reconstruction.values == () and "cosine_weights" not in pool.experiment.to_dict()
    first = R.decode(plan, counts, method=method, **QUERIES)
    second_queries = dict(cosine_weights=((1., 0.), (0., 1.)), sine_weights=((0., 0.), (0., 0.)))
    second = R.decode(plan, counts, method=method, **second_queries)
    assert reconstruction.amplitudes == first.amplitudes == second.amplitudes
    assert first.values == R.evaluate_observables(reconstruction.amplitudes, **QUERIES)
    np.testing.assert_array_equal(second.values, reconstruction.amplitudes[:2])
    assert counts == original_counts and plan.plan_id == identity
    population = R.population_estimate(pool, amplitudes)
    population_queries = R.population_estimate(pool, amplitudes, **QUERIES)
    assert population.values == ()
    assert population_queries.values == R.evaluate_observables(population.amplitudes, **QUERIES)
    with pytest.raises(ValueError, match="both"):
        R.decode(plan, counts, cosine_weights=(1., 0.))
    with pytest.raises(ValueError, match="matching"):
        R.evaluate_observables(reconstruction.amplitudes, cosine_weights=(1.,), sine_weights=(0.,))


def test_earlier_bundles_keep_count_identity_and_allow_new_queries(pool, tmp_path):
    from matched_sensing import Experiment as LegacyExperiment
    e = LegacyExperiment(**pool.experiment.to_dict(), **QUERIES)
    legacy = R.Pool(e, pool.duration_s, pool.start_time_s, pool.design_clock_s,
                    pool.sequences, pool.Z, pool.settings)
    plan = R.plan_acquisition(legacy, n_shots=400, seed=5)
    saved = R.load_bundle(R.export_bundle(plan, tmp_path/"earlier"))
    assert saved.plan_id == plan.plan_id
    counts = R.simulate(plan, np.zeros(4), seed=9)
    R.write_counts(tmp_path/"measured.csv", plan, counts)
    assert R.read_counts(tmp_path/"measured.csv", saved) == counts
    assert R.decode(saved, counts).values == R.decode(plan, counts, **QUERIES).values
    new = dict(cosine_weights=(1., 0.), sine_weights=(0., 0.))
    assert len(R.decode(saved, counts, **new).values) == 1
