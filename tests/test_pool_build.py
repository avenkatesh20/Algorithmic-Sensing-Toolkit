"""Independent integration, sampling-law compatibility, and pool diagnostics."""

import warnings

import numpy as np
import pytest

import randomized_sensing as R
from matched_sensing import response
from randomized_sensing import _grid
from randomized_sensing.pulses import Sequence, _repair


@pytest.mark.parametrize("factor", [2, 4, 8, 16])
def test_fft_samples_match_dense_multisine_with_phase_origin(factor):
    length, tick = 800, 2.5e-9
    duration = length*tick
    f = (np.arange(4)+20.5)/(factor/2*duration)
    experiment = R.Experiment(tuple(f), 28e3, clock_s=tick)
    grid = _grid.CellGrid(experiment, duration, 123e-9, tick, length)
    assert grid.fft[1] == factor*length
    theta = np.random.default_rng(4).uniform(0, 2*np.pi, (17, len(f)))
    t = 123e-9+(np.arange(length)+.5)*tick
    dense = np.cos(theta)@np.cos(2*np.pi*f[:, None]*t)-np.sin(theta)@np.sin(2*np.pi*f[:, None]*t)
    np.testing.assert_allclose(grid.samples(theta), dense, rtol=1e-9, atol=1e-10)
    assert _grid.fft_grid(f+.01, duration, length) is None


def test_fully_periodic_tones_use_unpadded_fft():
    e = R.Experiment((1e6, 1.5e6), 28e3, clock_s=2.5e-9)
    grid = _grid.CellGrid(e, 2e-6, 123e-9, 2.5e-9, 800)
    assert grid.fft[1] == 800
    theta = np.random.default_rng(4).uniform(0, 2*np.pi, (3, 2))
    t = 123e-9+(np.arange(800)+.5)*2.5e-9
    f = np.array(e.frequencies_hz)
    dense = np.cos(theta[:, :, None]+2*np.pi*f[None, :, None]*t).sum(axis=1)
    np.testing.assert_allclose(grid.samples(theta), dense, rtol=1e-9, atol=1e-10)


def test_long_shots_use_natural_period_and_fold_repaired_cells_exactly():
    f = 10e6+(np.arange(4)-1.5)*25e3
    e = R.Experiment(tuple(f), 28e3, clock_s=2.5e-9)
    grid = _grid.CellGrid(e, 280e-6, 123e-9, 2.5e-9, 112000)
    assert grid.fft[1] == 32000 < grid.length
    theta = np.random.default_rng(4).uniform(0, 2*np.pi, (5, 4))
    t = 123e-9+(np.arange(grid.length)+.5)*grid.tick
    dense = np.cos(theta)@np.cos(2*np.pi*f[:, None]*t)-np.sin(theta)@np.sin(2*np.pi*f[:, None]*t)
    np.testing.assert_allclose(grid.samples(theta), dense, rtol=1e-9, atol=1e-10)
    patterns = np.where(grid.samples(theta) >= 0, 1, -1).astype(np.int8)
    for pattern in patterns:
        _repair(pattern, 8)
    # Simulate a boundary repair that destroys periodicity. Folding must still
    # integrate the actual cells, including a partial final period.
    patterns[0, :19] *= -1
    patterns[1, -23:] *= -1
    vectors = grid.responses(patterns)
    for s, vector in zip(patterns, vectors):
        switches = (np.flatnonzero(s[1:] != s[:-1])+1)*grid.tick
        expected = s[0]*response(e, switches, 280e-6, 123e-9)
        np.testing.assert_allclose(vector, expected, rtol=0, atol=1e-10)


def legacy_sequences(experiment, duration, count, start, seed, cap):
    """Original 32-pattern dense sampler; response() is the independent verifier."""
    tick = experiment.clock_s
    length = round(duration/tick)
    dwell = max(1, int(np.ceil(max(experiment.min_gap_s, experiment.edge_guard_s)/tick)))
    t = start+(np.arange(length)+.5)*tick
    angle = 2*np.pi*np.asarray(experiment.frequencies_hz)[:, None]*t
    ct, st = np.cos(angle), np.sin(angle)
    rng, sequences = np.random.default_rng(seed), []
    while len(sequences) < count:
        theta = rng.uniform(0, 2*np.pi, (min(32, count-len(sequences)), len(ct)))
        for s in np.where(np.cos(theta)@ct-np.sin(theta)@st >= 0, 1, -1).astype(np.int8):
            _repair(s, dwell)
            pulses = np.flatnonzero(s[1:] != s[:-1])+1
            if cap is None or len(pulses) <= cap:
                sequences.append(Sequence(tuple((pulses*tick).tolist()), int(s[0])))
    return tuple(sequences)


@pytest.mark.parametrize("n", [4, 20, 100])
@pytest.mark.parametrize("duration", [40e-6, 80e-6])
def test_seeded_fft_sequences_and_exact_signed_responses_match_original(n, duration):
    f = 10e6+(np.arange(n)-(n-1)/2)*25e3
    e = R.Experiment(tuple(f), 28e3, clock_s=2.5e-9)
    size = max(512, 16*n)
    p = R.build_pool(e, duration_s=duration, pool_size=size, start_time_s=123e-9)
    expected = legacy_sequences(e, duration, size//2, 123e-9, 4, None)
    assert p.sequences == expected
    assert p.synthesis_method == "fft"
    for i, q in enumerate(expected):
        z = q.initial_sign*response(e, q.switch_times_s, duration, 123e-9)
        np.testing.assert_allclose(p.Z[i], z, rtol=0, atol=1e-10)
        assert R.audit(p, i)["timing_valid"]
    assert {q.initial_sign for q in p.sequences} == {-1, 1}
    np.testing.assert_array_equal(p.Z[len(expected):], -p.Z[:len(expected)])


def test_arbitrary_tones_and_small_memory_chunks_match_literal_integrals(monkeypatch):
    e = R.Experiment((.813456e6, 1.28391e6, 2.17361e6, 3.65731e6), 28e3,
                     clock_s=2.5e-9, edge_guard_s=30.1e-9, min_gap_s=20.1e-9)
    settings = dict(duration_s=7e-6, pool_size=128, start_time_s=123e-9)
    normal = R.build_pool(e, **settings)
    monkeypatch.setattr(_grid, "_TABLE_ELEMENTS", 400)
    monkeypatch.setattr(_grid, "_BLOCK_ELEMENTS", 10000)
    chunked = R.build_pool(e, **settings)
    expected = legacy_sequences(e, 7e-6, 64, 123e-9, 4, None)
    assert normal.synthesis_method == chunked.synthesis_method == "dense"
    assert normal.sequences == chunked.sequences == expected
    np.testing.assert_allclose(normal.Z, chunked.Z, rtol=0, atol=1e-12)
    for i, q in enumerate(expected):
        np.testing.assert_allclose(chunked.Z[i], q.initial_sign*response(e, q.switch_times_s, 7e-6, 123e-9),
                                   rtol=0, atol=1e-10)
        check = R.audit(chunked, i)
        assert check["timing_valid"] and check["minimum_interval_s"] >= 32.5e-9*(1-1e-12)


def test_rejections_preserve_random_draw_order_and_pulse_cap(tmp_path):
    e = R.Experiment((.8e6, 1.05e6, 1.3e6, 1.7e6), 28e3, clock_s=2.5e-9)
    p = R.build_pool(e, duration_s=20e-6, pool_size=128, max_pulses=50)
    assert p.settings["attempts"] > len(p.sequences)
    assert p.sequences == legacy_sequences(e, 20e-6, 64, 0., 4, 50)
    assert all(R.audit(p, i)["timing_valid"] for i in range(len(p.Z)))
    assert max(len(q.switch_times_s) for q in p.sequences) <= 50
    plan = R.plan_acquisition(p, n_shots=400)
    # Export and load independently reintegrate every base waveform.
    saved = R.load_bundle(R.export_bundle(plan, tmp_path/"fft"))
    assert saved.plan_id == plan.plan_id
    np.testing.assert_array_equal(saved.pool.Z, p.Z)
    counts = R.simulate(plan, np.zeros(8))
    assert R.decode(saved, counts) == R.decode(plan, counts)


def test_unresolved_tones_raise_before_returning_unstable_pool_and_allow_override(tmp_path):
    f = np.sort(np.random.default_rng(0).uniform(9.7e6, 10.3e6, 20))
    e = R.Experiment(tuple(f), 28e3, clock_s=2.5e-9)
    with pytest.warns(RuntimeWarning, match="unresolved"):
        with pytest.raises(ValueError, match="allow_ill_conditioned=True"):
            R.build_pool(e, duration_s=40e-6, pool_size=256)
    with pytest.warns(RuntimeWarning) as recorded:
        p = R.build_pool(e, duration_s=40e-6, pool_size=256, allow_ill_conditioned=True)
    assert len(recorded) == 2 and p.condition > 1e3
    assert p.condition == pytest.approx(np.linalg.cond(p.Sigma), rel=1e-7)
    plan = R.plan_acquisition(p, n_shots=100, seed=5)
    output = R.export_bundle(plan, tmp_path/"legacy_unstable")
    with pytest.warns(RuntimeWarning, match="covariance condition"):
        loaded = R.load_bundle(output)
    assert loaded.plan_id == plan.plan_id
    np.testing.assert_array_equal(loaded.pool.Z, p.Z)
    counts = R.simulate(plan, np.zeros(40))
    assert R.decode(loaded, counts) == R.decode(plan, counts)


def test_healthy_notebook_pool_has_no_warning():
    e = R.Experiment((.8e6, 1.05e6, 1.31e6, 1.72e6), 100e3, clock_s=2.5e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = R.build_pool(e, duration_s=20e-6, pool_size=128)
    assert p.condition < 20
    assert p.condition == pytest.approx(np.linalg.cond(p.Sigma))
    with pytest.raises(ValueError, match="must be a bool"):
        R.build_pool(e, duration_s=20e-6, allow_ill_conditioned=1)


def test_moderately_conditioned_pool_warns_but_does_not_require_override():
    e = R.Experiment((1e6, 1.001e6), 28e3, clock_s=2.5e-9)
    with pytest.warns(RuntimeWarning) as recorded:
        p = R.build_pool(e, duration_s=200e-6, pool_size=128)
    assert 20 < p.condition < 1e3 and len(recorded) == 2


def test_free_running_commensurate_period_warns_without_changing_plan():
    e = R.Experiment((1e6, 1.5e6), 28e3, clock_s=2.5e-9)
    p = R.build_pool(e, duration_s=2e-6, pool_size=128)
    with pytest.warns(RuntimeWarning, match="no phase diversity"):
        a = R.plan_acquisition(p, n_shots=100, field_mode="free_running", shot_period_s=4e-6)
    b = R.plan_acquisition(p, n_shots=100, field_mode="free_running",
                           start_times_s=p.start_time_s+np.arange(100)*4e-6)
    assert a.rows == b.rows and a.plan_id == b.plan_id
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        R.plan_acquisition(p, n_shots=100, field_mode="free_running", shot_period_s=4.0375e-6)
