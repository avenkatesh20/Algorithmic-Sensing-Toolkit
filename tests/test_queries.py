from dataclasses import replace

import numpy as np
import pytest

from matched_sensing import (
    Experiment, ReadoutError, compile_bank, compile_queries, query_space,
    plan_queries, decode_queries, simulate_queries, export_bundle, load_bundle,
    read_counts, write_counts,
)
from matched_sensing.queries import validate_query_plan


@pytest.fixture(scope="module")
def query_bank():
    c = np.array([[.5, -.3, 0, .4], [.1, .2, -.4, .3]])
    s = np.array([[.2, .4, .1, -.2], [-.3, .1, .2, .25]])
    c = np.vstack([c, c[0]+2*c[1], np.zeros(4)])
    s = np.vstack([s, s[0]+2*s[1], np.zeros(4)])
    e = Experiment((.8e6, 1.05e6, 1.31e6, 1.72e6), c, s,
                   100e3, t2_s=200e-6, amplitude_bound=1.7)
    return compile_queries(e, max_gain=4, max_duration_s=40e-6, max_pulses=128,
                           start_time_s=123e-9, seed=4)


def test_factorization_and_shared_rank(query_bank):
    e, s = query_bank.experiment, query_bank.space
    u, v = np.array(s.basis), np.array(s.coefficients)
    assert s.rank == len(query_bank.banks) == 2 < s.count == 4
    np.testing.assert_allclose(u.T@u, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(s.kappa*v@u.T, e.weights, atol=1e-14)
    assert s.kappa == pytest.approx(np.linalg.norm(e.weights, axis=1).max())
    assert np.linalg.norm(v, axis=1).max() == pytest.approx(1.)
    for i, bank in enumerate(query_bank.banks):
        np.testing.assert_allclose(bank.experiment.weights, u[:, i])
        assert bank.experiment.min_gap_s == 20e-9
        assert bank.experiment.amplitude_bound == 1.7
    with pytest.raises(ValueError, match="compile_queries"):
        compile_bank(e)


def test_query_budget_and_joint_reconstruction(query_bank):
    plan = plan_queries(query_bank, precision=(.08, .1, .15, .01), confidence=.95)
    assert np.all(np.array(plan.error_bounds) <= np.array(plan.precisions)*(1+1e-12))
    assert sum(1-p.confidence for p in plan.coordinate_plans) <= .05+1e-12
    assert plan.active_time_s == sum(p.active_time_s for p in plan.coordinate_plans)
    assert len({r["row_id"] for r in plan.rows}) == len(plan.rows)
    assert {r["coordinate_index"] for r in plan.rows} == {0, 1}
    for row in plan.rows:
        assert query_bank.sequences[row["sequence_index"]].duration_s == row["duration_s"]
    rng = np.random.default_rng(31)
    for i in range(25):
        a = rng.normal(size=8); a *= 1.7*rng.uniform(.1, 1)/np.linalg.norm(a)
        truth = query_bank.experiment.weights@a
        result = decode_queries(plan, simulate_queries(plan, a, seed=i))
        lo, hi = np.array(result.intervals).T
        assert np.all(lo <= truth) and np.all(truth <= hi)
        assert np.all(hi-lo <= 2*np.array(plan.precisions)+1e-12)
        assert result.values[2] == pytest.approx(result.values[0]+2*result.values[1], abs=1e-14)
        assert result.values[3] == 0 and result.intervals[3] == (0., 0.)
    shots = sum(row["shots"] for row in plan.rows)
    with pytest.raises(ReadoutError, match="total max_shots"):
        plan_queries(query_bank, precision=plan.precisions, max_shots=shots-1)
    with pytest.raises(ReadoutError, match="every"):
        decode_queries(plan, {})
    with pytest.raises(ValueError, match="amplitude_bound"):
        simulate_queries(plan, np.ones(8)*10)


def test_query_bundle_and_metadata_validation(query_bank, tmp_path):
    plan = plan_queries(query_bank, precision=.1)
    folder = export_bundle(plan, tmp_path/"queries")
    restored = load_bundle(folder)
    assert restored.bank.space == plan.bank.space
    assert restored.rows == plan.rows
    a = np.arange(8, dtype=float)/20
    counts = simulate_queries(plan, a, seed=4)
    path = folder/"measured.csv"
    write_counts(path, plan, counts)
    assert decode_queries(restored, read_counts(path, restored)) == decode_queries(plan, counts)
    assert (folder/"query_space.csv").exists() and (folder/"query_coefficients.csv").exists()
    with pytest.raises(ValueError, match="integer"):
        read_counts(folder/"counts_template.csv", restored)
    with pytest.raises(ValueError, match="separate"):
        write_counts(folder/"query_space.csv", plan, counts)
    bad_space = replace(plan.bank.space, kappa=2*plan.bank.space.kappa)
    with pytest.raises(ValueError, match="factorization"):
        validate_query_plan(replace(plan, bank=replace(plan.bank, space=bad_space)))
    with pytest.raises(ValueError, match="confidence"):
        validate_query_plan(replace(plan, confidence=.999))
    bad_bank = replace(query_bank.banks[0], experiment=replace(query_bank.banks[0].experiment, t2_s=None))
    with pytest.raises(ValueError, match="hardware"):
        plan_queries(replace(query_bank, banks=(bad_bank, *query_bank.banks[1:])))
    (folder/"query_coefficients.csv").write_text("edited\n")
    with pytest.raises(ValueError, match="changed"):
        load_bundle(folder)


def test_zero_queries_need_no_acquisition(tmp_path):
    e = Experiment((0., 1e6), ((0., 0.), (0., 0.)), ((0., 0.), (0., 0.)), 100e3)
    bank = compile_queries(e)
    assert bank.space.rank == 0 and bank.space.kappa == 0
    plan = plan_queries(bank, max_shots=0)
    assert plan.rows == [] and plan.active_time_s == 0
    folder = export_bundle(plan, tmp_path/"zeros")
    restored = load_bundle(folder)
    result = decode_queries(restored, read_counts(folder/"counts_template.csv", restored))
    assert result.values == (0., 0.) and result.intervals == ((0., 0.), (0., 0.))


def test_rank_truncation_is_charged_and_dc_sine_stays_zero():
    e = Experiment((0., 1e6), ((1., 0.), (1., 1e-6)), ((0., 0.), (0., 0.)), 100e3)
    exact = query_space(e)
    truncated = compile_queries(e, rank_rtol=1e-4, gains=(1.,), durations_s=(10e-6,), pulse_counts=(32,))
    assert exact.rank == 2 and truncated.space.rank == 1
    assert max(truncated.space.residual_bounds) > 4e-7
    assert truncated.banks[0].experiment.sine_weights[0] == 0
    with pytest.raises(ReadoutError, match="rank_rtol"):
        plan_queries(truncated, precision=1e-8)
    # A scalar query is also accepted by the shared-query interface.
    single = Experiment((1e6,), (2.,), (.5,), 100e3)
    assert query_space(single).rank == 1
    with pytest.raises(ValueError, match="matching"):
        replace(e, sine_weights=((0., 0.),))
    with pytest.raises(ValueError, match="rank_rtol"):
        query_space(e, rank_rtol=-1)
