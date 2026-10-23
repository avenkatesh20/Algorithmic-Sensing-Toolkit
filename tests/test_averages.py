"""Averaged physical outcomes preserve aggregation, weighting, and raw conventions."""

from dataclasses import replace
import csv
import json

import numpy as np
import pytest

import randomized_sensing as R


@pytest.fixture(scope="module")
def plan():
    e = R.Experiment((.813456e6, 1.28391e6), 100e3, t2_s=200e-6,
                     visibility=.8, readout_bias=.12, clock_s=7e-9)
    pool = R.build_pool(e, duration_s=7e-6, pool_size=128, seed=4)
    return R.plan_acquisition(pool, n_shots=4000, x_fraction=.25, seed=5)


def data(plan):
    counts = R.simulate(plan, [.12, -.08, .1, .15], seed=42)
    means = [2*counts[r["row_id"]]/r["shots"]-1 for r in plan.rows]
    return counts, means


@pytest.mark.parametrize("method", ["RQS", "RQS-ML"])
def test_grouped_averages_equal_counts_including_bias_and_parity(plan, method):
    counts, means = data(plan)
    original_id = plan.plan_id
    original_rows = [r.copy() for r in plan.rows]
    weights = dict(cosine_weights=((1., .2), (.1, -.3)), sine_weights=((.2, -.3), (.4, .1)))
    reference = R.decode(plan, counts, method=method, **weights)
    by_id = dict(zip((r["row_id"] for r in plan.rows), means))
    results = [R.decode_averages(plan, means, method=method, **weights),
               R.decode_averages(plan, by_id, method=method, **weights),
               R.decode(plan, np.array(means), method=method, **weights),
               R.decode(plan, R.AveragedOutcomes(by_id), method=method, **weights)]
    for result in results:
        np.testing.assert_allclose(result.amplitudes, reference.amplitudes, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(result.values, reference.values, atol=1e-9, rtol=1e-9)
        assert result.method == reference.method
    assert plan.plan_id == original_id and list(plan.rows) == original_rows
    assert {r["outcome_sign"] for r in plan.rows if r["basis"] == "Y"} == {-1, 1}


def test_rounded_means_and_actual_repetitions_do_not_change_sampling_weights(plan):
    _, means = data(plan)
    means[:2] = [.54, -.1]  # n=3 gives fractional positive statistics; do not round.
    first = R.decode_averages(plan, means, repetitions=3)
    second = R.decode_averages(plan, means, repetitions=np.arange(len(means))+7)
    assert first.amplitudes == second.amplitudes
    assert first.x_mean == second.x_mean
    # Additional observables reuse the same estimated amplitudes.
    weights = dict(cosine_weights=(1., 0.), sine_weights=(0., 1.))
    queried = R.decode_averages(plan, means, repetitions=3, **weights)
    assert queried.values == R.evaluate_observables(first.amplitudes, **weights)
    assert queried.amplitudes == first.amplitudes


def test_likelihood_repetition_weights_match_repeated_binary_data(plan):
    counts, means = data(plan)
    repeats = {r["row_id"]: 3*r["shots"] for r in plan.rows}
    expanded = replace(plan, rows=tuple(dict(r, shots=3*r["shots"]) for r in plan.rows))
    reference = R.decode(expanded, {key: 3*value for key, value in counts.items()}, method="RQS-ML")
    averaged = R.decode_averages(plan, means, repetitions=repeats, method="RQS-ML")
    np.testing.assert_allclose(averaged.amplitudes, reference.amplitudes, atol=1e-9, rtol=1e-9)
    assert averaged.fitted_contrast == pytest.approx(reference.fitted_contrast, abs=1e-10)


def test_average_csv_roundtrip_identity_and_legacy_bundle_compatibility(plan, tmp_path):
    folder = R.export_bundle(plan, tmp_path/"experiment")
    saved = R.load_bundle(folder)
    assert saved.plan_id == plan.plan_id
    _, means = data(plan)
    means[:2] = [.54, -.1]
    path = folder/"measured_averages.csv"
    R.write_averages(path, plan, means, repetitions=3)
    restored = R.read_averages(path, saved)
    assert list(restored.means.values()) == means
    assert set(restored.repetitions.values()) == {3}
    assert R.decode(saved, restored) == R.decode_averages(plan, means, repetitions=3)
    R.write_averages(tmp_path/"copy.csv", saved, restored)
    assert R.read_averages(tmp_path/"copy.csv", saved) == restored
    with pytest.raises(ValueError, match="finite number"):
        R.read_averages(folder/"averages_template.csv", saved)
    with pytest.raises(ValueError, match="separate"):
        R.write_averages(folder/"averages_template.csv", plan, means)
    with pytest.raises(ValueError, match="separate"):
        R.write_counts(folder/"averages_template.csv", plan, data(plan)[0])
    # Original bundles omit the new optional template; their IDs still load.
    manifest = json.loads((folder/"manifest.json").read_text())
    manifest["sha256"].pop("averages_template.csv")
    (folder/"averages_template.csv").unlink()
    (folder/"manifest.json").write_text(json.dumps(manifest))
    assert R.load_bundle(folder).plan_id == plan.plan_id


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.01, 1.01])
def test_invalid_mean_values_rejected(plan, invalid):
    _, means = data(plan)
    means[0] = invalid
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        R.decode_averages(plan, means)


@pytest.mark.parametrize("repetitions", [0, -1, 1.5, True])
def test_invalid_repetitions_rejected(plan, repetitions):
    with pytest.raises(ValueError, match="repetitions"):
        R.decode_averages(plan, data(plan)[1], repetitions=repetitions)


def test_missing_misaligned_or_duplicate_average_rows_rejected(plan, tmp_path):
    _, means = data(plan)
    with pytest.raises(ValueError, match="one real mean"):
        R.decode_averages(plan, means[:-1])
    with pytest.raises(ValueError, match="every acquisition row"):
        R.decode_averages(plan, {"wrong": .54})
    with pytest.raises(ValueError, match="repetitions"):
        R.decode_averages(plan, means, repetitions={"wrong": 100})
    path = tmp_path/"averages.csv"
    R.write_averages(path, plan, means)
    with path.open() as stream:
        reader = csv.DictReader(stream); columns = reader.fieldnames; rows = list(reader)
    for invalid in (dict(rows[0], plan_id="wrong"), dict(rows[1])):
        rows[0] = invalid
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
        with pytest.raises(ValueError, match="mismatched average row"):
            R.read_averages(path, plan)


def test_freely_running_data_retain_individual_timestamps(plan, tmp_path):
    running = R.plan_acquisition(plan.pool, n_shots=100, field_mode="free_running", shot_period_s=10e-6)
    with pytest.raises(ValueError, match="different phases"):
        R.decode_averages(running, np.zeros(len(running.rows)), repetitions=100)
    output = R.export_bundle(running, tmp_path/"running")
    assert not (output/"averages_template.csv").exists()
    assert R.load_bundle(output).plan_id == running.plan_id
