"""Uniform random acquisition, raw counts, and reuse across linear observables."""

from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json

import numpy as np

from .pulses import Pool, integer
from ._estimator import moment_estimate, polish
from ._timestamped import validate_times, fit as fit_timestamped


@dataclass(frozen=True)
class AcquisitionPlan:
    pool: Pool
    rows: tuple[dict, ...]
    seed: int
    field_mode: str = "phase_locked"

    @property
    def total_shots(self):
        return sum(r["shots"] for r in self.rows)

    @property
    def active_time_s(self):
        return self.total_shots*self.pool.duration_s

    @property
    def time_with_overhead_s(self):
        return self.active_time_s+self.total_shots*self.pool.experiment.overhead_s

    @property
    def scheduled_span_s(self):
        """First sensing start to last sensing end; excludes final readout overhead."""
        if self.field_mode == "phase_locked":
            return None
        return self.rows[-1]["start_time_s"]-self.rows[0]["start_time_s"]+self.pool.duration_s

    @property
    def plan_id(self):
        p = self.pool
        metadata = dict(experiment=p.experiment.to_dict(), duration_s=p.duration_s,
                        start_time_s=p.start_time_s, design_clock_s=p.design_clock_s,
                        settings=p.settings, rows=self.rows, seed=self.seed,
                        sequences=[(s.switch_times_s, s.initial_sign) for s in p.sequences])
        if self.field_mode != "phase_locked":
            metadata["field_mode"] = self.field_mode
        digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
        digest.update(p.Z.astype("<f8").tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class Estimate:
    amplitudes: tuple[float, ...]
    values: tuple[float, ...]
    method: str
    x_mean: float
    guard_floor: float | None
    floor_active: bool
    ml_applied: bool
    fitted_contrast: float | None
    fit_converged: bool | None = None
    response_condition: float | None = None


@dataclass(frozen=True)
class TimedCounts:
    """Raw per-shot positive counts and ACTUAL starts, keyed by acquisition row ID."""

    positive_counts: dict[str, int]
    start_times_s: dict[str, float]


def _times(plan, start_times_s=None):
    if start_times_s is None:
        start_times_s = [r["start_time_s"] for r in plan.rows]
    elif isinstance(start_times_s, Mapping):
        if set(start_times_s) != {r["row_id"] for r in plan.rows}:
            raise ValueError("Start timestamps must match every acquisition row exactly.")
        start_times_s = [start_times_s[r["row_id"]] for r in plan.rows]
    return validate_times(start_times_s, len(plan.rows), plan.pool.duration_s)


def validate_plan(plan):
    if plan.field_mode not in ("phase_locked", "free_running"):
        raise ValueError("field_mode must be 'phase_locked' or 'free_running'.")
    seen = set()
    for i, row in enumerate(plan.rows):
        index = integer(row["sequence_index"], "sequence_index")
        q = plan.pool.sequence(index)
        expected = q.initial_sign*(-1)**len(q.switch_times_s) if row["basis"] == "Y" else 1
        expected_id = f"shot_{i}" if plan.field_mode == "free_running" else f"{row['basis']}_{index}"
        if (row["basis"] not in ("X", "Y") or row["row_id"] in seen
                or row["row_id"] != expected_id
                or row["outcome_sign"] != expected):
            raise ValueError("Invalid acquisition basis, row identity, or outcome sign.")
        integer(row["shots"], "shots", 1)
        if plan.field_mode == "free_running" and row["shots"] != 1:
            raise ValueError("Freely running acquisitions require one timestamped outcome per shot.")
        seen.add(row["row_id"])
    if {r["basis"] for r in plan.rows} != {"X", "Y"}:
        raise ValueError("Acquisition requires both X and Y shots.")
    if plan.field_mode == "free_running":
        _times(plan)


def plan_acquisition(pool, *, n_shots, x_fraction=.2, seed=5,
                     field_mode="phase_locked", start_times_s=None, shot_period_s=None):
    """Freeze uniform draws WITH replacement.

    n_shots is a user-chosen budget, not an accuracy/confidence certificate.
    Physical pulses always start in the +1 frame. outcome_sign converts the
    physical Y mean (after bias subtraction) to the signed pool convention.
    phase_locked groups repeated sequence/basis shots. free_running retains one
    row per shot, with interleaved random X/Y bases and nominal start timestamps.
    Supply a full start schedule or a shot period; actual timestamps go in counts.
    """
    n_shots = integer(n_shots, "n_shots", 4)
    if not np.isfinite(x_fraction) or not 0 < x_fraction < 1:
        raise ValueError("x_fraction must lie strictly between 0 and 1.")
    nx = min(n_shots-1, max(1, int(round(n_shots*x_fraction))))
    seed = integer(seed, "seed")
    rng = np.random.default_rng(seed)
    rows = []
    if field_mode == "free_running":
        if start_times_s is not None and shot_period_s is not None:
            raise ValueError("Supply start_times_s or shot_period_s, not both.")
        if start_times_s is None:
            period = pool.duration_s+pool.experiment.overhead_s if shot_period_s is None else shot_period_s
            if not np.isfinite(period) or period < pool.duration_s+pool.experiment.overhead_s:
                raise ValueError("shot_period_s must include sensing duration and specified per-shot overhead.")
            start_times_s = pool.start_time_s+np.arange(n_shots)*period
        starts = validate_times(start_times_s, n_shots, pool.duration_s)
        indices = rng.integers(len(pool.Z), size=n_shots)
        bases = np.full(n_shots, "Y"); bases[rng.choice(n_shots, nx, replace=False)] = "X"
        for i, (index, basis, start) in enumerate(zip(indices, bases, starts)):
            index = int(index)
            q = pool.sequence(index)
            sign = q.initial_sign*(-1)**len(q.switch_times_s) if basis == "Y" else 1
            rows.append(dict(row_id=f"shot_{i}", sequence_index=index, basis=str(basis),
                             shots=1, outcome_sign=sign, start_time_s=float(start)))
        plan = AcquisitionPlan(pool, tuple(rows), seed, field_mode)
        validate_plan(plan)
        return plan
    if field_mode != "phase_locked":
        raise ValueError("field_mode must be 'phase_locked' or 'free_running'.")
    if start_times_s is not None or shot_period_s is not None:
        raise ValueError("Per-shot timing options require field_mode='free_running'.")
    for basis, shots in (("Y", n_shots-nx), ("X", nx)):
        multiplicities = np.bincount(rng.integers(len(pool.Z), size=shots), minlength=len(pool.Z))
        for index in np.flatnonzero(multiplicities):
            index = int(index)
            q = pool.sequence(index)
            sign = q.initial_sign*(-1)**len(q.switch_times_s) if basis == "Y" else 1
            rows.append(dict(row_id=f"{basis}_{index}", sequence_index=index, basis=basis,
                             shots=int(multiplicities[index]), outcome_sign=sign))
    plan = AcquisitionPlan(pool, tuple(rows), int(seed))
    validate_plan(plan)
    return plan


def validate_counts(plan, counts):
    validate_plan(plan)
    if plan.field_mode == "free_running":
        if not isinstance(counts, TimedCounts):
            raise ValueError("Freely running data require TimedCounts with actual start timestamps.")
        _times(plan, counts.start_times_s)
        positive = counts.positive_counts
    else:
        if isinstance(counts, TimedCounts):
            raise ValueError("Timestamped data require field_mode='free_running'.")
        positive = counts
    if set(positive) != {r["row_id"] for r in plan.rows}:
        raise ValueError("Counts must match every acquisition row exactly.")
    for row in plan.rows:
        value = integer(positive[row["row_id"]], "positive count")
        if value > row["shots"]:
            raise ValueError("A positive count exceeds its shot total.")


def _amplitudes(p, amplitudes):
    a = np.asarray(amplitudes, float)
    if a.shape != (2*p.N,) or not np.isfinite(a).all() or np.linalg.norm(a) > p.experiment.amplitude_bound*(1+1e-12):
        raise ValueError("amplitudes must have length 2N and obey amplitude_bound.")
    return a


def evaluate_observables(amplitudes, *, cosine_weights, sine_weights):
    """Evaluate any later-chosen linear queries from an amplitude estimate."""
    a = np.asarray(amplitudes, float)
    if a.ndim != 1 or not a.size or a.size % 2 or not np.isfinite(a).all():
        raise ValueError("amplitudes must be a finite, nonempty vector of length 2N.")
    c, s = (np.asarray(w, float) for w in (cosine_weights, sine_weights))
    if (c.ndim not in (1, 2) or c.shape != s.shape or not c.size
            or c.shape[-1] != len(a)//2 or not np.isfinite(c).all() or not np.isfinite(s).all()):
        raise ValueError("Weights must be matching finite (N,) vectors or (M,N) arrays.")
    return tuple((np.atleast_2d(np.concatenate((c, s), axis=-1))@a).tolist())


def _values(pool, amplitudes, cosine_weights, sine_weights):
    if cosine_weights is None and sine_weights is None:
        # Earlier bundles may carry default queries; new acquisitions have none.
        if not hasattr(pool.experiment, "weights"):
            return ()
        cosine_weights, sine_weights = pool.experiment.cosine_weights, pool.experiment.sine_weights
    elif cosine_weights is None or sine_weights is None:
        raise ValueError("Supply both cosine_weights and sine_weights after acquisition.")
    return evaluate_observables(amplitudes, cosine_weights=cosine_weights, sine_weights=sine_weights)


def _timed_estimate(plan, positive, times, method, cosine_weights, sine_weights):
    a, contrast, mu, condition = fit_timestamped(plan.pool, plan.rows, positive, times)
    values = _values(plan.pool, a, cosine_weights, sine_weights)
    return Estimate(tuple(a.tolist()), values, method, mu, None, False, True, contrast, True, condition)


def population_estimate(pool, amplitudes, *, cosine_weights=None, sine_weights=None, start_times_s=None):
    """Noise-free inversion diagnostic for a pool or timestamped acquisition plan.

    Freely running plans use exact per-shot outcome probabilities in the same
    likelihood fit, without inserting the true amplitudes into reconstruction.
    """
    if isinstance(pool, AcquisitionPlan):
        plan = pool
        validate_plan(plan)
        pool = plan.pool
        if plan.field_mode == "free_running":
            a = _amplitudes(pool, amplitudes)
            times = _times(plan, start_times_s)
            probabilities = _timed_probabilities(plan, a, times)
            positive = {r["row_id"]: float(p) for r, p in zip(plan.rows, probabilities)}
            return _timed_estimate(plan, positive, times, "TIMESTAMPED-ML-population", cosine_weights, sine_weights)
    if start_times_s is not None:
        raise ValueError("Per-shot timestamps require a freely running acquisition plan.")
    a = _amplitudes(pool, amplitudes)
    phase = pool.Z@a
    c = pool.experiment.contrast(pool.duration_s)
    mu = float(c*np.mean(np.cos(phase)))
    estimated, floor = moment_estimate(pool, c*np.sin(phase)@pool.Z/len(pool.Z), mu)
    values = _values(pool, estimated, cosine_weights, sine_weights)
    return Estimate(tuple(estimated.tolist()), values, "RQS-population", mu,
                    float(floor), bool(mu < floor), False, None)


def _timed_probabilities(plan, amplitudes, times):
    p = plan.pool
    z = p.responses_at(np.array([r["sequence_index"] for r in plan.rows]), times)
    phase = z@amplitudes
    is_y = np.array([r["basis"] == "Y" for r in plan.rows])
    signs = np.array([r["outcome_sign"] for r in plan.rows])
    value = np.where(is_y, signs*np.sin(phase), np.cos(phase))
    return np.clip((1+p.experiment.readout_bias+p.experiment.contrast(p.duration_s)*value)/2, 0, 1)


def simulate(plan, amplitudes, *, seed=42, start_times_s=None):
    """Raw physical +1 counts, with dephasing, visibility, offset, and pulse parity."""
    validate_plan(plan)
    p = plan.pool
    a = _amplitudes(p, amplitudes)
    rng, counts = np.random.default_rng(seed), {}
    if plan.field_mode == "free_running":
        times = _times(plan, start_times_s)
        plus = rng.binomial(1, _timed_probabilities(plan, a, times))
        return TimedCounts({r["row_id"]: int(k) for r, k in zip(plan.rows, plus)},
                           {r["row_id"]: float(t) for r, t in zip(plan.rows, times)})
    if start_times_s is not None:
        raise ValueError("Per-shot timestamps require field_mode='free_running'.")
    for row in plan.rows:
        phase = p.Z[row["sequence_index"]]@a
        value = row["outcome_sign"]*np.sin(phase) if row["basis"] == "Y" else np.cos(phase)
        probability = (1+p.experiment.readout_bias+p.experiment.contrast(p.duration_s)*value)/2
        counts[row["row_id"]] = int(rng.binomial(row["shots"], np.clip(probability, 0, 1)))
    return counts


def decode(plan, counts, *, method=None, cosine_weights=None, sine_weights=None):
    """Estimate amplitudes and optional later-chosen queries; no certified intervals.

    Weights do not alter the saved acquisition or amplitude reconstruction.
    Without weights, new query-independent plans return values=().
    """
    validate_counts(plan, counts)
    if plan.field_mode == "free_running":
        if method not in (None, "TIMESTAMPED-ML"):
            raise ValueError("Freely running data use method='TIMESTAMPED-ML' with actual per-shot responses.")
        return _timed_estimate(plan, counts.positive_counts, _times(plan, counts.start_times_s),
                               "TIMESTAMPED-ML", cosine_weights, sine_weights)
    method = "RQS" if method is None else method
    if method not in ("RQS", "RQS-ML"):
        raise ValueError("method must be 'RQS' or 'RQS-ML'.")
    p, v, mu, ny, nx = plan.pool, np.zeros(2*plan.pool.N), 0., 0, 0
    for row in plan.rows:
        weighted = 2*counts[row["row_id"]]-row["shots"]*(1+p.experiment.readout_bias)
        if row["basis"] == "Y":
            v += row["outcome_sign"]*weighted*p.Z[row["sequence_index"]]; ny += row["shots"]
        else:
            mu += weighted; nx += row["shots"]
    mu /= nx
    a, floor = moment_estimate(p, v/ny, mu)
    applied, contrast = False, None
    if method == "RQS-ML":
        a, applied, contrast = polish(p, a, plan.rows, counts)
    values = _values(p, a, cosine_weights, sine_weights)
    return Estimate(tuple(a.tolist()), values, method, float(mu), float(floor),
                    bool(mu < floor), applied, contrast)
