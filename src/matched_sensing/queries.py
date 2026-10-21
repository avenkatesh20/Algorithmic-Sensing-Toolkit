"""Shared query coordinates: K_j/kappa = U v_j, F_j = kappa v_j @ y."""

from dataclasses import dataclass, replace

import numpy as np

from .compiler import CompilationError, compile_bank
from .experiment import Experiment
from .readout import ReadoutError, decode, plan_readout, simulate, validate_plan


def _matrix(a):
    return tuple(map(tuple, np.asarray(a, float).tolist()))


@dataclass(frozen=True)
class QuerySpace:
    basis: tuple                 # U: (2N, r), columns are orthonormal
    coefficients: tuple          # rows are v_j: (M, r)
    kappa: float
    residual_bounds: tuple       # amplitude_bound * ||K_j - kappa U v_j||
    singular_values: tuple
    rank_rtol: float

    @property
    def rank(self):
        return len(self.basis[0])

    @property
    def count(self):
        return len(self.coefficients)


def _queries(experiment):
    return np.atleast_2d(experiment.weights)


def _residual_bounds(experiment, u, v, kappa, *, margin=1.0):
    k = _queries(experiment)
    # Include floating-point reconstruction error, in original observable units.
    padding = margin*64*np.finfo(float).eps*max(k.shape)*np.linalg.norm(k, axis=1)
    return experiment.amplitude_bound*(np.linalg.norm(k-kappa*v@u.T, axis=1)+padding)


def query_space(experiment, *, rank_rtol=None):
    """Find the numerical row span of the requested weights using an SVD.

    Each input row retains its original scale. Discarded directions are charged
    as worst-case observable bias, rather than silently changing the queries.
    """
    k = _queries(experiment)
    rtol = np.finfo(float).eps*max(k.shape) if rank_rtol is None else float(rank_rtol)
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rank_rtol must be finite and nonnegative.")
    kappa = float(np.max(np.linalg.norm(k, axis=1)))
    if not np.isfinite(kappa):
        raise ValueError("Observable norms exceed floating-point range.")
    n = len(experiment.frequencies_hz)
    active = np.ones(2*n, dtype=bool)
    active[n+np.flatnonzero(np.asarray(experiment.frequencies_hz) == 0)] = False
    reduced = k[:, active]
    _, singular, vh = np.linalg.svd(reduced/kappa if kappa else reduced, full_matrices=False)
    rank = int(np.count_nonzero(singular > rtol*singular[0])) if singular[0] else 0
    u = np.zeros((2*n, rank))
    u[active] = vh[:rank].T
    for i in range(rank):
        if u[np.argmax(abs(u[:, i])), i] < 0:
            u[:, i] *= -1
    v = k@u/kappa if kappa else np.zeros((len(k), 0))
    residual = _residual_bounds(experiment, u, v, kappa)
    return QuerySpace(_matrix(u), _matrix(v), kappa, tuple(residual), tuple(singular), rtol)


@dataclass(frozen=True)
class QueryBank:
    experiment: Experiment
    space: QuerySpace
    banks: tuple

    @property
    def sequences(self):
        return tuple(s for bank in self.banks for s in bank.sequences)

    @property
    def attempts(self):
        return tuple(dict(coordinate_index=i, attempts=bank.attempts)
                     for i, bank in enumerate(self.banks))


def _coordinate_experiment(experiment, u):
    n = len(experiment.frequencies_hz)
    return replace(experiment, cosine_weights=tuple(u[:n]), sine_weights=tuple(u[n:]))


def compile_queries(experiment, *, rank_rtol=None, **compiler_options):
    """Compile one coherent bank per shared coordinate, under common search limits.

    Pass max_gain, max_duration_s and max_pulses (or explicit candidate lists)
    through compiler_options; see compile_bank for the search convention.
    """
    space = query_space(experiment, rank_rtol=rank_rtol)
    banks = []
    for i, u in enumerate(np.asarray(space.basis).T):
        try:
            banks.append(compile_bank(_coordinate_experiment(experiment, u), **compiler_options))
        except CompilationError as exc:
            raise CompilationError(f"Query coordinate {i}: {exc}") from exc
    return QueryBank(experiment, space, tuple(banks))


def _validate_bank(bank):
    space = bank.space
    k = _queries(bank.experiment)
    u, v = np.asarray(space.basis, float), np.asarray(space.coefficients, float)
    r = space.rank
    if (u.shape != (k.shape[1], r) or v.shape != (len(k), r) or len(bank.banks) != r
            or not np.isfinite(u).all() or not np.isfinite(v).all()
            or not np.isfinite(space.rank_rtol) or space.rank_rtol < 0):
        raise ValueError("Invalid query-space dimensions or values.")
    kappa = float(np.max(np.linalg.norm(k, axis=1)))
    expected_v = k@u/kappa if kappa else np.zeros_like(v)
    if (not np.isclose(space.kappa, kappa, rtol=1e-12, atol=0)
            or not np.allclose(u.T@u, np.eye(r), rtol=0, atol=1e-12)
            or not np.allclose(v, expected_v, rtol=1e-12, atol=1e-14)):
        raise ValueError("Invalid query factorization.")
    residual = np.asarray(space.residual_bounds)
    # Leave room for differences between BLAS implementations after JSON reload.
    needed = _residual_bounds(bank.experiment, u, v, kappa, margin=0.5)
    if residual.shape != (len(k),) or not np.isfinite(residual).all() or np.any(residual < needed*(1-1e-12)):
        raise ValueError("Invalid query residual bounds.")
    for i, child in enumerate(bank.banks):
        if child.experiment != _coordinate_experiment(bank.experiment, u[:, i]):
            raise ValueError("A coordinate bank has mismatched weights or hardware parameters.")


@dataclass(frozen=True)
class QueryPlan:
    bank: QueryBank
    coordinate_plans: tuple
    precisions: tuple            # absolute errors, original observable units
    confidence: float            # simultaneous confidence for all queries

    @property
    def rows(self):
        rows, offset = [], 0
        for i, plan in enumerate(self.coordinate_plans):
            rows.extend(dict(row, row_id=f"q{i}_{row['row_id']}",
                             sequence_index=offset+row["sequence_index"], coordinate_index=i)
                        for row in plan.rows)
            offset += len(plan.bank.sequences)
        return rows

    @property
    def active_time_s(self):
        return sum(p.active_time_s for p in self.coordinate_plans)

    @property
    def time_with_overhead_s(self):
        return sum(p.time_with_overhead_s for p in self.coordinate_plans)

    @property
    def error_bounds(self):
        s = self.bank.space
        return tuple(s.kappa*abs(np.asarray(s.coefficients))@
                     np.array([p.precision for p in self.coordinate_plans])+s.residual_bounds)


def _coordinate_precision(bank, precisions):
    s = bank.space
    available = np.asarray(precisions)-s.residual_bounds
    lengths = s.kappa*np.sum(abs(np.asarray(s.coefficients)), axis=1)
    if np.any(available < 0) or np.any((lengths > 0) & (available <= 0)):
        raise ReadoutError("Rank truncation/roundoff exceeds the requested precision; reduce rank_rtol or relax precision.")
    return float(np.min(available[lengths > 0]/lengths[lengths > 0])) if s.rank else None


def plan_queries(bank, *, precision=0.05, confidence=0.95, max_shots=10_000_000, optimize="active"):
    """Plan shared measurements with simultaneous error guarantees for all queries.

    Uses conservative coordinate intervals and a union bound. This allocation
    does not implement the manuscript's optimal directional-variance strategy.
    max_shots caps the sum across all coordinates and both readout bases.
    """
    _validate_bank(bank)
    eps = np.asarray(precision, float)
    if eps.ndim == 0:
        eps = np.full(bank.space.count, float(eps))
    if (eps.shape != (bank.space.count,) or not np.isfinite(eps).all() or np.any(eps <= 0)
            or not 0 < confidence < 1 or isinstance(max_shots, bool)
            or not isinstance(max_shots, (int, np.integer)) or max_shots < 0
            or optimize not in ("active", "with_overhead")):
        raise ValueError("Require positive precision per query, confidence in (0,1), a nonnegative integer max_shots, and a valid optimize mode.")
    eta = _coordinate_precision(bank, eps)
    r = bank.space.rank
    if r and max_shots < 2*r:
        raise ReadoutError("The total shot budget is too small for the query coordinates.")
    plans = tuple(plan_readout(b, precision=eta, confidence=1-(1-confidence)/r,
                               max_shots=max_shots, optimize=optimize) for b in bank.banks)
    if sum(row["shots"] for p in plans for row in p.rows) > max_shots:
        raise ReadoutError("Shared coordinate plans exceed the total max_shots budget.")
    return QueryPlan(bank, plans, tuple(eps.tolist()), float(confidence))


def validate_query_plan(plan):
    _validate_bank(plan.bank)
    s = plan.bank.space
    eps = np.asarray(plan.precisions)
    if (eps.shape != (s.count,) or not np.isfinite(eps).all() or np.any(eps <= 0)
            or not 0 < plan.confidence < 1 or len(plan.coordinate_plans) != s.rank):
        raise ValueError("Invalid query readout plan.")
    eta = _coordinate_precision(plan.bank, eps)
    for bank, child in zip(plan.bank.banks, plan.coordinate_plans):
        if child.bank != bank:
            raise ValueError("The coordinate plan uses a different pulse bank.")
        validate_plan(child)
        if child.precision > eta*(1+1e-12):
            raise ValueError("Coordinate precision does not meet the query error budget.")
    if sum(1-p.confidence for p in plan.coordinate_plans) > (1-plan.confidence)*(1+1e-12):
        raise ValueError("Coordinate confidence does not meet the joint confidence budget.")


@dataclass(frozen=True)
class QueryEstimate:
    values: tuple
    intervals: tuple
    confidence: float
    coordinate_estimates: tuple


def decode_queries(plan, positive_counts):
    """Decode shared coordinates, then reconstruct every original observable."""
    validate_query_plan(plan)
    if set(positive_counts) != {r["row_id"] for r in plan.rows}:
        raise ReadoutError("Provide exactly one positive count for every acquisition row.")
    estimates = tuple(decode(p, {row["row_id"]:positive_counts[f"q{i}_{row['row_id']}"]
                                for row in p.rows}) for i, p in enumerate(plan.coordinate_plans))
    s = plan.bank.space
    weights = s.kappa*np.asarray(s.coefficients)
    centers = np.array([e.value for e in estimates])
    limits = np.array([e.interval for e in estimates]).reshape(s.rank, 2)
    values = weights@centers
    lower = np.minimum(weights*limits[:, 0], weights*limits[:, 1]).sum(axis=1)-s.residual_bounds
    upper = np.maximum(weights*limits[:, 0], weights*limits[:, 1]).sum(axis=1)+s.residual_bounds
    return QueryEstimate(tuple(values), _matrix(np.c_[lower, upper]), plan.confidence, estimates)


def simulate_queries(plan, amplitudes, *, seed=0):
    """Synthetic counts from literal pulses; truth is never passed to the decoder."""
    validate_query_plan(plan)
    a = np.asarray(amplitudes, float)
    e = plan.bank.experiment
    if (a.shape != (2*len(e.frequencies_hz),) or not np.isfinite(a).all()
            or np.linalg.norm(a) > e.amplitude_bound*(1+1e-12)):
        raise ValueError("Supply finite cosine/sine amplitudes within amplitude_bound.")
    seeds = np.random.SeedSequence(seed).spawn(len(plan.coordinate_plans))
    return {f"q{i}_{key}":count for i, (p, child_seed) in enumerate(zip(plan.coordinate_plans, seeds))
            for key, count in simulate(p, a, seed=child_seed).items()}
