"""Preplanned X/Y measurements, dephasing-aware shot budgets, and phase unwrapping."""

from dataclasses import dataclass

import numpy as np

from .compiler import PulseBank, audit


class ReadoutError(ValueError):
    """Counts are incomplete, ambiguous, or inconsistent with the declared model."""


@dataclass(frozen=True)
class ReadoutPlan:
    bank: PulseBank
    sequence_indices: tuple[int, ...]
    shots_per_basis: tuple[int, ...]
    phase_radii: tuple[float, ...]
    precision: float
    confidence: float
    optimize: str = "active"

    @property
    def rows(self):
        e = self.bank.experiment
        return [dict(row_id=f"stage{j}_{basis}", sequence_index=i, basis=basis,
                     analysis_phase_rad=np.pi/2 if basis == "X" else 0.0,
                     shots=n, duration_s=self.bank.sequences[i].duration_s,
                     contrast=float(e.contrast(self.bank.sequences[i].duration_s)),
                     start_time_s=self.bank.sequences[i].start_time_s)
                for j,(i,n) in enumerate(zip(self.sequence_indices,self.shots_per_basis))
                for basis in ("X", "Y")]

    @property
    def active_time_s(self):
        return float(sum(row["shots"]*row["duration_s"] for row in self.rows))

    @property
    def time_with_overhead_s(self):
        return self.active_time_s + self.bank.experiment.overhead_s*sum(row["shots"] for row in self.rows)


@dataclass(frozen=True)
class Estimate:
    value: float
    interval: tuple[float, float]
    confidence: float
    phase_branches: tuple[int, ...]


def _validate_bank(bank):
    if not bank.sequences or not np.isfinite(bank.phase_tolerance) or bank.phase_tolerance <= 0:
        raise ValueError("The pulse bank is empty or its tolerance is invalid.")
    gains = [s.gain for s in bank.sequences]
    if gains != sorted(set(gains)):
        raise ValueError("The pulse bank must have distinct increasing gains.")
    for sequence in bank.sequences:
        check = audit(bank.experiment, sequence)
        if not check["timing_valid"] or check["phase_error_bound"] > bank.phase_tolerance:
            raise ValueError("A pulse sequence fails its response/timing audit.")
        if abs(check["phase_error_bound"]-sequence.phase_error_bound) > 1e-10:
            raise ValueError("The saved phase error does not match the actual pulse times.")
        if not np.allclose(check["response"], sequence.phase_vector, rtol=0, atol=1e-10):
            raise ValueError("The saved response does not match the actual pulse times.")


def _proposal(bank, indices, precision, confidence, max_shots):
    sequences = [bank.sequences[i] for i in indices]
    gains = np.array([s.gain for s in sequences])
    if gains[0] >= np.pi:
        return None
    max_ratio = max(1.0, float(np.max(gains[1:]/gains[:-1]))) if len(gains) > 1 else 1.0
    common = min(np.pi/(2*(1+max_ratio)), (np.pi-gains[0])/2)
    radii = np.full(len(gains), common)
    radii[-1] = min(common, gains[-1]*precision/bank.experiment.observable_scale)
    statistical = radii - np.array([s.phase_error_bound for s in sequences])
    contrast = np.array([bank.experiment.contrast(s.duration_s) for s in sequences])
    if np.any(statistical <= 0) or np.any(contrast <= 0):
        return None
    # Hoeffding for both +/-1 sample means, union-bounded over all stages.
    denominator = (contrast*np.sin(statistical))**2
    if np.any(denominator == 0):
        return None
    required = np.ceil(4*np.log(4*len(gains)/(1-confidence))/denominator)
    if not np.isfinite(required).all() or 2*sum(required) > max_shots:
        return None
    return tuple(map(int, required)), tuple(map(float, radii))


def plan_readout(bank, *, precision=0.05, confidence=0.95, max_shots=10_000_000, optimize="active"):
    """Choose a gain chain and conservative X/Y shot counts from the compiled bank.

    Search contiguous chains and two-gain chains. Confidence is conditional on the
    signal norm, independent binary noise, calibrated contrast/bias, and phase-locked
    replay. The uncertainty budget includes each control's uniform phase error.
    """
    _validate_bank(bank)
    if (not np.isfinite(precision) or precision <= 0 or not 0 < confidence < 1
            or not isinstance(max_shots,(int,np.integer)) or isinstance(max_shots,bool) or max_shots < 2):
        raise ValueError("Require precision > 0, 0 < confidence < 1, and max_shots >= 2.")
    if optimize not in ("active", "with_overhead"):
        raise ValueError("optimize must be 'active' or 'with_overhead'.")
    paths = set()
    for i in range(len(bank.sequences)):
        for j in range(i,len(bank.sequences)):
            paths.add(tuple(range(i,j+1)))
            paths.add(tuple(sorted({i,j})))
    plans = []
    for indices in sorted(paths):
        proposal = _proposal(bank, indices, precision, confidence, max_shots)
        if proposal:
            shots, radii = proposal
            plans.append(ReadoutPlan(bank, indices, shots, radii, float(precision), float(confidence), optimize))
    if not plans:
        raise ReadoutError("No unambiguous gain chain fits the error and shot budgets. Include a gain below pi, reduce compilation error, or increase max_shots.")
    cost = lambda p:p.active_time_s if optimize == "active" else p.time_with_overhead_s
    return min(plans, key=cost)


def validate_plan(plan):
    _validate_bank(plan.bank)
    indices = plan.sequence_indices
    if not indices or list(indices) != sorted(set(indices)) or min(indices) < 0 or max(indices) >= len(plan.bank.sequences):
        raise ValueError("Invalid readout sequence indices.")
    if len(plan.shots_per_basis) != len(indices) or len(plan.phase_radii) != len(indices):
        raise ValueError("Incomplete readout plan.")
    if not np.isfinite(plan.precision) or plan.precision <= 0 or not 0 < plan.confidence < 1:
        raise ValueError("Invalid precision/confidence.")
    expected = _proposal(plan.bank, indices, plan.precision, plan.confidence, float("inf"))
    if expected is None or not np.allclose(plan.phase_radii, expected[1], rtol=1e-12, atol=0):
        raise ValueError("Invalid phase-error allocation.")
    if any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n < needed
           for n,needed in zip(plan.shots_per_basis,expected[0])):
        raise ValueError("The shot counts do not meet the declared confidence budget.")


def decode(plan, positive_counts):
    """Decode a row_id -> number-of-+1-outcomes mapping; the true field is never used."""
    validate_plan(plan)
    rows = plan.rows
    if set(positive_counts) != {r["row_id"] for r in rows}:
        raise ReadoutError("Provide exactly one positive count for every acquisition row.")
    for row in rows:
        k = positive_counts[row["row_id"]]
        if isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 0 <= k <= row["shots"]:
            raise ReadoutError(f"Invalid positive count for {row['row_id']}.")
    lo, hi = -1.0, 1.0
    branches = []
    bias = plan.bank.experiment.readout_bias
    for j,(index,n,radius) in enumerate(zip(plan.sequence_indices,plan.shots_per_basis,plan.phase_radii)):
        x = 2*positive_counts[f"stage{j}_X"]/n-1-bias
        y = 2*positive_counts[f"stage{j}_Y"]/n-1-bias
        if x == 0 and y == 0:
            raise ReadoutError("Undefined measured phase. Check counts, contrast and shot budget.")
        phase = np.arctan2(y,x)
        gain = plan.bank.sequences[index].gain
        first = int(np.ceil((gain*lo-phase-radius)/(2*np.pi)))
        last = int(np.floor((gain*hi-phase+radius)/(2*np.pi)))
        candidates = [(max(lo,(phase+2*np.pi*k-radius)/gain),
                       min(hi,(phase+2*np.pi*k+radius)/gain),k) for k in range(first,last+1)]
        candidates = [c for c in candidates if c[0] <= c[1]]
        if len(candidates) != 1:
            raise ReadoutError("The phases are inconsistent or ambiguous. Check the phase reference, calibration, field bound and counts.")
        lo, hi, branch = candidates[0]; branches.append(branch)
    scale = plan.bank.experiment.observable_scale
    return Estimate(float(scale*(lo+hi)/2), (float(scale*lo),float(scale*hi)),
                    plan.confidence, tuple(branches))


def simulate(plan, amplitudes, *, seed=0):
    """Optional synthetic counts using the actual compiled Fourier responses."""
    validate_plan(plan)
    e = plan.bank.experiment
    a = np.asarray(amplitudes,float)
    if a.shape != e.weights.shape or not np.isfinite(a).all() or np.linalg.norm(a) > e.amplitude_bound*(1+1e-12):
        raise ValueError("Supply finite [cosine amplitudes, sine amplitudes] within amplitude_bound.")
    rng = np.random.default_rng(seed); counts = {}
    for row in plan.rows:
        sequence = plan.bank.sequences[row["sequence_index"]]
        phase = np.asarray(sequence.phase_vector) @ a
        mean = e.readout_bias + row["contrast"]*np.sin(phase+row["analysis_phase_rad"])
        probability = np.clip((1+mean)/2,0,1)
        counts[row["row_id"]] = int(rng.binomial(row["shots"],probability))
    return counts

