"""Full-response synthesis by optimizing positive gaps between X(pi) pulses."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .experiment import Experiment


class CompilationError(RuntimeError):
    """The declared search found no usable control; this is not an impossibility proof."""


@dataclass(frozen=True)
class Sequence:
    gain: float
    duration_s: float
    start_time_s: float
    switch_times_s: tuple[float, ...]
    phase_vector: tuple[float, ...]
    phase_error_bound: float


@dataclass(frozen=True)
class PulseBank:
    experiment: Experiment
    sequences: tuple[Sequence, ...]
    phase_tolerance: float
    attempts: tuple[dict, ...] = ()
    search: dict | None = None


def _integrals(experiment, left, right, start=0.0, dtype=float):
    """Exact Fourier integrals, including DC; shape (intervals, 2N)."""
    f = np.asarray(experiment.frequencies_hz, dtype=dtype)
    left, right = np.atleast_1d(left).astype(dtype), np.atleast_1d(right).astype(dtype)
    width = right - left
    pi = np.arccos(dtype(-1))
    angle = 2*pi*(dtype(start) + (left + right)/2)[:, None]*f
    scale = width[:, None]*np.sinc(width[:, None]*f)
    return np.c_[scale*np.cos(angle), scale*np.sin(angle)]


def response(experiment, switch_times_s, duration_s, start_time_s=0.0):
    """Reintegrate actual +/-1 intervals, using extended precision where available."""
    edges = np.r_[0.0, switch_times_s, duration_s]
    integrals = _integrals(experiment, edges[:-1], edges[1:], start_time_s, np.longdouble)
    pi = np.arccos(np.longdouble(-1))
    return np.asarray(2*pi*experiment.coupling_hz*((-1.0)**np.arange(len(integrals)) @ integrals), float)


def audit(experiment, sequence):
    """Check timing, optional clock, and uniform phase error over the field norm ball."""
    e = np.asarray(sequence.switch_times_s, float)
    t, start = sequence.duration_s, sequence.start_time_s
    if not np.isfinite([t, start, sequence.gain]).all() or t <= 0 or start < 0 or sequence.gain <= 0:
        raise ValueError("Invalid gain, duration, or phase-reference start time.")
    if e.ndim != 1 or len(e) % 2 or not np.isfinite(e).all():
        raise ValueError("Use a finite even number of pulse times, restoring the pulse frame.")
    timing = not len(e) or bool(e[0] > 0 and e[-1] < t and
        e[0] >= experiment.edge_guard_s and t-e[-1] >= experiment.edge_guard_s
        and np.all(np.diff(e) >= experiment.min_gap_s))
    minimum_gap = float(np.min(np.diff(e))) if len(e) > 1 else None
    if experiment.clock_s:
        ticks = np.r_[e, t]/experiment.clock_s
        integer = np.rint(ticks).astype(np.int64)
        gap = int(np.ceil(experiment.min_gap_s/experiment.clock_s))
        guard = int(np.ceil(experiment.edge_guard_s/experiment.clock_s))
        timing = bool(np.all(abs(ticks-integer) < 1e-7)) and (not len(e) or bool(
            integer[0] >= max(1,guard) and integer[-2] <= integer[-1]-max(1,guard)
            and np.all(np.diff(integer[:-1]) >= max(1,gap))))
        minimum_gap = float(np.min(np.diff(integer[:-1]))*experiment.clock_s) if len(e) > 1 else None
    v = response(experiment, e, t, start)
    padding = 64*np.finfo(float).eps*max(1.0, 2*np.pi*experiment.coupling_hz*t)*np.sqrt(len(v))*(len(e)+1)
    error = experiment.amplitude_bound*(np.linalg.norm(v-experiment.target(sequence.gain)) + padding)
    return dict(timing_valid=timing, phase_error_bound=float(error),
                minimum_gap_s=minimum_gap, response=v.tolist())


def _objective(experiment, gain, duration, pulses, start, minimum):
    slack = duration - sum(minimum)
    sign = (-1.0)**np.arange(pulses)
    omega = 2*np.pi*experiment.coupling_hz
    base = omega*_integrals(experiment, 0.0, duration, start)[0]

    def evaluate(x):
        p = np.exp(x-np.max(x)); p /= p.sum()
        edges = np.cumsum(minimum + slack*p)[:-1]
        v = base + 2*omega*(sign @ _integrals(experiment, np.zeros(pulses), edges, start))
        error = experiment.amplitude_bound*(v-experiment.target(gain))
        f = 2*np.pi*np.asarray(experiment.frequencies_hz)
        angles = (start+edges)[:, None]*f
        features = np.c_[np.cos(angles), np.sin(angles)]
        edge_gradient = 2*omega*experiment.amplitude_bound*sign*(features @ error)
        gap_gradient = np.r_[np.cumsum(edge_gradient[::-1])[::-1], 0.0]
        gradient = slack*p*(gap_gradient-p @ gap_gradient)
        return 0.5*float(error @ error), gradient, edges
    return evaluate


def _clock(experiment, edges, duration, gain, start, steps=100):
    """Snap, repair gaps, then improve by feasible one-tick moves; re-audit afterward."""
    tick = experiment.clock_s
    end = int(round(duration/tick))
    gap = max(1, int(np.ceil(experiment.min_gap_s/tick)))
    guard = max(1, int(np.ceil(experiment.edge_guard_s/tick)))
    if end < 2*guard + (len(edges)-1)*gap:
        return None
    times = np.rint(edges/tick).astype(np.int64)
    for i in range(len(times)):
        times[i] = max(times[i], guard if not i else times[i-1]+gap)
    for i in range(len(times)-1, -1, -1):
        times[i] = min(times[i], end-guard if i == len(times)-1 else times[i+1]-gap)
    omega = 2*np.pi*experiment.coupling_hz
    target = experiment.target(gain)
    for _ in range(steps):
        e = times*tick
        v = response(experiment, e, duration, start)
        error = v-target
        best = (0.0, None, None)
        for shift in (-1, 1):
            valid = (times+shift >= np.r_[guard, times[:-1]+gap]) & (times+shift <= np.r_[times[1:]-gap, end-guard])
            changes = 2*omega*((-1.0)**np.arange(len(e)))[:, None]*_integrals(experiment, e, e+shift*tick, start)
            improvements = -(2*changes @ error + np.sum(changes*changes, axis=1))
            if np.any(valid):
                i = int(np.argmax(np.where(valid, improvements, -np.inf)))
                if improvements[i] > best[0]:
                    best = (improvements[i], i, shift)
        if best[1] is None or best[0] < 1e-15:
            break
        times[best[1]] += best[2]
    return times*tick


def _search_candidates(experiment, gains, durations, counts, max_gain, max_duration_s, max_pulses):
    limits = {}
    for name, supplied, value, default in (
            ("max_gain", gains, max_gain, 4.0),
            ("max_duration_s", durations, max_duration_s, 40e-6),
            ("max_pulses", counts, max_pulses, 128)):
        if supplied is not None:
            if value is not None:
                raise ValueError(f"Supply either {name} or its explicit candidates, not both.")
            continue
        value = default if value is None else value
        if name == "max_pulses":
            valid = isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value >= 0
        else:
            valid = np.isscalar(value) and not isinstance(value, bool) and np.isfinite(value) and value > 0
        if not valid:
            raise ValueError(f"{name} must be a nonnegative integer." if name == "max_pulses"
                             else f"{name} must be positive and finite.")
        limits[name] = int(value) if name == "max_pulses" else float(value)
    if gains is None:
        cap, gain, gains = limits["max_gain"], 0.5, []
        while gain < cap:
            gains.append(gain)
            gain *= 2
        gains.append(cap)
    if durations is None:
        durations = []
        for fraction in (0.25, 0.5, 1.0):
            duration = fraction*limits["max_duration_s"]
            if experiment.clock_s:
                tick = experiment.clock_s
                ticks = int(np.floor(duration/tick))
                if ticks*tick > duration:
                    ticks -= 1
                duration = ticks*tick
            if duration > 0:
                durations.append(duration)
        if not durations:
            raise ValueError("max_duration_s is shorter than clock_s.")
    if counts is None:
        cap = limits["max_pulses"]
        counts = [2*(cap//divisor) for divisor in (8, 4, 2)]
    return gains, durations, counts, limits


def compile_bank(experiment, *, max_gain=None, max_duration_s=None, max_pulses=None,
                 gains=None, durations_s=None, pulse_counts=None,
                 start_time_s=0.0, phase_tolerance=1e-3, maxiter=1200, restarts=2, seed=0):
    """Search within gain, duration, and pulse-count limits (defaults: 4, 40 us, 128).

    Automatically generate intermediate candidates and keep the shortest passing
    duration per gain. Explicit candidate lists remain available for advanced use;
    do not supply both a limit and a list for the same quantity. These local fits
    are not globally shortest controls. Every returned sequence passes a response
    and timing audit, including after clock rounding, and stays within the limits.
    """
    experiment.require_single()
    gains, durations, counts, limits = _search_candidates(
        experiment, gains, durations_s, pulse_counts, max_gain, max_duration_s, max_pulses)
    gains, durations, counts = sorted(set(gains)), sorted(set(durations)), sorted(set(counts))
    if not gains or not durations or not counts or not np.isfinite(gains+durations).all():
        raise ValueError("Supply nonempty finite gains, durations, and pulse counts.")
    if min(gains) <= 0 or min(durations) <= 0 or phase_tolerance <= 0 or not np.isfinite(phase_tolerance):
        raise ValueError("Gains, durations, and phase_tolerance must be positive.")
    if not np.isfinite(start_time_s) or start_time_s < 0 or maxiter < 1 or restarts < 1:
        raise ValueError("Invalid start time or optimizer budget.")
    if any(isinstance(n, bool) or int(n) != n or n < 0 or n % 2 for n in counts):
        raise ValueError("Pulse counts must be nonnegative even integers.")
    if experiment.clock_s and any(abs(t/experiment.clock_s-round(t/experiment.clock_s)) > 1e-7 for t in durations):
        raise ValueError("Candidate durations must be multiples of clock_s.")
    rng = np.random.default_rng(seed); sequences, attempts = [], []
    for gain in gains:
        chosen = None
        for duration in durations:
            for pulses in map(int, counts):
                margin = 64*np.finfo(float).eps*duration
                minimum = np.r_[experiment.edge_guard_s+margin,
                                np.full(max(0, pulses-1), experiment.min_gap_s+margin),
                                experiment.edge_guard_s+margin] if pulses else np.zeros(1)
                if sum(minimum) >= duration:
                    attempts.append(dict(gain=gain, duration_s=duration, pulses=pulses, reason="timing capacity"))
                    continue
                evaluate = _objective(experiment, gain, duration, pulses, start_time_s, minimum)
                for restart in range(restarts if pulses else 1):
                    x = rng.normal(0, 0.12, pulses+1)
                    if pulses:
                        result = minimize(lambda z:evaluate(z)[:2], x, jac=True, method="L-BFGS-B",
                            options=dict(maxiter=maxiter, ftol=1e-15, gtol=1e-10, maxcor=30, maxls=40))
                        x = result.x
                    edges = evaluate(x)[2]
                    if experiment.clock_s and pulses:
                        edges = _clock(experiment, edges, duration, gain, start_time_s)
                    if edges is None:
                        continue
                    v = response(experiment, edges, duration, start_time_s)
                    sequence = Sequence(float(gain), float(duration), float(start_time_s),
                                        tuple(edges.tolist()), tuple(v.tolist()), 0.0)
                    checked = audit(experiment, sequence)
                    attempts.append(dict(gain=gain, duration_s=duration, pulses=pulses,
                                         restart=restart, **{k:v for k,v in checked.items() if k != "response"}))
                    if checked["timing_valid"] and checked["phase_error_bound"] <= phase_tolerance:
                        chosen = Sequence(sequence.gain, sequence.duration_s, sequence.start_time_s,
                            sequence.switch_times_s, sequence.phase_vector, checked["phase_error_bound"])
                        break
                if chosen:
                    break
            if chosen:
                break
        if chosen:
            sequences.append(chosen)
    if not sequences:
        raise CompilationError("No candidate passed. Increase duration/pulse limits or restarts; inspect your timing and response target.")
    search = dict(gains=list(map(float,gains)),durations_s=list(map(float,durations)),
                  pulse_counts=list(map(int,counts)),start_time_s=float(start_time_s),
                  maxiter=int(maxiter),restarts=int(restarts),seed=int(seed),method="L-BFGS-B on positive gaps")
    if limits:
        search["limits"] = limits
    return PulseBank(experiment, tuple(sequences), float(phase_tolerance), tuple(attempts),search)
