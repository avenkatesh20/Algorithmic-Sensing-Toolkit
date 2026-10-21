"""Signal convention, requested observable, and calibrated hardware parameters."""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class Experiment:
    """b(t)=sum(a*cos(2*pi*f*t)+b*sin(2*pi*f*t)); H=2*pi*coupling*b(t)*Z/2.

    Frequencies are in Hz, times in seconds, and ||[a,b]|| <= amplitude_bound.
    Each weight vector defines one observable; (M, N) arrays define M observables.
    Results are cosine_weights @ a + sine_weights @ b, in the supplied units.
    Pulses are ideal instantaneous X(pi) rotations, specified by their times.
    """

    frequencies_hz: tuple[float, ...]
    cosine_weights: tuple[float, ...] | tuple[tuple[float, ...], ...]
    sine_weights: tuple[float, ...] | tuple[tuple[float, ...], ...]
    coupling_hz: float
    t2_s: float | None = None
    amplitude_bound: float = 1.0
    min_gap_s: float = 20e-9
    edge_guard_s: float = 20e-9
    clock_s: float | None = None
    visibility: float = 1.0
    readout_bias: float = 0.0
    overhead_s: float = 0.0

    def __post_init__(self):
        f = np.asarray(self.frequencies_hz, dtype=float)
        if f.ndim != 1 or not f.size or not np.isfinite(f).all():
            raise ValueError("frequencies_hz must be a finite, nonempty vector.")
        object.__setattr__(self, "frequencies_hz", tuple(f.tolist()))
        c, s = (np.asarray(w, dtype=float) for w in (self.cosine_weights, self.sine_weights))
        if (c.ndim not in (1, 2) or c.shape != s.shape or not c.size
                or c.shape[-1] != len(f) or not np.isfinite(c).all() or not np.isfinite(s).all()):
            raise ValueError("Weights must be matching finite (N,) vectors or (M,N) arrays.")
        for name, a in (("cosine_weights", c), ("sine_weights", s)):
            value = tuple(a.tolist()) if a.ndim == 1 else tuple(map(tuple, a.tolist()))
            object.__setattr__(self, name, value)
        if np.any(f < 0) or len(np.unique(f)) != len(f):
            raise ValueError("Frequencies must be distinct and nonnegative.")
        if np.any(s[..., f == 0] != 0):
            raise ValueError("A DC sine coefficient is unobservable; set its weight to zero.")
        if c.ndim == 1 and not np.linalg.norm(self.weights):
            raise ValueError("The observable must be nonzero.")
        for name in ("coupling_hz", "amplitude_bound"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for name in ("min_gap_s", "edge_guard_s", "overhead_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative and finite.")
        for name in ("t2_s", "clock_s"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite, or None.")
        if not (np.isfinite(self.visibility) and 0 < self.visibility <= 1):
            raise ValueError("visibility must be in (0, 1].")
        if not np.isfinite(self.readout_bias) or abs(self.readout_bias) + self.visibility > 1:
            raise ValueError("Require |readout_bias| + visibility <= 1.")

    @property
    def weights(self):
        return np.concatenate((self.cosine_weights, self.sine_weights), axis=-1)

    def require_single(self):
        if self.weights.ndim != 1:
            raise ValueError("Use compile_queries for multiple observables.")

    @property
    def observable_scale(self):
        self.require_single()
        return float(np.linalg.norm(self.weights) * self.amplitude_bound)

    def target(self, gain):
        """A gain g produces phase g*x, x=observable/observable_scale in [-1,1]."""
        self.require_single()
        return gain * self.weights / (np.linalg.norm(self.weights) * self.amplitude_bound)

    def contrast(self, duration_s):
        return self.visibility * np.exp(-duration_s / self.t2_s) if self.t2_s else self.visibility

    def to_dict(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

