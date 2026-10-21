# Randomized spectroscopy

`randomized_sensing.Experiment` specifies known frequencies and hardware
parameters. No observables are supplied during pulse design or acquisition.
After collecting counts, choose any linear observables and pass their
`cosine_weights` and `sine_weights` to `decode`. The same counts support
different lists of queries without rebuilding the pool or acquiring more data.

`decode(plan, counts)` returns the amplitude estimate with empty `values`.
`decode(plan, counts, cosine_weights=..., sine_weights=...)` also returns the
requested `K @ amplitudes`. Alternatively, call `evaluate_observables` on an
already decoded amplitude vector. Weights accept matching `(N,)` vectors or
`(M,N)` matrices, with one row per query. `population_estimate` accepts the same
optional weights for validation on a known synthetic field.

New acquisition bundles contain no observable weights. Earlier bundles that
stored weights remain readable, retain their count identities, and allow new
weights to override the earlier queries during decoding.

## Pulses and acquisition

Each distinct waveform is the sign of a multisine with independent phases
uniform on `[0, 2*pi)`. Short runs are merged, and the response of the repaired
waveform is integrated exactly over every supplied frequency. The pool also
includes the negative of every response. Unlike a Fourier-grid construction,
this implementation supports arbitrary **positive** frequencies and durations
on the chosen timing grid. DC is currently unsupported. All input tones must
be listed, including tones with zero observable weight.

`build_pool` takes `duration_s`, `max_pulses`, `pool_size`, and `seed`.
The default pool has `max(8192, 64*N)` entries; half are independent patterns
and half their negatives. There are no cyclic shifts. Waveforms exceeding the
pulse cap are rejected; this changes the pool law and can affect estimator
bias. A failed search or poorly conditioned covariance returns an error.

`clock_s` sets the hardware grid. If it is `None`, the explicit
`design_clock_s` (default 2.5 ns) is a numerical synthesis grid. The duration
must contain an integer number of cells. Every free interval conservatively
obeys `max(min_gap_s, edge_guard_s)`, rounded **up** to that grid. A larger
edge guard therefore also restricts internal intervals. Controls are ideal
instantaneous X(pi) pulses; pulse widths are not modeled.

Physical modulation always starts at +1. Negative pool responses are realized
by a decoder sign convention, with the same physical waveform. Odd pulse
counts also invert physical Y readout. `outcome_sign` accounts for both:

```
corrected_Y_mean = outcome_sign * (raw_Y_mean - readout_bias)
```

No initial or final correction pulse is added. X means need only bias
subtraction. The decoder performs these corrections: supply **raw physical**
positive-outcome counts. The exported `RUN.txt` specifies the analysis convention.

`plan_acquisition` draws entries uniformly with replacement. The default
`field_mode="phase_locked"` groups repeat sequence/basis pairs into count rows;
each replay must reproduce the starting tone phases used in the pool.
Its fixed `n_shots` budget includes both X and Y shots. Grouping assumes a fixed
field and independent outcomes.
Active time is `n_shots * duration_s`; elapsed time also includes preparation,
readout, reset and phase-reference waits. The reported overhead total adds
only the specified `overhead_s` per shot.

## Freely running fields

Set `field_mode="free_running"` to acquire without resetting or repeating the
source phases. The frequencies and global Fourier amplitudes must remain fixed;
their phases evolve continuously according to the recorded start times.
Observable weights can still be chosen after acquisition.

```python
plan = R.plan_acquisition(pool, n_shots=20_000, field_mode="free_running",
                          shot_period_s=20e-6, seed=5)
```

`shot_period_s` includes sensing and preparation/readout/reset time. It must be
at least `duration_s + overhead_s`; its default is that sum. Alternatively,
pass a complete `start_times_s` schedule. These are **planned** times. Each
physical shot has its own acquisition row with `shots=1`, and X/Y bases are
randomly interleaved. `scheduled_span_s` reports the planned interval from the
first sensing start to the last sensing end; it excludes final readout overhead.

In a copy of `counts_template.csv`, record the raw `plus` outcome (0 or 1) and
the **actual** `start_time_s` for every shot. `acquisition.csv` separately labels
the nominal schedule as `planned_start_time_s`. The template leaves measured
timestamps blank; the decoder rejects missing, nonfinite, negative, or overlapping
start times. Outcomes at different starts cannot be merged into a count row.

Use elapsed times in seconds from one common stable clock origin, not Unix
timestamps. That origin defines the cosine/sine amplitudes and downstream query
weights; the source's phase at zero may be unknown and is part of those amplitudes.
Timestamp uncertainty produces a phase uncertainty `2*pi*f*delta_t`, so use
sufficient precision for the highest frequency. Timestamp uncertainty, frequency
drift, and changes to relative pulse timings are not estimated by this model.

For a stored complex response `Z_j = z_cos_j + i*z_sin_j` at pool reference time
`t0`, the response at measured start `t` is exactly

```text
Z_j(t) = Z_j(t0) * exp(i*2*pi*f_j*(t-t0))
```

`pool.responses_at(indices, times)` performs this analytic rotation. It preserves
the same physical pulse waveform and its timing constraints. Absolute pulse
reference times are left blank in the freely running pulse export because one
waveform is reused at different starts.

`read_counts` returns `TimedCounts(positive_counts, start_times_s)` for this mode;
both dictionaries use acquisition row IDs. `simulate` can accept an alternative
`start_times_s` array or row-ID mapping to model delays from the plan.
`write_counts` preserves those measured times. Saved phase-locked bundles keep
their existing count format and identities.

`decode(plan, counts)` selects `TIMESTAMPED-ML` for freely running data. It uses
the actual responses and calibrated offset in a binary likelihood, jointly
fitting amplitudes and contrast within the amplitude norm bound. Three
deterministic starting points come only from the data and calibrations. It
requires at least `3*2N` total shots and a full-rank Y-response design; failures
of these checks or fit convergence are reported as errors.

The fixed-pool Gaussian/moment identity is not assumed for this reconstruction.
`RQS` and `RQS-ML` remain the phase-locked estimators; requesting them on freely
running data raises an error. This bounded local likelihood fit does not certify
a global optimum, uniqueness, precision, or confidence interval.

For validation, `population_estimate(plan, amplitudes, start_times_s=...)` passes
exact per-shot outcome probabilities through the same fit. This removes shot
noise and checks numerical inversion for that field and schedule. The decoder
receives those probabilities, not the true amplitudes. The notebook also supports
repeated noisy trials using the same actual timestamp table.

## Phase-locked postprocessing

With signed responses `z` and pool covariance `Sigma = mean(z z.T)`, RQS forms
`v = mean(y*z)`, `mu = mean(x)`, and `Sigma^-1 @ v / mu`, then applies a
per-tone magnitude correction based on the pool's Bessel moments. Contrast
cancels in the unguarded ratio. The calibrated contrast and amplitude bound
set a lower guard on `mu`; `floor_active=True` exposes when it binds.
The calibrated additive readout offset is subtracted separately.

This moment inversion is exact under an ideal Gaussian response law. Real
finite, repaired pools only approximately obey the assumed random-phase law,
and the per-tone correction does not remove all cross-tone correlations.
Systematic error can therefore remain with arbitrarily many shots.
`RQS-ML` adds up to 20 local likelihood iterations using the actual response
matrix and binary counts, jointly fitting amplitudes and contrast. With fewer
than `3 * 2N` shots, it returns the moment estimate unchanged. This local fit
does not certify a unique/global likelihood solution.

No uniform precision or confidence intervals are claimed. Choose the shot
budget and duration using independent validation for the signals of interest;
inspect both shot noise and noiseless population-moment error. Adding shots
alone does not eliminate model bias. The notebook includes both checks.

In phase-locked mode, the starting tone phases must be reproducible for every
waveform trigger. Arbitrary frequency spacings do not imply a convenient common
recurrence. Freely running mode instead requires accurate per-shot timestamps.
Bundles include pulse times, intervals, acquisition counts, experiment inputs,
the full pool and file hashes; decoding needs no files from Downloads.
