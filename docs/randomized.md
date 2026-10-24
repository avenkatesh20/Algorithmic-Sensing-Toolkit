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
includes the negative of every response. This implementation supports arbitrary
**positive** frequencies and durations
on the chosen timing grid. DC is currently unsupported. All input tones must
be listed, including tones with zero observable weight.

`build_pool` takes `duration_s`, `max_pulses`, `pool_size`, and `seed`.
The default pool has `max(8192, 64*N)` entries; half are independent patterns
and half their negatives. There are no cyclic shifts. Waveforms exceeding the
pulse cap are rejected; this changes the pool law and can affect estimator
bias. A failed search returns an error.

Pool building automatically uses FFTs when all `K*f*duration_s` are integers
for a common `K` in `{1,2,4,8,16}`, to floating-point tolerance. Fully periodic
tones use `K=1`, avoiding unnecessary zero padding. The FFT includes
the cell-midpoint and `start_time_s` phases. Otherwise, dense trigonometric
tables support arbitrary frequencies. Responses are exact integrals of the
**repaired, signed** cells: FFT coefficients on a compatible grid, or cell-table
matrix products off grid. Dense tables are cached up to about 64 MB, then
chunked in time; pattern batches and FFT workspaces are also bounded. No
optional transform dependency is needed. `pool.synthesis_method` reports
`"fft"` or `"dense"`. To use multiple FFT workers locally:

```python
from scipy.fft import set_workers

with set_workers(4):
    pool = R.build_pool(experiment, duration_s=40e-6, seed=4)
```

For long commensurate shots, common tone-bin factors reduce the transform to
the multisine's natural period. The full waveform still undergoes timing repair;
its repaired cells are folded into that period for exact response integration,
including boundary changes and partial final periods. Every pool entry retains
its independently sampled phases and complete physical pulse sequence.

The default respects SciPy's worker setting (normally one). Each waveform has
independent random phases; there is no cyclic-shift augmentation. Closed-form
interval integration remains the independent verifier in `audit` and bundle
export/load, with the existing response-error tolerance. Large exports can
therefore take longer than pool building.

### Conditioning

Before building, a warning flags `min_frequency_spacing*duration_s < 1`, a
resolution heuristic rather than a guarantee of failure. After building,
`pool.condition` stores the covariance eigenvalue ratio. Values above **20**
warn that weak directions amplify noise; values above **1000** reject a new
build. Increasing duration, increasing tone separation, relaxing timing, or
increasing an undersized pool can help. A low condition number alone does not
certify the estimator's approximation accuracy.

For diagnostic use, `build_pool(..., allow_ill_conditioned=True)` permits
condition numbers above 1000 with a warning. Numerically singular covariances
(condition at least `1e10`) are still rejected. Loading existing bundles uses
the warning behavior, preserving earlier valid saved data and plan identities.
Do not rebuild an acquired plan to read its measurements: changed summation
roundoff can change new response bytes and hence new plan IDs even when the
pulse times match. See [implementation changes](../CHANGELOG.md).

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
positive-outcome counts or averaged outcomes. The exported `RUN.txt` specifies
the analysis convention.

`plan_acquisition` draws entries uniformly with replacement. The default
`field_mode="phase_locked"` groups repeat sequence/basis pairs into count rows;
each replay must reproduce the starting tone phases used in the pool.
Its fixed `n_shots` budget includes both X and Y shots. Grouping assumes a fixed
field and independent outcomes.
Active time is `n_shots * duration_s`; elapsed time also includes preparation,
readout, reset and phase-reference waits. The reported overhead total adds
only the specified `overhead_s` per shot.

For `free_running` schedules made with `shot_period_s`, a warning flags a
period that repeats **every** tone phase (all `f*period` are integers within
`1e-9` cycles). Timestamps then add no phase diversity. This does not imply a
rank-deficient design: distinct random waveforms may still span the signal.
Choose a noncommensurate period if phase diversity is desired; the actual
timestamped response and full-rank checks remain unchanged. Explicit start-time
tables are not modified or subjected to this regular-period heuristic.

## Averaged outcomes

For phase-locked acquisition, each sequence/basis row can be measured repeatedly
and reduced to its raw physical mean in `[-1,1]`. For binary outcomes this is
`mean = 2*plus/repetitions - 1`. For example, 77 positive outcomes out of 100
give `0.54`; 45 out of 100 give `-0.1`. The decoder accepts rounded means and
does not round them back into integer counts.

```python
# One mean per acquisition.csv row, in the same order. Supply the complete list.
result = R.decode_averages(plan, averages, repetitions=100,
                           cosine_weights=cosine_weights,
                           sine_weights=sine_weights)
```

`averages` can be a list/array or a dictionary keyed by acquisition `row_id`.
Values for X and Y measurement bases are separate, even when their pulses match.
Supply raw means before readout-bias subtraction or pulse-frame/sign correction.
The existing corrections, contrast ratio, small-X guard, and per-tone correction
are applied once. The moment accumulators are linear in the recorded means;
the contrast normalization and per-tone inversion are unchanged.

`repetitions` accepts one positive integer for all rows, a matching array, or a
row-ID dictionary. If omitted, it defaults to each row's planned `shots`.
The random-draw multiplicities in the frozen plan remain the RQS integration
weights. Repeating one acquired waveform more often improves its mean estimate;
it does not change the original sampling distribution. Actual repetition counts
instead weight the optional `method="RQS-ML"` likelihood refinement. For exact
binary averages, means and counts are equivalent sufficient statistics. For
rounded or analog means, refinement uses a weighted Bernoulli objective and
does not certify a noise model, precision, or confidence interval.

Phase-locked exports include `averages_template.csv`, with columns
`plan_id,row_id,repetitions,mean`. Keep its IDs, fill each mean, and update
repetitions if needed. Save a copy and load it with
`R.read_averages("measured_averages.csv", plan)`. The returned `AveragedOutcomes`
can be passed directly to `R.decode`. `R.write_averages` accepts the same arrays,
mappings, and optional repetitions. Original count CSVs and older bundles remain
readable. An ordered array passed directly to `R.decode` is interpreted as means;
plain dictionaries passed to `R.decode` retain the positive-count convention.

The plan's reported shot/time totals describe its original acquisition schedule.
If actual repetitions differ, active time is `sum(repetitions)*duration_s`;
add `overhead_s` per repetition for the specified overhead total.

Every repetition included in a mean must reproduce the same field phases.
Freely running shots at different phases cannot be merged into one mean and
one timestamp. Keep their individual timestamped outcomes for the existing
freely running decoder.

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

`decode(plan, counts)` selects **RQS** for freely running data. For each Y
shot, it computes `u = Sigma^-1 @ z` in the pool reference frame, then rotates
`u` to the actual starting time. The order is important: covariance whitening
and the per-tone rotation do not generally commute. Corrected Y outcomes weight
these rotated vectors. Their mean is divided by the bias-corrected X mean,
using the same guard floor and per-tone Bessel-map correction as phase-locked
RQS. Contrast cancels in the unguarded ratio.

The moment estimate needs only one X and one Y shot, including when the total
is below `6*N` or the measured Y design is rank deficient. This does not promise
accuracy with very few shots. Finite pools, timing repair, covariance anisotropy,
and departures from the random-phase model can leave bias; phase rotations do
not remove those approximations. `pool.rotate_vectors(vectors, times)` exposes
the same long-double/remainder rotation for arbitrary `(...,2N)` vectors.
Bank-frame solves are cached up to 128 MB; larger banks use bounded batches.

`method="TIMESTAMPED-ML"` requests the moment estimate followed by the existing
constrained amplitude/contrast likelihood fit. The moment result supplies a
fourth starting point, in addition to the three data/calibration starts. A full
fit still needs at least `3*2N` total shots and a full-rank Y-response design.
If these prerequisites fail, the result contains the RQS amplitudes with
`ml_applied=False`, `fit_converged=None`, and `refinement_skipped_reason`.
Missing/invalid timestamps and actual likelihood convergence failures still
raise errors. `RQS-ML` remains the phase-locked refinement name.

```python
result = R.decode(plan, measured)  # free-running RQS, with no 6N floor
refined = R.decode(plan, measured, method="TIMESTAMPED-ML")
print(refined.ml_applied, refined.refinement_skipped_reason)
```

Binary count files are unchanged. For normalized continuous readout, pass
`TimedOutcomes(raw_values, actual_start_times)` to `decode`: one raw value in
`[-1,1]` and one actual timestamp per acquisition row. Each value must describe
its recorded starting phase. It cannot average shots with different phases.
The moment accumulators accept these values directly; optional refinement uses
fractional Bernoulli sufficient statistics, without certifying an analog noise
model or a confidence interval.

`population_estimate(plan, amplitudes, start_times_s=...)` now feeds exact
per-shot expected outcomes into free-running RQS by default. It removes binary
outcome noise while retaining the finite waveform/schedule draw variation.
Use `method="TIMESTAMPED-ML"` for the optional noise-free likelihood check;
its same eligibility/fallback rules apply. The true amplitudes generate the
synthetic data but are not passed to either reconstruction method.

The [scaling benchmark](../benchmarks/free_running_scaling.py) separates
calibration signals from held-out evaluation on fresh banks, compares the
phase-locked and free-running moment estimates, and optionally evaluates the
former ML-only shot restriction and the new refinement. It reports evaluated
and extrapolated T95 budgets separately, nonconvergence, trial counts, and
fitted slopes. A sqrt(N) law is a regime-dependent expectation to test, not a
software guarantee. Weak signals can reveal the old shot floor more clearly.

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
