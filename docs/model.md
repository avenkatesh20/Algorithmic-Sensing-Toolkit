# Model and conventions

All frequencies are in Hz; all times are in seconds. The field is fixed
throughout an acquisition:

\[
b(t)=\sum_j[a_j\cos(2\pi f_jt)+b_j\sin(2\pi f_jt)],\qquad
H(t)=\tfrac12(2\pi\,\texttt{coupling_hz})b(t)Z.
\]

The requested observables are `cosine_weights @ a + sine_weights @ b`.
Weights can be vectors for one observable or matching `(M,N)` arrays for a list.
The amplitude vector is ordered `[a_1,...,a_N,b_1,...,b_N]` and obeys
`norm(amplitudes) <= amplitude_bound`. We preserve the supplied weights:
normalization inside the compiler is reversed by the decoder.

## Pulses

Each sequence starts in the +1 modulation frame. An instantaneous X(pi)
pulse changes its sign. An even pulse count restores that frame before
X/Y analysis. `min_gap_s` constrains adjacent pulse times; `edge_guard_s`
constrains the first and last free-evolution intervals. An optional
`clock_s` places pulses and the sequence endpoint on that clock grid.
Rounded sequences are repaired and rechecked against the full response.

`start_time_s` is relative to the field's phase origin, not a wall-clock
timestamp. The compiler includes it in every sine/cosine integral. Reuse
the waveform only when every tone has the same starting phase, for example
through phase-locked triggering, a common recurrence, or a resettable source.
For a different start phase, compile at the new reference time. Arbitrary
frequency spacings are supported; a common recurrence is not assumed.

The compiler directly fits `V = gain * weights / (norm(weights)*amplitude_bound)`.
For any allowed field, the ideal phase is `gain * observable / observable_scale`.
Its response error is integrated over all listed modes, including unwanted
tones. Accepted controls satisfy the timing constraints and a uniform phase
error bound. A failed finite search does not establish physical infeasibility.
The returned control uses the shortest successful candidate duration for that
gain, not a globally shortest waveform.

`max_gain`, `max_duration_s`, and `max_pulses` bound each compiled sequence
(defaults: 4, 40 us, and 128). Gains start at 0.5 and double, with the cap
included; a cap below 0.5 gives that single gain. Durations use one quarter,
one half, and all of the duration cap, rounded down to the optional clock.
Pulse counts use the same fractions, rounded down to even integers; an odd
cap therefore permits at most the preceding even count. Duplicate candidates
and zero durations are removed. A zero pulse cap allows pulse-free evolution.

These are finite candidate grids, not an exhaustive or continuous search over
all values below the limits. The maximum gain is included as a target but may
not compile within the other limits; successful lower gains remain available.
All resolved candidates and requested limits are recorded in the bundle.
For advanced control, supply explicit `gains`, `durations_s`, or `pulse_counts`
instead of the corresponding maximum; the two forms cannot be combined for
the same quantity. Limits apply separately to every query coordinate and gain,
while `max_shots` in the readout planner limits the total acquisition.

## Counts and gain selection

With `x = observable / observable_scale`, an ideal response has phase `gain*x`.
The recorded +1 probabilities are

\[
p_X=\tfrac12[1+\texttt{readout_bias}+C\cos(\phi)],\quad
p_Y=\tfrac12[1+\texttt{readout_bias}+C\sin(\phi)],\qquad
C=\texttt{visibility}\,e^{-\tau/T_2}.
\]

`t2_s=None` removes dephasing. Set visibility and bias from calibrated readout.
The backend must implement this sign convention for its X/Y analysis pulses.

The planner searches declared gain chains, charging both X and Y shots.
Shot counts include the dephasing penalty and the audited phase-error bound.
The decoder intersects phase intervals while resolving 2*pi aliases. It
returns an estimate in the original observable units and a confidence interval.
Inconsistent or ambiguous counts produce an error rather than an arbitrary branch.

For a statistical phase radius `theta`, Hoeffding and a union bound give
`n >= 4*log(4*S/(1-confidence)) / (C*sin(theta))**2` shots per basis for `S`
stages. Add the compilation error to the statistical phase radius. The gain
chain leaves a strict alias margin; its final interval has half-width at most
the requested precision. Confidence is conditional on the amplitude bound,
correct frequencies and phase reference, independent binary outcomes, and
the stated calibrated noise model. Calibration uncertainty is not included.

Active interrogation time and active time plus `overhead_s` per shot are
reported separately. `optimize="with_overhead"` includes the latter in gain
selection. Additional trigger waits are not predicted. Reset the sensor before
each shot; a long total acquisition does not represent one coherent evolution.

## Shared query subspace

Let `K = concatenate([cosine_weights, sine_weights], axis=1)`, with one row
per query, and `A = [a,b]`. An SVD finds an orthonormal basis `U` for its
numerical row span, of dimension `r`. With `kappa = max_j norm(K_j)` and
`v_j = U.T @ K_j / kappa`, the requested outputs are

\[
y=U^T A,\qquad F_j=\kappa v_j^T y.
\]

`compile_queries` compiles one coherent bank for each column of `U`, retaining
all input frequencies and hardware constraints. `plan_queries` measures these
`r` shared coordinates; dependent queries require no additional acquisition.
Exported rows use `q{i}_stage{j}_{basis}` and global sequence indices, all
zero-based. `decode_queries` combines their estimates using the original weights.

The numerical residual is bounded by
`B_j = amplitude_bound * norm(K_j - kappa * U @ v_j)`, with a floating-point
margin. The default SVD cutoff is `eps * max(M,2N)` times the largest singular
value. Increasing `rank_rtol` can discard weak directions, but their residuals
remain in the error budget. Zero queries need no shots.

For requested absolute errors `epsilon_j`, the planner gives each coordinate
the precision `eta = min_j (epsilon_j - B_j)/(kappa * norm(v_j,1))`, ignoring
zero rows. It splits the total failure probability across the `r` coordinates.
Thus, simultaneously for every query, the error is at most
`kappa * norm(v_j,1) * eta + B_j <= epsilon_j`. The decoder propagates the
actual coordinate intervals, expanding them by `B_j`. It needs no assumption
of independence between reconstructed queries.

This implements the manuscript's shared-subspace factorization. The fixed X/Y
readout and interval allocation are conservative; they do not implement its
optimal directional-variance/minimax allocation or certify its asymptotic cost.
Coordinate gain chains are optimized individually within the declared search,
not jointly over all possible bases, waveforms, and measurement allocations.

## Current scope

The matched workflow supports one or several specified linear observables.
It uses full coherent responses and fixed X/Y measurements, not the research
scripts' adaptive terminal measurement or the manuscript's clipped-filter
plus short-Ramsey residual construction. Finite pulse width, control errors,
timing jitter, field drift, and colored noise require an extended physical
model. The package exports timings and counts, not instrument-specific drivers.

The separate [randomized workflow](randomized.md) estimates all Fourier
amplitudes from random multisine controls, then evaluates every requested
observable. Its RQS/RQS-ML estimates do not carry these matched confidence
interval guarantees.
