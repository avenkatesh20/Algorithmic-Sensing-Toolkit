# Changes

## Unreleased

- Randomized pools use FFT synthesis and exact FFT cell integration for
  compatible tones, with bounded-memory dense synthesis and exact cell-table
  integration for arbitrary frequencies. Random phases remain independent;
  cyclic shifts are not used. Timing repair, pulse caps, physical signs, and
  independent interval-integration audits are unchanged.
- The seed's phase-draw order is preserved, including pulse-cap rejections.
  FFT roundoff can change a sign at a multisine zero; response summation also
  changes last bits. Newly built pools can therefore have different `plan_id`s
  even with unchanged pulse times. Load the original saved acquisition bundle
  when decoding existing data. Bundle schema, identity hashing, and measured
  count/average formats are unchanged; legacy responses are never recomputed.
- Pools expose `condition` and `synthesis_method`. Unresolved-tone warnings
  precede synthesis; covariance condition above 20 warns and above 1000 rejects
  new builds unless `allow_ill_conditioned=True`. Existing nonsingular bundles
  still load with a warning. The numerical singularity limit remains `1e10`.
- Freely running regular schedules warn when their period repeats every tone
  phase. Actual timestamp handling, rank checks, and all estimator math are
  unchanged.
- Phase-locked randomized decoding accepts raw averaged outcomes in `[-1,1]`
  through `decode_averages`, `AveragedOutcomes`, and average CSV files, alongside
  the original positive-count inputs.
