# Randomized-sensing checks

Use Python >=3.10 from the toolkit root. The experimental pulse constraints and
binary readout model are the same as in the library.

For routine implementation checks, use the [correctness validation](validation.md)
and small paired likelihood comparison. Exhaustive likelihood scaling sweeps
are optional and are not needed to run the experimental notebook.

Saved results: [RQS scaling](results/free_running_scaling_RQS.md) and
[small paired likelihood check](results/free_running_refinement_check.md).
The JSON files beside these reports retain budgets, errors, failures, and
search histories; SVG/PNG figures accompany the scaling report.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/build_pool.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/free_running_scaling.py --ml-trials 0 --output benchmarks/results/free_running_scaling_RQS.json
```

The scaling script uses 128 calibration signals, then 400 held-out signals over
two fresh banks. Half are dense Gaussian directions and half are single tones,
each with norm 0.65 (or 0.2 in the noiseless stress test). It searches duration
in 40 us steps and X fractions 0.1/0.2/0.3. Free-running periods are duration
plus 1.2345 us, with recorded uniform 0–10 ns trigger jitter.

The 95th percentile of the maximum error over all 2N amplitudes predicts the
next shot budget. Calibration and held-out searches stop within 5% or after
`--max-rounds`. Results retain evaluated budgets, estimated T95, success rates,
histories, convergence flags, and fitted slopes. Report nonconverged points;
do not interpret an extrapolated budget as a separately measured success rate.
Completed conditions are checkpointed. Reuse the same output to resume;
different configurations need a different output path.
These are per-amplitude L-infinity tests at fixed signal L2 norm, not precision
guarantees for arbitrary later observables. Reported sensing time is n*tau;
classical compilation, bank statistics, and decoding time are separate.

Omit `--ml-trials 0` to also run the former ML-only eligibility rule and the new
optional refinement on 400 held-out trials each. These compare methods at the
RQS-selected durations and X fractions; they do **not** independently optimize
the duration for ML-only sensing. Likelihood fitting is substantially slower
than moment decoding on this machine. Failures and skipped fits are reported.
Moment trials can run concurrently; full likelihood trials run serially.
The benchmark stores verified pulse-time tuples as float64 arrays and
materializes the identical tuples on access. This reduces Python-object memory
when holding two large banks. No pulse times or responses are changed, and
plan hashing, export and independent audits still see the original values.
Bank synthesis uses a bounded eight-million-element batching budget locally
to keep four FFT workers busy on long shots. The public library default is
unchanged; the random phase stream and physical controls are preserved.
Correction-table preparation uses two independent column-block workers in the
benchmark. It performs the same sums and interpolation as the serial library
default and is checked for bit-identical results.

For a smaller paired check at the completed RQS budgets:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/free_running_refinement_check.py
```

This uses four new trials per condition, split across two additional banks and
the two signal types. It compares errors and classical decoding time, and
records old ML-only refusals below the shot floor. Four trials are illustrative;
they do not estimate T95, confidence, or a refinement scaling exponent.

The dashed `6N` curve is only the old ML eligibility lower bound at the plotted
durations, not a measured ML T95. If duration scales as sqrt(N), this curve
scales as N^1.5. The shot floor alone does not imply that slope under arbitrary
retuning of duration. The same distinction applies to the weak-signal test.
Reference values from the supplied fixes note are labeled separately from
measured toolkit results. No dynamical-decoupling baseline is run here.
