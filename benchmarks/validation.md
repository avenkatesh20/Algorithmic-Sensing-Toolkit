# Free-running correctness checks

Routine validation uses deterministic regression tests and small paired RQS/ML
comparisons. Exhaustive likelihood scaling runs are optional reproduction work.

From the toolkit root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m pytest -q
```

The tests check:

- **Independent physical data:** integrate the actual +/-1 pulse intervals
  analytically, without the simulator, stored response table, rotation helper,
  or outcome-sign metadata. Decode these continuous expected outcomes with
  both RQS and TIMESTAMPED-ML, with no dephasing and with T2=200 us. The cases
  include nonzero readout bias, visibility below one, odd/even pulse counts,
  both virtual signs, shifted triggers, and recorded jitter. RQS must recover
  each amplitude within 0.015; likelihood recovery and contrast must agree
  within 2e-5. These tolerances test this synthetic example, not arbitrary data.
- **Estimator mathematics:** whiten before rotation; compare with an explicit
  per-shot sum. Reference-frame and commensurate schedules agree with
  phase-locked RQS within 1e-12. Both field modes share the contrast guard and
  per-tone correction.
- **Likelihood:** check the analytic gradient against finite differences;
  verify that adding the fourth start retains the objective quality of the
  legacy three starts. Better likelihood need not mean lower amplitude error
  on an individual noisy trial.
- **Eligibility and failures:** 50-tone acquisitions with 2 or 200 shots
  decode below the former 300-shot floor. Optional ML returns the same RQS
  amplitudes when shots or Y rank are insufficient. Actual fit failures and
  missing/invalid measured timestamps still raise errors.
- **Controls and compatibility:** compare pulse responses with independent
  interval integrals; enforce clock, minimum gaps, edge guards and pulse caps;
  check count/mean input paths, bundle round-trips and plan identities. Serial
  and parallel correction-table preparation give identical phase-locked
  results. The notebook's phase-locked and free-running paths also execute.

The saved [paired comparison](results/free_running_refinement_check.md) contains
60 trials across 15 conditions, with fresh banks and both dense/single-tone
fields. Forty-four trials applied ML; it reduced amplitude max error in 32.
The other 16 returned identical RQS estimates below the old floor, where the
former ML-only path refused. These small runs check behavior; they do not
measure T95 or a likelihood scaling exponent.

The larger [RQS-only report](results/free_running_scaling_RQS.md) is retained as
additional numerical evidence. It includes reference discrepancies and search
convergence flags. No exhaustive 400-trial likelihood sweep is needed for this
correctness validation, and none is claimed.
