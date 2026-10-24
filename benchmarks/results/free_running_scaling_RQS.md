# Free-running RQS scaling checks

Binary projection noise is included; 'noiseless' means no dephasing.
Precision is max absolute error <=0.1 over all 2N amplitudes, at fixed signal L2 norm.
This is not an accuracy guarantee for arbitrary later linear observables.

Calibration signals: 128; held-out signals: 400 over 2 fresh banks.
Durations and X fractions are selected before held-out evaluation; only the shot budget is then searched.
T95 is the error-based estimate at the last evaluated budget. Success rates refer to the evaluated n.

## noiseless, norm 0.65

| N | tau (us) | X fraction | n locked evaluated/estimated | n free evaluated/estimated | T95 locked (ms) | T95 free (ms) | Free success | Budget agreement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 10 | 40 | 0.1 | 1673/1683 | 1574/1644 | 67.32 | 65.76 | 0.943 | True |
| 20 | 80 | 0.2 | 1663/1623 | 1666/1696 | 129.84 | 135.68 | 0.938 | True |
| 50 | 80 | 0.1 | 2550/2639 | 2700/2685 | 211.12 | 214.80 | 0.950 | True |
| 100 | 120 | 0.1 | 2700/2799 | 2700/2730 | 335.88 | 327.60 | 0.948 | True |
| 200 | 200 | 0.1 | 2642/2529 | 2464/2464 | 505.80 | 492.80 | 0.950 | True |
| 300 | 280 | 0.1 | 2502/2485 | 2502/2434 | 695.80 | 681.52 | 0.955 | True |
| 500 | 240 | 0.1 | 3958/3933 | 3969/3937 | 943.92 | 944.88 | 0.955 | True |

Fitted log-log slope, N>=50, phase_locked_RQS: **0.652**.
Fitted log-log slope, N>=50, free_running_RQS: **0.645**.
Free-running exponent split: duration **0.544** + estimated shots **0.101**.

Phase-locked deviations outside the note's 15% reference tolerance: N=10: -25.1%; N=50: -17.7%.

Free/phase-locked estimated T95 ratio: 0.974–1.045.

Evaluated RQS budgets below the former 6N eligibility floor: none at these sizes and settings.

Calibration candidate searches reaching the round limit: N=100 (X fraction: 0.3); N=500 (tau: 240).
These are knob-search candidates; final held-out budget agreement is reported separately.

## t2_1ms, norm 0.65

| N | tau (us) | X fraction | n locked evaluated/estimated | n free evaluated/estimated | T95 locked (ms) | T95 free (ms) | Free success | Budget agreement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 10 | 40 | 0.1 | 1950/1880 | 1846/1871 | 75.20 | 74.84 | 0.943 | True |
| 20 | 40 | 0.1 | 3533/3414 | 3490/3636 | 136.56 | 145.44 | 0.932 | True |
| 50 | 80 | 0.1 | 3095/3234 | 3268/3156 | 258.72 | 252.48 | 0.955 | True |
| 100 | 120 | 0.1 | 3510/3558 | 3510/3532 | 426.96 | 423.84 | 0.948 | True |
| 200 | 200 | 0.1 | 3944/4019 | 3955/3852 | 803.80 | 770.40 | 0.955 | True |
| 300 | 160 | 0.1 | 6791/6486 | 6421/6573 | 1037.76 | 1051.68 | 0.938 | True |
| 500 | 240 | 0.1 | 6669/6555 | 6606/6579 | 1573.20 | 1578.96 | 0.953 | True |

Fitted log-log slope, N>=50, phase_locked_RQS: **0.790**.
Fitted log-log slope, N>=50, free_running_RQS: **0.802**.
Free-running exponent split: duration **0.449** + estimated shots **0.353**.

Phase-locked deviations outside the note's 15% reference tolerance: N=10: -24.0%; N=20: -25.2%.

Free/phase-locked estimated T95 ratio: 0.958–1.065.

Evaluated RQS budgets below the former 6N eligibility floor: none at these sizes and settings.

Calibration candidate searches reaching the round limit: N=10 (tau: 40, 80); N=50 (tau: 80, 120); N=100 (tau: 160).
These are knob-search candidates; final held-out budget agreement is reported separately.

## noiseless, norm 0.2

| N | tau (us) | X fraction | n locked evaluated/estimated | n free evaluated/estimated | T95 locked (ms) | T95 free (ms) | Free success | Budget agreement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 20 | 200 | 0.3 | 207/216 | 202/207 | 43.20 | 41.40 | 0.935 | True |
| 50 | 280 | 0.2 | 261/249 | 256/254 | 69.72 | 71.12 | 0.953 | True |
| 100 | 400 | 0.1 | 284/295 | 312/297 | 118.00 | 118.80 | 0.963 | True |
| 200 | 600 | 0.2 | 320/311 | 296/307 | 186.60 | 184.20 | 0.930 | True |
| 300 | 720 | 0.1 | 324/340 | 324/340 | 244.80 | 244.80 | 0.932 | True |

Fitted log-log slope, N>=50, phase_locked_RQS: **0.698**.
Fitted log-log slope, N>=50, free_running_RQS: **0.684**.
Free-running exponent split: duration **0.535** + estimated shots **0.148**.

Free/phase-locked estimated T95 ratio: 0.958–1.020.

Evaluated RQS budgets below the former 6N eligibility floor: N=50, 100, 200, 300.

Calibration exhausted its budget-search rounds at N=20; see histories.

Calibration candidate searches reaching the round limit: N=20 (X fraction: 0.1, 0.3); N=200 (tau: 600); N=200 (X fraction: 0.1).
These are knob-search candidates; final held-out budget agreement is reported separately.

## Likelihood comparison

Full likelihood evaluation trials configured: 0.
When this is zero, no ML T95 or refinement exponent has been measured by this run.
The plotted 6N line is an eligibility lower bound at the RQS-selected durations, not ML performance.
It scales as N^1.5 only when those durations scale as sqrt(N); it is not a universal bound under duration retuning.
The optional four-trial paired check is saved separately in free_running_refinement_check.json.
That small check illustrates errors, skipped fits, and classical time; it cannot estimate T95 or a scaling exponent.

Completed conditions in this file: 19.
Results are checkpointed. Inspect convergence flags and raw histories in the accompanying JSON.
