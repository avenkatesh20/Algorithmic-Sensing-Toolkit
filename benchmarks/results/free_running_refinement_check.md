# Paired likelihood checks

4 new trials per condition across two fresh banks, alternating dense and single-tone fields.
These are illustrative paired errors and classical times, not T95 estimates or fitted scaling exponents.
Each method receives the same binary outcomes and actual trigger timestamps.

| Case | Norm | N | Shots / old floor | Fits applied | Old refusals | Median max error RQS / refined | Improved trials | Mean refinement seconds |
| :--- | ---: | ---: | :---: | :---: | ---: | :---: | ---: | ---: |
| noiseless | 0.65 | 20 | 1666 / 120 | 4/4 | 0 | 0.0772 / 0.0515 | 4 | 0.101 |
| t2_1ms | 0.65 | 20 | 3490 / 120 | 4/4 | 0 | 0.0739 / 0.0600 | 3 | 0.118 |
| noiseless | 0.65 | 50 | 2700 / 300 | 4/4 | 0 | 0.0763 / 0.0659 | 4 | 0.202 |
| t2_1ms | 0.65 | 50 | 3268 / 300 | 4/4 | 0 | 0.0848 / 0.0655 | 4 | 0.233 |
| noiseless | 0.65 | 100 | 2700 / 600 | 4/4 | 0 | 0.0823 / 0.0718 | 4 | 1.400 |
| t2_1ms | 0.65 | 100 | 3510 / 600 | 4/4 | 0 | 0.0831 / 0.0731 | 2 | 2.752 |
| noiseless | 0.65 | 200 | 2464 / 1200 | 4/4 | 0 | 0.0859 / 0.0810 | 3 | 16.734 |
| t2_1ms | 0.65 | 200 | 3955 / 1200 | 4/4 | 0 | 0.0815 / 0.0854 | 2 | 33.097 |
| noiseless | 0.2 | 20 | 202 / 120 | 4/4 | 0 | 0.0546 / 0.0547 | 1 | 0.032 |
| noiseless | 0.2 | 50 | 256 / 300 | 0/4 | 4 | 0.0770 / 0.0770 | 0 | 0.002 |
| noiseless | 0.2 | 100 | 312 / 600 | 0/4 | 4 | 0.0604 / 0.0604 | 0 | 0.004 |
| noiseless | 0.2 | 200 | 296 / 1200 | 0/4 | 4 | 0.0809 / 0.0809 | 0 | 0.007 |
| noiseless | 0.2 | 300 | 324 / 1800 | 0/4 | 4 | 0.0726 / 0.0726 | 0 | 0.014 |
| noiseless | 0.65 | 300 | 2502 / 1800 | 4/4 | 0 | 0.0768 / 0.0853 | 2 | 65.805 |
| t2_1ms | 0.65 | 300 | 6421 / 1800 | 4/4 | 0 | 0.0834 / 0.0760 | 3 | 64.889 |

A likelihood fit can improve or worsen the amplitude max error on an individual trial.
Below the shot floor the optional refinement returns the identical moment estimate; the former ML-only path refuses.
No full 400-trial ML success curve has been measured by this small check. Raw results include failures and skip reasons.
