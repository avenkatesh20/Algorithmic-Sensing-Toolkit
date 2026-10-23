# Pool-build validation

Measured locally on 2026-10-23: macOS 14, Apple Silicon, Python 3.11,
NumPy 1.26.4 and SciPy 1.14.1. Four FFT workers, one BLAS thread, seed 4.
Frequencies are centered at 10 MHz with 25 kHz spacing; coupling is 28 kHz,
the hardware clock is 2.5 ns, minimum intervals/edge guards are 20 ns, and
there is no additional pulse-count cap. Pool sizes use the unchanged default.

| Tones | Duration (us) | Signed entries | Build (s) | Covariance condition |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 40 | 8192 | 1.87 | 1.83 |
| 200 | 40 | 12800 | 2.77 | 2.80 |
| 500 | 40 | 32000 | 9.08 | 3.83 |
| 50 | 80 | 8192 | 1.90 | 1.83 |
| 200 | 80 | 12800 | 3.32 | 2.80 |
| 500 | 80 | 32000 | 11.47 | 3.09 |

The N=200, 40 us result is above the note's approximate two-second target;
the N=500, 80 us build is well below one minute. Timings vary with thread
settings and machine load. The default respects SciPy's worker configuration;
the notebook demonstrates a local four-worker context.

Each benchmark independently audits eight entries, including negative responses,
after the timed build. All sampled timings passed; the largest absolute
response discrepancy was `1.71e-13`. These times exclude export and loading,
which still independently reintegrate **every base sequence** with the original
closed-form verifier. Large exports can take longer than pool building.
The temporary table/FFT memory is bounded, but the saved pool and complete pulse
lists still scale with the number and length of waveforms.

Reproduce from the toolkit root, using Python >=3.10:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/build_pool.py
```

The regression suite checks FFT samples against dense multisines, phase offsets,
the original random-draw order and pulse-cap rejection behavior, exact signed
responses at N=4/20/100 and multiple durations, off-grid chunked integration,
timing/edge guards, all-entry export audits, conditioning errors/overrides,
legacy-bundle identity, and the free-running commensurability warning.
The randomized notebook is also executed in its default configuration and
checked with count CSVs, average CSVs, pasted means, and a smaller free-running
configuration. Estimator math and physical readout signs are unchanged.
