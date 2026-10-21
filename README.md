# Algorithmic sensing toolkit

Design single-qubit pulse sequences for known signal frequencies, export an
acquisition plan, and decode experimental counts with classical postprocessing.

## Start

Requires Python 3.10+. From this folder:

```sh
python -m pip install -e ".[notebook,test]"
jupyter lab
```

| Notebook | When observables are chosen | Workflow |
|---|---|---|
| [matched_experiment.ipynb](notebooks/matched_experiment.ipynb) | Before sensing | Compile shared query coordinates and plan shots for a requested precision |
| [randomized_experiment.ipynb](notebooks/randomized_experiment.ipynb) | After sensing | Collect random X/Y measurements, then evaluate different observables from the same counts |

Run a notebook from top to bottom. In the randomized notebook, edit the weights
in **step 6** and rerun that section onward to try different observables.

## Inputs and conventions

List all possible input frequencies, including unwanted tones. Frequencies are
in Hz and times in seconds. The field and requested linear observables are

```text
b(t) = sum_j [a_j*cos(2*pi*f_j*t) + b_j*sin(2*pi*f_j*t)]
F = cosine_weights @ a + sine_weights @ b
H(t) = (2*pi*coupling_hz) * b(t) * Z/2
```

Weights have shape `(N,)` for one observable or `(M,N)` for several. The bound
`amplitude_bound` applies to the joint Euclidean norm of all cosine and sine
amplitudes. The randomized workflow currently requires positive frequencies.

Both workflows accept `t2_s` (`None` removes dephasing), `min_gap_s`,
`edge_guard_s`, an optional hardware `clock_s`, calibrated `visibility` and
`readout_bias`, and per-shot `overhead_s`. Active interrogation time and time
including the specified overhead are reported separately.

## Matched sensing

Supply observable weights before compilation. Multiple observables share an
SVD query basis, so dependent queries reuse measured coordinates.

```python
import matched_sensing as M

experiment = M.Experiment(
    frequencies_hz=(.8e6, 1.05e6, 1.31e6, 1.72e6),
    cosine_weights=((.5, -.3, 0., .4), (.1, .2, -.4, .3)),
    sine_weights=((.2, .4, .1, -.2), (-.3, .1, .2, .25)),
    coupling_hz=100e3, t2_s=200e-6,
    min_gap_s=20e-9, edge_guard_s=20e-9,
)
bank = M.compile_queries(experiment, max_gain=4,
                         max_duration_s=40e-6, max_pulses=128)
plan = M.plan_queries(bank, precision=.08, confidence=.95)
M.export_bundle(plan, "outputs/matched_experiment")
```

After acquisition, load the saved plan and your completed counts file:

```python
plan = M.load_bundle("outputs/matched_experiment")
counts = M.read_counts("measured_counts.csv", plan)
result = M.decode_queries(plan, counts)
print(result.values, result.intervals)
```

`precision` is an absolute error per query; `confidence` applies simultaneously
to all queries under the stated model. Numerical rank and synthesis errors
enter the error budget. Compilation searches a finite set of candidates below
the supplied limits; it does not certify the globally fastest sequence.
See [the matched model](docs/model.md).

## Randomized sensing

Set only frequencies and hardware parameters before acquisition:

```python
import randomized_sensing as R

experiment = R.Experiment(
    frequencies_hz=(.8e6, 1.05e6, 1.31e6, 1.72e6),
    coupling_hz=100e3, t2_s=200e-6,
    min_gap_s=20e-9, edge_guard_s=20e-9, clock_s=2.5e-9,
)
pool = R.build_pool(experiment, duration_s=10e-6,
                    max_pulses=None, seed=4)
plan = R.plan_acquisition(pool, n_shots=20_000, x_fraction=.2, seed=5)
R.export_bundle(plan, "outputs/randomized_experiment")
```

After acquisition, choose observables during decoding. Changing these weights
reuses the same saved plan and counts:

```python
plan = R.load_bundle("outputs/randomized_experiment")
counts = R.read_counts("measured_counts.csv", plan)
result = R.decode(plan, counts, method="RQS-ML",
                  cosine_weights=(.5, -.3, 0., .4),
                  sine_weights=(.2, .4, .1, -.2))
print(result.values)
```

`R.decode(plan, counts)` returns amplitudes without selecting observables.
`R.evaluate_observables(result.amplitudes, cosine_weights=..., sine_weights=...)`
evaluates additional queries without repeating amplitude reconstruction.

The pulse pool uses repaired random multisines and their signed responses.
`max_pulses=None` leaves the pulse count limited by duration and timing constraints;
a numerical cap adds a tighter limit. Without a hardware clock, `design_clock_s`
sets the synthesis grid. Every interval conservatively obeys the larger of
`min_gap_s` and `edge_guard_s`, rounded up to that grid.

In the notebook, select `FIELD_MODE="free_running"` to log every shot's actual
start time instead of reproducing the source phases. For example:

```python
plan = R.plan_acquisition(pool, n_shots=20_000, field_mode="free_running",
                          shot_period_s=20e-6, seed=5)
```

The export has one row per shot. Fill both `plus` (0 or 1) and actual
`start_time_s` in the counts CSV, using elapsed seconds from a common stable
clock origin. The decoder uses measured times even when they differ from the
planned schedule. `R.decode(plan, counts)` automatically selects timestamped
likelihood reconstruction for this mode. Relative pulse timings stay unchanged.

For phase-locked acquisitions, RQS uses approximate moment inversion with a per-tone correction; RQS-ML adds
a local likelihood fit. The notebook includes simulation, population-bias checks,
and optional repeated trials for the selected observables. Its fixed shot budget
does not certify a precision or confidence interval. See [the randomized model](docs/randomized.md).

## Hardware files

| File | Purpose |
|---|---|
| `pulses.csv`, `intervals.csv` | Pulse times and modulation intervals |
| `acquisition.csv` | Sequence, physical X/Y basis, and repetitions |
| `counts_template.csv` | Copy and fill with raw physical +1 outcome counts; actual start times for freely running acquisitions |
| `bundle.json`, `manifest.json`, `RUN.txt` | Settings, integrity hashes, and execution conventions |
| `query_space.csv`, `query_coefficients.csv` | Shared basis and query coefficients for matched multi-query sensing |
| `compilation.json` | Matched compilation diagnostics |
| `pool.npz`, `sequences.csv` | Randomized response pool and acquired-sequence summaries |

Read `RUN.txt` before running hardware. Save measured counts separately from the
blank template, preserving plan IDs and shot totals. The randomized decoder
handles pulse-frame and response signs; supply raw counts without correcting them.

Pulses are ideal instantaneous rotations. Frequencies and Fourier amplitudes
must remain fixed. Matched and phase-locked randomized sensing require repeatable
starting tone phases; freely running randomized sensing uses a stable clock and
actual per-shot timestamps. Finite pulse widths, unrecorded timing jitter, calibration
uncertainty, and phase-reference waiting times are not modeled.

## Project layout

- `src/matched_sensing/`: matched compilation, query sharing, and readout.
- `src/randomized_sensing/`: random pulse pools, acquisition, and later query evaluation.
- `notebooks/`: editable experimental workflows.
- `docs/`: physical models and conventions.
- `tests/`: numerical and export/decoding checks; run `python -m pytest`.

Generated `outputs/`, caches, `build/`, and `dist/` are excluded by `.gitignore`.
The notebooks regenerate example outputs locally. License selection is pending.
