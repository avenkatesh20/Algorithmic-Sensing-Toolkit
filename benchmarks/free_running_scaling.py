"""Calibrate on 128 signals; evaluate on >=400 held-out signals and fresh banks.

Run from the toolkit root with PYTHONPATH=src. Results checkpoint after each
completed condition; reruns resume those conditions. Full likelihood checks
can be expensive: --ml-trials 0 runs only the two moment estimators first.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.fft import set_workers

import randomized_sensing as R
from randomized_sensing import _grid as design_grid
from randomized_sensing._timestamped import fit as strict_fit


PAPER = {"noiseless": {10: 89.9, 20: 145.7, 50: 256.4, 100: 335.1, 200: 538.6, 300: 658., 500: 973.4},
         "t2_1ms": {10: 98.9, 20: 182.6, 50: 265.9, 100: 439., 200: 838.7, 300: 1073., 500: 1691.9}}


@dataclass(frozen=True)
class PackedSequence:
    """Benchmark-only storage: preserve every float64 time, materialize on access."""
    times: np.ndarray
    initial_sign: int

    @property
    def switch_times_s(self):
        return tuple(self.times.tolist())


def pack_sequences(pool):
    pool.sequences = tuple(PackedSequence(np.asarray(q.switch_times_s, float), q.initial_sign)
                           for q in pool.sequences)


def signals(n, norm, count, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(count, 2*n))
    a *= norm/np.linalg.norm(a, axis=1)[:, None]
    for i in range(1, count, 2):
        j, angle = rng.integers(n), rng.uniform(0, 2*np.pi)
        a[i] = 0.
        a[i, j], a[i, n+j] = norm*np.cos(angle), norm*np.sin(angle)
    return a


def bank(n, tau, t2, seed):
    f = 10e6+(np.arange(n)-(n-1)/2)*25e3
    e = R.Experiment(tuple(f), 28e3, t2_s=t2, clock_s=2.5e-9)
    before = perf_counter()
    previous_block = design_grid._BLOCK_ELEMENTS
    try:
        # Larger bounded batches keep four FFT workers busy for long shots.
        design_grid._BLOCK_ELEMENTS = 8_000_000
        with set_workers(4):
            p = R.build_pool(e, duration_s=tau*1e-6, seed=seed)
    finally:
        design_grid._BLOCK_ELEMENTS = previous_block
    # Independent physical verification, outside quantum active time.
    for i in np.linspace(0, len(p.Z)-1, 8, dtype=int):
        check = R.audit(p, int(i))
        assert check["timing_valid"] and check["response_error"] < 1e-9*max(1., abs(p.Z[i]).max())
    pack_sequences(p)
    p._map_workers = 2  # identical column blocks, bounded two-worker map preparation
    R.population_estimate(p, np.zeros(2*n))  # warm the identical shared Bessel map
    print(f"  bank N={n} tau={tau}us seed={seed}: {perf_counter()-before:.1f}s incl. map", flush=True)
    return p


def batch(pool, amplitudes, shots, fraction, mode, method, seed, workers):
    def trial(i):
        options = dict(field_mode="free_running", shot_period_s=pool.duration_s+1.2345e-6) if mode == "free_running" else {}
        plan = R.plan_acquisition(pool, n_shots=shots, x_fraction=fraction, seed=seed+3*i, **options)
        starts = None
        if mode == "free_running":
            starts = np.array([r["start_time_s"] for r in plan.rows])
            starts += np.random.default_rng(seed+3*i+1).uniform(0, 10e-9, len(starts))
        measured = R.simulate(plan, amplitudes[i], seed=seed+3*i+2, start_times_s=starts)
        try:
            if method == "ML-only":
                # Exactly the former eligibility checks and legacy three starts.
                a, _, _, _ = strict_fit(pool, plan.rows, measured.positive_counts,
                                        np.array([measured.start_times_s[r["row_id"]] for r in plan.rows]))
                applied = True
            else:
                result = R.decode(plan, measured, method=method)
                a, applied = np.asarray(result.amplitudes), result.ml_applied
            return float(np.max(abs(a-amplitudes[i]))), bool(applied), None
        except (ValueError, RuntimeError) as exc:
            return float("inf"), False, str(exc)
    # Warm the bank-frame solve once before concurrent, read-only trials.
    if mode == "free_running" and method != "ML-only":
        warm = R.plan_acquisition(pool, n_shots=2, field_mode=mode, shot_period_s=pool.duration_s+1.2345e-6)
        R.decode(warm, R.simulate(warm, amplitudes[0]))
    with ThreadPoolExecutor(max_workers=workers if method == "RQS" else 1) as executor:
        rows = list(executor.map(trial, range(len(amplitudes))))
    return np.array([r[0] for r in rows]), sum(r[1] for r in rows), [r[2] for r in rows if r[2]]


def search(pool, amplitudes, initial, fraction, mode, method, args, seed, floor=2):
    budget, history = max(floor, round(initial)), []
    for iteration in range(args.max_rounds):
        errors, applied, failures = batch(pool, amplitudes, budget, fraction, mode, method,
                                           seed+100_000*iteration, args.workers)
        q = float(np.quantile(errors, .95))
        history.append(dict(n=budget, q95=q, success=float(np.mean(errors <= args.epsilon)),
                            ml_applied=applied, failures=failures[:3]))
        if not np.isfinite(q):
            return dict(n=budget, t95_s=None, converged=False, history=history)
        target = max(floor, int(np.ceil(budget*(q/args.epsilon)**2)))
        if abs(target-budget) <= .05*budget:
            return dict(n=budget, t95_s=budget*pool.duration_s, converged=True, history=history)
        budget = target
    # Keep extrapolated and actually evaluated budgets explicitly separate.
    return dict(n=history[-1]["n"], t95_s=history[-1]["n"]*pool.duration_s,
                converged=False, next_n=budget, history=history)


def calibrate(n, norm, t2, args):
    a = signals(n, norm, args.calibration_trials, 1000+n)
    tau0 = 1.5*np.sqrt(n)/(2*np.pi*28e3*norm)
    tau = tau0 if t2 is None else tau0/(np.sqrt(1+(tau0/t2)**2)+tau0/t2)
    start = max(40, 40*round(tau*1e6/40))
    initial = 2700*(norm/.65)**2
    cache = Path("outputs/free_running_calibration")/f"{args.output.stem}-{n}-{norm}-{t2}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(cache.read_text()) if cache.exists() else {}
    settings = dict(calibration_trials=args.calibration_trials, epsilon=args.epsilon,
                    max_rounds=args.max_rounds, max_neighbors=args.max_neighbors)
    if state.get("settings") != settings:
        state = dict(settings=settings)
    records = {int(k): v for k, v in state.get("duration_candidates", {}).items()}
    retained = {}

    def candidate(duration):
        if duration not in records:
            p = bank(n, duration, t2, 4)
            estimate = search(p, a, initial, .2, "free_running", "RQS", args, 10_000)
            records[duration] = estimate
            state["duration_candidates"] = records
            save(cache, state)
            print(f"  calibration tau={duration}us: n={estimate['n']} converged={estimate['converged']}", flush=True)
            value = estimate["t95_s"] or float("inf")
            if value < retained.get("cost", float("inf")):
                retained.clear()
                retained.update(pool=p, tau=duration, cost=value)
            else:
                del p
            gc.collect()
        return records[duration]["t95_s"] or float("inf")

    current = start
    candidate(current)
    for _ in range(args.max_neighbors):
        neighbours = [t for t in (current-40, current+40) if t >= 40]
        for t in neighbours:
            candidate(t)
        best = min([current]+neighbours, key=candidate)
        if best == current:
            break
        current = best
    p = retained["pool"] if retained.get("tau") == current else bank(n, current, t2, 4)
    choices = {float(k): v for k, v in state.get("fraction_candidates", {}).items()}
    for fraction in (.1, .2, .3):
        if fraction not in choices:
            choices[fraction] = search(p, a, records[current]["n"], fraction, "free_running", "RQS", args, 30_000)
            state["fraction_candidates"] = choices
            save(cache, state)
    fraction = min(choices, key=lambda f: choices[f]["t95_s"] or float("inf"))
    chosen = choices[fraction]
    locked = state.get("locked")
    if locked is None:
        locked = search(p, a, chosen["n"], fraction, "phase_locked", "RQS", args, 40_000)
        state["locked"] = locked
        save(cache, state)
    ml = None
    if args.ml_trials:
        ml = search(p, a, max(6*n, chosen["n"]), fraction, "free_running", "ML-only", args, 50_000, floor=6*n)
    result = dict(tau_us=current, x_fraction=fraction, free=chosen, locked=locked, ml_only=ml,
                  duration_candidates=records, fraction_candidates=choices)
    del p
    retained.clear()
    gc.collect()
    return result


def evaluate(n, norm, t2, calibration, args):
    methods = {"phase_locked_RQS": ("phase_locked", "RQS", calibration["locked"]["n"]),
               "free_running_RQS": ("free_running", "RQS", calibration["free"]["n"])}
    if args.ml_trials and calibration["ml_only"]:
        ml = calibration["ml_only"]["n"]
        nr = calibration["free"]["n"]
        methods["old_ML_only"] = ("free_running", "ML-only", ml)
        methods["free_running_refined"] = ("free_running", "TIMESTAMPED-ML", min(nr, ml))
    counts = {name: args.ml_trials if "ML" in name or "refined" in name else args.evaluation_trials for name in methods}
    # Reuse the same fresh banks during the held-out shot-budget search, but use
    # new outcome noise at each budget. No duration or X-fraction is retuned.
    pools = [bank(n, calibration["tau_us"], t2, 104+100*b) for b in range(args.banks)]
    results = {}
    for name, (mode, method, budget) in methods.items():
        history = []
        for iteration in range(args.max_rounds):
            collected, used, failures = [], 0, []
            for b, p in enumerate(pools):
                count = counts[name]//args.banks
                a = signals(n, norm, count, 100_000+n+10_000*b)
                errors, applied, bad = batch(p, a, budget, calibration["x_fraction"], mode, method,
                                               1_000_000+10_000*b+100_000*iteration, args.workers)
                collected.extend(errors.tolist())
                used += applied
                failures.extend(bad[:3])
            errors = np.asarray(collected)
            q = float(np.quantile(errors, .95))
            predicted = int(np.ceil(budget*(q/args.epsilon)**2)) if np.isfinite(q) else None
            # Only the former ML-only protocol imposes a shot floor. The new
            # optional refinement must remain free to fall back below it.
            floor = 6*n if name == "old_ML_only" else 2
            if predicted is not None:
                predicted = max(floor, predicted)
            history.append(dict(n=budget, q95=q, success=float(np.mean(errors <= args.epsilon))))
            print(f"  held-out {name}: n={budget}, q95={q:.4f}, next={predicted}, trials={len(errors)}", flush=True)
            if predicted is None or abs(predicted-budget) <= .05*budget:
                break
            if iteration+1 < args.max_rounds:
                budget = predicted
        results[name] = dict(n=budget, tau_us=calibration["tau_us"], x_fraction=calibration["x_fraction"],
                             trials=len(errors), q95=q, median=float(np.median(errors)),
                             success=float(np.mean(errors <= args.epsilon)), active_time_s=budget*p_duration(calibration),
                             estimated_t95_s=None if predicted is None else predicted*p_duration(calibration),
                             estimated_n=predicted, budget_agreement=None if predicted is None else abs(predicted-budget)/budget,
                             floor_bound=name == "old_ML_only" and budget == 6*n,
                             converged=predicted is not None and abs(predicted-budget) <= .05*budget,
                             ml_applied=used, failures=failures, history=history)
    del pools
    gc.collect()
    return results


def p_duration(calibration):
    return calibration["tau_us"]*1e-6


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2)+"\n")
    temporary.replace(path)


def plots(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    scenarios = [("noiseless", .65), ("t2_1ms", .65), ("noiseless", .2)]
    labels = {"phase_locked_RQS": "Phase-locked RQS", "free_running_RQS": "Free-running RQS",
              "old_ML_only": "Former ML only", "free_running_refined": "RQS + likelihood"}
    slopes = {}
    for ax, (case, norm) in zip(axes, scenarios):
        points = [r for r in report["rows"] if r["case"] == case and r["norm"] == norm]
        names = sorted({name for r in points for name in r.get("evaluation", {})})
        for name in names:
            rows = sorted([r for r in points if name in r.get("evaluation", {})], key=lambda r: r["N"])
            x = np.array([r["N"] for r in rows])
            y = np.array([r["evaluation"][name]["estimated_t95_s"] or np.nan for r in rows])*1e3
            use = (x >= 50) & np.isfinite(y)
            slope = float(np.polyfit(np.log(x[use]), np.log(y[use]), 1)[0]) if use.sum() >= 2 else None
            slopes[f"{case}_{norm}_{name}"] = slope
            label = labels.get(name, name)
            if slope is not None:
                label += f": slope {slope:.2f}"
            ax.loglog(x, y, "o-", label=label)
        completed = sorted([r for r in points if "evaluation" in r], key=lambda r: r["N"])
        if completed:
            xx = np.array([r["N"] for r in completed])
            bound = np.array([6*r["N"]*r["calibration"]["tau_us"]*1e-3 for r in completed])
            ax.loglog(xx, bound, "k--", lw=1, label="6N floor at these durations (bound)")
            if norm == .65:
                reference = [(int(v), PAPER[case][int(v)]) for v in xx if int(v) in PAPER[case]]
                if reference:
                    rx, ry = zip(*reference)
                    ax.loglog(rx, ry, ":", color=".6", label="Note: phase-locked reference")
        title = "No dephasing" if case == "noiseless" else r"$T_2=1$ ms"
        ax.set(title=f"{title}, norm={norm:g}", xlabel="Number of tones", ylabel="Estimated T95 (ms)")
        if names:
            ax.legend(fontsize=7)
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)
    report["slopes"] = slopes
    summary(report, path.with_suffix(".md"))


def summary(report, path):
    def milliseconds(value):
        return "failed" if value is None else f"{1000*value:.2f}"

    lines = ["# Free-running RQS scaling checks", "",
             "Binary projection noise is included; 'noiseless' means no dephasing.",
             "Precision is max absolute error <=0.1 over all 2N amplitudes, at fixed signal L2 norm.",
             "This is not an accuracy guarantee for arbitrary later linear observables.", "",
             f"Calibration signals: {report['config']['calibration_trials']}; held-out signals: "
             f"{report['config']['evaluation_trials']} over {report['config']['banks']} fresh banks.",
             "Durations and X fractions are selected before held-out evaluation; only the shot budget is then searched.",
             "T95 is the error-based estimate at the last evaluated budget. Success rates refer to the evaluated n.", ""]
    for case, norm in (("noiseless", .65), ("t2_1ms", .65), ("noiseless", .2)):
        rows = sorted([r for r in report["rows"] if r["case"] == case and r["norm"] == norm
                       and "evaluation" in r], key=lambda r: r["N"])
        if not rows:
            continue
        lines += [f"## {case}, norm {norm}", "",
                  "| N | tau (us) | X fraction | n locked evaluated/estimated | n free evaluated/estimated | T95 locked (ms) | T95 free (ms) | Free success | Budget agreement |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
        mismatches = []
        for row in rows:
            a, b = (row["evaluation"][k] for k in ("phase_locked_RQS", "free_running_RQS"))
            lines.append(f"| {row['N']} | {b['tau_us']} | {b['x_fraction']} | {a['n']}/{a['estimated_n']} | {b['n']}/{b['estimated_n']} | "
                         f"{milliseconds(a['estimated_t95_s'])} | {milliseconds(b['estimated_t95_s'])} | "
                         f"{b['success']:.3f} | {a['converged'] and b['converged']} |")
            reference = row.get("paper_reference_t95_s")
            if reference and a["estimated_t95_s"] is not None and abs(a["estimated_t95_s"]/reference-1) > .15:
                mismatches.append(f"N={row['N']}: {100*(a['estimated_t95_s']/reference-1):+.1f}%")
        lines.append("")
        for method in ("phase_locked_RQS", "free_running_RQS"):
            slope = report["slopes"].get(f"{case}_{norm}_{method}")
            if slope is not None:
                lines.append(f"Fitted log-log slope, N>=50, {method}: **{slope:.3f}**.")
        fitted = [r for r in rows if r["N"] >= 50]
        if len(fitted) >= 2:
            x = np.log([r["N"] for r in fitted])
            duration = np.polyfit(x, np.log([r["calibration"]["tau_us"] for r in fitted]), 1)[0]
            counts = np.polyfit(x, np.log([r["evaluation"]["free_running_RQS"]["estimated_n"] for r in fitted]), 1)[0]
            lines += [f"Free-running exponent split: duration **{duration:.3f}** + estimated shots **{counts:.3f}**."]
        if mismatches:
            lines += ["", "Phase-locked deviations outside the note's 15% reference tolerance: "+"; ".join(mismatches)+"."]
        ratios = [r["evaluation"]["free_running_RQS"]["estimated_t95_s"] /
                  r["evaluation"]["phase_locked_RQS"]["estimated_t95_s"] for r in rows]
        lines += ["", f"Free/phase-locked estimated T95 ratio: {min(ratios):.3f}–{max(ratios):.3f}."]
        excluded = [str(r["N"]) for r in rows if r["evaluation"]["free_running_RQS"]["n"] < 6*r["N"]]
        lines += ["", "Evaluated RQS budgets below the former 6N eligibility floor: "+
                  ("N="+", ".join(excluded) if excluded else "none at these sizes and settings")+"."]
        calibration_failures = [str(r["N"]) for r in rows if not r["calibration"]["free"]["converged"]]
        if calibration_failures:
            lines += ["", "Calibration exhausted its budget-search rounds at N="+", ".join(calibration_failures)+"; see histories."]
        candidates = []
        for row in rows:
            for group, label in (("duration_candidates", "tau"), ("fraction_candidates", "X fraction")):
                failed = [str(k) for k, v in row["calibration"][group].items() if not v["converged"]]
                if failed:
                    candidates.append(f"N={row['N']} ({label}: {', '.join(failed)})")
        if candidates:
            lines += ["", "Calibration candidate searches reaching the round limit: "+"; ".join(candidates)+".",
                      "These are knob-search candidates; final held-out budget agreement is reported separately."]
        lines.append("")
    lines += ["## Likelihood comparison", "",
              f"Full likelihood evaluation trials configured: {report['config']['ml_trials']}.",
              "When this is zero, no ML T95 or refinement exponent has been measured by this run.",
              "The plotted 6N line is an eligibility lower bound at the RQS-selected durations, not ML performance.",
              "It scales as N^1.5 only when those durations scale as sqrt(N); it is not a universal bound under duration retuning.",
              "The optional four-trial paired check is saved separately in free_running_refinement_check.json.",
              "That small check illustrates errors, skipped fits, and classical time; it cannot estimate T95 or a scaling exponent.", "",
              f"Completed conditions in this file: {sum('evaluation' in r for r in report['rows'])}.",
              "Results are checkpointed. Inspect convergence flags and raw histories in the accompanying JSON."]
    path.write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="*", type=int, default=[10, 20, 50, 100, 200, 300, 500])
    parser.add_argument("--stress-sizes", nargs="*", type=int, default=[20, 50, 100, 200, 300])
    parser.add_argument("--cases", nargs="+", choices=["noiseless", "t2_1ms"], default=["noiseless", "t2_1ms"])
    parser.add_argument("--calibration-trials", type=int, default=128)
    parser.add_argument("--evaluation-trials", type=int, default=400)
    parser.add_argument("--ml-trials", type=int, default=400)
    parser.add_argument("--banks", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=.1)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/free_running_scaling.json"))
    args = parser.parse_args()
    if (min(args.calibration_trials, args.evaluation_trials, args.banks, args.workers, args.max_rounds) < 1
            or args.ml_trials < 0 or args.max_neighbors < 0 or args.epsilon <= 0):
        parser.error("Trial counts, banks, workers, rounds and epsilon must be positive; ML trials/neighbours may be zero.")
    if args.evaluation_trials % args.banks or args.ml_trials % args.banks:
        parser.error("Trial counts must split equally across fresh banks.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("output", "sizes", "stress_sizes", "cases")}
    report = json.loads(args.output.read_text()) if args.output.exists() else dict(config=config, rows=[])
    if report["config"] != config:
        parser.error("Use a separate output for different trial/search configurations.")
    cases = [(case, .65, n) for n in args.sizes for case in args.cases]
    if "noiseless" in args.cases:
        cases += [("noiseless", .2, n) for n in args.stress_sizes]
    for case, norm, n in cases:
        existing = next((r for r in report["rows"] if r["case"] == case and r["norm"] == norm and r["N"] == n), None)
        if existing and "evaluation" in existing:
            continue
        print(f"START {case} norm={norm} N={n}", flush=True)
        t2 = None if case == "noiseless" else 1e-3
        row = existing if existing else dict(case=case, norm=norm, N=n)
        if "calibration" not in row:
            row["calibration"] = calibrate(n, norm, t2, args)
            report["rows"].append(row)
            save(args.output, report)
        row["evaluation"] = evaluate(n, norm, t2, row["calibration"], args)
        if norm == .65:
            row["paper_reference_t95_s"] = PAPER[case].get(n, 0)/1000
        save(args.output, report)
        plots(report, args.output)
        save(args.output, report)
        print(f"DONE {case} norm={norm} N={n}", flush=True)
    plots(report, args.output)
    save(args.output, report)


if __name__ == "__main__":
    main()
