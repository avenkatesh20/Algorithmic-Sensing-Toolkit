"""Small paired likelihood check; illustrative errors, never a T95 estimate.

Reads completed RQS scaling conditions and uses new banks, signals and noise.
The full 400-trial likelihood/T95 search remains in free_running_scaling.py.
"""

import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np

# Import the sibling benchmark, rather than any unrelated installed package.
from free_running_scaling import bank, signals, save
import randomized_sensing as R
from randomized_sensing._timestamped import fit as strict_fit, RefinementUnavailable


def summary(report, path):
    lines = ["# Paired likelihood checks", "",
             f"{report['trials']} new trials per condition across two fresh banks, alternating dense and single-tone fields.",
             "These are illustrative paired errors and classical times, not T95 estimates or fitted scaling exponents.",
             "Each method receives the same binary outcomes and actual trigger timestamps.", "",
             "| Case | Norm | N | Shots / old floor | Fits applied | Old refusals | Median max error RQS / refined | Improved trials | Mean refinement seconds |",
             "| :--- | ---: | ---: | :---: | :---: | ---: | :---: | ---: | ---: |"]
    for row in report["rows"]:
        results = [t["results"] for t in row["trials"]]
        fitted = [r for r in results if r["TIMESTAMPED-ML"]["ml_applied"]]
        valid = [r for r in results if r["TIMESTAMPED-ML"]["max_error"] is not None]
        old_refusals = sum(bool(r["former_ML_only"].get("refused")) for r in results)
        improved = sum(r["TIMESTAMPED-ML"]["max_error"] < r["RQS"]["max_error"] for r in valid)
        base = np.median([r["RQS"]["max_error"] for r in results])
        refined = np.median([r["TIMESTAMPED-ML"]["max_error"] for r in valid]) if valid else float("nan")
        seconds = np.mean([r["TIMESTAMPED-ML"]["classical_s"] for r in results])
        lines.append(f"| {row['case']} | {row['norm']} | {row['N']} | {row['n']} / {row['former_minimum_shots']} | "
                     f"{len(fitted)}/{len(results)} | {old_refusals} | {base:.4f} / {refined:.4f} | {improved} | {seconds:.3f} |")
    lines += ["", "A likelihood fit can improve or worsen the amplitude max error on an individual trial.",
              "Below the shot floor the optional refinement returns the identical moment estimate; the former ML-only path refuses.",
              "No full 400-trial ML success curve has been measured by this small check. Raw results include failures and skip reasons."]
    path.write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("benchmarks/results/free_running_scaling_RQS.json"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/free_running_refinement_check.json"))
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100, 200, 300, 500])
    parser.add_argument("--trials", type=int, default=4)
    args = parser.parse_args()
    if args.trials < 4 or args.trials % 4:
        parser.error("Use a multiple of four trials, split over two fresh banks and signal types.")
    source = json.loads(args.source.read_text())
    report = json.loads(args.output.read_text()) if args.output.exists() else dict(
        trials=args.trials, kind="illustrative paired errors; not T95", rows=[])
    if report["trials"] != args.trials:
        parser.error("Use another output path for a different trial count.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for condition in source["rows"]:
        if condition["N"] not in args.sizes or "evaluation" not in condition:
            continue
        key = (condition["N"], condition["case"], condition["norm"])
        if any((r["N"], r["case"], r["norm"]) == key for r in report["rows"]):
            continue
        n, case, norm = key
        settings = condition["evaluation"]["free_running_RQS"]
        shots, tau, fraction = settings["n"], settings["tau_us"], settings["x_fraction"]
        row = dict(N=n, case=case, norm=norm, n=shots, tau_us=tau, x_fraction=fraction,
                   active_time_s=shots*tau*1e-6, former_minimum_shots=6*n, trials=[])
        print("START paired check", key, "shots", shots, flush=True)
        for b in range(2):
            p = bank(n, tau, None if case == "noiseless" else 1e-3, 304+100*b)
            fields = signals(n, norm, args.trials//2, 7000+n+100*b)
            for i, a in enumerate(fields):
                plan = R.plan_acquisition(p, n_shots=shots, x_fraction=fraction,
                                          field_mode="free_running", shot_period_s=p.duration_s+1.2345e-6,
                                          seed=8000+100*b+i)
                times = np.array([r["start_time_s"] for r in plan.rows])
                times += np.random.default_rng(9000+100*b+i).uniform(0, 10e-9, len(times))
                measured = R.simulate(plan, a, seed=10000+100*b+i, start_times_s=times)
                results = {}
                for method in ("RQS", "TIMESTAMPED-ML"):
                    start = perf_counter()
                    try:
                        estimate = R.decode(plan, measured, method=method)
                        results[method] = dict(max_error=float(np.max(abs(np.asarray(estimate.amplitudes)-a))),
                                               ml_applied=estimate.ml_applied,
                                               skipped=estimate.refinement_skipped_reason,
                                               classical_s=perf_counter()-start)
                    except RuntimeError as exc:
                        if method == "RQS":
                            raise
                        results[method] = dict(max_error=None, ml_applied=False, failed=str(exc),
                                               classical_s=perf_counter()-start)
                start = perf_counter()
                try:
                    legacy, _, _, _ = strict_fit(p, plan.rows, measured.positive_counts, times)
                    results["former_ML_only"] = dict(max_error=float(np.max(abs(legacy-a))),
                                                       classical_s=perf_counter()-start, refused=None)
                except RefinementUnavailable as exc:
                    results["former_ML_only"] = dict(max_error=None, classical_s=perf_counter()-start,
                                                       refused=str(exc))
                except RuntimeError as exc:
                    results["former_ML_only"] = dict(max_error=None, classical_s=perf_counter()-start,
                                                       failed=str(exc))
                row["trials"].append(dict(bank=b, signal="dense" if i % 2 == 0 else "one_tone", results=results))
                print(" ", key, b, i, results, flush=True)
            del p
            gc.collect()
        report["rows"].append(row)
        save(args.output, report)
        summary(report, args.output.with_suffix(".md"))
        print("DONE paired check", key, flush=True)
    summary(report, args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
