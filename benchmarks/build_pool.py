"""Run from the toolkit root: PYTHONPATH=src python benchmarks/build_pool.py."""

import argparse
from time import perf_counter

import numpy as np
from scipy.fft import set_workers

import randomized_sensing as R


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[50, 200, 500])
    parser.add_argument("--durations-us", nargs="+", type=float, default=[40, 80])
    parser.add_argument("--fft-workers", type=int, default=4)
    args = parser.parse_args()
    print("N   duration_us   entries   build_s   condition   sampled_response_error", flush=True)
    with set_workers(args.fft_workers):
        for duration in args.durations_us:
            for n in args.sizes:
                f = 10e6+(np.arange(n)-(n-1)/2)*25e3
                experiment = R.Experiment(tuple(f), 28e3, clock_s=2.5e-9)
                before = perf_counter()
                pool = R.build_pool(experiment, duration_s=duration*1e-6, seed=4)
                elapsed = perf_counter()-before
                indices = np.linspace(0, len(pool.Z)-1, 8, dtype=int)
                checks = [R.audit(pool, int(i)) for i in indices]
                assert all(c["timing_valid"] and c["response_error"] <=
                           1e-9*max(1., np.max(abs(pool.Z[i]))) for i, c in zip(indices, checks))
                error = max(c["response_error"] for c in checks)
                print(f"{n:<3d} {duration:11g} {len(pool.Z):9d} {elapsed:9.3f} "
                      f"{pool.condition:11.3f} {error:24.3g}", flush=True)
                del pool


if __name__ == "__main__":
    main()
