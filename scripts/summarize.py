#!/usr/bin/env python3
"""Parse experiment CSVs and produce a comparison table.

Usage: python3 scripts/summarize.py [results-dir]
"""

import csv
import glob
import os
import statistics
import sys


def read_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows


def summarize(rows):
    wall = [r["wall_us"] for r in rows]
    imb = [r["imbalance_pct"] for r in rows]
    gf = [r["gflops"] for r in rows]

    # Steady state = second half (skip initial exploration / cache warmup)
    half = len(wall) // 2
    steady_wall = wall[half:]
    steady_gf = gf[half:]

    return {
        "median_us": statistics.median(wall),
        "steady_us": statistics.median(steady_wall),
        "p5_us": sorted(wall)[max(0, len(wall) // 20)],
        "p95_us": sorted(wall)[min(len(wall) - 1, len(wall) * 19 // 20)],
        "median_imb": statistics.median(imb),
        "max_imb": max(imb),
        "median_gflops": statistics.median(gf),
        "steady_gflops": statistics.median(steady_gf),
    }


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    csvs = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
    csvs = [c for c in csvs if "_looptimes" not in c]

    if not csvs:
        print(f"No CSV files found in {results_dir}/")
        return 1

    # Find baseline for overhead calculation
    baseline_us = None
    summaries = {}
    for path in csvs:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            rows = read_csv(path)
            s = summarize(rows)
            summaries[name] = s
            if name == "baseline_static":
                baseline_us = s["steady_us"]
        except Exception as e:
            print(f"  {name}: ERROR — {e}")

    hdr = (
        f"{'Configuration':<25} {'Median':>9} {'Steady':>9} "
        f"{'Ovhd%':>7} {'Imb.med%':>9} {'GF/s':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name in sorted(summaries):
        s = summaries[name]
        ovhd = ""
        if baseline_us and baseline_us > 0:
            ovhd = f"{(s['steady_us'] / baseline_us - 1) * 100:+.1f}%"
        print(
            f"{name:<25} {s['median_us']:>9.0f} {s['steady_us']:>9.0f} "
            f"{ovhd:>7} {s['median_imb']:>9.2f} {s['steady_gflops']:>7.2f}"
        )

    print()
    print("Median/Steady = median wall time (us) over all / second-half steps.")
    print("Ovhd% = steady-state overhead vs baseline_static.")
    print("Imb.med% = median load imbalance (max-min)/mean thread time.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
