#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


SWEEP_SPECS = (
    ("repeats_sweep", "--repeats"),
    ("decim_sweep", "--decim"),
    ("max_samples_sweep", "--max-samples"),
    ("threads_sweep", "threads"),
)


def parse_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run_cmd(cmd, env=None):
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return proc.stdout


def extract_metrics(text: str):
    patterns = {
        "samples_in": r"Samples in:\s*([0-9]+)",
        "samples_out": r"Samples out per repeat:\s*([0-9]+)",
        "time_s": r"Main compute time \[s\]:\s*([-+eE0-9.]+)",
        "gflops": r"Approx FIR throughput \[GFLOP/s\]:\s*([-+eE0-9.]+)",
        "energy_sum": r"Energy sum \(all repeats\):\s*([-+eE0-9.]+)",
        "hits": r"Block-hit count \(all repeats\):\s*([0-9]+)",
        "checksum": r"Checksum:\s*([-+eE0-9.]+)",
    }

    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if not m:
            raise ValueError(f"Could not parse metric '{key}' from program output")
        val = m.group(1)
        if key in ("samples_in", "samples_out", "hits"):
            out[key] = int(val)
        else:
            out[key] = float(val)
    return out


def run_case(args, repeats, decim, max_samples, threads):
    cmd = [str(args.exe)]

    if args.input:
        cmd += ["--input", args.input, "--format", args.format]
        if max_samples > 0:
            cmd += ["--max-samples", str(max_samples)]
    else:
        synthetic_samples = max_samples if max_samples > 0 else args.synthetic_samples
        cmd += ["--samples", str(synthetic_samples), "--seed", str(args.seed)]

    cmd += [
        "--repeats",
        str(repeats),
        "--decim",
        str(decim),
        "--taps",
        str(args.taps),
        "--block",
        str(args.block),
        "--check-outputs",
        str(args.check_outputs),
    ]

    env = os.environ.copy()
    env[args.thread_env] = str(threads)

    stdout = run_cmd(cmd, env=env)
    metrics = extract_metrics(stdout)
    metrics.update(
        {
            "repeats": repeats,
            "decim": decim,
            "max_samples": max_samples,
            "threads": threads,
        }
    )
    return metrics


def run_all_sweeps(args, impl_name):
    repeats_vals = parse_list(args.repeats_list)
    decim_vals = parse_list(args.decim_list)
    max_samples_vals = parse_list(args.max_samples_list)
    threads_vals = parse_list(args.threads_list)

    rows = []

    print(f"[{impl_name}] [1/4] repeats sweep")
    for r in repeats_vals:
        m = run_case(
            args,
            repeats=r,
            decim=args.base_decim,
            max_samples=args.base_max_samples,
            threads=args.base_threads,
        )
        m["mode"] = "repeats_sweep"
        m["impl"] = impl_name
        rows.append(m)
        print(
            f"  repeats={r:>4} threads={m['threads']:>3} "
            f"time={m['time_s']:.6f}s gflops={m['gflops']:.3f}"
        )

    print(f"[{impl_name}] [2/4] decim sweep")
    for d in decim_vals:
        m = run_case(
            args,
            repeats=args.base_repeats,
            decim=d,
            max_samples=args.base_max_samples,
            threads=args.base_threads,
        )
        m["mode"] = "decim_sweep"
        m["impl"] = impl_name
        rows.append(m)
        print(
            f"  decim={d:>4} threads={m['threads']:>3} "
            f"time={m['time_s']:.6f}s gflops={m['gflops']:.3f}"
        )

    print(f"[{impl_name}] [3/4] max-samples sweep")
    for n in max_samples_vals:
        m = run_case(
            args,
            repeats=args.base_repeats,
            decim=args.base_decim,
            max_samples=n,
            threads=args.base_threads,
        )
        m["mode"] = "max_samples_sweep"
        m["impl"] = impl_name
        rows.append(m)
        print(
            f"  max_samples={n:>9} threads={m['threads']:>3} "
            f"time={m['time_s']:.6f}s gflops={m['gflops']:.3f}"
        )

    print(f"[{impl_name}] [4/4] threads sweep")
    for t in threads_vals:
        m = run_case(
            args,
            repeats=args.base_repeats,
            decim=args.base_decim,
            max_samples=args.base_max_samples,
            threads=t,
        )
        m["mode"] = "threads_sweep"
        m["impl"] = impl_name
        rows.append(m)
        print(f"  threads={t:>3} time={m['time_s']:.6f}s gflops={m['gflops']:.3f}")

    return rows


def write_csv(path: Path, rows):
    fieldnames = [
        "impl",
        "mode",
        "repeats",
        "decim",
        "max_samples",
        "threads",
        "samples_in",
        "samples_out",
        "time_s",
        "gflops",
        "energy_sum",
        "hits",
        "checksum",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def get_mode_x(row, mode):
    if mode == "repeats_sweep":
        return row["repeats"]
    if mode == "decim_sweep":
        return row["decim"]
    if mode == "threads_sweep":
        return row["threads"]
    return row["max_samples"]


def make_plot(path: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable, skipped plotting: {e}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    impls = sorted({r["impl"] for r in rows})

    for idx, (mode, xlabel) in enumerate(SWEEP_SPECS):
        ax = axes[idx]
        rr = [r for r in rows if r["mode"] == mode]

        if not rr:
            ax.set_visible(False)
            continue

        for impl in impls:
            ii = [r for r in rr if r["impl"] == impl]
            ii.sort(key=lambda x: get_mode_x(x, mode))
            if not ii:
                continue
            xs = [get_mode_x(r, mode) for r in ii]
            ys = [r["time_s"] for r in ii]
            ax.plot(xs, ys, marker="o", label=impl)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Main compute time [s]")
        ax.grid(True, alpha=0.3)
        if len(impls) > 1:
            ax.legend()

    fig.suptitle("IQ benchmark parameter sweeps")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot: {path}")


def make_speedup_plot(path: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable, skipped speedup plotting: {e}")
        return

    impls = sorted({r["impl"] for r in rows})
    if len(impls) < 2:
        return

    ref_impl = impls[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for idx, (mode, xlabel) in enumerate(SWEEP_SPECS):
        ax = axes[idx]
        mode_rows = [r for r in rows if r["mode"] == mode]
        if not mode_rows:
            ax.set_visible(False)
            continue

        ref_rows = [r for r in mode_rows if r["impl"] == ref_impl]
        ref_map = {get_mode_x(r, mode): r["time_s"] for r in ref_rows}
        if not ref_map:
            ax.set_visible(False)
            continue

        for impl in impls[1:]:
            ii = [r for r in mode_rows if r["impl"] == impl]
            ii.sort(key=lambda x: get_mode_x(x, mode))
            xs = []
            ys = []
            for r in ii:
                x = get_mode_x(r, mode)
                ref_time = ref_map.get(x)
                if ref_time is None or r["time_s"] <= 0.0:
                    continue
                xs.append(x)
                ys.append(ref_time / r["time_s"])
            if xs:
                ax.plot(xs, ys, marker="o", label=f"{impl} vs {ref_impl}")

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Speedup (time_ref / time_impl)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Speedup relative to {ref_impl}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved speedup plot: {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Run IQ executable parameter sweeps and plot runtime.")
    ap.add_argument("--exe", type=Path, default=Path("./iq_baseline"), help="Path to executable")
    ap.add_argument(
        "--compare-exe",
        type=Path,
        default=None,
        help="Optional second executable for comparison plots",
    )
    ap.add_argument("--input", default="", help="Optional IQ input file")
    ap.add_argument("--format", default="ci16", choices=["ci16", "cf32"], help="Input format")
    ap.add_argument("--seed", type=int, default=12345, help="Seed for synthetic mode")
    ap.add_argument(
        "--synthetic-samples",
        type=int,
        default=2_000_000,
        help="Used only when --input is omitted",
    )
    ap.add_argument("--taps", type=int, default=64)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--check-outputs", type=int, default=1024)

    ap.add_argument("--base-repeats", type=int, default=3)
    ap.add_argument("--base-decim", type=int, default=4)
    ap.add_argument("--base-max-samples", type=int, default=4_000_000)
    ap.add_argument("--base-threads", type=int, default=1)

    ap.add_argument("--repeats-list", default="1,2,3,4,6")
    ap.add_argument("--decim-list", default="2,4,8,16")
    ap.add_argument("--max-samples-list", default="500000,1000000,2000000,4000000")
    ap.add_argument("--threads-list", default="1,2,4,8")
    ap.add_argument(
        "--thread-env",
        default="OMP_NUM_THREADS",
        help="Environment variable used to set thread count",
    )

    ap.add_argument("--out-csv", type=Path, default=Path("bench_results.csv"))
    ap.add_argument("--out-png", type=Path, default=Path("bench_results.png"))
    ap.add_argument(
        "--out-speedup-png",
        type=Path,
        default=Path("bench_speedup.png"),
        help="Speedup plot path (used when --compare-exe is set)",
    )
    args = ap.parse_args()

    args.exe = args.exe.resolve()
    if args.compare_exe is not None:
        args.compare_exe = args.compare_exe.resolve()
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            print(f"Current working directory: {Path.cwd()}")
            return 1
        args.input = str(input_path)

    if not args.exe.exists():
        print(f"Executable not found: {args.exe}")
        return 1
    if args.compare_exe is not None and not args.compare_exe.exists():
        print(f"Compare executable not found: {args.compare_exe}")
        return 1

    rows = []

    print(f"Using thread control env: {args.thread_env}")

    primary_impl_name = args.exe.stem
    rows.extend(run_all_sweeps(args, primary_impl_name))

    if args.compare_exe is not None:
        original_exe = args.exe
        args.exe = args.compare_exe
        rows.extend(run_all_sweeps(args, args.exe.stem))
        args.exe = original_exe

    write_csv(args.out_csv, rows)
    print(f"Saved CSV: {args.out_csv}")

    make_plot(args.out_png, rows)
    if args.compare_exe is not None:
        make_speedup_plot(args.out_speedup_png, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
