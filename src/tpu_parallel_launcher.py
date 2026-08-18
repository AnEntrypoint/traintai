"""CONFIRMED NOT VIABLE AS DESIGNED (round 44, 2026-08-18): real Kaggle
testing found every one of 8 subprocess.Popen children crashes with a
real SIGABRT, "Could not find SliceBuilder port 8471 in any of the 0
ports provided in tpu_process_addresses=local" -- the same error class
the TPU_VISIBLE_CHIPS env-var approach hit. xm.xla_device(n) pinning
working within ONE process (round9tpusmoke v6: xla:0..7 all resolved
cleanly) does NOT generalize to N independent OS subprocesses each
calling it once -- this pod's TPU runtime coordination service appears
to accept only one real claimant process, not N. Kept in the tree as a
record of a real, tested-and-failed approach (see AGENTS.md round 44,
device.py's get_device() docstring) -- do not resurrect this design
without first understanding why the runtime rejects multi-process
claims, e.g. investigating whether a single coordinator process could
own the TPU and dispatch work to per-chip worker threads instead of
separate OS processes.

Each variant is a real, independent train.py argv list (e.g. a
lever-isolation sweep: same base config, one hyperparameter varied per
variant) -- this launcher does not itself decide what varies; callers
build VARIANTS.

Usage: define VARIANTS below (or import run_variants and call it), then
`python3 src/tpu_parallel_launcher.py`.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run_variants(variants, chip_count=8):
    """variants: list of (tag, extra_argv_list). Launches each as a real
    subprocess pinned to a distinct chip (variant i -> chip i % chip_count),
    waits for all to finish, returns a list of
    {tag, chip, returncode, wall_seconds, stdout_tail, stderr_tail}."""
    if len(variants) > chip_count:
        print(
            f"WARNING: {len(variants)} variants > {chip_count} chips -- "
            f"variants will queue in batches of {chip_count}, not run fully "
            f"concurrently.",
            flush=True,
        )

    results = []
    for batch_start in range(0, len(variants), chip_count):
        batch = variants[batch_start : batch_start + chip_count]
        procs = []
        for i, (tag, extra_argv) in enumerate(batch):
            env = dict(os.environ)
            env["TRAINTAI_DEVICE"] = "xla"
            env["TRAINTAI_NO_SPMD"] = "1"
            env["TRAINTAI_XLA_DEVICE_INDEX"] = str(i)
            argv = ["python3", os.path.join(HERE, "train.py")] + extra_argv + ["--tag", tag]
            print(f"launching chip {i}: {' '.join(argv)}", flush=True)
            t0 = time.time()
            proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            procs.append((tag, i, t0, proc))

        for tag, chip, t0, proc in procs:
            stdout, stderr = proc.communicate()
            results.append(
                {
                    "tag": tag,
                    "chip": chip,
                    "returncode": proc.returncode,
                    "wall_seconds": round(time.time() - t0, 1),
                    "stdout_tail": stdout[-2000:],
                    "stderr_tail": stderr[-2000:],
                }
            )
            print(
                f"chip {chip} ({tag}) done: exit={proc.returncode} "
                f"wall={results[-1]['wall_seconds']}s",
                flush=True,
            )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tpu_parallel_results.json")
    args = ap.parse_args()

    # Real smoke-test default: 2 tiny variants (lr sweep) to verify the
    # per-process chip pinning genuinely works before any real sweep is
    # built on top of this launcher -- callers should replace VARIANTS
    # with a real lever-isolation design once this is confirmed.
    base = [
        "--arm", "ple", "--vocab", "32768", "--d-model", "96", "--n-layers", "6",
        "--n-heads", "4", "--ple-dim", "128", "--fixed-ffn", "66",
        "--seq-len", "2048", "--steps", "10", "--batch-size", "4",
        "--eval-every", "5", "--optimizer", "adamw",
    ]
    variants = [
        ("smoke-chip0-lr1e-3", base + ["--lr", "1e-3"]),
        ("smoke-chip1-lr5e-4", base + ["--lr", "5e-4"]),
    ]
    results = run_variants(variants)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
