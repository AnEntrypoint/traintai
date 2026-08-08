"""Run up to 8 independent train.py invocations concurrently, one per
real v5e-8 TPU chip, using TRAINTAI_XLA_DEVICE_INDEX for per-process
chip pinning (confirmed working in-process: round9tpusmoke v6 showed
xm.xla_device(n) resolving cleanly for n=0..7 with no env-var pinning
crash, unlike the TPU_VISIBLE_CHIPS subprocess approach which SIGABRTs
on this pod's runtime).

SPMD sharding is disabled for every child (TRAINTAI_NO_SPMD=1) -- SPMD
crashes with a real, confirmed SIGSEGV on this pod (see device.py's
setup_spmd_mesh docstring), and each child here only needs its own
single chip, not a mesh across all of them.

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
