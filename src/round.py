"""One repeatable training round, end to end.

  UV_NO_SYNC=1 uv run python src/round.py --prev runs/ple-st-r14-grpo.pt --tag st-r16

Stages (each logged to runs/<tag>*.log):
  1. prepare  rebuild train/val bins from the current data mix (st_prepare)
  2. sft      top-up SFT from --prev (train.py, 300 steps unless --steps)
  3. grpo     adherence RL with the full-coverage reward (npc_grpo)
  4. forge    720-rollout flaw dashboard (npc_forge)
  5. simeval  held-out oracle-choice + format metrics (sim_eval)

Prints one summary block at the end: forge pass rate + flaw histogram and
the sim-eval rates, so rounds are comparable at a glance. Run ONE round at
a time -- two concurrent torch jobs have crashed this box.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")
PY = [sys.executable]


def run_stage(name, cmd, log):
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.run(PY + cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    tail = ""
    with open(log) as f:
        lines = f.read().strip().splitlines()
        tail = lines[-1] if lines else ""
    print(f"[{name}] exit={p.returncode} {dt:.0f}s :: {tail[:110]}", flush=True)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--grpo-steps", type=int, default=200)
    ap.add_argument("--skip-prepare", action="store_true")
    args = ap.parse_args()
    args.prev = os.path.abspath(args.prev)

    sft_ckpt = os.path.join(RUNS, f"ple-{args.tag}-s0.pt")
    best_ckpt = os.path.join(RUNS, f"ple-{args.tag}-s0-best.pt")
    grpo_ckpt = os.path.join(RUNS, f"ple-{args.tag}-grpo.pt")

    if not args.skip_prepare:
        bins = os.path.join(HERE, "..", "data", "train_v32768.bin")
        if not os.path.exists(bins):
            if run_stage("tokens", ["../data/prepare.py", "--vocab", "32768"],
                         os.path.join(RUNS, f"{args.tag}-tokens.log")):
                return
        if run_stage("prepare", ["st_prepare.py"], os.path.join(RUNS, f"{args.tag}-prepare.log")):
            return
    rc = run_stage("sft", ["train.py", "--arm", "ple", "--vocab", "32768", "--d-model", "96",
                           "--n-layers", "6", "--n-heads", "4", "--ple-dim", "128", "--fixed-ffn", "66",
                           "--data-suffix", "_npc", "--init-from", args.prev,
                           "--steps", str(args.steps), "--tag", args.tag],
                   os.path.join(RUNS, f"{args.tag}.log"))
    if rc:
        return
    rc = run_stage("grpo", ["npc_grpo.py", sft_ckpt, "--st", "150", "--sim", "100",
                            "--steps", str(args.grpo_steps), "--out", grpo_ckpt],
                   os.path.join(RUNS, f"{args.tag}-grpo.log"))
    if rc:
        return
    run_stage("forge", ["npc_forge.py", grpo_ckpt, "--cards", "60", "--k", "6"],
              os.path.join(RUNS, f"{args.tag}-forge.log"))
    run_stage("simeval", ["sim_eval.py", grpo_ckpt],
              os.path.join(RUNS, f"{args.tag}-simeval.log"))

    print(f"\n=== round {args.tag} summary ===")
    for stage, path in (("forge", f"{args.tag}-forge.log"), ("simeval", f"{args.tag}-simeval.log")):
        full = os.path.join(RUNS, path)
        if os.path.exists(full):
            with open(full) as f:
                body = f.read()
            head = body.find("===")
            print(body[head:] if head >= 0 else body[-800:])


if __name__ == "__main__":
    main()
