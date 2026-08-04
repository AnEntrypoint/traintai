"""One repeatable training round, end to end -- linear or branched.

  UV_NO_SYNC=1 uv run python src/round.py --prev runs/ple-st-r14-grpo.pt --tag st-r16

Stages (each logged to runs/<tag>*.log):
  1. actionforge  rejection-sample oracle-matching actions from --prev
  2. prepare      rebuild train/val bins from the current data mix (st_prepare)
  3. sft          top-up SFT from --prev (train.py, 300 steps unless --steps)
  4. grpo         adherence RL with the full-coverage reward (npc_grpo)
  5. forge        720-rollout flaw dashboard (npc_forge)
  6. simeval      held-out oracle-choice + format metrics (sim_eval)

Prints one summary block at the end: forge pass rate + flaw histogram and
the sim-eval rates, so rounds are comparable at a glance.

Branching (--branches N > 1): runs N independent rounds from the SAME
--prev checkpoint, each with a distinct lineage tag and a different SFT
seed (so their weight trajectories genuinely diverge, not just their
RNG-driven data sampling), SEQUENTIALLY -- never concurrently, per the
measured RAM discipline below (two concurrent torch jobs have crashed
this box twice; branching multiplies that risk by N if run in parallel,
so it does not). After all N branches complete, compares them by a real
measured fitness metric (forge pass rate, tie-broken by sim_eval oracle
match) -- never a judged/simulated score -- and prints a ranked summary;
it does NOT auto-promote a "winner" checkpoint anywhere, since AGENTS.md's
ship status is a human-reviewed decision recorded in this file, not an
automated overwrite.

Lineage tags: branched runs are named <tag>-b<N> (e.g. st-r25-b0,
st-r25-b1), verified disjoint from every existing runs/*.pt filename
pattern (plain st-rN tags never contain "-b", so no collision is
possible by construction -- see the lineage-id-collision PRD discovery).
"""

import argparse
import glob
import os
import re
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


def check_tag_collision(tag):
    """Verifies `tag` doesn't collide with any existing runs/*.pt lineage
    -- plain st-rN tags never contain '-b', and branch tags are always
    <base>-b<N>, so the only real collision risk is re-using an EXACT tag
    that already has a checkpoint on disk. Refuses loudly rather than
    silently overwriting a prior round's result."""
    existing = glob.glob(os.path.join(RUNS, f"ple-{tag}-*.pt"))
    if existing:
        raise SystemExit(f"tag '{tag}' already has checkpoints on disk ({existing[:3]}...); "
                          f"choose a new tag, never reuse one -- AGENTS.md's round log depends on tags being unique")


def run_one_round(tag, prev, steps, grpo_steps, skip_prepare, skip_action_forge,
                   action_forge_scenarios, sft_seed=None):
    """Runs stages 1-6 for a single tag/lineage. Returns a dict with the
    checkpoint path and, if forge/simeval produced parseable numbers, the
    measured fitness fields -- or None if the round failed before
    producing a checkpoint at all (SFT or GRPO stage failure)."""
    check_tag_collision(tag)
    sft_ckpt = os.path.join(RUNS, f"ple-{tag}-s0.pt")
    best_ckpt = os.path.join(RUNS, f"ple-{tag}-s0-best.pt")
    grpo_ckpt = os.path.join(RUNS, f"ple-{tag}-grpo.pt")

    if not skip_prepare:
        bins = os.path.join(HERE, "..", "data", "train_v32768.bin")
        if not os.path.exists(bins):
            if run_stage("tokens", ["../data/prepare.py", "--vocab", "32768"],
                         os.path.join(RUNS, f"{tag}-tokens.log")):
                return None
        if not skip_action_forge:
            run_stage("actionforge", ["npc_action_forge.py", prev,
                                       "--scenarios", str(action_forge_scenarios)],
                       os.path.join(RUNS, f"{tag}-actionforge.log"))
        if run_stage("prepare", ["st_prepare.py"], os.path.join(RUNS, f"{tag}-prepare.log")):
            return None

    sft_cmd = ["train.py", "--arm", "ple", "--vocab", "32768", "--d-model", "96",
               "--n-layers", "6", "--n-heads", "4", "--ple-dim", "128", "--fixed-ffn", "66",
               "--data-suffix", "_npc", "--init-from", prev,
               "--steps", str(steps), "--tag", tag]
    if sft_seed is not None:
        sft_cmd += ["--seed", str(sft_seed)]
    rc = run_stage("sft", sft_cmd, os.path.join(RUNS, f"{tag}.log"))
    if rc:
        return None
    grpo_in = best_ckpt if os.path.exists(best_ckpt) else sft_ckpt
    print(f"[grpo] input {os.path.basename(grpo_in)}", flush=True)
    rc = run_stage("grpo", ["npc_grpo.py", grpo_in, "--st", "150", "--sim", "100",
                            "--steps", str(grpo_steps), "--out", grpo_ckpt],
                   os.path.join(RUNS, f"{tag}-grpo.log"))
    if rc:
        return None
    run_stage("forge", ["npc_forge.py", grpo_ckpt, "--cards", "60", "--k", "6"],
              os.path.join(RUNS, f"{tag}-forge.log"))
    run_stage("simeval", ["sim_eval.py", grpo_ckpt],
              os.path.join(RUNS, f"{tag}-simeval.log"))

    return {
        "tag": tag,
        "ckpt": grpo_ckpt,
        "forge_pass_rate": parse_forge_pass_rate(os.path.join(RUNS, f"{tag}-forge.log")),
        "oracle_match_rate": parse_oracle_match_rate(os.path.join(RUNS, f"{tag}-simeval.log")),
    }


def parse_forge_pass_rate(log_path):
    """Extracts the forge dashboard's 'pass rate: N/M = P%' line -- the
    same measured number round.py's own summary block already prints,
    parsed back out for branch comparison instead of re-deriving it."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        m = re.search(r"pass rate: \d+/\d+ = (\d+)%", f.read())
    return int(m.group(1)) if m else None


def parse_oracle_match_rate(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        m = re.search(r"oracle match\s*:\s*\d+/\d+ = (\d+)%", f.read())
    return int(m.group(1)) if m else None


def print_round_summary(tag):
    print(f"\n=== round {tag} summary ===")
    for stage, path in (("forge", f"{tag}-forge.log"), ("simeval", f"{tag}-simeval.log")):
        full = os.path.join(RUNS, path)
        if os.path.exists(full):
            with open(full) as f:
                body = f.read()
            head = body.find("===")
            print(body[head:] if head >= 0 else body[-800:])


def rank_branches(results):
    """Real measured fitness metric for comparing branches: forge pass
    rate first (the project's primary quality signal, per AGENTS.md's
    entire round-log history), sim_eval oracle match as tie-break.
    Never an LLM judge, never a synthesized score -- both fields are
    parsed directly out of the same forge/sim_eval dashboards every
    linear round already produces and trusts."""
    scored = [r for r in results if r and r["forge_pass_rate"] is not None]
    scored.sort(key=lambda r: (r["forge_pass_rate"], r["oracle_match_rate"] or 0), reverse=True)
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--grpo-steps", type=int, default=200)
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-action-forge", action="store_true",
                     help="skip rejection-sampling oracle-matching actions from --prev before prepare")
    ap.add_argument("--action-forge-scenarios", type=int, default=800)
    ap.add_argument("--branches", type=int, default=1,
                     help="run N divergent branches from --prev sequentially instead of one linear round")
    args = ap.parse_args()
    args.prev = os.path.abspath(args.prev)

    if args.branches <= 1:
        result = run_one_round(args.tag, args.prev, args.steps, args.grpo_steps,
                                args.skip_prepare, args.skip_action_forge,
                                args.action_forge_scenarios)
        if result:
            print_round_summary(args.tag)
        return

    # Branched: N sequential rounds from the same --prev, distinct
    # lineage tags and SFT seeds so weight trajectories genuinely
    # diverge (not just RNG-driven data-order noise within one seed).
    # SEQUENTIAL, never concurrent -- see module docstring.
    print(f"=== branched round {args.tag}: {args.branches} sequential branches from {os.path.basename(args.prev)} ===")
    results = []
    for i in range(args.branches):
        branch_tag = f"{args.tag}-b{i}"
        print(f"\n--- branch {i + 1}/{args.branches}: {branch_tag} ---", flush=True)
        r = run_one_round(branch_tag, args.prev, args.steps, args.grpo_steps,
                           args.skip_prepare, args.skip_action_forge,
                           args.action_forge_scenarios, sft_seed=1000 + i)
        results.append(r)
        if r:
            print_round_summary(branch_tag)

    ranked = rank_branches(results)
    print(f"\n=== branch comparison: {args.tag} ({len(ranked)}/{args.branches} produced a scored checkpoint) ===")
    if not ranked:
        print("ALL BRANCHES FAILED before producing a forge-scored checkpoint -- no winner, nothing to promote. "
              "Do not treat any branch's partial checkpoint as a result.")
        return
    parent_note = ""
    for rank, r in enumerate(ranked, 1):
        print(f"  #{rank} {r['tag']}: forge {r['forge_pass_rate']}% | oracle_match {r['oracle_match_rate']}% | {r['ckpt']}")
    best = ranked[0]
    # Flat-or-regressed check: if every branch is <= the ship baseline
    # this project tracks (AGENTS.md's r16 74% floor), this round is a
    # null result, same discipline as "r14->r15 flat, more identical
    # rounds buy nothing" -- print it plainly instead of implying any
    # branch is an improvement just because it ranked #1 among losers.
    SHIP_FLOOR = 74
    if best["forge_pass_rate"] is not None and best["forge_pass_rate"] <= SHIP_FLOOR:
        print(f"\nNULL RESULT: best branch ({best['tag']}, forge {best['forge_pass_rate']}%) "
              f"did not beat the {SHIP_FLOOR}% ship floor. Do not record this as a ship candidate; "
              f"see AGENTS.md's 'more identical rounds buy nothing' precedent.")
    else:
        print(f"\nBest branch: {best['tag']} (forge {best['forge_pass_rate']}%) -- "
              f"NOT auto-promoted; record in AGENTS.md as a measured result and decide ship status by hand, "
              f"same as every prior round in this project's history.")


if __name__ == "__main__":
    main()
