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
match) -- never a judged/simulated score.

Autonomous multi-generation mode (--generations N and/or --hours H, with
--branches > 1): after each generation's branches are ranked, AUTO-
PROMOTES a winner as the next generation's --prev and keeps going --
genuinely unattended operation, per user direction ("hours of training
without needing a human in the loop"). Promotion is diversity-aware, not
always-the-single-leader: with --promote-strategy top-k-random (default),
the next --prev is drawn randomly from the top PROMOTE_TOP_K branches
(not always rank #1), so repeated generations don't collapse onto one
lineage's local optimum. A generation whose best branch is a NULL RESULT
(does not beat the running best-ever forge score) still promotes that
generation's best branch forward (training must continue for hours
unattended; a human is not there to intervene), but is marked
NON-IMPROVING in the auto-appended AGENTS.md entry so the record stays
honest even though nothing paused to ask about it. Stops after
--generations generations, or when --hours wall-clock is exceeded,
whichever comes first; with neither flag set, --branches > 1 runs
exactly one generation and stops (the pre-existing single-generation
behavior, unattended-loop is opt-in via a budget flag).

Lineage tags: branched runs are named <tag>-g<G>-b<N> (generation G,
branch N), verified disjoint from every existing runs/*.pt filename
pattern (plain st-rN tags never contain "-g" or "-b", so no collision is
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
                   action_forge_scenarios, sft_seed=None, skip_tournament=False,
                   tournament_roster=8, tournament_ticks=8, tournament_k=4,
                   sft_batch_size=None, sft_lr=None):
    """Runs stages 1-6 for a single tag/lineage. Returns a dict with the
    checkpoint path and, if forge/simeval produced parseable numbers, the
    measured fitness fields -- or None if the round failed before
    producing a checkpoint at all (SFT or GRPO stage failure)."""
    check_tag_collision(tag)
    seed_suffix = sft_seed if sft_seed is not None else 0
    sft_ckpt = os.path.join(RUNS, f"ple-{tag}-s{seed_suffix}.pt")
    best_ckpt = os.path.join(RUNS, f"ple-{tag}-s{seed_suffix}-best.pt")
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
        if not skip_tournament:
            # Survival-sim tournament self-play (sim_tournament.py) --
            # composes with this pipeline the same way actionforge does:
            # a pre-prepare data-generation pass writing st_survival.jsonl,
            # which st_prepare.py's mixture already reads if present. Not
            # a fork of round.py's own stage sequence -- one more source
            # feeding the same prepare step every other source uses.
            run_stage("tournament", ["sim_tournament.py", prev,
                                      "--roster", str(tournament_roster),
                                      "--ticks", str(tournament_ticks),
                                      "--k", str(tournament_k)],
                       os.path.join(RUNS, f"{tag}-tournament.log"))
        if run_stage("prepare", ["st_prepare.py"], os.path.join(RUNS, f"{tag}-prepare.log")):
            return None

    sft_cmd = ["train.py", "--arm", "ple", "--vocab", "32768", "--d-model", "96",
               "--n-layers", "6", "--n-heads", "4", "--ple-dim", "128", "--fixed-ffn", "66",
               "--data-suffix", "_npc", "--init-from", prev,
               "--steps", str(steps), "--tag", tag]
    if sft_seed is not None:
        sft_cmd += ["--seed", str(sft_seed)]
    if sft_batch_size is not None:
        sft_cmd += ["--batch-size", str(sft_batch_size)]
        if sft_lr is not None:
            sft_cmd += ["--lr", str(sft_lr)]
        else:
            # Linear LR scaling (Goyal et al.): preserves the r16-r24
            # recipe's training dynamics when batch size changes rather
            # than silently diverging from it -- default train.py lr is
            # 1e-3 at batch_size=32, so a larger batch scales lr by the
            # same ratio unless the caller passes an explicit --sft-lr.
            scaled_lr = 1e-3 * (sft_batch_size / 32)
            sft_cmd += ["--lr", str(scaled_lr)]
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


PROMOTE_TOP_K = 3  # random pick among the top K branches each generation, not always rank #1
SHIP_FLOOR = 74     # AGENTS.md's r16 ship baseline; a generation whose best doesn't beat this is NON-IMPROVING


def run_generation(gen_idx, base_tag, prev, n_branches, steps, grpo_steps,
                    skip_prepare, skip_action_forge, action_forge_scenarios, rng,
                    skip_tournament=False):
    """Runs one generation's N branches sequentially from `prev`, ranks
    them, and returns (ranked_results, promoted_result_or_None).
    Promotion picks randomly among the top PROMOTE_TOP_K ranked branches
    (diversity-preserving -- see module docstring) rather than always the
    single leader, so an unattended multi-generation run does not
    deterministically collapse onto one lineage's local optimum."""
    gen_tag = f"{base_tag}-g{gen_idx}"
    print(f"\n=== generation {gen_idx}: {n_branches} branches from {os.path.basename(prev)} (tag {gen_tag}) ===")
    results = []
    for i in range(n_branches):
        branch_tag = f"{gen_tag}-b{i}"
        print(f"\n--- branch {i + 1}/{n_branches}: {branch_tag} ---", flush=True)
        r = run_one_round(branch_tag, prev, steps, grpo_steps,
                           skip_prepare, skip_action_forge,
                           action_forge_scenarios, sft_seed=1000 + gen_idx * 100 + i,
                           skip_tournament=skip_tournament)
        results.append(r)
        if r:
            print_round_summary(branch_tag)

    ranked = rank_branches(results)
    print(f"\n=== generation {gen_idx} comparison ({len(ranked)}/{n_branches} produced a scored checkpoint) ===")
    if not ranked:
        print("ALL BRANCHES FAILED before producing a forge-scored checkpoint -- "
              "nothing to promote this generation; the run must stop, there is no valid --prev to continue from.")
        return ranked, None
    for rank, r in enumerate(ranked, 1):
        print(f"  #{rank} {r['tag']}: forge {r['forge_pass_rate']}% | oracle_match {r['oracle_match_rate']}% | {r['ckpt']}")

    pool = ranked[:PROMOTE_TOP_K]
    check_generation_diversity(pool)
    promoted = rng.choice(pool)
    best = ranked[0]
    if best["forge_pass_rate"] is not None and best["forge_pass_rate"] <= SHIP_FLOOR:
        print(f"\nNON-IMPROVING generation: best branch ({best['tag']}, forge {best['forge_pass_rate']}%) "
              f"did not beat the {SHIP_FLOOR}% ship floor -- promoting {promoted['tag']} anyway to keep the "
              f"unattended run going (per user direction: no human-in-the-loop pause), but this generation "
              f"is NOT a ship candidate.")
    else:
        print(f"\nPromoted (top-{PROMOTE_TOP_K} random pick, not always the leader, for diversity): "
              f"{promoted['tag']} (forge {promoted['forge_pass_rate']}%)")
    return ranked, promoted


def check_generation_diversity(pool):
    """Real measured behavioral-distance check (branch_diversity.py) on
    the top-PROMOTE_TOP_K checkpoints -- the actual candidates promotion
    picks from. Best-effort: import/measurement failures print a warning
    and let the generation loop continue rather than blocking an
    unattended multi-hour run on a diagnostic step. A collapse warning is
    informational, printed into the same log every generation already
    produces -- it does not itself change anything about this generation
    (round.py's per-branch --seed spread and top-K-random promotion are
    already the mitigations; this is the measurement that would tell a
    future session/human whether they're working)."""
    if len(pool) < 2:
        return
    try:
        import branch_diversity
        ckpts = [r["ckpt"] for r in pool if os.path.exists(r["ckpt"])]
        if len(ckpts) < 2:
            print("diversity check skipped: fewer than 2 promotable checkpoints exist on disk")
            return
        _, mean_divergence = branch_diversity.measure_diversity(ckpts)
        print(f"\nbranch diversity (top-{len(ckpts)} promotable checkpoints): "
              f"{mean_divergence:.0%} mean pairwise divergence")
        if mean_divergence < 0.05:
            print("COLLAPSE WARNING: this generation's top branches are near-identical despite "
                  "distinct SFT seeds -- see branch_diversity.py for the same check run standalone.")
    except Exception as e:
        print(f"diversity check failed (non-fatal, continuing): {e}")


def append_agents_md_entry(gen_idx, base_tag, ranked, promoted):
    """Auto-appends a real-numbers entry to AGENTS.md for this generation,
    same discipline as every hand-written round-result section already in
    the file -- automated instead of human-typed, since the run is
    unattended, but never a claim without a measurement behind it."""
    path = os.path.join(HERE, "..", "AGENTS.md")
    lines = [f"\n## Autonomous generation {gen_idx} ({base_tag}, auto-logged)\n\n"]
    if not ranked:
        lines.append("All branches failed before producing a scored checkpoint. No promotion.\n")
    else:
        for rank, r in enumerate(ranked, 1):
            marker = " <- PROMOTED" if promoted and r["tag"] == promoted["tag"] else ""
            lines.append(f"- #{rank} `{r['tag']}`: forge {r['forge_pass_rate']}%, "
                          f"oracle match {r['oracle_match_rate']}%{marker}\n")
        best = ranked[0]
        if best["forge_pass_rate"] is not None and best["forge_pass_rate"] <= SHIP_FLOOR:
            lines.append(f"- NON-IMPROVING vs the {SHIP_FLOOR}% ship floor; not a ship candidate.\n")
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(lines)


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
    ap.add_argument("--skip-tournament", action="store_true",
                     help="skip the sim_tournament.py survival-sim self-play data pass before prepare")
    ap.add_argument("--branches", type=int, default=1,
                     help="run N divergent branches from --prev sequentially instead of one linear round")
    ap.add_argument("--generations", type=int, default=1,
                     help="with --branches > 1, keep auto-promoting and running new generations up to N (unattended)")
    ap.add_argument("--hours", type=float, default=None,
                     help="with --branches > 1, keep auto-promoting generations until this many wall-clock hours elapse")
    ap.add_argument("--seed", type=int, default=7, help="promotion-choice RNG seed")
    ap.add_argument("--sft-batch-size", type=int, default=None,
                     help="override train.py's SFT batch size (default 32). Measured this session: "
                          "32 -> 9.7GB peak, extrapolated ~44 -> ~13.3GB on a real 16GB T4 (unverified "
                          "on hardware -- measured on a 6GB local card and extrapolated). --sft-lr is "
                          "auto-scaled linearly unless also given explicitly.")
    ap.add_argument("--sft-lr", type=float, default=None,
                     help="override the auto-scaled SFT learning rate when --sft-batch-size is set")
    args = ap.parse_args()
    args.prev = os.path.abspath(args.prev)

    if args.branches <= 1:
        result = run_one_round(args.tag, args.prev, args.steps, args.grpo_steps,
                                args.skip_prepare, args.skip_action_forge,
                                args.action_forge_scenarios,
                                skip_tournament=args.skip_tournament,
                                sft_batch_size=args.sft_batch_size, sft_lr=args.sft_lr)
        if result:
            print_round_summary(args.tag)
        return

    import random
    rng = random.Random(args.seed)
    t_start = time.time()
    prev = args.prev
    gen_idx = 0
    while True:
        ranked, promoted = run_generation(gen_idx, args.tag, prev, args.branches,
                                           args.steps, args.grpo_steps,
                                           args.skip_prepare, args.skip_action_forge,
                                           args.action_forge_scenarios, rng,
                                           skip_tournament=args.skip_tournament)
        append_agents_md_entry(gen_idx, args.tag, ranked, promoted)
        if promoted is None:
            print("\nStopping: no branch produced a valid checkpoint this generation.")
            break
        gen_idx += 1
        elapsed_hours = (time.time() - t_start) / 3600
        hit_generations = args.generations and gen_idx >= args.generations
        hit_hours = args.hours and elapsed_hours >= args.hours
        if not (args.generations > 1 or args.hours):
            # neither budget flag set: run exactly one generation and stop
            # (pre-existing single-generation behavior; the unattended
            # multi-generation loop is opt-in via a budget flag)
            break
        if hit_generations or hit_hours:
            print(f"\nStopping: budget reached (generations={gen_idx}, elapsed={elapsed_hours:.2f}h).")
            break
        prev = promoted["ckpt"]
        print(f"\n>>> continuing unattended: generation {gen_idx} starts from {os.path.basename(prev)}")


if __name__ == "__main__":
    main()
