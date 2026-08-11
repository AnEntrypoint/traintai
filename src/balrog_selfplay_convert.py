"""Converts BALROG's real per-episode eval output (the
`<task>_run_NN.csv` + `<task>_run_NN.json` pair every real `eval.py` run
already writes, per balrog-inspect/balrog/evaluator.py:283-383 --
`save_trajectories` in config.yaml is a stale/unread field, this CSV+JSON
pair is the actual always-on trajectory record) into BALROG-shaped SFT
rows, filtered by real episode return so only genuinely successful
self-play rollouts against our OWN checkpoint enter the training mixture
-- a rejection-sampling flywheel on top of balrog_demo_convert.py's
expert-demo bootstrap, mirroring npc_action_forge.py's oracle-correctness
gate for the BALROG side of the mixture.

Input: an eval.py results directory, structure confirmed directly from
evaluator.py: `<results_dir>/<env_name>/<task>/<task>_run_NN.{csv,json}`.
The CSV's real header (evaluator.py:289): Step,Action,Reasoning,
Observation,Reward,Done -- escapechar is the literal character used by
evaluator.py's csv.writer.

Per-row messages/action shape and build_row() truncation policy are
REUSED from balrog_demo_convert.py (same instruction_prompt_for()
dispatch, same build_prompt() import) so self-play rows are byte-for-byte
consistent with the expert-demo rows already in the mixture.

Output: data/npc/balrog_selfplay.jsonl, one {"text": ...} row per kept
(prompt-prefix, next-action) step from a KEPT episode -- an entire
episode is kept or dropped as a unit, not scored per-step, since a
step's own local correctness isn't independently knowable from the CSV
alone.

Selection is rank-based per task (--top-frac), not an absolute return
threshold. Real round 10 evidence: episode_return is continuous
(BALROG's own per-step reward accumulation, e.g. -0.97, -1.0, not a
binary pass/fail), but with a weak checkpoint every real episode across
all 117 in round 10 scored <= 0.0 -- an absolute `> 0.0` "real progress"
threshold correctly rejected all of them (total output rows: 0), which
is mechanically honest but means the self-play flywheel cannot bootstrap
at all until the checkpoint is already competent, a chicken-and-egg gap.
Ranking keeps the RELATIVELY best episodes per task (highest
episode_return, real signal even when every episode is a net failure)
so the flywheel has something to reinforce from turn one, the same
spirit as npc_action_forge.py's rejection sampling operating on relative
not absolute correctness early in training. --min-return remains
available as an optional absolute floor for callers who want the
stricter "only genuine progress" gate back (e.g. once the checkpoint is
strong enough that a meaningful fraction of episodes clear it).
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

sys.path.insert(0, HERE)
from balrog_demo_convert import _build_messages, build_row, instruction_prompt_for  # noqa: E402
from model import Config  # noqa: E402

DEFAULT_CAP = 20000
ENVS = ["babyai", "crafter", "babaisai", "textworld", "minihack", "nle"]


def find_episode_pairs(results_dir, env_name):
    env_dir = os.path.join(results_dir, env_name)
    if not os.path.isdir(env_dir):
        return []
    pairs = []
    for json_path in sorted(glob.glob(os.path.join(env_dir, "**", "*_run_*.json"), recursive=True)):
        csv_path = json_path[: -len(".json")] + ".csv"
        if os.path.exists(csv_path):
            pairs.append((csv_path, json_path))
    return pairs


def task_of(results_dir, env_name, json_path):
    env_dir = os.path.join(results_dir, env_name)
    rel = os.path.relpath(json_path, env_dir)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else None


def load_episode_return(json_path):
    with open(json_path, encoding="utf-8") as f:
        log = json.load(f)
    return float(log.get("episode_return", 0.0))


def load_parse_success_rate(json_path):
    """1 - failed_candidates/num_steps for one episode, or 1.0 if the log
    predates this field (no evidence of failure -> don't penalize it).
    Real root cause found 2026-08-11 (rounds 27/29): iterating self-play
    a second hop off an already-self-play-tuned checkpoint regresses real
    parse-success, twice, independently -- ranking by episode_return alone
    can keep episodes with high in-game reward but a high per-step
    hallucinated-continuation rate, reinforcing the model's own format
    drift rather than the general stop-after-action skill. Blending this
    into selection targets the actual property this campaign optimizes.
    """
    with open(json_path, encoding="utf-8") as f:
        log = json.load(f)
    num_steps = log.get("num_steps")
    failed = log.get("failed_candidates")
    if not num_steps or failed is None:
        return 1.0
    return 1.0 - (len(failed) / num_steps)


INVALID_ACTION_MARKER = "Your previous output did not contain a valid action."
INVALID_ACTION_RE = re.compile(re.escape(INVALID_ACTION_MARKER) + r" Defaulted to action: (.*?)\n")


def replay_csv_steps(csv_path, env_name, task):
    """Yields (messages, action) pairs from a real eval-run CSV, mirroring
    balrog_demo_convert.py's replay_steps() shape exactly so build_row()
    needs no changes. The instruction prompt is reconstructed the same
    way (instruction_prompt_for()); the first CSV row's own Observation
    column supplies the same first_long_term_context balrog_demo_convert.py
    derives from the first .npz entry.

    Real, confirmed bug fixed here: evaluator.py's CSV writer logs the
    RAW model completion in the Action column (evaluator.py:329,
    `action = response.completion` runs AFTER the real validated action
    was already used to step the env), not the validated action that
    was actually taken -- so a row where the model's raw output was
    garbage still has that garbage recorded as "the action". The real
    validated fallback action surfaces as a warning string injected
    into THIS SAME ROW's own Observation column ("Your previous output
    did not contain a valid action. Defaulted to action: <X>",
    evaluator.py:317-328 -- `env.step(action)` with the ALREADY-
    validated action runs first, producing the observation the marker
    gets prepended to, and only then does line 329 overwrite `action`
    with the raw completion for the CSV write -- so the marker and the
    garbage action always land in the same row, not adjacent rows).
    Training on the raw column verbatim was teaching the model to
    reproduce its own worst completions -- both as a training TARGET
    and, just as damaging, as HISTORY CONTEXT for every later step in
    the same episode. Fixed by recovering BALROG's own real validated
    action from the marker text and using THAT for history whenever a
    row was invalid, and by skipping invalid rows entirely as training
    targets -- the model only ever sees and is trained to reproduce
    real, valid actions, both in its own past and as the thing to
    predict next."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, escapechar="˘")
        rows = list(reader)

    if not rows:
        return

    first_long = rows[0]["Observation"]
    instruction = instruction_prompt_for(env_name, task, first_long)

    history = []
    last_long = first_long

    for row in rows:
        action = row["Action"]
        m = INVALID_ACTION_RE.search(row["Observation"])
        was_invalid = m is not None

        if not was_invalid:
            messages = _build_messages(instruction, history, last_long, "")
            yield messages, action

        history.append((last_long, m.group(1) if was_invalid else action))
        last_long = row["Observation"]

        if row.get("Done", "").strip().lower() in ("true", "1"):
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True,
                     help="a real eval.py results directory "
                          "(<results-dir>/<env_name>/<task>/<task>_run_NN.{csv,json})")
    ap.add_argument("--out", default=os.path.join(DATA, "npc", "balrog_selfplay.jsonl"))
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--top-frac", type=float, default=0.25,
                     help="keep the top fraction (by episode_return, "
                          "highest first) of each task's episodes -- "
                          "relative selection so the flywheel has real "
                          "episodes to reinforce even when a weak "
                          "checkpoint makes every episode a net failure "
                          "(round 10: 117/117 episodes scored <=0.0). "
                          "At least 1 episode per task with >=1 real "
                          "episode is always kept.")
    ap.add_argument("--min-return", type=float, default=None,
                     help="optional absolute floor: additionally require "
                          "episode_return strictly greater than this "
                          "(BALROG's reward is 0 or negative on pure "
                          "failure, so >0.0 is the honest 'this episode "
                          "made real progress' bar). Combines with "
                          "--top-frac (both must pass); unset by default "
                          "so early-training rounds are not empty.")
    ap.add_argument("--parse-rank-weight", type=float, default=0.0,
                     help="blend an episode's parse-success rate "
                          "(1 - failed_candidates/num_steps) into the "
                          "ranking key alongside episode_return, as "
                          "rank_key = episode_return + weight * "
                          "parse_success_rate (parse_success_rate is in "
                          "[0,1], so this is roughly comparable in scale "
                          "to episode_return's typical [-1,1] range). "
                          "0.0 (default) keeps the original return-only "
                          "ranking; a positive value biases selection "
                          "toward episodes with cleaner stop-after-action "
                          "formatting, not just higher in-game reward --"
                          "the real lever candidate raised after rounds "
                          "27/29 showed iterating self-play a second hop "
                          "off return-only selection regresses parse "
                          "success.")
    args = ap.parse_args()

    seq_len = args.seq_len if args.seq_len is not None else Config().seq_len

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    stats = {env: {"episodes": 0, "kept_episodes": 0, "kept_rows": 0, "skipped_rows": 0} for env in ENVS}
    total_rows = 0
    capped = False

    with open(args.out, "w", encoding="utf-8") as f:
        for env_name in ENVS:
            pairs = find_episode_pairs(args.results_dir, env_name)
            stats[env_name]["episodes"] = len(pairs)

            by_task = {}
            for csv_path, json_path in pairs:
                task = task_of(args.results_dir, env_name, json_path)
                episode_return = load_episode_return(json_path)
                rank_key = episode_return
                if args.parse_rank_weight:
                    rank_key += args.parse_rank_weight * load_parse_success_rate(json_path)
                by_task.setdefault(task, []).append((episode_return, rank_key, csv_path, json_path))

            selected = []
            for task, episodes in by_task.items():
                episodes.sort(key=lambda e: e[1], reverse=True)
                n_keep = max(1, round(len(episodes) * args.top_frac))
                selected.extend(episodes[:n_keep])

            for episode_return, _rank_key, csv_path, json_path in selected:
                if capped:
                    break
                if args.min_return is not None and episode_return <= args.min_return:
                    continue

                task = task_of(args.results_dir, env_name, json_path)
                try:
                    steps = list(replay_csv_steps(csv_path, env_name, task))
                except Exception as e:
                    print(f"  [{env_name}] failed to replay {csv_path}: {e}", file=sys.stderr)
                    continue

                if not steps:
                    continue
                stats[env_name]["kept_episodes"] += 1

                for messages, action in steps:
                    if total_rows >= args.cap:
                        capped = True
                        break
                    text, kept = build_row(messages, action, tok, seq_len)
                    if kept:
                        f.write(json.dumps({"text": text}) + "\n")
                        stats[env_name]["kept_rows"] += 1
                        total_rows += 1
                    else:
                        stats[env_name]["skipped_rows"] += 1
            if capped:
                break

    print()
    print(f"{'game':<12} {'episodes':>9} {'kept_ep':>8} {'kept_rows':>10} {'skipped_rows':>13}")
    for env_name in ENVS:
        s = stats[env_name]
        print(f"{env_name:<12} {s['episodes']:>9} {s['kept_episodes']:>8} {s['kept_rows']:>10} {s['skipped_rows']:>13}")
    print(f"\ntotal output rows: {total_rows} "
          f"(cap={args.cap}, top_frac={args.top_frac}, min_return={args.min_return}) -> {args.out}")


if __name__ == "__main__":
    main()
