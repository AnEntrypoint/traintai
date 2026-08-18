"""Builds a small, high-repetition SFT dataset that drills ONLY the
direction-word action steps from BALROG's real demo data -- the exact
confusion class real per-episode eval data (rounds 38-42, see AGENTS.md)
showed dominates ~77% of ALL failed actions across every game: babaisai's
bare up/down/left/right, minihack/nle's bare north/east/south/west, and
crafter's "Move North"-style phrases all compete for the same handful of
direction-word output slots, and the model reliably picks the WRONG
game's convention even though the correct action list is always present
in-context (balrog_server.py's truncate_prompt_ids() already guarantees
the instruction survives truncation -- this is a genuine learned-weight
bias, not a missing-information problem).

Five real rounds (38-42) already ruled out fixing this via BALROG's
overall mixture SHARE (row-count or token-count balancing) -- babaisai
is a hard-threshold responder, crafter regresses everything when zeroed,
and no configuration beat round 38's plain baseline. This script targets
the actual confused DECISION rather than the game's overall representation:
pull every real direction-action row (already-correct
prompt+action pairs) out of balrog_demos.jsonl per game, and emit a
dedicated, small, HEAVILY repeated dataset of just those steps, so a
short low-LR fine-tuning pass (round.py's existing --init-from continued-
training pattern) concentrates many extra gradient updates specifically
on the confused direction-word boundary, without touching the rest of
the mixture's balance at all.

Real per-game direction-action sets, reproduced from
balrog_demo_convert.py's own _BABAISAI_ACTIONS/_BABYAI_ACTIONS/
_CRAFTER_ACTION_DICT/_MINIHACK_ACTIONS dicts (verbatim from each real
BALROG environment's own action space, not re-derived).

Output: data/npc/balrog_direction_drill.jsonl, one {"text": ...} row per
kept (repeated) direction-action step. Reads directly from an existing
balrog_demos.jsonl (already-rendered "...assistant: <action>" rows, same
format st_prepare.py's read stage consumes) -- no re-conversion or
records.zip access needed.
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")

DEFAULT_CAP = 20000
DEFAULT_REPEAT = 4

# Real direction-word action sets per game, verbatim from
# balrog_demo_convert.py's _BABAISAI_ACTIONS/_MINIHACK_ACTIONS/
# _CRAFTER_ACTION_DICT keys (babyai's own directions are phrase-shaped,
# "go forward"/"turn left", not bare words, so it is not part of the
# bare-direction-word confusion class this drill targets).
DIRECTION_ACTIONS = {
    "babaisai": {"up", "down", "left", "right"},
    "crafter": {"Move North", "Move South", "Move East", "Move West"},
    "minihack_nle_textworld": {"north", "south", "east", "west"},
}

GAME_MARKERS = [
    ("babaisai", "Baba Is You"),
    ("babyai", "navigation game"),
    ("crafter", "Move North"),
]


def _game_of(text):
    for name, marker in GAME_MARKERS:
        if marker in text:
            return name
    return "minihack_nle_textworld"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _row_action(text):
    """Extracts the target action from an already-rendered
    "...\\nassistant: <action>" row, mirroring balrog_server.py's own
    build_prompt() trailing-cue convention exactly (the row is
    build_prompt(messages) + " " + action, so splitting on the LAST
    "assistant:" and stripping recovers the real action string)."""
    if "assistant:" not in text:
        return None
    return text.rsplit("assistant:", 1)[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default=os.path.join(NPC, "balrog_demos.jsonl"),
                     help="an existing balrog_demos.jsonl to pull direction-action rows from")
    ap.add_argument("--out", default=os.path.join(NPC, "balrog_direction_drill.jsonl"))
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP,
                     help="max output rows per game, after repetition")
    ap.add_argument("--repeat", type=int, default=DEFAULT_REPEAT,
                     help="how many times each real direction-action row is "
                          "repeated in the output -- concentrates extra "
                          "gradient updates on the confused decision "
                          "without needing new/synthetic data, matching "
                          "the flywheel's existing 'more real repetitions, "
                          "not more variety' pattern for a hard behavior.")
    args = ap.parse_args()

    if not os.path.exists(args.demos):
        raise SystemExit(f"{args.demos} not found -- run balrog_demo_convert.py first")

    by_game = {g: [] for g in DIRECTION_ACTIONS}
    total_seen = 0
    for row in read_jsonl(args.demos):
        text = row["text"]
        total_seen += 1
        game = _game_of(text)
        if game not in DIRECTION_ACTIONS:
            continue
        action = _row_action(text)
        if action in DIRECTION_ACTIONS[game]:
            by_game[game].append(text)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total_out = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for game, rows in by_game.items():
            if not rows:
                continue
            repeated = (rows * args.repeat)[: args.cap]
            for text in repeated:
                f.write(json.dumps({"text": text}) + "\n")
                total_out += 1

    print(f"{'game':<25} {'unique rows':>12} {'output rows':>12}")
    for game, rows in by_game.items():
        out_n = min(len(rows) * args.repeat, args.cap)
        print(f"{game:<25} {len(rows):>12} {out_n:>12}")
    print(f"\nscanned {total_seen} demo rows -> {total_out} direction-drill rows "
          f"(repeat={args.repeat}, cap={args.cap}/game) -> {args.out}")


if __name__ == "__main__":
    main()
