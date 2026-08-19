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
kept (repeated) direction-action step, EQUAL row count per game
(wrap-around cycling, not proportional to raw availability -- round 43
found the naive repeat-then-cap scheme let minihack/nle's larger real
pool dominate the drill and overcorrect babyai/babaisai toward compass
words, the opposite of the intended fix). Reads directly from an
existing balrog_demos.jsonl (already-rendered "...assistant: <action>"
rows, same format st_prepare.py's read stage consumes) -- no
re-conversion or records.zip access needed.
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")

DEFAULT_CAP = 20000

# Real direction-word action sets per game, verbatim from
# balrog_demo_convert.py's _BABAISAI_ACTIONS/_MINIHACK_ACTIONS/
# _CRAFTER_ACTION_DICT keys (babyai's own directions are phrase-shaped,
# "go forward"/"turn left", not bare words -- but round 49's real
# failure data (see AGENTS.md) shows babyai is EQUALLY vulnerable to
# the same compass-word/go+direction confusion as every other game
# (1775 real failures, 87% bare compass words or "go X" phrases) --
# it just had no drill signal of its own to correct it. Added below
# with its own real phrase-shaped targets.
DIRECTION_ACTIONS = {
    "babaisai": {"up", "down", "left", "right"},
    "crafter": {"Move North", "Move South", "Move East", "Move West"},
    "minihack_nle_textworld": {"north", "south", "east", "west"},
    "babyai": {"go forward", "turn left", "turn right"},
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
                     help="total output rows across all games; split "
                          "EQUALLY per game by default (cap // num_games), "
                          "each game's own rows cycling via wrap-around if "
                          "it has fewer unique rows than its target share --"
                          " concentrates extra gradient updates on the "
                          "confused decision equally across games, not "
                          "proportional to a game's raw row availability "
                          "(round 43's real finding: the old repeat-then-"
                          "cap scheme let minihack/nle's larger raw pool "
                          "dominate, overcorrecting babyai/babaisai "
                          "TOWARD compass words instead of away from "
                          "them -- see AGENTS.md).")
    ap.add_argument("--game-share", default="",
                     help='optional JSON dict overriding the equal-share '
                          'default with explicit per-game fractions of '
                          '`cap`, e.g. \'{"crafter": 0.4, "babaisai": 0.3, '
                          '"babyai": 0.15, "minihack_nle_textworld": 0.15}\''
                          ' (missing games default to an equal split of '
                          'the remaining share). Real motivation: crafter '
                          'stayed flat (~0.3%% parse-success) across every '
                          'round tested despite equal drill share -- its '
                          'real instruction prompt is unusually heavy (a '
                          '22-item achievements list + 16 action '
                          'definitions vs babaisai\'s 5 actions/no '
                          'achievements), a plausible reason its own '
                          'direction words get less effective reinforcement '
                          'per row than other games\' -- worth testing a '
                          'higher crafter share before concluding the '
                          'mechanism itself cannot help crafter.')
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

    # Real finding (round 43, 2026-08-18, see AGENTS.md): the original
    # `(rows * repeat)[:cap]` scheme let each game's real unique-row
    # count determine its final drill share -- minihack_nle_textworld's
    # much larger real pool (4529 unique rows vs babyai's 1651/
    # babaisai's 1088) dominated the drill dataset even after repeat=4,
    # so extra gradient concentration disproportionately reinforced
    # minihack/nle's OWN convention model-wide, overcorrecting babyai/
    # babaisai TOWARD compass words instead of away from them (real
    # eval: babyai 5.48%->2.40%, babaisai 4.05%->3.18%, both worse).
    # Fixed by giving every game an EQUAL final row count (wrap-around
    # cycling through each game's own unique rows, same wrap-around
    # principle as balrog_demo_convert.py's original row-balance fix)
    # instead of letting repeat*unique_rows vary by game.
    games = list(by_game.keys())
    if args.game_share:
        override = json.loads(args.game_share)
        unknown = set(override) - set(games)
        if unknown:
            raise SystemExit(
                f"--game-share names unknown game(s) {sorted(unknown)} -- "
                f"real keys are {sorted(games)} (babyai is intentionally "
                f"absent: its own direction confusion is 'go X' phrases, "
                f"not bare direction words, a different drill target)."
            )
        remaining_games = [g for g in games if g not in override]
        remaining_share = max(0.0, 1.0 - sum(override.values()))
        equal_remainder = remaining_share / len(remaining_games) if remaining_games else 0.0
        target_share = {g: override.get(g, equal_remainder) for g in games}
    else:
        target_share = {g: 1.0 / len(games) for g in games}
    target_rows = {g: int(args.cap * target_share[g]) for g in games}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    total_out = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for game, rows in by_game.items():
            if not rows:
                continue
            n = 0
            i = 0
            while n < target_rows[game]:
                f.write(json.dumps({"text": rows[i % len(rows)]}) + "\n")
                i += 1
                n += 1
                total_out += 1

    print(f"{'game':<25} {'unique rows':>12} {'output rows':>12}")
    for game, rows in by_game.items():
        out_n = target_rows[game] if rows else 0
        print(f"{game:<25} {len(rows):>12} {out_n:>12}")
    _mode = f"game_share={target_share}" if args.game_share else "equal share"
    print(f"\nscanned {total_seen} demo rows -> {total_out} direction-drill rows "
          f"({_mode}, wrap-around, total cap={args.cap}) -> {args.out}")


if __name__ == "__main__":
    main()
