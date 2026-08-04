"""Shared converter for the Kaggle Game Arena dataset family (kaggle/*-gameplay,
all CC BY 4.0, LLM-vs-LLM game transcripts in Kaggle-Environments replay
format). Per user direction: ALL 13 named games (ultimate-tic-tac-toe,
bargaining, lines-of-action, coin-game, checkers, clobber,
game-arena-dots-and-boxes, dark-hex, word-association, five-in-a-row,
poker-heads-up, werewolf, chess-text) are folded into the SAME sparse ~5%
overfitting-prevention interleave role kaggle_wiki_convert.py already
plays, not given individual large per-dataset caps the way
kaggle_werewolf.jsonl was first wired in with -- this module supersedes
treating any one game as a bigger source than the others.

Two content shapes exist across the family, both handled here:
  1. Move-log games (tic-tac-toe, chess, checkers, dots-and-boxes,
     dark-hex, clobber, lines-of-action, five-in-a-row, coin-game):
     structured OpenSpiel actionHistory, no natural-language text.
     Rendered as a compact "state -> chosen move" line, not a dialog
     card -- there is no persona to speak as, so this does not force an
     ST Description/Scenario shape onto pure notation.
  2. Language-heavy games (werewolf, bargaining, poker, word-association):
     real reasoning/message text inside action.kwargs.message or
     action.call_details[].response. Rendered as ST-format cards, the
     same shape kaggle_werewolf_convert.py already used.

LLM-generated content throughout (Claude/GPT/Gemini/Grok playing each
other) -- never counted toward the real-data ratio AGENTS.md tracks.
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
OUT = os.path.join(NPC, "kaggle_gamearena.jsonl")
HOLD = os.path.join(NPC, "kaggle_gamearena_holdout.jsonl")

# Sparse interleave target, matching kaggle_wiki.jsonl's ~5%-of-mixture
# sizing rationale (Distribution Smoothing / Emergent Misalignment
# Prevention literature) -- this caps the COMBINED output of all 13
# games, not per-game.
MAX_ROWS = 900
HOLDOUT_CAP = 40

MAX_TEXT_CHARS = 400
MIN_TEXT_CHARS = 16

LANGUAGE_GAMES = {"werewolf", "bargaining", "poker-heads-up", "word-association"}
MOVE_LOG_GAMES = {"ultimate-tic-tac-toe", "lines-of-action", "coin-game", "checkers",
                   "clobber", "game-arena-dots-and-boxes", "dark-hex",
                   "five-in-a-row", "chess-text"}


def truncate_at_sentence(text, limit):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    best = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: best + 1] if best > limit * 0.4 else cut.rstrip() + "."


def extract_chat_turns(game):
    """Werewolf-style: action.action_type == 'ChatAction', kwargs.message."""
    for step in game.get("steps", []):
        for agent_step in step:
            action = agent_step.get("action")
            if not isinstance(action, dict):
                continue
            kw = action.get("kwargs")
            if isinstance(kw, dict) and kw.get("message"):
                yield kw.get("actor_id") or "Player", kw["message"]


def extract_response_turns(game):
    """Bargaining/poker-style: action.call_details[0].response holds the
    model's reasoning text, action.actionString/actionSubmittedToString
    the structured move it concluded with. No per-agent name field exists
    in this schema (unlike Werewolf's kwargs.actor_id) -- the agent's
    index within the step (agent_step position in the per-tick list) is
    the only stable identity available, rendered as "Player N"."""
    for step in game.get("steps", []):
        for idx, agent_step in enumerate(step):
            action = agent_step.get("action")
            if not isinstance(action, dict):
                continue
            details = action.get("call_details")
            if isinstance(details, list) and details:
                resp = details[0].get("response")
                if resp:
                    yield f"Player {idx + 1}", resp


def game_kind(title):
    t = (title or "").lower()
    for name in LANGUAGE_GAMES:
        if name.replace("-", " ") in t or name in t:
            return "language"
    return "move_log"


def convert_language_game(game, title):
    """Renders language-heavy turns as ST-format cards, one per turn with
    a short rolling dialog window -- same shape kaggle_werewolf_convert.py
    already used, generalized across the whole family so bargaining/poker/
    word-association get the identical treatment werewolf did."""
    turns = list(extract_chat_turns(game)) or list(extract_response_turns(game))
    rows = []
    window = []
    for actor, text in turns:
        trimmed = truncate_at_sentence(text.strip(), MAX_TEXT_CHARS)
        if len(trimmed) >= MIN_TEXT_CHARS:
            desc = f"{actor}, playing {title}, reasoning under strategic uncertainty."
            scen = f"A round of {title} in progress."
            lines = [f"Description: {desc}", f"Scenario: {scen}", "<START>"]
            for prior_actor, prior_text in window[-3:]:
                lines.append(f"{prior_actor}: {prior_text}")
            lines.append(f"{actor}: {trimmed}")
            rows.append("\n".join(lines) + "\n")
        window.append((actor, trimmed))
        window = window[-4:]
    return rows


def convert_movelog_game(game, title):
    """Renders move-log games as compact state->move continuation lines,
    NOT an ST dialog card -- there is no persona to speak as in pure
    chess/checkers notation, so forcing a Description/Scenario/<START>
    shape onto it would be a fabricated card, not a faithful rendering of
    the source."""
    rows = []
    for step in game.get("steps", []):
        for agent_step in step:
            info = agent_step.get("info", {})
            move = info.get("actionSubmittedToString")
            if move and len(move) >= 3:
                rows.append(f"{title}: the position calls for {move}.\n")
    return rows


def clean_title(raw_title):
    """Kaggle-Environments titles the OpenSpiel-backed games as
    'Open Spiel: <snake_case_name>'; strip that wrapper and underscores
    so it reads as prose ('Open Spiel: dots_and_boxes' -> 'Dots And
    Boxes') instead of leaking implementation naming into training text."""
    t = raw_title or "a game"
    if t.lower().startswith("open spiel:"):
        t = t.split(":", 1)[1].strip()
    return t.replace("_", " ").title()


def convert_game_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            game = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    raw_title = game.get("title") or "a game"
    title = clean_title(raw_title)
    kind = game_kind(raw_title)
    if kind == "language":
        return convert_language_game(game, title)
    return convert_movelog_game(game, title)


def main(source_dirs):
    """source_dirs: list of directories, each containing *.json game
    files for one dataset (e.g. one dir per downloaded kaggle/*-gameplay
    dataset). All games across all dirs feed the same combined, capped
    output -- this is the point: 13 sources sharing one small sparse
    interleave slot, not 13 separate large caps."""
    rng = random.Random(61)
    rows = []
    for d in source_dirs:
        if not os.path.isdir(d):
            print(f"skipping missing dir {d}")
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".json"):
                rows.extend(convert_game_file(os.path.join(d, fname)))

    holdout = []
    out = []
    for r in rows:
        (holdout if rng.random() < 0.04 else out).append(r)
    rng.shuffle(out)
    rng.shuffle(holdout)
    out, holdout = out[:MAX_ROWS], holdout[:HOLDOUT_CAP]

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps({"text": r}) + "\n")
    with open(HOLD, "w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps({"text": r}) + "\n")
    print(f"wrote {len(out)} training + {len(holdout)} held-out Game Arena rows "
          f"from {len(source_dirs)} dataset dirs, {len(rows)} raw turns before cap")


if __name__ == "__main__":
    main(sys.argv[1:] if len(sys.argv) > 1 else ["."])
