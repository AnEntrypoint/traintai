"""Convert kaggle/werewolf-gameplay (Kaggle, CC BY 4.0) into ST-format
training conversations.

Real LLM-vs-LLM Werewolf transcripts (Claude/GPT/Gemini/Grok playing each
other) in Kaggle-Environments replay format: each *.json file is one game
with `steps` -- a list of per-tick agent-step dicts. A ChatAction's
`kwargs` carries `actor_id` (persona name), `message` (the public
in-character line), and `raw_prompt` (the full game-state + role + timeline
context for that turn). This is LLM-generated content -- same caveat class
as gpt-roleplay-realm (not real human dialogue) -- but the value here is
different: real strategic multi-agent reasoning under uncertainty and
social negotiation, directly relevant to the survival-sim's TALK verb,
not just RP-style variety. Never counted toward the "real data" ratio
AGENTS.md tracks.

Each game file becomes multiple training rows, one per ChatAction turn:
the raw_prompt's role/game-state section becomes Description/Scenario,
prior public messages in the timeline become the dialog lead-in, and this
turn's message becomes the target line. Rows are per-turn (not whole-game
transcripts) to keep each row well under the model's 512-token window --
matching every other source's chunking discipline in this project.

Output: data/npc/kaggle_werewolf.jsonl (training) + kaggle_werewolf_holdout.jsonl.
"""

import json
import os
import re
import sys

from kaggle_source_convert import run_conversion

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
OUT = os.path.join(NPC, "kaggle_werewolf.jsonl")
HOLD = os.path.join(NPC, "kaggle_werewolf_holdout.jsonl")

MAX_ROWS = 2000
HOLDOUT_CAP = 80
MAX_MESSAGE_CHARS = 400
MIN_MESSAGE_CHARS = 16


def truncate_at_sentence(text, limit):
    """Cuts at the last sentence boundary before limit, matching
    pippa_convert.py's first_sentence() precedent -- a mid-sentence hard
    cut ('...still leaning toward' with no period) teaches the model to
    stop generating mid-thought, the opposite of the clean-stop behavior
    every other source in this mixture is built to reinforce."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    best = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: best + 1] if best > limit * 0.4 else cut.rstrip() + "."


def iter_chat_turns(game):
    """Yields (actor_id, role_name, message) for every ChatAction in one
    game's steps, in chronological order."""
    for step in game.get("steps", []):
        for agent_step in step:
            action = agent_step.get("action")
            if not isinstance(action, dict) or action.get("action_type") != "ChatAction":
                continue
            kw = action.get("kwargs", {})
            msg = kw.get("message")
            actor = kw.get("actor_id")
            raw_prompt = kw.get("raw_prompt", "")
            if not msg or not actor:
                continue
            m = re.search(r'"your_role_name":\s*"([^"]+)"', raw_prompt)
            role = m.group(1) if m else "Villager"
            yield actor, role, msg.strip()


def convert_game(game):
    """Returns a list of rendered ST-format rows, one per chat turn, each
    with a short rolling window of prior turns as dialog lead-in (keeps
    context realistic without requiring the full raw_prompt, which can be
    thousands of tokens of game-state JSON unsuited to this model's
    512-token window)."""
    turns = list(iter_chat_turns(game))
    rows = []
    window = []
    for actor, role, msg in turns:
        trimmed = truncate_at_sentence(msg, MAX_MESSAGE_CHARS)
        if len(trimmed) >= MIN_MESSAGE_CHARS:
            desc = f"{actor}, playing Werewolf as a {role}, reasoning under social pressure."
            scen = "A round-robin discussion among the surviving players, deciding who to trust."
            lines = [f"Description: {desc}", f"Scenario: {scen}", "<START>"]
            for prior_actor, prior_msg in window[-3:]:
                lines.append(f"{prior_actor}: {prior_msg}")
            lines.append(f"{actor}: {trimmed}")
            rows.append("\n".join(lines) + "\n")
        window.append((actor, trimmed))
        window = window[-4:]
    return rows


def main(source_dir):
    def gen_rows():
        for fname in sorted(os.listdir(source_dir)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(source_dir, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    game = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            for row in convert_game(game):
                yield {"text": row}

    def convert(row):
        return row["text"]

    run_conversion(gen_rows(), convert, OUT, HOLD,
                   max_rows=MAX_ROWS, holdout_cap=HOLDOUT_CAP, seed=53,
                   holdout_fraction=0.04, label="Werewolf gameplay turns")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
