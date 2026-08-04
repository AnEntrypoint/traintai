"""Convert PIPPA (PygmalionAI, apache-2.0) rows into ST-format conversations.

PIPPA rows carry bot_name / bot_greeting / bot_definitions; the definitions
hold {{char}}:/{{user}}: turns. Rows with at least 2 exchanges become real
training conversations (card header + name-prefixed turns); a disjoint
slice is written to pippa_holdout.jsonl as a REAL-data eval set that never
enters the bins -- the genuine-improvement gate against overfitting.

Outputs: data/npc/pippa_st.jsonl (training) + data/npc/pippa_holdout.jsonl.
"""

import os
import re

from kaggle_source_convert import read_jsonl_rows, run_conversion

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
SRC = os.path.join(NPC, "pippa_deduped.jsonl")
OUT = os.path.join(NPC, "pippa_st.jsonl")
HOLD = os.path.join(NPC, "pippa_holdout.jsonl")

MAX_ROWS = 9000
HOLDOUT = 300
MAX_TURN_CHARS = 400
MIN_EXCHANGES = 2


def parse_turns(defs):
    parts = re.split(r"\{\{char}}:|\{\{user}}:", defs)
    roles = re.findall(r"\{\{(char|user)}}:", defs)
    turns = []
    for role, text in zip(roles, parts[1:]):
        text = " ".join(text.split()).strip()
        if not text or len(text) > MAX_TURN_CHARS:
            return None
        turns.append(("npc" if role == "char" else "user", text))
    return turns


def first_sentence(text, limit=260):
    for sep in (". ", "! ", "? "):
        i = text.find(sep)
        if 20 < i < limit:
            return text[: i + 1]
    return text[:limit]


def convert(row):
    name = row.get("bot_name", "").strip()
    if not name or len(name) > 40 or not re.match(r"^[A-Z][A-Za-z' .-]{1,39}$", name):
        return None
    turns = parse_turns(row.get("bot_definitions", ""))
    if not turns or sum(1 for r, _ in turns if r == "user") < MIN_EXCHANGES:
        return None
    greet = " ".join((row.get("bot_greeting") or "").split()).strip()
    desc = first_sentence(greet) if greet else f"{name}, a character."
    lines = [f"Description: {desc}",
             "Personality: as the card shows, consistently.",
             f"Scenario: a meeting with {name}.",
             "<START>"]
    started = False
    for role, text in turns:
        speaker = name if role == "npc" else "Player"
        if role == "npc" and not started and text.lower().startswith(("description:", "scenario:")):
            continue
        started = True
        lines.append(f"{speaker}: {text}")
    if not started or len(lines) < 7:
        return None
    return "\n".join(lines) + "\n"


def main():
    run_conversion(read_jsonl_rows(SRC), convert, OUT, HOLD,
                   max_rows=MAX_ROWS, holdout_cap=HOLDOUT, seed=7,
                   holdout_fraction=0.04, label="PIPPA conversations")


if __name__ == "__main__":
    main()
