"""Shared pipeline for converting a Kaggle dataset into ST-format training
data + a held-out non-overfit gate, following pippa_convert.py's exact
shape (shuffle -> holdout-split -> cap -> write two jsonl files -> print
counts). Every *_convert.py for a new Kaggle source (names/lore/dialog/
fantasy) composes this instead of re-implementing the split/cap/write
loop, so a future 5th source is a `convert_row` function plus a call to
`run_conversion`, not a new copy of the whole pipeline.

Per-source specifics (row parsing, decontamination beyond the shared
clean() filter, schema handling) stay in each source's own module -- this
file owns only the part that was genuinely identical across sources.
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")

DEFAULT_HOLDOUT_FRACTION = 0.04


def run_conversion(rows, convert_row, out_path, holdout_path, max_rows,
                    holdout_cap, seed=7, holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
                    label="rows"):
    """rows: iterable of raw source records. convert_row(row) -> rendered
    ST-format text or None (reject). Splits into train/holdout BEFORE
    capping (matches pippa_convert.py exactly: holdout draws from the full
    converted pool at holdout_fraction, train draws from the rest, both
    capped independently), so the holdout set is a genuine random sample
    of the source, not a tail-end slice."""
    rng = random.Random(seed)
    out, holdout = [], []
    for row in rows:
        c = convert_row(row)
        if not c:
            continue
        (holdout if rng.random() < holdout_fraction else out).append(c)
    rng.shuffle(out)
    rng.shuffle(holdout)
    out, holdout = out[:max_rows], holdout[:holdout_cap]
    with open(out_path, "w", encoding="utf-8") as f:
        for c in out:
            f.write(json.dumps({"text": c}) + "\n")
    with open(holdout_path, "w", encoding="utf-8") as f:
        for c in holdout:
            f.write(json.dumps({"text": c}) + "\n")
    print(f"wrote {len(out)} training + {len(holdout)} held-out {label}")
    return len(out), len(holdout)


def read_jsonl_rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
