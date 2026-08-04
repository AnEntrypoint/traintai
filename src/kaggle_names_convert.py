"""Convert isaacbenge/fantasy-for-markov-generator (Kaggle, CC0-1.0) into
a decontaminated fantasy-name pool for sim_world.py's NAME_POOL.

The dataset has 12 category CSVs, one bare name per line, no header. Most
categories (GeneralWest, Nordic, Egyptian, Baltic, Greece, Italiano,
MiddleEastern, Mayan, Asianish) are real historical/cultural first-name
lists -- fine as raw names but not what this pulls, since sim_world.py's
existing NAME_POOL is deliberately invented-fantasy-flavored, not drawn
from any real culture's actual name stock. Only WeirdGoblin.csv and
LongDragon_or_Minotaur.csv (623 combined rows) are unambiguously
synthetic/fantasy-generated names, so those are the two files this
script uses.

Decontamination beyond st_prepare.py's clean() TOXIC-substring filter:
this session found synthetic_names.jsonl/st_conversations.jsonl already
contain real celebrity/franchise names ("Natasha Romanoff", "Mike Isaac")
that caused incoherent generations when trained on (r24 diagnosis). A
small blocklist catches the same failure mode here even though the
source files themselves are synthetic -- a defense-in-depth check, not
because this particular dataset is expected to contain celebrities.

Output: data/npc/kaggle_names.jsonl, one {"name": "..."} row per accepted
name. sim_world.py's NAME_POOL stays the small hand-authored fallback;
loading this file is an opt-in expansion (see sim_world.py's
load_name_pool()), not a silent replacement.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
OUT = os.path.join(NPC, "kaggle_names.jsonl")

# Downloaded via: kaggle datasets download isaacbenge/fantasy-for-markov-generator --unzip -p <dir>
SOURCE_FILES = ("WeirdGoblin.csv", "LongDragon_or_Minotaur.csv")

MIN_LEN = 3
MAX_LEN = 20

# Defense-in-depth: real people/franchise characters that must never enter
# a name pool regardless of source, matching the contamination pattern
# found in synthetic_names.jsonl this session (case-insensitive substring).
BLOCKLIST = ("romanoff", "widow", "stark", "skywalker", "potter", "gandalf",
             "frodo", "batman", "superman", "spider", "vader")


def clean_name(raw):
    name = raw.strip()
    if not (MIN_LEN <= len(name) <= MAX_LEN):
        return None
    if not re.fullmatch(r"[A-Za-z' -]+", name):
        return None
    low = name.lower()
    if any(b in low for b in BLOCKLIST):
        return None
    return name[0].upper() + name[1:]


def main(source_dir):
    seen = set()
    names = []
    for fname in SOURCE_FILES:
        path = os.path.join(source_dir, fname)
        if not os.path.exists(path):
            print(f"skipping missing {path}")
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                n = clean_name(line)
                if n and n not in seen:
                    seen.add(n)
                    names.append(n)
    with open(OUT, "w", encoding="utf-8") as f:
        for n in names:
            f.write(json.dumps({"name": n}) + "\n")
    print(f"wrote {len(names)} decontaminated fantasy names to {OUT} "
          f"(from {len(SOURCE_FILES)} source files)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
