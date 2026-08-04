"""Convert ffatty/plain-text-wikipedia-simpleenglish (Kaggle, MIT) into a
sparse non-game general-text interleave source.

Per the interleaving literature the user cited this session (Emergent
Misalignment Prevention / Distribution Smoothing): mixing a SMALL fraction
(~5%) of general benign data uniformly into a specialized training mix
counters over-adaptation without diluting the specialized signal -- this
plays the same architectural role TinyStories already plays in
st_prepare.py, but with REAL encyclopedic text (Simple English Wikipedia,
249K articles, MIT-licensed) instead of synthetic children's stories, so
it is a genuinely distinct interleave source, not a duplicate of the
existing TinyStories slice.

AllCombined.txt is one flat file: "Title\\n\\nBody text...\\n\\nNext
Title\\n\\n...". This script chunks article bodies the same way
kaggle_fantasy_convert.py chunks book text (~350 words/chunk, well under
the model's 512-token seq_len), decontaminates via the shared clean()
convention, and caps the OUTPUT SIZE deliberately small (target ~5% of a
typical round's mixture) rather than pulling in the full 31M-token corpus
-- this is a sparse interleave ingredient, not a bulk general-corpus swap.

Output: data/npc/kaggle_wiki.jsonl (training) + kaggle_wiki_holdout.jsonl.
"""

import json
import os
import random
import sys

from kaggle_source_convert import run_conversion

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
OUT = os.path.join(NPC, "kaggle_wiki.jsonl")
HOLD = os.path.join(NPC, "kaggle_wiki_holdout.jsonl")

CHUNK_WORDS = 350
MAX_ROWS = 1500  # deliberately sparse -- see module docstring, ~5% interleave target not a bulk corpus
HOLDOUT_CAP = 60
MIN_ARTICLE_WORDS = 60  # drop stub articles (too short to be useful continuation material)


def read_articles(path):
    """Yields (title, body) for each article in AllCombined.txt. Articles
    are separated by a blank line after the title and the next title
    starts at the next non-blank line following the body -- detected by
    the same 'short standalone line preceded and followed by a blank
    line' shape the dataset's own README excerpt shows, so this reads the
    file as one pass rather than assuming a stricter delimiter exists."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    blocks = text.split("\n\n")
    i = 0
    while i < len(blocks) - 1:
        title = blocks[i].strip()
        body = blocks[i + 1].strip()
        if title and body and len(title.split()) <= 8 and not title[0].islower():
            yield title, body
            i += 2
        else:
            i += 1


def chunk_article(title, body):
    words = body.split()
    if len(words) < MIN_ARTICLE_WORDS:
        return []
    if len(words) <= CHUNK_WORDS:
        return [f"{title}\n\n{body}"]
    chunks = []
    for start in range(0, len(words), CHUNK_WORDS):
        chunk = " ".join(words[start : start + CHUNK_WORDS])
        if len(chunk.split()) >= MIN_ARTICLE_WORDS:
            chunks.append(f"{title}\n\n{chunk}" if start == 0 else chunk)
    return chunks


def main(source_txt):
    rng = random.Random(41)

    def gen_chunks():
        for title, body in read_articles(source_txt):
            for chunk in chunk_article(title, body):
                yield {"content": chunk}

    # Reservoir-sample down to a bounded working set before the shared
    # run_conversion shuffle/cap, since the full article stream is far
    # larger than MAX_ROWS and materializing all of it is wasteful for a
    # source whose whole point is being a small sparse slice.
    reservoir = []
    seen = 0
    target_pool = (MAX_ROWS + HOLDOUT_CAP) * 8
    for row in gen_chunks():
        seen += 1
        if len(reservoir) < target_pool:
            reservoir.append(row)
        else:
            j = rng.randint(0, seen - 1)
            if j < target_pool:
                reservoir[j] = row

    def convert(row):
        return row["content"]

    run_conversion(reservoir, convert, OUT, HOLD,
                   max_rows=MAX_ROWS, holdout_cap=HOLDOUT_CAP, seed=41,
                   holdout_fraction=0.04, label="Simple English Wikipedia chunks")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AllCombined.txt")
