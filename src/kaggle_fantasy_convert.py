"""Convert mehhti/classic-fantasy-and-adventure-literature-corpus (Kaggle,
uploader-asserted CC0-1.0) into a TinyStories-slot-style general-coherence
text supplement.

The dataset's own claim ("published prior to 1931 = Public Domain") does
not match how US copyright actually works for the 1927-1930 window (a
95-years-from-publication rule, not a flat 1931 cutoff) -- direct
inspection of the 124 titles found "A Farewell to Arms" (Hemingway, 1929,
copyright historically enforced by the estate) and "The Great Gatsby"
(Fitzgerald, 1925, did not enter the US public domain until 2021) among
otherwise-safe 19th-century classics. Per this project's kaggle-license-
audit discipline (never assume permissive by default, PIPPA was checked
before being pulled in), this script uses an EXPLICIT ALLOWLIST of titles
that are unambiguously public domain (pre-1900, or long-established
public-domain status) and are genuinely fantasy/adventure/mythology genre
-- not the uploader's full 124-book claim, and not a same-shape excludelist
that would need updating every time a new risky title is noticed.

Output: data/npc/kaggle_fantasy.jsonl (training) + kaggle_fantasy_holdout.jsonl,
each row a chunk of book text (not a full ST conversation card -- this is
prose-continuation supplement material, the same role TinyStories plays
in st_prepare.py, not dialog training data).
"""

import csv
import json
import os
import sys

from kaggle_source_convert import run_conversion

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
OUT = os.path.join(NPC, "kaggle_fantasy.jsonl")
HOLD = os.path.join(NPC, "kaggle_fantasy_holdout.jsonl")

MAX_ROWS = 3000
HOLDOUT_CAP = 100
CHUNK_WORDS = 350  # measured ~511-539 BPE tokens at 400 words; 350 keeps chunks under the 512-token seq_len with margin

# Explicit allowlist: unambiguously public-domain (pre-1900 or
# long-settled status) AND genuinely fantasy/adventure/mythology genre.
# Excludes general literary fiction (Middlemarch, Wuthering Heights),
# non-fiction/memoir (Frederick Douglass narrative), and anything from
# the risky 1900s-1930s copyright window (Gatsby, Farewell to Arms).
ALLOWED_TITLES = {
    "A Journey to the Centre of the Earth by Verne Jules",
    "A Princess of Mars by Edgar Rice Burroughs",
    "Alices Adventures in Wonderland by Carroll Lewis",
    "Around the World in Eighty Days by Verne Jules",
    "At the Back of the North Wind by George MacDonald",
    "Beowulf (Anonymous)",
    "Don Quixote by Miguel de Cervantes",
    "Dorothy and the Wizard in Oz by L. Frank Baum",
    "Five Children and It by E. Nesbit",
    "Four Arthurian Romances by Chrétien de Troyes active 12th century",
    "Frankenstein by Mary Shelley",
    "Grimms Fairy Tales",
    "Gullivers Travels",
    "Hans Christian Andersens Fairy Tales",
    "King Solomons Mines by H. Rider Haggard",
    "Ozma of Oz by L. Frank Baum",
    "Peter Pan by J. M. Barrie",
    "Phantastes",
    "Robin Hood by J. Walker McSpadden",
    "She A History of Adventure by H. Rider Haggard",
    "Tarzan of the Apes by Edgar Rice Burroughs",
    "The Arabian Nights Entertainments by Andrew Lang",
    "The Blue Fairy Book by Andrew Lang",
    "The Book of Wonder by Lord Dunsany",
    "The Call of the Wild by Jack London",
    "The Enchanted Castle",
    "The Gods of Pegana by Lord Dunsany",
    "The House of the Wolfings by William Morris",
    "The Iliad by Homer",
    "The Jungle Book",
    "The Lost World by Arthur Conan Doyle",
    "The Marvelous Land of Oz",
    "The Merry Adventures of Robin Hood by Howard Pyle",
    "The Odyssey by Homer",
    "The Princess and Curdie by George MacDonald",
    "The Princess and the Goblin",
    "The Red Fairy Book by Andrew Lang",
    "The Return of Tarzan by Edgar Rice Burroughs",
    "The Romance of Tristan and Iseult by Bédier Joseph",
    "The Story of King Arthur and His Knights by Howard Pyle",
    "The Swiss Family Robinson by Johann David Wyss",
    "The Thousand and One Nights",
    "The Time Machine by H. G. Wells",
    "The War of the Worlds by H. G. Wells",
    "The Water-Babies by Charles Kingsley",
    "The Well at the Worlds End by William Morris",
    "The Wind in the Willows",
    "The Wonderful Wizard of Oz by L. Frank Baum",
    "The Wood Beyond the World by William Morris",
    "The Worm Ouroboros by E. R. Eddison",
    "Thuvia maid of Mars by Burroughs Edgar Rice",
    "Treasure Island by Robert Louis Stevenson",
    "Twenty Thousand Leagues Under the Sea by Jules Verne",
}


def read_source_rows(path):
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 3:
                continue
            yield {"title": row[0], "word_count": row[1], "content": row[2]}


def chunk_words(row):
    """Split one book's content into ~CHUNK_WORDS-word chunks, dropping
    the first chunk (Project Gutenberg boilerplate / title page noise is
    concentrated at the very start of most public-domain texts)."""
    words = row["content"].split()
    chunks = []
    for i in range(CHUNK_WORDS, len(words) - CHUNK_WORDS, CHUNK_WORDS):
        chunk = " ".join(words[i : i + CHUNK_WORDS])
        chunks.append(chunk)
    return chunks


def main(source_csv):
    def gen_chunks():
        for row in read_source_rows(source_csv):
            if row["title"] not in ALLOWED_TITLES:
                continue
            for chunk in chunk_words(row):
                yield {"title": row["title"], "content": chunk}

    def convert(chunk_row):
        return chunk_row["content"]

    n_out, n_hold = run_conversion(gen_chunks(), convert, OUT, HOLD,
                                    max_rows=MAX_ROWS, holdout_cap=HOLDOUT_CAP,
                                    seed=23, holdout_fraction=0.03,
                                    label="fantasy-corpus chunks (allowlisted titles only)")
    n_titles = len(ALLOWED_TITLES)
    print(f"allowlist: {n_titles} titles out of the source's 124 "
          f"({n_titles / 124:.0%}) -- see module docstring for the license/genre reasoning")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle_story_dataset.csv")
