"""Build the single-purpose ST training bins.

Mixture (interleaved, anti-overfit by construction):
  - chimbiwide real roleplay conversations, re-rendered to name-prefix ST
  - PIPPA real roleplay (apache-2.0, PygmalionAI), converted and beat-stripped
  - authored exemplars (st_authored*.jsonl, all ST features covered)
  - world-DB grounded conversations (st_world.jsonl: items, quests, places)
  - economy-sim oracle conversations (st_sim.jsonl: DEAL/GOTO decisions,
    abstention cases, dialog-only)
  - forge rejection-sampled rollouts, CAPPED (self-distillation limit)
  - action-forge rollouts (npc_action_forge.py): the model's own GOTO/DEAL/
    BUY responses that exactly match the sim_econ oracle, CAPPED -- data-
    side successor to the GRPO action-reward arc (r17-r22, all measured
    dead; see AGENTS.md)
  - gold multi-sentence chains (st_chains.py): explicitly anchor-word-
    chained NPC responses targeting chain_depth, measured ~0.1 baseline
    across r16-r23
  - a capped combinatorial-template subset from st_data.py output (~15%)
  - kaggle_fantasy_convert.py output (kaggle_fantasy.jsonl): 350-word
    chunks from an EXPLICIT-ALLOWLIST subset (53 of 124 titles) of
    mehhti/classic-fantasy-and-adventure-literature-corpus (Kaggle) --
    only titles that are both unambiguously public domain and genuinely
    fantasy/adventure/mythology genre; see that module's docstring for
    why the full 124-title claim was not trusted as-is
  - kaggle_wiki_convert.py output (kaggle_wiki.jsonl): a SPARSE (~5% of
    mixture) interleave of ffatty/plain-text-wikipedia-simpleenglish
    (Kaggle, MIT) -- Distribution Smoothing / Emergent Misalignment
    Prevention literature: a small uniform fraction of general benign
    real-world text counters over-adaptation to the specialized mix
    better than a large block would, so this stays deliberately capped
    small (KAGGLE_WIKI_CAP), not scaled up like the fantasy corpus
  - kaggle_werewolf_convert.py output (kaggle_werewolf.jsonl): real
    LLM-vs-LLM Werewolf social-deduction transcripts (Kaggle, CC BY 4.0,
    Claude/GPT/Gemini/Grok playing each other) -- LLM-generated, same
    caveat class as any synthetic source (never counted toward the real-
    data ratio AGENTS.md tracks), but teaches strategic multi-agent
    reasoning and negotiation through dialogue, directly relevant to the
    survival-sim's TALK verb rather than just RP-style variety
  - a TinyStories token slice (~20%) so the model keeps general coherence

All sources pass a decontamination filter (TOXIC substrings: the old fixed
template and second-person meta-narrative seams) and have *action beats*
stripped from response lines (dialog-only output target) before
tokenization. data/npc/pippa_holdout.jsonl, kaggle_fantasy_holdout.jsonl,
kaggle_wiki_holdout.jsonl, and kaggle_werewolf_holdout.jsonl are NEVER
included -- they are the real-data generalization gates.

Output: data/train_npc.bin + data/val_npc.bin (uint16 + eot).
"""

import json
import os
import re

import numpy as np
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")
TOK = os.path.join(DATA, "bpe32768.json")
TEMPLATE_CAP = 6000
FORGE_CAP = 2500
ACTION_FORGE_CAP = 1500
KAGGLE_FANTASY_CAP = 3000
KAGGLE_WIKI_CAP = 900  # sparse interleave target ~5% of a typical round's mixture, per Distribution Smoothing literature
KAGGLE_WEREWOLF_CAP = 2000
TINYSTORIES_TOKENS = 4_000_000

TOXIC = ("i deal in what this place provides",
         "say what you need and i will name a price",
         "you are a new friend",
         "they say you ",
         "the user is seeking")

BEAT = re.compile(r"\s*\*[^*]+\*\s*")


def clean(text):
    low = text.lower()
    return not any(t in low for t in TOXIC)


def strip_beats(text):
    out = []
    for ln in text.split("\n"):
        if ln.startswith(("Description:", "Personality:", "Scenario:")):
            out.append(ln)
        else:
            out.append(BEAT.sub(" ", ln).strip() if ln.strip() else ln)
    return "\n".join(out)


def name_of(system_text):
    m = re.search(r"You are ([A-Z][A-Za-z' -]{1,40}?)[.,]", system_text)
    if m:
        return m.group(1).strip()
    m = re.search(r"Background: ([A-Z][A-Za-z' -]{1,40}?) (?:is|was)", system_text)
    return m.group(1).strip() if m else "NPC"


def rerender_real(row):
    msgs = row["messages"]
    system = ""
    turns = []
    for m in msgs:
        role, content = m.get("role"), m.get("content", "")
        if role == "system" or (not system and role == "user" and "roleplay mode" in content.lower()):
            system = content
            continue
        turns.append((role, content))
    if not system or not turns:
        return None
    name = name_of(system)
    desc = system
    for marker in ("Roleplaying Instructions:", "Roleplay Instructions:"):
        i = desc.find(marker)
        if i > 0:
            desc = desc[:i].rstrip(" .\n")
    lines = [f"Description: {desc}", "<START>"]
    for role, content in turns:
        speaker = name if role != "user" else "Player"
        lines.append(f"{speaker}: {content.strip()}")
    return "\n".join(lines) + "\n"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)




def _bulk_encoder(tok):
    try:
        import gigatoken as gt
        g = gt.Tokenizer(tok)
        return lambda texts: [list(r) for r in g.encode_batch(texts)]
    except Exception:
        return lambda texts: [e.ids for e in tok.encode_batch(texts)]

def main():
    tok = Tokenizer.from_file(TOK)
    encode = _bulk_encoder(tok)
    eot = tok.token_to_id("<|endoftext|>")
    texts = []

    for path in ("npc_dialogue.jsonl", "rpg-quests-dialogue.jsonl"):
        for row in read_jsonl(os.path.join(NPC, path)):
            r = rerender_real(row)
            if r and clean(r):
                texts.append(strip_beats(r))
    n_real = len(texts)

    for path in ("st_authored.jsonl", "st_authored2.jsonl"):
        for row in read_jsonl(os.path.join(NPC, path)):
            if clean(row["text"]):
                texts.append(strip_beats(row["text"]))
    n_auth = len(texts) - n_real

    n_world = 0
    for row in read_jsonl(os.path.join(NPC, "st_world.jsonl")):
        if clean(row["text"]):
            texts.append(strip_beats(row["text"]))
            n_world += 1

    n_sim = 0
    sim_path = os.path.join(NPC, "st_sim.jsonl")
    if os.path.exists(sim_path):
        for row in read_jsonl(sim_path):
            if clean(row["text"]):
                texts.append(row["text"])
                n_sim += 1

    n_pippa = 0
    pippa_path = os.path.join(NPC, "pippa_st.jsonl")
    if os.path.exists(pippa_path):
        for row in read_jsonl(pippa_path):
            if clean(row["text"]):
                texts.append(strip_beats(row["text"]))
                n_pippa += 1

    n_forge = 0
    forge_path = os.path.join(NPC, "st_forge_data.jsonl")
    if os.path.exists(forge_path):
        for row in read_jsonl(forge_path):
            if clean(row["text"]):
                texts.append(strip_beats(row["text"]))
                n_forge += 1
                if n_forge >= FORGE_CAP:
                    break

    n_action_forge = 0
    action_forge_path = os.path.join(NPC, "st_action_forge.jsonl")
    if os.path.exists(action_forge_path):
        for row in read_jsonl(action_forge_path):
            if clean(row["text"]):
                texts.append(row["text"])
                n_action_forge += 1
                if n_action_forge >= ACTION_FORGE_CAP:
                    break

    n_chains = 0
    chains_path = os.path.join(NPC, "st_chains.jsonl")
    if os.path.exists(chains_path):
        for row in read_jsonl(chains_path):
            if clean(row["text"]):
                texts.append(strip_beats(row["text"]))
                n_chains += 1

    n_kaggle_fantasy = 0
    kaggle_fantasy_path = os.path.join(NPC, "kaggle_fantasy.jsonl")
    if os.path.exists(kaggle_fantasy_path):
        for row in read_jsonl(kaggle_fantasy_path):
            if clean(row["text"]):
                texts.append(row["text"])
                n_kaggle_fantasy += 1
                if n_kaggle_fantasy >= KAGGLE_FANTASY_CAP:
                    break

    n_kaggle_wiki = 0
    kaggle_wiki_path = os.path.join(NPC, "kaggle_wiki.jsonl")
    if os.path.exists(kaggle_wiki_path):
        for row in read_jsonl(kaggle_wiki_path):
            if clean(row["text"]):
                texts.append(row["text"])
                n_kaggle_wiki += 1
                if n_kaggle_wiki >= KAGGLE_WIKI_CAP:
                    break

    n_kaggle_werewolf = 0
    kaggle_werewolf_path = os.path.join(NPC, "kaggle_werewolf.jsonl")
    if os.path.exists(kaggle_werewolf_path):
        for row in read_jsonl(kaggle_werewolf_path):
            if clean(row["text"]):
                texts.append(strip_beats(row["text"]))
                n_kaggle_werewolf += 1
                if n_kaggle_werewolf >= KAGGLE_WEREWOLF_CAP:
                    break

    tmpl = [strip_beats(row["text"]) for row in read_jsonl(os.path.join(NPC, "st_conversations.jsonl")) if clean(row["text"])]
    texts.extend(tmpl[:TEMPLATE_CAP])
    print(f"real {n_real} | authored {n_auth} | world {n_world} | sim {n_sim} | pippa {n_pippa} | "
          f"forge {n_forge} | action_forge {n_action_forge} | chains {n_chains} | "
          f"kaggle_fantasy {n_kaggle_fantasy} | kaggle_wiki {n_kaggle_wiki} | "
          f"kaggle_werewolf {n_kaggle_werewolf} | "
          f"template {min(len(tmpl), TEMPLATE_CAP)} | total {len(texts)}")

    ids = []
    for i, enc in enumerate(encode(texts)):
        ids.extend(enc)
        ids.append(eot)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(texts)}, {len(ids) / 1e6:.1f}M tokens", flush=True)

    ts = np.memmap(os.path.join(DATA, "train_v32768.bin"), dtype=np.uint16, mode="r")
    ids.extend(ts[:TINYSTORIES_TOKENS].tolist())
    print(f"added {TINYSTORIES_TOKENS / 1e6:.1f}M TinyStories tokens; total {len(ids) / 1e6:.1f}M")

    arr = np.array(ids, dtype=np.uint16)
    n_val = max(1, int(len(arr) * 0.005))
    arr[:-n_val].tofile(os.path.join(DATA, "train_npc.bin"))
    arr[-n_val:].tofile(os.path.join(DATA, "val_npc.bin"))
    print(f"train {len(arr) - n_val:,} / val {n_val:,}")


if __name__ == "__main__":
    main()
