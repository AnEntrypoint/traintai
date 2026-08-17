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
  - kaggle_gamearena_convert.py output (kaggle_gamearena.jsonl): a SPARSE
    (~5% of mixture, same role as kaggle_wiki.jsonl) interleave combining
    ALL 13 user-named Kaggle Game Arena datasets (ultimate-tic-tac-toe,
    bargaining, lines-of-action, coin-game, checkers, clobber,
    dots-and-boxes, dark-hex, word-association, five-in-a-row, poker,
    werewolf, chess -- all CC BY 4.0, Claude/GPT/Gemini/Grok playing each
    other) into ONE combined capped source rather than each dataset
    getting its own large allocation. Two content shapes: language-heavy
    games (werewolf/bargaining/poker/word-association) render as ST-format
    dialog cards; pure move-log games (chess/checkers/etc, no natural-
    language content at all) render as compact state->move continuation
    lines, not a fabricated persona card. LLM-generated throughout --
    never counted toward the real-data ratio AGENTS.md tracks, but
    teaches strategic multi-agent reasoning under uncertainty.
  - kaggle_werewolf_convert.py output (kaggle_werewolf.jsonl): Werewolf's
    OWN separate, larger capped allocation (KAGGLE_WEREWOLF_CAP=2000), IN
    ADDITION TO its share of the combined kaggle_gamearena.jsonl pool
    above -- richest/most-tested single source in the family, kept at
    higher weight by explicit choice rather than folded down to an equal
    per-game share
  - a TinyStories token slice (~20%) so the model keeps general coherence

All sources pass a decontamination filter (TOXIC substrings: the old fixed
template and second-person meta-narrative seams) and have *action beats*
stripped from response lines (dialog-only output target) before
tokenization. data/npc/pippa_holdout.jsonl, kaggle_fantasy_holdout.jsonl,
kaggle_wiki_holdout.jsonl, kaggle_gamearena_holdout.jsonl, and
kaggle_werewolf_holdout.jsonl are NEVER included -- they are the
real-data generalization gates.

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
KAGGLE_GAMEARENA_CAP = 900  # same sparse ~5% role, combined across all 13 Kaggle Game Arena datasets -- see kaggle_gamearena_convert.py
KAGGLE_WEREWOLF_CAP = 2000  # Werewolf's own larger, separately-capped allocation, IN ADDITION TO its share of kaggle_gamearena.jsonl's combined pool
BALROG_DEMOS_CAP = int(os.environ.get("BALROG_DEMOS_CAP", 3000))  # balrog_demo_convert.py output: real BALROG expert-demo trajectories re-rendered as "Observation: ... assistant: <action>" SFT rows, so the model sees this prompt shape at least once before a BALROG eval round instead of only ever being evaluated on it untrained. Overridable via env var for lever-isolation sweeps that vary the demo-data mixture ratio without editing this file between runs. Reverted to 3000 from a round-14 attempt at 12000 (2026-08-10): real per-episode parse-success-rate measurement (see below) showed raising the row-count cap 3000->12000 moved parse-success from ~6% to ~5%, i.e. NO improvement (within noise, if not slightly worse) -- refuting the "just add more demo rows" hypothesis with real evidence across rounds 9/13/14. Root cause instead traced to train.py's Batcher/model.py loss having NO prompt masking at all (ignore_index=-1 support exists but is never used) -- every token, including each BALROG row's huge Observation/instruction prompt (200-2000+ tokens), gets equal gradient weight against the 1-3 actual completion tokens that matter for the stop-after-action behavior. Real fix is per-row loss masking (BALROG_ROW_MASKING below), not a bigger cap.
BALROG_ROW_MASKING = os.environ.get("BALROG_ROW_MASKING", "1") == "1"  # mask the PROMPT portion of every BALROG demo/self-play row's loss (ignore_index=-1 via a parallel mask.bin), so gradient concentrates on the completion tokens (the action) instead of being diluted across each row's huge Observation/instruction prompt. The real, measured-necessary fix (2026-08-10) after BALROG_DEMOS_CAP alone was shown to not move real parse-success rate at all across rounds 9/13/14 (~5-7%, flat). Disable via env var to isolate/compare against the unmasked baseline.
BALROG_DEMOS_ONLY = os.environ.get("BALROG_DEMOS_ONLY", "") == "1"  # when set, skip every other source and TinyStories entirely -- an isolation-test mixture of pure BALROG demo data, for measuring whether diluting demo data with the rest of the mixture is itself a lever
BALROG_SELFPLAY_CAP = int(os.environ.get("BALROG_SELFPLAY_CAP", 1500))  # balrog_selfplay_convert.py output: our OWN checkpoint's real self-play rollouts, filtered to episode_return>0 -- a rejection-sampling flywheel on top of the expert-demo bootstrap, same self-distillation-limit convention as FORGE_CAP/ACTION_FORGE_CAP
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

    n_kaggle_gamearena = 0
    kaggle_gamearena_path = os.path.join(NPC, "kaggle_gamearena.jsonl")
    if os.path.exists(kaggle_gamearena_path):
        for row in read_jsonl(kaggle_gamearena_path):
            if clean(row["text"]):
                texts.append(strip_beats(row["text"]))
                n_kaggle_gamearena += 1
                if n_kaggle_gamearena >= KAGGLE_GAMEARENA_CAP:
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

    # Real finding (2026-08-12): reading the first BALROG_DEMOS_CAP rows
    # of balrog_demos.jsonl as a flat prefix looked balanced by ROW count
    # (balrog_demo_convert.py's round-robin write means the first N rows
    # cycle evenly through every game), but babaisai (avg ~1192 tokens/
    # row) and babyai (avg ~889 tokens/row) are real, substantially
    # shorter than minihack/nle (avg ~1974 tokens/row) -- so an
    # equal-ROW prefix still gives babyai/babaisai ~45-55% FEWER real
    # training TOKENS than minihack/nle, even though row counts match.
    # Round 38's real eval confirmed this: the round-robin row-balance
    # fix alone did NOT move babaisai/babyai's near-zero parse-success
    # (a token-level cap tested in balrog_demo_convert.py's WRITE stage
    # was found not to help either, since st_prepare.py's fixed-size
    # row-count prefix read never reaches deep enough into the file to
    # benefit from write-side token balancing -- the read side must fix
    # this directly). Fixed here: tag each row by its source game (using
    # the same real prompt markers balrog_demo_convert.py's own
    # instruction_prompt_for() emits), then select per-game until each
    # game reaches an equal token-count quota within the overall
    # BALROG_DEMOS_CAP*avg_row_len token budget, not a fixed row-count
    # prefix -- so babyai/babaisai get written MORE times (more
    # repetition of their smaller real raw pools) to match minihack/
    # nle's real per-game token exposure.
    BALROG_GAME_MARKERS = [
        ("babaisai", "Baba Is You"),
        ("babyai", "navigation game"),
        ("crafter", "Move North"),
    ]

    def _balrog_game_of(text):
        for name, marker in BALROG_GAME_MARKERS:
            if marker in text:
                return name
        return "minihack_nle_textworld"

    n_balrog_demos = 0
    balrog_demos = []
    balrog_demos_path = os.path.join(NPC, "balrog_demos.jsonl")
    if os.path.exists(balrog_demos_path) and BALROG_DEMOS_CAP > 0:
        all_demo_rows = [row["text"] for row in read_jsonl(balrog_demos_path) if clean(row["text"])]
        if all_demo_rows:
            by_game = {}
            for text in all_demo_rows:
                by_game.setdefault(_balrog_game_of(text), []).append(text)
            _tok_bulk = _bulk_encoder(tok)
            game_lens = {g: [len(e) for e in _tok_bulk(rows)] for g, rows in by_game.items()}
            avg_row_len = sum(n for lens in game_lens.values() for n in lens) / len(all_demo_rows)
            token_budget = BALROG_DEMOS_CAP * avg_row_len
            per_game_budget = token_budget / len(by_game)
            indices = {g: 0 for g in by_game}
            tokens_written = {g: 0 for g in by_game}
            active = list(by_game.keys())
            while active and n_balrog_demos < BALROG_DEMOS_CAP * len(by_game):
                wrote_any = False
                for g in list(active):
                    rows, lens = by_game[g], game_lens[g]
                    if tokens_written[g] >= per_game_budget:
                        active.remove(g)
                        continue
                    i = indices[g] % len(rows)
                    balrog_demos.append(rows[i])
                    tokens_written[g] += lens[i]
                    indices[g] += 1
                    n_balrog_demos += 1
                    wrote_any = True
                if not wrote_any:
                    break
            print(f"  balrog_demos token-balanced selection: "
                  + ", ".join(f"{g}={indices[g]}rows/{tokens_written[g]}tok" for g in by_game))

    n_balrog_selfplay = 0
    balrog_selfplay = []
    balrog_selfplay_path = os.path.join(NPC, "balrog_selfplay.jsonl")
    if os.path.exists(balrog_selfplay_path):
        for row in read_jsonl(balrog_selfplay_path):
            if clean(row["text"]):
                balrog_selfplay.append(row["text"])
                n_balrog_selfplay += 1
                if n_balrog_selfplay >= BALROG_SELFPLAY_CAP:
                    break

    balrog_row_flags = []  # parallel to texts: True for BALROG demo/self-play rows, used to build the loss mask below

    if BALROG_DEMOS_ONLY:
        # Isolation-test mixture: pure BALROG demo data, nothing else,
        # no TinyStories -- measures whether the rest of the mixture
        # dilutes/interferes with format-learning from demo data alone,
        # a real lever candidate this can't distinguish from without it.
        texts = list(balrog_demos) + list(balrog_selfplay)
        balrog_row_flags = [True] * len(texts)
        tmpl = []
        print(f"BALROG_DEMOS_ONLY=1 -- balrog_demos {n_balrog_demos} | balrog_selfplay {n_balrog_selfplay} | total {len(texts)}")
    else:
        balrog_row_flags = [False] * len(texts)
        texts.extend(balrog_demos)
        balrog_row_flags.extend([True] * len(balrog_demos))
        texts.extend(balrog_selfplay)
        balrog_row_flags.extend([True] * len(balrog_selfplay))
        tmpl = [strip_beats(row["text"]) for row in read_jsonl(os.path.join(NPC, "st_conversations.jsonl")) if clean(row["text"])]
        texts.extend(tmpl[:TEMPLATE_CAP])
        balrog_row_flags.extend([False] * min(len(tmpl), TEMPLATE_CAP))
        print(f"real {n_real} | authored {n_auth} | world {n_world} | sim {n_sim} | pippa {n_pippa} | "
              f"forge {n_forge} | action_forge {n_action_forge} | chains {n_chains} | "
              f"kaggle_fantasy {n_kaggle_fantasy} | kaggle_wiki {n_kaggle_wiki} | "
              f"kaggle_gamearena {n_kaggle_gamearena} | kaggle_werewolf {n_kaggle_werewolf} | "
              f"balrog_demos {n_balrog_demos} | balrog_selfplay {n_balrog_selfplay} | "
              f"template {min(len(tmpl), TEMPLATE_CAP)} | total {len(texts)}")

    # Real fix (2026-08-10): BALROG rows end "...\nassistant: <action>" (see
    # balrog_demo_convert.py's build_row()) -- everything up to and including
    # "assistant:" is the PROMPT (huge Observation/instruction text, 200-2000+
    # tokens); only the action itself (1-3 tokens) is the actual completion
    # signal that matters for the model's stop-after-action behavior.
    # Unmasked training gives the prompt equal gradient weight to the
    # completion, diluting the exact signal BALROG_DEMOS_CAP alone was
    # measured (rounds 9/13/14, real parse-success rate flat ~5-7%) not to
    # fix. Mask the prompt span (ignore_index=-1 in model.py's loss) for
    # BALROG rows only -- every other source's rows stay fully unmasked,
    # unchanged from every prior round's training regime.
    n_masked_rows = 0
    n_masked_tokens = 0
    ids = []
    mask = []  # parallel to ids: 1 = trainable, 0 = masked out of loss
    ASSISTANT_CUE = "assistant:"
    for i, (text, enc, is_balrog) in enumerate(zip(texts, encode(texts), balrog_row_flags)):
        row_mask = [1] * len(enc)
        if BALROG_ROW_MASKING and is_balrog:
            cue_pos = text.rfind(ASSISTANT_CUE)
            if cue_pos != -1:
                prompt_text = text[: cue_pos + len(ASSISTANT_CUE)]
                prompt_len = len(tok.encode(prompt_text).ids)
                prompt_len = min(prompt_len, len(enc))  # defensive: never mask past the row's own token count
                if prompt_len > 0:
                    row_mask[:prompt_len] = [0] * prompt_len
                    n_masked_rows += 1
                    n_masked_tokens += prompt_len
        ids.extend(enc)
        mask.extend(row_mask)
        ids.append(eot)
        mask.append(1)  # EOT itself stays trainable -- it IS the stop signal
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(texts)}, {len(ids) / 1e6:.1f}M tokens", flush=True)

    if BALROG_ROW_MASKING:
        print(f"loss-masked {n_masked_rows} BALROG rows, {n_masked_tokens / 1e6:.2f}M prompt tokens excluded from loss")

    if not BALROG_DEMOS_ONLY:
        ts = np.memmap(os.path.join(DATA, "train_v32768.bin"), dtype=np.uint16, mode="r")
        ts_ids = ts[:TINYSTORIES_TOKENS].tolist()
        ids.extend(ts_ids)
        mask.extend([1] * len(ts_ids))
        print(f"added {TINYSTORIES_TOKENS / 1e6:.1f}M TinyStories tokens; total {len(ids) / 1e6:.1f}M")

    arr = np.array(ids, dtype=np.uint16)
    mask_arr = np.array(mask, dtype=np.uint8)
    assert len(arr) == len(mask_arr), f"token/mask length mismatch: {len(arr)} vs {len(mask_arr)}"
    n_val = max(1, int(len(arr) * 0.005))
    out_tag = os.environ.get("ST_PREPARE_OUT_TAG", "npc")  # override to build multiple named mixture variants side by side (e.g. lever-sweep runs), without each overwriting the shared default train_npc.bin/val_npc.bin
    arr[:-n_val].tofile(os.path.join(DATA, f"train_{out_tag}.bin"))
    arr[-n_val:].tofile(os.path.join(DATA, f"val_{out_tag}.bin"))
    mask_arr[:-n_val].tofile(os.path.join(DATA, f"train_{out_tag}.mask.bin"))
    mask_arr[-n_val:].tofile(os.path.join(DATA, f"val_{out_tag}.mask.bin"))
    print(f"train {len(arr) - n_val:,} / val {n_val:,} -> train_{out_tag}.bin/val_{out_tag}.bin "
          f"(+ .mask.bin sidecar, {mask_arr.mean():.4f} mean trainable-fraction)")


if __name__ == "__main__":
    main()
