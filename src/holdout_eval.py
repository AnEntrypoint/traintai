"""Held-out real-data generalization gate: teacher-forced ppl on every
*_holdout.jsonl in data/npc/.

Each holdout file never enters the training bins (produced by its
matching *_convert.py alongside the training slice, e.g.
pippa_convert.py -> pippa_holdout.jsonl, kaggle_fantasy_convert.py ->
kaggle_fantasy_holdout.jsonl), so teacher-forced perplexity on it
measures real-data generalization rather than mixture memorization.
Originally PIPPA-only; extended to scan every *_holdout.jsonl present so
a new Kaggle source's holdout gate is automatic on being added (drop
<name>_holdout.jsonl next to its training file, no code change needed
here) rather than requiring a new hardcoded path per source. Run per
checkpoint after a round's SFT stage and compare against the ship
checkpoint:

  UV_NO_SYNC=1 uv run python src/holdout_eval.py runs/ple-st-r19-s0-best.pt
"""

import argparse
import glob
import json
import math
import os

import torch
from tokenizers import Tokenizer

from device import get_device
from model import Config, TinyLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")


def score_holdout(model, tok, cfg, device, path):
    rows = [json.loads(l)["text"] for l in open(path, encoding="utf-8") if l.strip()]
    tot_lp = 0.0
    tot_tok = 0
    for text in rows:
        ids = tok.encode(text).ids[: cfg.seq_len]
        if len(ids) < 8:
            continue
        t = torch.tensor([ids], device=device)
        with torch.no_grad():
            logits, _ = model(t[:, :-1])
        lp = torch.log_softmax(logits.float(), dim=-1)
        tok_lp = lp.gather(-1, t[:, 1:].unsqueeze(-1)).squeeze(-1)
        tot_lp += tok_lp.sum().item()
        tot_tok += tok_lp.numel()
    nll = -tot_lp / max(1, tot_tok)
    return len(rows), tot_tok, nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()

    device = get_device()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = TinyLM(cfg).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))

    holdout_paths = sorted(glob.glob(os.path.join(NPC, "*_holdout.jsonl")))
    if not holdout_paths:
        print("no *_holdout.jsonl files found in data/npc/")
        return

    for path in holdout_paths:
        name = os.path.basename(path)
        n_rows, tot_tok, nll = score_holdout(model, tok, cfg, device, path)
        if tot_tok == 0:
            print(f"{name}: 0 rows scored (empty or all-too-short), skipped")
            continue
        print(f"{name}: rows {n_rows} | tokens scored {tot_tok} | "
              f"teacher-forced ppl {math.exp(nll):.2f} (mean nll {nll:.4f})")


if __name__ == "__main__":
    main()
