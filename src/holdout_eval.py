"""Held-out real-data generalization gate: teacher-forced ppl on PIPPA holdout.

data/npc/pippa_holdout.jsonl never enters the training bins (pippa_convert.py),
so teacher-forced perplexity on it measures real-data generalization rather
than mixture memorization. Run per checkpoint after a round's SFT stage and
compare against the ship checkpoint:

  UV_NO_SYNC=1 uv run python src/holdout_eval.py runs/ple-st-r19-s0-best.pt
"""

import argparse
import json
import math
import os

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ck["cfg"])
    model = TinyLM(cfg).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))

    path = os.path.join(DATA, "npc", "pippa_holdout.jsonl")
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
    print(f"holdout rows: {len(rows)} | tokens scored: {tot_tok}")
    print(f"teacher-forced ppl: {math.exp(nll):.2f} (mean nll {nll:.4f})")


if __name__ == "__main__":
    main()
