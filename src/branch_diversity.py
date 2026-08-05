"""Measured branch-diversity check for round.py's --branches output.

A real, checkable behavioral-distance metric between N branch
checkpoints: run every checkpoint greedily (temperature effectively 0,
deterministic) against the SAME fixed probe set of real BALROG game
observations, and measure the fraction of probes where any two
checkpoints' first generated completion differs. Mean pairwise
divergence near 0 means the branches have collapsed to functionally
identical policies despite distinct --seed values -- the exact failure
mode round.py's per-branch --seed diversity source is meant to prevent,
now actually measured rather than assumed.

Not a training or selection script -- read-only comparison, printed as a
dashboard in the same style as npc_forge.py's flaw histogram.

Probes come from probes.json, a real dump of BALROG environment
observations (see balrog-smoke/dump_probes.py, which must be run inside
a BALROG kernel/environment -- BALROG's own gym==0.23 pin conflicts with
this repo's numpy>=2.5.1, so the two never share a Python environment,
per this session's explicit architecture decision). No hardcoded/
hand-authored probe text -- if probes.json is missing, this fails loudly
rather than falling back to synthetic strings.
"""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM
from device import get_device

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PROBES_PATH = os.path.join(HERE, "..", "data", "balrog_probes.json")


def build_probes():
    """Loads the fixed real-BALROG-observation probe set dumped by
    balrog-smoke/dump_probes.py. Every checkpoint compared is judged
    against literally the same real game states -- raises if the dump
    hasn't been generated yet, rather than silently falling back to
    synthetic text."""
    if not os.path.exists(PROBES_PATH):
        raise FileNotFoundError(
            f"{PROBES_PATH} not found -- run balrog-smoke/dump_probes.py inside a BALROG "
            "environment first and copy its probes.json output here as data/balrog_probes.json."
        )
    with open(PROBES_PATH, encoding="utf-8") as f:
        return json.load(f)


def greedy_complete(model, tok, prompt, tokens, device):
    ids = torch.tensor([tok.encode(prompt).ids], device=device)
    with torch.no_grad():
        for _ in range(tokens):
            logits, _ = model(ids)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
    text = tok.decode(ids[0, -tokens:].tolist())
    stops = [c for c in (text.find("\nPlayer:"), text.find("Player:")) if c >= 0]
    return text[: min(stops)].strip() if stops else text.strip()


def load_checkpoint(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model


def measure_diversity(ckpt_paths, device=None, tok=None, tokens=48):
    """Reusable core: returns (pair_divergence dict, mean_divergence
    float) for 2+ checkpoint paths. Importable by round.py's generation
    loop so collapse detection runs automatically every generation, not
    only via the standalone CLI. Keyed by index, not basename -- two
    different checkpoint PATHS can share a basename in different runs/
    subdirs, and comparing the same path against itself (a legitimate
    sanity check) must not silently collapse to one dict entry and
    divide by zero pairs."""
    if len(ckpt_paths) < 2:
        raise ValueError("need at least 2 checkpoints to measure diversity between")
    device = device or get_device()
    tok = tok or Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))
    probes = build_probes()

    names = [f"{i}:{os.path.basename(p)}" for i, p in enumerate(ckpt_paths)]
    completions = {}
    for name, path in zip(names, ckpt_paths):
        model = load_checkpoint(path, device)
        completions[name] = [greedy_complete(model, tok, p, tokens, device) for p in probes]
        del model

    pair_divergence = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            diffs = sum(1 for ca, cb in zip(completions[a], completions[b]) if ca != cb)
            pair_divergence[(a, b)] = diffs / len(probes)

    mean_divergence = sum(pair_divergence.values()) / len(pair_divergence)
    return pair_divergence, mean_divergence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="two or more checkpoint paths to compare")
    ap.add_argument("--tokens", type=int, default=48)
    args = ap.parse_args()

    if len(args.ckpts) < 2:
        raise SystemExit("need at least 2 checkpoints to measure diversity between")

    pair_divergence, mean_divergence = measure_diversity(args.ckpts, tokens=args.tokens)
    print(f"=== branch diversity ({len(args.ckpts)} checkpoints, {len(build_probes())} real BALROG probes, greedy decode) ===")
    for (a, b), d in sorted(pair_divergence.items(), key=lambda kv: kv[1]):
        print(f"  {a} vs {b}: {d:.0%} of probes diverge")
    print(f"mean pairwise divergence: {mean_divergence:.0%}")
    if mean_divergence < 0.05:
        print("COLLAPSE WARNING: branches are near-identical on this probe set despite distinct seeds -- "
              "the --seed diversity source is not producing behaviorally distinct policies at this step count. "
              "Consider more SFT/GRPO steps per branch, or a genuinely different data mix per branch, not just seed.")


if __name__ == "__main__":
    main()
