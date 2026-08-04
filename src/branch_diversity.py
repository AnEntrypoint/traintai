"""Measured branch-diversity check for round.py's --branches output.

A real, checkable behavioral-distance metric between N branch
checkpoints: run every checkpoint greedily (temperature effectively 0,
deterministic) against the SAME fixed probe set of survival-sim turns,
and measure the fraction of probes where any two checkpoints' first
generated action-line (or dialog, if no action) differs. Mean pairwise
divergence near 0 means the branches have collapsed to functionally
identical policies despite distinct --seed values -- the exact failure
mode round.py's per-branch --seed diversity source is meant to prevent,
now actually measured rather than assumed.

Not a training or selection script -- read-only comparison, printed as a
dashboard in the same style as npc_forge.py's flaw histogram.
"""

import argparse
import os
import random

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM
from device import get_device
from sim_world import SurvivalWorld
from st_world import PLACES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

N_PROBES = 20
PROBE_SEED = 5  # fixed, disjoint from every round's own seeds, so every checkpoint sees the identical probe set


def build_probes():
    """A fixed, reproducible set of N_PROBES rendered turns (same shape
    sim_baseline.py uses), generated once with a seed no training/
    tournament run ever uses, so every checkpoint compared is judged
    against literally the same inputs."""
    rng = random.Random(PROBE_SEED)
    w = SurvivalWorld(rng)
    places = [p[0] for p in PLACES]
    agents = [w.spawn_agent(rng.choice(places)) for _ in range(6)]
    probes = []
    for _ in range(N_PROBES):
        w.tick()
        a = rng.choice([ag for ag in agents if ag.alive] or agents)
        probes.append(w.render_turn(a))
    return probes


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="two or more checkpoint paths to compare")
    ap.add_argument("--tokens", type=int, default=48)
    args = ap.parse_args()

    if len(args.ckpts) < 2:
        raise SystemExit("need at least 2 checkpoints to measure diversity between")

    device = get_device()
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))
    probes = build_probes()

    # Keyed by index, not basename -- two different checkpoint PATHS can
    # share a basename in different runs/ subdirs, and comparing the same
    # path against itself (a legitimate sanity check) must not silently
    # collapse to one dict entry and divide by zero pairs.
    names = [f"{i}:{os.path.basename(p)}" for i, p in enumerate(args.ckpts)]
    completions = {}
    for name, path in zip(names, args.ckpts):
        model = load_checkpoint(path, device)
        completions[name] = [greedy_complete(model, tok, p, args.tokens, device) for p in probes]
        del model

    pair_divergence = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            diffs = sum(1 for ca, cb in zip(completions[a], completions[b]) if ca != cb)
            pair_divergence[(a, b)] = diffs / len(probes)

    mean_divergence = sum(pair_divergence.values()) / len(pair_divergence)
    print(f"=== branch diversity ({len(names)} checkpoints, {N_PROBES} fixed probes, greedy decode) ===")
    for (a, b), d in sorted(pair_divergence.items(), key=lambda kv: kv[1]):
        print(f"  {a} vs {b}: {d:.0%} of probes diverge")
    print(f"mean pairwise divergence: {mean_divergence:.0%}")
    if mean_divergence < 0.05:
        print("COLLAPSE WARNING: branches are near-identical on this probe set despite distinct seeds -- "
              "the --seed diversity source is not producing behaviorally distinct policies at this step count. "
              "Consider more SFT/GRPO steps per branch, or a genuinely different data mix per branch, not just seed.")


if __name__ == "__main__":
    main()
