"""Phase 2 baseline: measure the current checkpoint's emission/validity
rates on the new survival verbs (TRAVEL/ATTACK/TALK/USE/WAIT) before any
tournament-generated training data exists.

This is the r16-baseline-equivalent for the survival-sim arc: every later
phase's "did it help" claim is measured against the numbers this script
prints, the same way the GOTO/DEAL/BUY action-accuracy arc was anchored to
r16's "GOTO 9%/DEAL 5%, abstention 71%" starting point (AGENTS.md).

Not a training script -- pure measurement, single-rollout (no selection,
no injection into any .jsonl).
"""

import argparse
import os
import random

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM
from device import get_device
from npc_score import ACTION_RE, parse_action
from sim_world import SurvivalWorld
from st_world import PLACES, TRAVEL_GRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

VALID_VERBS = ("TRAVEL", "ATTACK", "TALK", "USE", "WAIT")


def gen(model, ids, tokens, temperature, top_k, device):
    with torch.no_grad():
        for _ in range(tokens):
            logits, _ = model(ids)
            z = logits[:, -1, :] / temperature
            th = z.topk(top_k, dim=-1).values[:, -1:]
            z = z.masked_fill(z < th, float("-inf"))
            nxt = torch.multinomial(torch.softmax(z, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--turns", type=int, default=200)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))

    rng = random.Random(args.seed)
    w = SurvivalWorld(rng)
    places = [p[0] for p in PLACES]
    agents = [w.spawn_agent(rng.choice(places)) for _ in range(8)]

    n = 0
    emitted = 0
    valid = 0
    place_names = {p[0] for p in PLACES}
    by_verb = {v: {"emitted": 0, "valid": 0} for v in VALID_VERBS}
    no_action = 0
    samples = []

    for i in range(args.turns):
        w.tick()
        a = rng.choice([ag for ag in agents if ag.alive] or agents)
        prompt = w.render_turn(a)
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        plen = ids.shape[1]
        out = gen(model, ids, args.tokens, args.temperature, 40, device)
        text = tok.decode(out[0, plen:].tolist())
        stops = [c for c in (text.find("\nPlayer:"), text.find("Player:")) if c >= 0]
        if stops:
            text = text[: min(stops)]
        lines = [l for l in text.strip().split("\n") if l.strip()]
        action_lines = [l for l in lines if l.strip().startswith("[")]
        n += 1
        if not action_lines:
            no_action += 1
            if len(samples) < 8:
                samples.append((a.name, "(no action)", None, text.strip().replace("\n", " / ")[:100]))
            continue
        action = parse_action(action_lines[0])
        emitted += 1
        if action is None:
            continue
        verb = action[0]
        if verb not in by_verb:
            continue  # emitted an old verb (GOTO/DEAL/BUY) -- not scored here
        by_verb[verb]["emitted"] += 1
        is_valid = False
        if verb == "TRAVEL":
            is_valid = action[1] in place_names and action[1] in TRAVEL_GRAPH.get(a.place, {})
        elif verb == "ATTACK":
            is_valid = action[1] is not None and any(o.name == action[1] for o in w.agents_at(a.place) if o.name != a.name)
        elif verb == "TALK":
            is_valid = action[1] is not None and any(o.name == action[1] for o in w.agents_at(a.place) if o.name != a.name)
        elif verb == "USE":
            is_valid = action[1] is not None
        elif verb == "WAIT":
            is_valid = True
        if is_valid:
            valid += 1
            by_verb[verb]["valid"] += 1
        if len(samples) < 8:
            samples.append((a.name, action_lines[0], is_valid, text.strip().replace("\n", " / ")[:100]))

    print(f"=== survival-verb baseline ({n} turns, ckpt={os.path.basename(args.ckpt)}) ===")
    print(f"no action (WAIT-equivalent/abstain): {no_action}/{n} = {no_action/n:.0%}")
    print(f"any action emitted: {emitted}/{n} = {emitted/n:.0%}")
    print(f"valid new-verb action: {valid}/{n} = {valid/n:.0%}")
    for v, d in by_verb.items():
        if d["emitted"]:
            print(f"  {v:8s}: emitted {d['emitted']:3d}, valid {d['valid']:3d} ({d['valid']/d['emitted']:.0%})")
        else:
            print(f"  {v:8s}: emitted 0")
    print()
    print("samples:")
    for name, act, ok, text in samples:
        mark = "-- " if ok is None else ("OK " if ok else "X  ")
        print(f"  [{mark}] {name} :: {act} :: {text!r}")


if __name__ == "__main__":
    main()
