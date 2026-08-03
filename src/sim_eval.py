"""Simulation eval: format adherence + oracle-choice accuracy, held out.

Runs the model on the sim_econ.py --scenarios set (seed disjoint from the
training data) and measures, separately per the research discipline (dialog
quality and command correctness regress differently):

  format_rate    output is dialog lines plus at most one [VERB ...] line
  beats_rate     outputs containing *action beats* (dialog-only target: 0)
  abstain_acc    oracle says no action -> model emits none
  goto_acc       oracle [GOTO: p] -> model emits exactly that place
  deal_acc       oracle [DEAL: i p] -> same item and price within +-30%
  invalid_action emitted action with unknown place/item or bad syntax

These are the simulation-format adherence numbers the training loop must
move -- respond vs interact distinction, not just dialog quality.
"""

import argparse
import json
import os
import re

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM
from npc_score import ACTION_RE, oracle_ok, parse_action
from st_world import ITEMS, PLACES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")

PLACE_NAMES = {p[0] for p in PLACES}
ITEM_NAMES = {i[0] for items in ITEMS.values() for i in items}


def gen(model, ids, tokens, temperature, top_k):
    outs = []
    with torch.no_grad():
        for _ in range(tokens):
            logits, _ = model(ids)
            z = logits[:, -1, :] / temperature
            th = z.topk(top_k, dim=-1).values[:, -1:]
            z = z.masked_fill(z < th, float("-inf"))
            nxt = torch.multinomial(torch.softmax(z, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
            outs.append(nxt)
    return torch.cat(outs, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))

    scenarios = [json.loads(l) for l in open(os.path.join(NPC, "sim_scenarios.jsonl"), encoding="utf-8") if l.strip()]
    if args.limit:
        scenarios = scenarios[: args.limit]

    n = fmt = beats = acts = invalid = oracle_hits = 0
    price_n = price_ok = 0
    by_kind = {"none": [0, 0], "GOTO": [0, 0], "DEAL": [0, 0]}
    samples = []
    for s in scenarios:
        ids = torch.tensor([tok.encode(s["prompt"]).ids], device=device)
        out = gen(model, ids, args.tokens, args.temperature, 40)
        text = tok.decode(out[0].tolist())
        stops = [c for c in (text.find("\nPlayer:"), text.find("Player:")) if c >= 0]
        if stops:
            text = text[: min(stops)]
        lines = [l for l in text.strip().split("\n") if l.strip()]
        action_lines = [l for l in lines if l.strip().startswith("[")]
        action = parse_action(action_lines[0]) if action_lines else None
        oracle = s["oracle_action"]
        if oracle and oracle.startswith("[DEAL:"):
            oparts = oracle[7:-1].rsplit(" ", 1)
            oprice = int(oparts[1])
            quoted = re.findall(r"(\d+)\s*gold", text)
            if quoted:
                price_n += 1
                price_ok += any(abs(int(q) - oprice) <= 0.3 * oprice for q in quoted)
        n += 1
        if len(action_lines) <= 1 and (not action_lines or action is not None):
            fmt += 1
        if re.search(r"\*[^*]+\*", text):
            beats += 1
        if action_lines:
            acts += 1
            if action is None:
                invalid += 1
            elif action[0] == "GOTO" and action[1] not in PLACE_NAMES:
                invalid += 1
            elif action[0] == "DEAL" and action[1] not in ITEM_NAMES:
                invalid += 1
        kind = "none" if oracle is None else ("GOTO" if oracle.startswith("[GOTO") else "DEAL")
        ok = oracle_ok(oracle, action)
        by_kind[kind][0] += ok
        by_kind[kind][1] += 1
        oracle_hits += ok
        if len(samples) < 6:
            samples.append((s["keeper"], s["prompt"].rsplit("Player:", 1)[-1].strip()[:40],
                            oracle, text.strip().replace("\n", " / ")[:110], ok))

    print(f"scenarios: {n}")
    print(f"format rate      : {fmt}/{n} = {fmt / n:.0%}")
    print(f"action-beats rate: {beats}/{n} = {beats / n:.0%}  (dialog-only target: 0)")
    print(f"action rate      : {acts}/{n} = {acts / n:.0%}")
    print(f"invalid actions  : {invalid}/{max(1, acts)} emitted")
    print(f"oracle match     : {oracle_hits}/{n} = {oracle_hits / n:.0%}")
    if price_n:
        print(f"price fidelity   : {price_ok}/{price_n} quoted prices within 30% of oracle")
    for k, (h, t) in by_kind.items():
        if t:
            print(f"  {k:5s}: {h}/{t} = {h / t:.0%}")
    for keeper, q, oracle, text, ok in samples:
        print(f"  [{'OK' if ok else 'X '}] {keeper} q={q!r} oracle={oracle} :: {text!r}")


if __name__ == "__main__":
    main()
