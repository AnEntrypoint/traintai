"""The action forge: rejection-sample the model's own oracle-correct actions.

npc_forge.py rejection-samples on dialog quality (identity/sale/quest/lore/
object/place probes graded by rule-based flaw checks). This is the same
generate-grade-inject loop but for the open lever named in AGENTS.md:
action accuracy (GOTO/DEAL/BUY) is a measured-dead lever for GRPO reward
shaping (r17-r22) -- the remaining path is data-side, rejection-sampling
the model's own VALID oracle-matching actions into the flywheel.

Each cycle:
  1. GENERATE: sample sim_econ.py training-distribution scenarios (same
     seed family as st_sim.jsonl, NOT the held-out sim_scenarios.jsonl eval
     set -- that must stay disjoint) and let the model complete them.
  2. GRADE: oracle_ok() from npc_score.py, the exact same check sim_eval.py
     uses, so "passes the forge" and "passes the eval" mean the same thing.
  3. INJECT: only exact oracle matches become SFT rows (st_action_forge.jsonl);
     everything else is dropped -- the model's own mistakes are not useful
     supervision for a lever that failed via reward shaping already.
"""

import argparse
import json
import os
import random

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM
from device import get_device
from npc_score import oracle_ok, parse_action

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")


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
    ap.add_argument("--scenarios", type=int, default=800)
    ap.add_argument("--k", type=int, default=4, help="samples per scenario")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=97, help="disjoint from st_sim.jsonl (41) and sim_scenarios.jsonl (1041)")
    ap.add_argument("--out", default=os.path.join(NPC, "st_action_forge.jsonl"))
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyLM(Config(**ck["cfg"])).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))

    from sim_econ import World, generate, convo_to_scenario
    rng = random.Random(args.seed)
    world = World(rng)
    convos = generate(rng, args.scenarios, world=world, business=False)
    scenarios = [s for s in (convo_to_scenario(c) for c in convos) if s is not None]

    n_gen = n_hit = 0
    kind_hit = {"none": 0, "GOTO": 0, "DEAL": 0, "BUY": 0}
    kind_tot = {"none": 0, "GOTO": 0, "DEAL": 0, "BUY": 0}
    with open(args.out, "a", encoding="utf-8") as f_out:
        for s in scenarios:
            ids = torch.tensor([tok.encode(s["prompt"]).ids], device=device)
            plen = ids.shape[1]
            out = gen(model, ids.repeat(args.k, 1), args.tokens, args.temperature, 40)
            oracle = s["oracle_action"]
            kind = "none" if oracle is None else ("GOTO" if oracle.startswith("[GOTO") else ("BUY" if oracle.startswith("[BUY") else "DEAL"))
            for row in out:
                text = tok.decode(row[plen:].tolist())
                stops = [c for c in (text.find("\nPlayer:"), text.find("Player:")) if c >= 0]
                if stops:
                    text = text[: min(stops)]
                lines = [l for l in text.strip().split("\n") if l.strip()]
                action_lines = [l for l in lines if l.strip().startswith("[")]
                action = parse_action(action_lines[0]) if action_lines else None
                n_gen += 1
                kind_tot[kind] += 1
                if len(action_lines) <= 1 and oracle_ok(oracle, action):
                    n_hit += 1
                    kind_hit[kind] += 1
                    convo = f"{s['prompt']} {text.strip()}\nPlayer:\n"
                    f_out.write(json.dumps({"text": convo}) + "\n")

    print(f"=== action forge ({n_gen} rollouts over {len(scenarios)} scenarios, k={args.k}) ===")
    print(f"oracle-match injected: {n_hit}/{n_gen} = {n_hit / max(1, n_gen):.0%}")
    for k in kind_tot:
        if kind_tot[k]:
            print(f"  {k:5s}: {kind_hit[k]}/{kind_tot[k]} = {kind_hit[k] / kind_tot[k]:.0%}")
    print(f"wrote to {args.out}")


if __name__ == "__main__":
    main()
