"""Phase 3: tournament/population self-play, generation only -- no
training happens in this script. Per the plan's binding decision
(population/tournament style, not simple oracle-match rejection sampling
or single-episode reward): sample K rollout variants per agent-turn at
varying temperature, fork the world state K ways, advance a short
horizon, score every fork with a real measured fitness function, and
keep the top fraction of branches as candidate SFT rows.

Fitness is every term a direct, cheap, checkable state-delta read off
SurvivalWorld/Agent -- no LLM judge, no random.uniform, per AGENTS.md's
anti-fake-metric discipline (a past notebook-era incident faked metrics
this way and is a permanent cautionary tale in this project).

This does not replace npc_forge.py or npc_action_forge.py -- it is a new,
additional flywheel source (data/npc/st_survival.jsonl) feeding the same
st_prepare.py mixture pattern every other source already uses.
"""

import argparse
import copy
import json
import os
import random

import torch
from tokenizers import Tokenizer

from model import Config, TinyLM
from device import get_device
from npc_score import parse_action
from sim_world import SurvivalWorld
from st_world import PLACES, TRAVEL_GRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")
OUT = os.path.join(NPC, "st_survival.jsonl")

VERBS = ("TRAVEL", "ATTACK", "TALK", "USE", "WAIT")


def gen_one(model, ids, tokens, temperature, top_k, device):
    with torch.no_grad():
        for _ in range(tokens):
            logits, _ = model(ids)
            z = logits[:, -1, :] / temperature
            th = z.topk(top_k, dim=-1).values[:, -1:]
            z = z.masked_fill(z < th, float("-inf"))
            nxt = torch.multinomial(torch.softmax(z, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
    return ids


def resolve_action(world, agent, action):
    """Applies a parsed action to a forked world's agent. Invalid/absent
    actions are engine-tolerated no-ops, same pattern as GOTO/DEAL bad
    args elsewhere in this project -- the model's own mistakes cost it
    fitness (no progress that turn), not a crash."""
    if action is None:
        return "wait"
    verb, arg, _ = action
    if verb == "TRAVEL" and arg in TRAVEL_GRAPH.get(agent.place, {}):
        res = world.resolve_travel(agent, arg)
        return "travel" if res["arrived"] else "wait"
    if verb == "ATTACK" and arg:
        target = next((o for o in world.agents_at(agent.place) if o.name == arg), None)
        if target:
            world.resolve_attack(agent, target)
            return "attack"
        return "wait"
    if verb == "USE" and arg:
        trade_res = world.resolve_trade(agent, arg)
        if trade_res["bought"]:
            return "trade"
        forage_res = world.resolve_forage(agent)
        return "use" if forage_res["gained"] else "wait"
    if verb == "USE":
        res = world.resolve_forage(agent)
        return "use" if res["gained"] else "wait"
    if verb == "TALK":
        return "talk"
    return "wait"


def fitness_of(branch):
    """Every term a direct state-delta count, never a judged score."""
    f = branch["ticks_survived"]
    f += 2 * branch["trades_completed"]
    f += 2 * branch["combats_survived"]
    f += 1 * branch["new_places_reached"]
    if branch["died"]:
        f -= 5
    return f


def run_branch(model, tok, world, agent_name, horizon, temperature, tokens, device):
    """Advances one forked world by `horizon` agent-turns for the named
    agent, alternating other agents on a fixed round-robin so the world
    keeps moving around the branch's protagonist. Returns a fitness
    record and the list of rendered (prompt, completion) turns for
    injection if this branch survives selection."""
    turns = []
    ticks_survived = 0
    trades_completed = 0
    combats_survived = 0
    places_seen = {world.agents[agent_name].place}
    died = False

    other_names = [n for n in world.agents if n != agent_name]
    for step in range(horizon):
        agent = world.agents.get(agent_name)
        if agent is None or not agent.alive:
            died = True
            break
        world.tick()
        ticks_survived += 1
        if not agent.alive:
            died = True
            break
        prompt = world.render_turn(agent)
        ids = torch.tensor([tok.encode(prompt).ids], device=device)
        plen = ids.shape[1]
        out = gen_one(model, ids, tokens, temperature, 40, device)
        text = tok.decode(out[0, plen:].tolist())
        stops = [c for c in (text.find("\nPlayer:"), text.find("Player:")) if c >= 0]
        if stops:
            text = text[: min(stops)]
        lines = [l for l in text.strip().split("\n") if l.strip()]
        action_lines = [l for l in lines if l.strip().startswith("[")]
        action = parse_action(action_lines[0]) if action_lines else None
        hp_before = agent.hp
        kind = resolve_action(world, agent, action)
        if kind == "attack" and agent.alive and agent.hp >= hp_before - 3:
            combats_survived += 1
        elif kind == "travel":
            places_seen.add(agent.place)
        elif kind == "trade":
            trades_completed += 1
        turns.append({"prompt": prompt, "completion": text.strip()})
        # give other agents in the world one turn each too, on a simple
        # scripted needs-greedy policy (not the model under test), so the
        # world keeps moving and forage/trade competition is real
        for other_name in other_names:
            other = world.agents.get(other_name)
            if other and other.alive:
                if other.hunger < 60 or other.thirst < 60:
                    world.resolve_forage(other)
    return {
        "agent": agent_name,
        "ticks_survived": ticks_survived,
        "trades_completed": trades_completed,
        "combats_survived": combats_survived,
        "new_places_reached": len(places_seen) - 1,
        "died": died,
        "turns": turns,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--roster", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=8, help="horizon per branch")
    ap.add_argument("--k", type=int, default=4, help="rollout variants per agent")
    ap.add_argument("--keep-frac", type=float, default=0.25)
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--out", default=OUT)
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
    base_world = SurvivalWorld(rng)
    places = [p[0] for p in PLACES]
    roster = [base_world.spawn_agent(rng.choice(places)) for _ in range(args.roster)]
    roster_names = [a.name for a in roster]

    temperatures = [0.5, 0.7, 0.9, 1.1][: args.k] or [0.7]
    while len(temperatures) < args.k:
        temperatures.append(temperatures[-1])

    all_branches = []
    branch_seed = args.seed
    for agent_name in roster_names:
        for temp in temperatures:
            forked = copy.deepcopy(base_world)
            # deepcopy also copies base_world.rng's exact state -- every
            # fork would otherwise draw the identical random sequence
            # (tick() events, travel encounter rolls), making branches
            # indistinguishable regardless of temperature. Re-seed each
            # fork's rng distinctly so forks genuinely diverge.
            branch_seed += 1
            forked.rng = random.Random(branch_seed)
            forked.econ.rng = forked.rng
            forked.names.rng = forked.rng
            branch = run_branch(model, tok, forked, agent_name, args.ticks, temp,
                                 args.tokens, device)
            branch["temperature"] = temp
            branch["fitness"] = fitness_of(branch)
            all_branches.append(branch)

    all_branches.sort(key=lambda b: b["fitness"], reverse=True)
    n_keep = max(1, int(len(all_branches) * args.keep_frac))
    survivors = all_branches[:n_keep]

    n_rows = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for b in survivors:
            for t in b["turns"]:
                if len(t["completion"]) < 8:
                    continue
                convo = f"{t['prompt']} {t['completion']}\nPlayer:\n"
                f.write(json.dumps({"text": convo}) + "\n")
                n_rows += 1

    fits = [b["fitness"] for b in all_branches]
    print(f"=== tournament ({len(all_branches)} branches, roster={args.roster}, "
          f"k={args.k}, horizon={args.ticks}) ===")
    print(f"fitness: min {min(fits)} max {max(fits)} mean {sum(fits)/len(fits):.1f} "
          f"spread {max(fits) - min(fits)}")
    print(f"survivors kept: {len(survivors)}/{len(all_branches)} "
          f"({args.keep_frac:.0%} target)")
    print(f"rows written: {n_rows} to {args.out}")
    died = sum(1 for b in all_branches if b["died"])
    print(f"branches that died: {died}/{len(all_branches)}")


if __name__ == "__main__":
    main()
