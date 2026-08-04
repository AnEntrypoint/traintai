"""Phase 3: tournament/population self-play, generation only -- no
training happens in this script. Per the plan's binding decision
(population/tournament style, not simple oracle-match rejection sampling
or single-episode reward): sample K rollout variants per agent-turn at
varying temperature, fork the world state K ways, advance all forks in
BATCHED LOCKSTEP (one tick at a time, one GPU forward pass per tick
covering every still-alive branch), score every fork with a real measured
fitness function, and keep the top fraction of branches as candidate SFT
rows.

Batched lockstep (not per-branch-sequential) is the real performance
lever at this model's tiny size: forks are independent until they
complete, so tick N of every branch can share one GPU call instead of
480 sequential single-sample calls (12 branches x 40 ticks, measured in
this session's un-batched testing). This relies on model.py's attn_mask
support (build_causal_padding_mask) so prompts of different token
lengths -- which they always are, agent names/hunger/thirst/place text
vary every turn -- can share a batch without corrupting attention.

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

from model import Config, TinyLM, build_causal_padding_mask
from device import get_device
from npc_score import parse_action
from sim_world import SurvivalWorld
from st_world import PLACES, TRAVEL_GRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
NPC = os.path.join(DATA, "npc")
OUT = os.path.join(NPC, "st_survival.jsonl")

VERBS = ("TRAVEL", "ATTACK", "TALK", "USE", "WAIT")


class Branch:
    """One forked world + the protagonist agent under test in it, tracked
    through the batched lockstep loop."""

    def __init__(self, agent_name, world, temperature):
        self.agent_name = agent_name
        self.world = world
        self.temperature = temperature
        self.other_names = [n for n in world.agents if n != agent_name]
        self.turns = []
        self.ticks_survived = 0
        self.trades_completed = 0
        self.combats_survived = 0
        self.places_seen = {world.agents[agent_name].place}
        self.died = False
        self.alive = True

    def agent(self):
        return self.world.agents.get(self.agent_name)


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


def fitness_of(b):
    """Every term a direct state-delta count, never a judged score."""
    f = b.ticks_survived
    f += 2 * b.trades_completed
    f += 2 * b.combats_survived
    f += 1 * len(b.places_seen) - 1
    if b.died:
        f -= 5
    return f


def batched_generate(model, prompt_ids_list, tokens, temperatures, top_k, device):
    """One batched autoregressive generation over N prompts of different
    lengths, left-padded to the batch max and masked via
    build_causal_padding_mask so shorter prompts don't attend to pad
    tokens. Each row samples at its own branch's temperature. Returns a
    list of new-token-id lists, one per input prompt, stopping each row
    independently once it would exceed `tokens` new tokens (all rows run
    the same fixed number of steps; per-row completions are simply
    truncated to their own budget, no early-stop optimization -- correct
    over premature, matches the simplicity of every other generation loop
    in this project)."""
    n = len(prompt_ids_list)
    max_len = max(len(p) for p in prompt_ids_list)
    pad_id = 0
    batch = torch.full((n, max_len), pad_id, dtype=torch.long, device=device)
    pad_mask = torch.zeros(n, max_len, dtype=torch.bool, device=device)
    for i, p in enumerate(prompt_ids_list):
        # left-pad: real content ends at max_len, so every row's "current
        # position" for next-token prediction is the same column index
        offset = max_len - len(p)
        batch[i, offset:] = torch.tensor(p, dtype=torch.long, device=device)
        pad_mask[i, offset:] = True

    temps = torch.tensor(temperatures, device=device).view(n, 1)
    generated = [[] for _ in range(n)]
    with torch.no_grad():
        for _ in range(tokens):
            mask = build_causal_padding_mask(pad_mask)
            logits, _ = model(batch, attn_mask=mask)
            z = logits[:, -1, :] / temps
            th = z.topk(top_k, dim=-1).values[:, -1:]
            z = z.masked_fill(z < th, float("-inf"))
            nxt = torch.multinomial(torch.softmax(z, dim=-1), 1)
            for i in range(n):
                generated[i].append(nxt[i, 0].item())
            batch = torch.cat([batch, nxt], dim=1)
            pad_mask = torch.cat([pad_mask, torch.ones(n, 1, dtype=torch.bool, device=device)], dim=1)
    return generated


def run_tournament(model, tok, branches, horizon, tokens, device):
    """Advances every branch in lockstep: each tick, batch every
    still-alive branch's rendered prompt into one GPU call, decode, and
    resolve actions independently per branch."""
    for step in range(horizon):
        live = [b for b in branches if b.alive]
        if not live:
            break
        for b in live:
            b.world.tick()
            b.ticks_survived += 1
            agent = b.agent()
            if agent is None or not agent.alive:
                b.died = True
                b.alive = False

        live = [b for b in branches if b.alive]
        if not live:
            break

        prompts = [b.world.render_turn(b.agent()) for b in live]
        prompt_ids = [tok.encode(p).ids for p in prompts]
        temps = [b.temperature for b in live]
        generated = batched_generate(model, prompt_ids, tokens, temps, 40, device)

        for b, prompt, gen_ids in zip(live, prompts, generated):
            text = tok.decode(gen_ids)
            stops = [c for c in (text.find("\nPlayer:"), text.find("Player:")) if c >= 0]
            if stops:
                text = text[: min(stops)]
            lines = [l for l in text.strip().split("\n") if l.strip()]
            action_lines = [l for l in lines if l.strip().startswith("[")]
            action = parse_action(action_lines[0]) if action_lines else None
            agent = b.agent()
            hp_before = agent.hp
            kind = resolve_action(b.world, agent, action)
            if kind == "attack" and agent.alive and agent.hp >= hp_before - 3:
                b.combats_survived += 1
            elif kind == "travel":
                b.places_seen.add(agent.place)
            elif kind == "trade":
                b.trades_completed += 1
            b.turns.append({"prompt": prompt, "completion": text.strip()})
            # scripted needs-greedy background policy for other agents in
            # this branch's world, same as the sequential version
            for other_name in b.other_names:
                other = b.world.agents.get(other_name)
                if other and other.alive and (other.hunger < 60 or other.thirst < 60):
                    b.world.resolve_forage(other)


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

    branches = []
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
            branches.append(Branch(agent_name, forked, temp))

    run_tournament(model, tok, branches, args.ticks, args.tokens, device)

    for b in branches:
        b.fitness = fitness_of(b)
    branches.sort(key=lambda b: b.fitness, reverse=True)
    n_keep = max(1, int(len(branches) * args.keep_frac))
    survivors = branches[:n_keep]

    n_rows = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for b in survivors:
            for t in b.turns:
                if len(t["completion"]) < 8:
                    continue
                convo = f"{t['prompt']} {t['completion']}\nPlayer:\n"
                f.write(json.dumps({"text": convo}) + "\n")
                n_rows += 1

    fits = [b.fitness for b in branches]
    print(f"=== tournament ({len(branches)} branches, roster={args.roster}, "
          f"k={args.k}, horizon={args.ticks}, batched lockstep) ===")
    print(f"fitness: min {min(fits)} max {max(fits)} mean {sum(fits)/len(fits):.1f} "
          f"spread {max(fits) - min(fits)}")
    print(f"survivors kept: {len(survivors)}/{len(branches)} "
          f"({args.keep_frac:.0%} target)")
    print(f"rows written: {n_rows} to {args.out}")
    died = sum(1 for b in branches if b.died)
    print(f"branches that died: {died}/{len(branches)}")


if __name__ == "__main__":
    main()
