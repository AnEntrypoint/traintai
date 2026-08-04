"""Multi-agent survival layer wrapping sim_econ.World.

Phase 1 of the survival-sim plan: core mechanics only, no LLM in the loop
yet (that is npc_action_forge.py-style Phase 2/3 work). This module adds
agents with hunger/thirst/HP/inventory/skills to the existing economy
sim, deliberately as a wrapper around sim_econ.World and st_world.py's
data tables rather than a fork -- shop stock/pricing/demand stays exactly
sim_econ.World.tick()'s existing logic; this module only adds the agent
layer on top (spawn/decay/combat/travel/forage), plus a narratable event
log in the same style as World.recent.

Deliberately minimal per the project's own prior rejection of "mana/combat
scope creep" (AGENTS.md): flat skill XP counters (no unlock trees), a
single deterministic combat roll (no multi-round exchanges, no elemental
types), magic deferred entirely (no new resource pool -- USE-item flavor
effects gated by the lore skill are a later phase, not built here).
"""

import json
import os
import random
from dataclasses import dataclass, field

from sim_econ import World
from st_world import ITEMS, PLACES, SHOPKEEPERS, TRAVEL_GRAPH

PLACE_DESC = {p[0]: p[1] for p in PLACES}

# One-line bio flavor per agent, cycled by index rather than stored per-
# Agent (keeps Agent itself free of static flavor text -- the bio is a
# rendering-time concern, not simulation state).
BIO_TEMPLATES = [
    "a wanderer who trusts a blade more than a promise",
    "a fallen scholar working off an old debt",
    "a trapper who knows every game trail for a day's walk",
    "a former guard who left the wall for reasons kept close",
    "a peddler between routes, counting coin twice",
]

# Fixed unique-name pool. Deliberately NOT drawn from st_data.py's
# synthetic_names.jsonl / st_conversations.jsonl pool -- those are the
# confirmed source of "Natasha Romanoff"/"Mike Isaac"-style contamination
# found this session (real people/character names leaking into training
# data). This pool is hand-authored, in-setting, and disjoint from the
# five named SHOPKEEPERS so a spawned agent is never confused for one.
NAME_POOL = [
    "Brenna Kest", "Otho Vane", "Sera Quill", "Fenrick Dell", "Ymma Stroud",
    "Cael Brix", "Odessa Marrow", "Tobin Ashe", "Livia Corr", "Grask Pell",
    "Wenna Tarr", "Dain Wolfe", "Petra Slate", "Rurik Fen", "Ilsa Bram",
    "Corvin Hale", "Meret Sil", "Ansel Dry", "Thora Vex", "Bram Ostler",
]

_KAGGLE_NAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "data", "npc", "kaggle_names.jsonl")


def load_name_pool():
    """Extends NAME_POOL with kaggle_names_convert.py's decontaminated
    output when present (data/npc/kaggle_names.jsonl -- 608 fantasy/goblin
    names from isaacbenge/fantasy-for-markov-generator, CC0-1.0). Falls
    back to the hand-authored NAME_POOL alone if the file hasn't been
    generated yet, so sim_world.py never hard-depends on the Kaggle pull
    having run."""
    if not os.path.exists(_KAGGLE_NAMES_PATH):
        return list(NAME_POOL)
    extra = []
    with open(_KAGGLE_NAMES_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                extra.append(json.loads(line)["name"])
            except (json.JSONDecodeError, KeyError):
                continue
    return list(NAME_POOL) + extra

HUNGER_DECAY = 3
THIRST_DECAY = 4
STARVE_DAMAGE = 2
FORAGE_HUNGER_GAIN = 15
FORAGE_THIRST_GAIN = 20
FORAGE_CAP_PER_PLACE = 3  # diminishing-returns commons: N free forages before a place is picked clean this "season"

SKILLS = ("blade", "herbalism", "lore", "haggle", "travel")


STARTING_GOLD = 40


@dataclass
class Agent:
    name: str
    place: str
    hp: int = 20
    hp_max: int = 20
    hunger: int = 100
    thirst: int = 100
    gold: int = STARTING_GOLD
    inventory: dict = field(default_factory=dict)
    skills: dict = field(default_factory=lambda: {s: 0 for s in SKILLS})
    alive: bool = True

    def is_starving(self):
        return self.hunger <= 0 or self.thirst <= 0


class NameRegistry:
    def __init__(self, rng, pool=None):
        self.rng = rng
        self.available = list(pool) if pool is not None else load_name_pool()
        self.rng.shuffle(self.available)
        self.taken = set()

    def pop(self):
        if not self.available:
            raise RuntimeError("name pool exhausted -- extend NAME_POOL")
        name = self.available.pop()
        self.taken.add(name)
        return name

    def release(self, name):
        self.taken.discard(name)


class SurvivalWorld:
    """Wraps sim_econ.World with an agent layer. econ is the existing
    shop/stock/demand simulation, untouched; this class only adds agents,
    their needs, and place-graph travel/foraging on top."""

    def __init__(self, rng):
        self.rng = rng
        self.econ = World(rng)
        self.names = NameRegistry(rng)
        self.agents = {}  # name -> Agent
        self.forage_used = {}  # (place, season_tick // 20) -> count
        self.recent = []  # shared narratable event log, same 6-entry cap as World.recent
        self.tick_n = 0

    def _log(self, msg):
        self.recent.append(msg)
        self.recent = self.recent[-6:]

    def spawn_agent(self, place):
        name = self.names.pop()
        a = Agent(name=name, place=place)
        self.agents[name] = a
        return a

    def agents_at(self, place):
        return [a for a in self.agents.values() if a.alive and a.place == place]

    # -- needs -----------------------------------------------------------

    def tick_needs(self, agent):
        if not agent.alive:
            return
        agent.hunger = max(0, agent.hunger - HUNGER_DECAY)
        agent.thirst = max(0, agent.thirst - THIRST_DECAY)
        if agent.is_starving():
            agent.hp -= STARVE_DAMAGE
            if agent.hp <= 0:
                agent.alive = False
                self._log(f"{agent.name} died of starvation at {agent.place}")

    # -- combat ------------------------------------------------------------

    def resolve_attack(self, attacker, defender):
        """One deterministic roll, one HP delta. hit_chance is a simple
        skill-discounted coin flip -- no multi-round exchange, no damage
        types, per the project's explicit prior rejection of combat scope
        creep. Returns a structured outcome dict, same shape discipline as
        sim_econ's convo_* functions returning structured (line, action)
        pairs."""
        if not attacker.alive or not defender.alive:
            return {"hit": False, "dmg": 0, "defender_hp_after": defender.hp, "fled": False, "reason": "target unavailable"}
        if defender.place != attacker.place:
            return {"hit": False, "dmg": 0, "defender_hp_after": defender.hp, "fled": False, "reason": "not present"}
        hit_chance = min(0.85, 0.45 + 0.03 * min(attacker.skills["blade"], 15))
        hit = self.rng.random() < hit_chance
        dmg = 0
        fled = False
        if hit:
            dmg = self.rng.randint(2, 6)
            defender.hp = max(0, defender.hp - dmg)
            attacker.skills["blade"] += 1
            if defender.hp <= 0:
                defender.alive = False
                self._log(f"{attacker.name} struck {defender.name} down at {attacker.place}")
            elif self.rng.random() < 0.2:
                fled = True
                defender.place = self.rng.choice(list(TRAVEL_GRAPH[defender.place]))
                self._log(f"{defender.name} fled {attacker.place} toward {defender.place} after {attacker.name}'s attack")
        else:
            self._log(f"{attacker.name} swung at {defender.name} and missed, at {attacker.place}")
        return {"hit": hit, "dmg": dmg, "defender_hp_after": defender.hp, "fled": fled, "reason": None}

    # -- travel ------------------------------------------------------------

    def shortest_path(self, src, dst):
        """BFS shortest path by tick-weight over TRAVEL_GRAPH (small,
        uniform-ish weights -- BFS with weight accumulation is exact here;
        a real Dijkstra would be needed only if weights varied enough for
        BFS's expansion order to matter, which this graph is too small
        for). Returns (path_list, total_ticks) or (None, None) if
        unreachable (should not happen post Phase-0 connectivity fix)."""
        if src == dst:
            return [src], 0
        frontier = [(src, [src], 0)]
        best = {src: 0}
        while frontier:
            frontier.sort(key=lambda t: t[2])
            cur, path, cost = frontier.pop(0)
            if cur == dst:
                return path, cost
            for nb, w in TRAVEL_GRAPH[cur].items():
                nc = cost + w
                if nb not in best or nc < best[nb]:
                    best[nb] = nc
                    frontier.append((nb, path + [nb], nc))
        return None, None

    def resolve_travel(self, agent, dest):
        """Advances the agent along the shortest path immediately (ticks
        are the sim's unit of both needs-decay and travel-cost, so a
        TRAVEL action consumes `cost` ticks of hunger/thirst decay up
        front rather than the caller looping tick-by-tick -- keeps one
        LLM turn == one resolved action, matching the Markovian per-turn
        rendering the 512-token budget requires). One encounter roll per
        edge traversed."""
        if dest not in TRAVEL_GRAPH:
            return {"arrived": False, "reason": "unknown place", "ticks": 0, "encounters": 0}
        path, cost = self.shortest_path(agent.place, dest)
        if path is None:
            return {"arrived": False, "reason": "unreachable", "ticks": 0, "encounters": 0}
        encounters = 0
        for _ in range(cost):
            self.tick_needs(agent)
            if not agent.alive:
                return {"arrived": False, "reason": "died en route", "ticks": cost, "encounters": encounters}
            if self.rng.random() < 0.08:
                encounters += 1
                agent.hp = max(0, agent.hp - self.rng.randint(1, 3))
                self._log(f"{agent.name} was waylaid on the road toward {dest}")
        agent.place = dest
        agent.skills["travel"] += 1
        self._log(f"{agent.name} reached {dest} after {cost} ticks on the road")
        return {"arrived": True, "reason": None, "ticks": cost, "encounters": encounters}

    # -- foraging / scarcity ------------------------------------------------

    def resolve_forage(self, agent):
        """Diminishing-returns commons: FORAGE_CAP_PER_PLACE free
        hunger/thirst top-ups per place per ~20-tick season, then
        foraging yields nothing (the commons is picked clean) -- this is
        the scarcity knob distinct from sim_econ's shop stock, satisfying
        "difficult atmosphere to train against" without a second economy
        engine."""
        season_key = (agent.place, self.tick_n // 20)
        used = self.forage_used.get(season_key, 0)
        if used >= FORAGE_CAP_PER_PLACE:
            return {"gained": False, "reason": "picked clean this season"}
        self.forage_used[season_key] = used + 1
        agent.hunger = min(100, agent.hunger + FORAGE_HUNGER_GAIN)
        agent.thirst = min(100, agent.thirst + FORAGE_THIRST_GAIN)
        agent.skills["herbalism"] += 1
        return {"gained": True, "reason": None}

    # -- trade ------------------------------------------------------------

    def resolve_trade(self, agent, item_name):
        """Spends the agent's gold against sim_econ's shop stock (trades
        are trade-keyed, not place-keyed, matching sim_econ.py's own
        abstraction level -- st_world.py's PLACES/TRAVEL_GRAPH are
        physical travel nodes distinct from sim_econ's shop economy, so
        this does not invent a new place<->shop mapping). Buying
        provisioner food/drink items also restores hunger/thirst, making
        USE-to-trade a genuine alternative to free foraging once an
        agent has coin -- the risk/reward the scarcity economy is meant
        to create."""
        for trade, stock in self.econ.stock.items():
            if item_name in stock and stock[item_name] > 0:
                item = next((i for i in ITEMS[trade] if i[0] == item_name), None)
                if item is None:
                    continue
                price = item[1]
                if agent.gold < price:
                    return {"bought": False, "reason": "cannot afford"}
                agent.gold -= price
                stock[item_name] -= 1
                agent.inventory[item_name] = agent.inventory.get(item_name, 0) + 1
                agent.skills["haggle"] += 1
                if trade == "provisioner":
                    agent.hunger = min(100, agent.hunger + FORAGE_HUNGER_GAIN)
                    agent.thirst = min(100, agent.thirst + FORAGE_THIRST_GAIN)
                self._log(f"{agent.name} bought {item_name} for {price} gold")
                return {"bought": True, "reason": None, "price": price}
        return {"bought": False, "reason": "not in stock anywhere"}

    # -- world step ----------------------------------------------------------

    def tick(self):
        """Advances shared world state (econ stock/demand/shocks) and
        agent needs decay for every living agent by one tick."""
        self.econ.tick()
        for a in self.agents.values():
            self.tick_needs(a)
        self.tick_n += 1

    # -- rendering ------------------------------------------------------------

    def render_turn(self, agent, last_event=None):
        """Renders one agent's turn as a compact, Markovian prompt --
        extends sim_econ.convo_to_scenario()'s header+trailing-name-colon
        pattern with a Status/Here/Recent block built from live world
        state instead of static card text. Deliberately re-renders fresh
        every call rather than accumulating a transcript: the model's
        512-token context window has no room for multi-turn history, so
        every turn must stand alone (state in, action out)."""
        bio = BIO_TEMPLATES[hash(agent.name) % len(BIO_TEMPLATES)]
        place_desc = PLACE_DESC.get(agent.place, "")
        others = [o.name for o in self.agents_at(agent.place) if o.name != agent.name][:2]
        here_bits = []
        if others:
            here_bits.append(", ".join(others))
        here = "; ".join(here_bits) if here_bits else "no one else"
        recent = last_event or (self.recent[-1] if self.recent else "the road is quiet")
        lines = [
            f"Description: {agent.name}, {bio}.",
            f"Scenario: {agent.place} -- {place_desc}.",
            f"Status: HP {agent.hp}/{agent.hp_max}, hunger {agent.hunger}, thirst {agent.thirst}, gold {agent.gold}",
            f"Here: {here}",
            f"Recent: {recent}",
            "<START>",
            "Player:",
        ]
        return "\n".join(lines)
