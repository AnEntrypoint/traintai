"""Economy simulation -> oracle decisions -> ST-format training conversations.

The sim is the decision oracle the user directive calls for: every generated
conversation embeds a situation where exactly one choice is correct (buy at
this price or walk, this place stocks it or it does not, haggle floor or
refusal), and the NPC's line -- dialog only, no action beats -- carries that
choice, closed by at most one machine-readable action:

  [DEAL: <item> <gold>]  a sale/purchase closes; valid only when the item is
                         actually in stock in this scene (engine tolerates a
                         bad one as a bad life decision and ignores it)
  [GOTO: <place>]        travel advice; valid only for a known world place

Situations include the negative cases on purpose: asks for out-of-stock
items (must redirect, never DEAL), asks for items nobody stocks (no GOTO),
haggling below the floor (must refuse). Format: dialog line(s) first, then
the action on its own line -- trivially parseable, and an engine that never
sees an action just gets plain dialog.

--rows N writes data/npc/st_sim.jsonl; --scenarios N writes held-out
oracle-labeled eval scenarios to data/npc/sim_scenarios.jsonl (different
seed; the genuine-improvement test set for src/sim_eval.py).
"""

import argparse
import json
import os
import random

from st_world import EXPANDED_ITEMS as ITEMS
from st_world import EVENTS, LINEAGE, ORIGIN_PLACE, PLACES, SHOPKEEPERS

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
OUT = os.path.join(NPC, "st_sim.jsonl")
SCEN = os.path.join(NPC, "sim_scenarios.jsonl")

KEEPER_TRADE = {v[0]: k for k, v in SHOPKEEPERS.items()}
MARKUP = {"Dorn": 1.2, "Magistra Vool": 1.15, "Sage Willowbark": 1.0, "Hettie": 1.0, "Snik": 1.3}
FLOOR = 0.85
RUMORS = [
    "the pass is snowed in and nothing moves till it clears",
    "a caravan came through last week and bought half the valley's stock",
    "the festival crowd has coin and no patience",
    "the river trade is slow, so everything local runs cheap",
    "the mine reopened and suddenly everyone needs supplies",
    "bandits on the north road have merchants paying for guards",
]

QUOTE = [
    "{item}, {price} -- {detail}.",
    "That one is {price}. {detail_cap}.",
    "For you? {price}, and it is worth every coin -- {detail}.",
]
STOCK_LIST = [
    "Depends on your coin. {list2}.",
    "A few things worth your coin. {list2}.",
    "Everything on the shelf is for sale. {list2}.",
]
ACCEPT = [
    "Sold, and well bought. Care for it and it outlives you.",
    "Done. You will not find a fairer price on this road.",
    "Pleasure doing business. It is yours.",
    "Sensible choice. Wrap it up, it goes with you.",
]
DECLINE_HAGGLE = [
    "Below {floor} I lose money on the shelf itself. The price is {price}.",
    "I would sooner keep it. {price} is already the honest number.",
    "No. At that number I am paying you to take it. {price}, firm.",
]
HAGGLE_OK = [
    "You bargain like a tax collector. {price}, and that is the floor.",
    "Fine -- {price}, because I like your face. Not a copper less.",
    "{price} and we both pretend you won.",
]
NO_STOCK = [
    "That I do not have, and I will not pretend otherwise.",
    "Out of that, I am afraid. The shelf does not lie.",
    "Not in stock. I sell what I have, not what I wish I had.",
]
RESTOCK_ASK = [
    "There is one thing I am short of myself. {item} -- my supplier in {place} has it, and my legs do not. Bring it back and I pay {reward}.",
    "Since you look like you walk roads: I need {item} from {place}. Fetch it and {reward} is yours, no haggling.",
    "You want work? {place} holds what I need -- {item}. Return with it and I will make it {reward} for your trouble.",
]

GOTO_LEAD = [
    "For that you want {place}.",
    "Not here -- but {place} will have it.",
    "You will find that in {place}, if the road is kind.",
]
NOWHERE = [
    "Nowhere I know of sells that, and I know the roads.",
    "That I have never seen for sale in these parts.",
    "No shop between here and the coast carries that, friend.",
]
BUSINESS = [
    "{rumor_cap}, so trade is what it is this week.",
    "Honestly? {rumor_cap}. You plan around it or you starve.",
    "Fair to middling. {rumor_cap}.",
]


class World:
    """Tick-based scarcity: stock depletes with sales and regrows with
    caravans, and every change leaves a narratable reason the NPC can cite.
    Adopted from the evolutionary-marathon notebooks, with the random-choice
    'oracle' parts removed -- stock changes are mechanical, but every
    conversation-facing decision still comes from the price oracle."""

    def __init__(self, rng):
        self.rng = rng
        self.stock = {trade: {i[0]: rng.randint(0, 6) for i in items} for trade, items in ITEMS.items()}
        self.recent = []

    def tick(self):
        trade = self.rng.choice(list(self.stock))
        item = self.rng.choice(list(self.stock[trade]))
        r = self.rng.random()
        if r < 0.45 and self.stock[trade][item] > 0:
            self.stock[trade][item] -= 1
            if self.stock[trade][item] == 0:
                keeper = SHOPKEEPERS[trade][0]
                self.recent.append(f"a traveler bought the last {item} off {keeper} not two days past")
        elif r >= 0.6:
            self.stock[trade][item] = min(8, self.stock[trade][item] + 2)
            self.recent.append(f"a caravan came through with fresh {item}")
        self.recent = self.recent[-6:]

    def reason_for(self, item_name):
        for ev in reversed(self.recent):
            if item_name in ev:
                return ev
        return None


def price_str(p):
    return f"{p} gold"


def list2(items, demand, markup):
    a, b = items
    pa = round(a[1] * demand * markup)
    pb = round(b[1] * demand * markup)
    return f"{a[0]}, {price_str(pa)} -- {a[2]}. There is also {b[0]} at {price_str(pb)}"


def card_lines(keeper, desc, scen):
    return [f"Description: {desc}",
            f"Personality: plainspoken, busy, knows every item on the shelves.",
            f"Scenario: {scen}", "<START>"]


def convo_sale(rng, shop, keeper, desc, stock, demand):
    avail = [i for i in ITEMS[shop] if stock.get(i[0], 0) > 0]
    if not avail:
        return None
    star = rng.choice(avail)
    others = [i for i in avail if i is not star]
    price = round(star[1] * demand * MARKUP[keeper])
    q2, a2 = None, None
    if others and rng.random() < 0.6:
        listline = rng.choice(STOCK_LIST).format(list2=list2([star, others[0]], demand, MARKUP[keeper]))
    else:
        listline = rng.choice(QUOTE).format(item=star[0], price=price_str(price),
                                            detail=star[2], detail_cap=star[2].capitalize())
    kind = rng.random()
    if kind < 0.45:
        q2, a2 = f"Player: I'll take the {star[0]}.", (rng.choice(ACCEPT), f"[DEAL: {star[0]} {price}]")
    elif kind < 0.75:
        floor = round(star[1] * demand * FLOOR * MARKUP[keeper])
        offer = round(price * rng.uniform(0.5, 0.8))
        if rng.random() < 0.5 and offer >= floor:
            q2, a2 = (f"Player: {price_str(offer)}, final offer.",
                      (rng.choice(HAGGLE_OK).format(price=price_str(offer), floor=price_str(floor)),
                       f"[DEAL: {star[0]} {offer}]"))
        else:
            q2, a2 = (f"Player: How about {price_str(offer)}?",
                      (rng.choice(DECLINE_HAGGLE).format(floor=price_str(floor), price=price_str(price)), None))
    else:
        q2 = f"Player: Tell me about the {star[0]}."
        a2 = (f"{star[3].capitalize()}. That is the honest of it, and it costs me nothing to tell. "
              f"The item itself is {price_str(price)}."), None
    lines = card_lines(keeper, desc, f"{keeper}'s shop, the day's trade underway.")
    lines += [f"{keeper}: {rng.choice(['Morning, if it is one.', 'Welcome. Mind the step.', 'In or out, the door costs heat.'])}",
              "Player: What do you have for sale?",
              f"{keeper}: {listline}", q2]
    if a2[1]:
        lines += [f"{keeper}: {a2[0]}", a2[1]]
    else:
        lines += [f"{keeper}: {a2[0]}"]
    return "\n".join(lines) + "\n"


def convo_missing(rng, shop, keeper, desc, stock, demand, world=None):
    have = {i[0] for i, q in stock.items() if q > 0}
    missing = [i for s in ITEMS.values() for i in s if i[0] not in have]
    if not missing:
        return None
    want = rng.choice(missing)
    holders = [trade for trade, items in ITEMS.items() if any(i[0] == want[0] for i in items)]
    keeper_trade = KEEPER_TRADE[keeper]
    reason = world.reason_for(want[0]) if world else None
    if reason:
        no_stock_line = f"That I do not have, not today -- {reason}."
    else:
        no_stock_line = rng.choice(NO_STOCK)
    lines = card_lines(keeper, desc, f"{keeper}'s shop, the day's trade underway.")
    lines += [f"{keeper}: Welcome. Mind the step.",
              f"Player: Do you have {want[0]}?",
              f"{keeper}: {no_stock_line}"]
    real = [t for t in holders if t != keeper_trade]
    if real and rng.random() < 0.8:
        place = rng.choice([p[0] for p in PLACES])
        lines += [f"Player: Any idea where I would?",
                  f"{keeper}: {rng.choice(GOTO_LEAD).format(place=place)}", f"[GOTO: {place}]"]
    else:
        lines += [f"Player: Where could I find it?",
                  f"{keeper}: {rng.choice(NOWHERE)}"]
    return "\n".join(lines) + "\n"


def convo_restock(rng, shop, keeper, desc, world):
    empty = [name for name, q in world.stock[shop].items() if q == 0]
    if not empty:
        return None
    want = rng.choice(empty)
    place = ORIGIN_PLACE[shop]
    fair = next(i[1] for i in ITEMS[shop] if i[0] == want)
    reward = round(fair * 1.3)
    ask = rng.choice(RESTOCK_ASK).format(item=want, place=place, reward=price_str(reward))
    lines = card_lines(keeper, desc, f"{keeper}'s shop, shelves thin this week.")
    lines += [f"{keeper}: Welcome. Mind the empty shelf, the week has been long.",
              "Player: Do you have a quest for me?",
              f"{keeper}: {ask}",
              "Player: Where do I find it?",
              f"{keeper}: {rng.choice(GOTO_LEAD).format(place=place)} The keeper there knows my name.",
              f"[GOTO: {place}]"]
    return "\n".join(lines) + "\n"


def convo_business(rng, keeper, desc, rumor):
    greet = "Day to you. Mind the step."
    if rng.random() < 0.25 and keeper in LINEAGE:
        greet = f"Day to you. {LINEAGE[keeper].capitalize()}, for what that is worth."
    lines = card_lines(keeper, desc, f"{keeper}'s shop, between customers.")
    lines += [f"{keeper}: {greet}",
              "Player: How's business?",
              f"{keeper}: {rng.choice(BUSINESS).format(rumor_cap=rumor[0].upper() + rumor[1:])}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--scenarios", type=int, default=0)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--out", default=None, help="scenario output path (default sim_scenarios.jsonl)")
    args = ap.parse_args()
    scen_path = args.out or SCEN
    rng = random.Random(args.seed if not args.scenarios else args.seed + 1000)

    out, scenarios = [], []
    n_target = args.scenarios if args.scenarios else args.rows
    world = World(rng)
    for _ in range(n_target):
        for _ in range(3):
            world.tick()
        shop = rng.choice(list(ITEMS))
        keeper, desc = SHOPKEEPERS[shop]
        stock = world.stock[shop]
        demand = rng.uniform(0.8, 1.6)
        rumor = world.recent[-1] if world.recent and rng.random() < 0.7 else rng.choice(RUMORS)
        r = rng.random()
        if r < 0.42:
            c = convo_sale(rng, shop, keeper, desc, stock, demand)
        elif r < 0.68:
            c = convo_missing(rng, shop, keeper, desc, stock, demand, world)
        elif r < 0.82:
            c = convo_restock(rng, shop, keeper, desc, world)
        else:
            c = convo_business(rng, keeper, desc, rumor)
        if c is None:
            continue
        if args.scenarios:
            header = c.split("<START>\n", 1)[0]
            rest = c.split("<START>\n", 1)[1]
            first = rest.split("\n", 1)[0]
            body = rest.split("\n")[1:]
            player_idx = [i for i, l in enumerate(body) if l.startswith("Player:")]
            if not player_idx:
                continue
            last = player_idx[-1]
            action = next((l for l in body[last:] if l.startswith("[")), None)
            keeper = first.split(":", 1)[0]
            prompt = header + "<START>\n" + first + "\n" + "\n".join(body[: last + 1]) + "\n" + keeper + ":"
            scenarios.append({"prompt": prompt, "oracle_action": action, "keeper": keeper})
        else:
            out.append(c)

    if args.scenarios:
        with open(scen_path, "w", encoding="utf-8") as f:
            for s in scenarios:
                f.write(json.dumps(s) + "\n")
        print(f"wrote {len(scenarios)} eval scenarios to {scen_path}")
    else:
        rng.shuffle(out)
        with open(OUT, "w", encoding="utf-8") as f:
            for c in out:
                f.write(json.dumps({"text": c}) + "\n")
        n_deal = sum(1 for c in out if "[DEAL:" in c)
        n_goto = sum(1 for c in out if "[GOTO:" in c)
        n_act = sum(1 for c in out if "*" in c)
        print(f"wrote {len(out)} sim conversations to {OUT} "
              f"(DEAL {n_deal}, GOTO {n_goto}, action-beats {n_act})")


if __name__ == "__main__":
    main()
