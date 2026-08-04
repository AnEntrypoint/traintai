"""Gold multi-sentence chains: the deepest measured gap (AGENTS.md).

chain_depth (npc_forge.py) counts consecutive sentences from the start of a
response that each share a 5+-letter content word (regex [a-z]{5,}) with
the card blob (Description + Scenario) or the words accumulated so far,
breaking at the first sentence with zero overlap. r16-r23 forge dashboards
measure ~0.1 across the board -- checked directly: even the hand-authored
st_authored.jsonl rows mostly score 0, because natural in-character
phrasing rarely repeats a literal 5+-letter card word in sentence one.

This generator forces the anchor explicitly instead of hoping phrasing
lines up: every chain's Scenario line names an anchor word (an item name,
a place, a keeper's trade noun), sentence 1 repeats that anchor word
verbatim, and every following sentence repeats a word introduced by an
earlier sentence in the chain -- item name -> its property -> its
provenance -> the origin place or event that provenance cites. Verified
against the real chain_depth() function before being counted as a fix
(see AGENTS.md discipline: no number lands without a model run behind it,
but the DATA must pass its own target metric before it is worth training on).
"""

import json
import os
import random
import re

from st_world import EXPANDED_ITEMS, EVENTS, ORIGIN_PLACE, LINEAGE, SHOPKEEPERS, PLACES

HERE = os.path.dirname(os.path.abspath(__file__))
NPC = os.path.join(HERE, "..", "data", "npc")
OUT = os.path.join(NPC, "st_chains.jsonl")


def _anchor_word(s):
    """Longest 5+-letter lowercase word in s, for a scenario line to name
    and sentence 1 to be built around."""
    words = re.findall(r"[A-Za-z]{5,}", s)
    return max(words, key=len).lower() if words else None


def item_chain(item, shop):
    """Scenario anchors on the item name; s1 repeats it; s2 repeats a word
    from s1 (the item's own noun via property text); s3 repeats a word
    from s2 or the origin place cited in provenance; s4 closes on the
    origin place, which s3's provenance line already names. Returns None
    if the item name has no 5+-letter word to anchor on (chain_depth's
    word regex is [a-z]{5,}, so a short name like "Salt pork side" -> the
    fallback last-word "side" never overlaps the card either)."""
    name, price, prop, prov = item
    anchor = _anchor_word(name)
    if anchor is None:
        return None
    origin = ORIGIN_PLACE[shop]
    scenario = f"a {name} sits on the counter between you."
    s1 = f"The {name} -- {price} gold, and worth asking after."
    s2 = f"That {anchor} {prop}."
    s3 = prov[0].upper() + prov[1:] + "."
    s4 = f"{origin} work like that does not come through here twice."
    return name, scenario, [s1, s2, s3, s4]


def lineage_chain(rng, shop):
    """Scenario anchors on the shop's trade word; s1 repeats it via the
    lineage claim; s2 repeats the origin place; s3 repeats the origin
    place again while citing the event; s4 closes on the event word."""
    keeper, desc = SHOPKEEPERS[shop]
    lineage = LINEAGE[keeper]
    origin = ORIGIN_PLACE[shop]
    ev, yr, cons = rng.choice(EVENTS)
    scenario = f"{keeper} is asked about the shop's history in {origin}."
    s1 = f"{lineage.capitalize()}, and {origin} remembers every one of them."
    s2 = f"{origin} still tells of {ev} ({yr})."
    s3 = f"That was the year {cons}."
    s4 = f"That's the history of this shop, straight as I can tell it."
    return keeper, scenario, [s1, s2, s3, s4]


def place_chain(rng, place):
    """Scenario anchors on the place name; s1 repeats it; s2 repeats the
    destination it leads to; s3 repeats that destination via the event
    tied to the route; s4 closes on the destination again."""
    name, desc, exits = place
    dest = rng.choice(exits)
    ev, yr, cons = rng.choice(EVENTS)
    scenario = f"a traveler asks the way out of {name}."
    s1 = f"{name} -- {desc}."
    s2 = f"The road out of {name} runs to {dest}, if that's where you're bound."
    s3 = f"{dest.split(',')[0].split()[-1] if ',' not in dest else dest} saw {ev} ({yr}): {cons}."
    s4 = f"Travelers still talk about {dest} because of it."
    return name, scenario, [s1, s2, s3, s4]


def render(rng, keeper, desc, scenario, q, chain):
    # No filler opener: chain_depth walks from the response's first
    # sentence, so sentence 1 must be an anchor sentence, not a throwaway.
    lines = [f"Description: {desc}", f"Scenario: {scenario}", "<START>",
             f"Player: {q}",
             f"{keeper}: " + " ".join(chain)]
    return "\n".join(lines) + "\n"


def main():
    rng = random.Random(23)
    out = []
    for shop, (keeper, desc) in SHOPKEEPERS.items():
        pool = [it for it in EXPANDED_ITEMS[shop] if _anchor_word(it[0]) is not None]
        n_skipped = len(EXPANDED_ITEMS[shop]) - len(pool)
        if n_skipped:
            print(f"  {shop}: skipping {n_skipped} items with no 5+-letter anchor word")
        rng.shuffle(pool)
        for item in pool[: min(80, len(pool))]:
            result = item_chain(item, shop)
            if result is None:
                continue
            name, scenario, chain = result
            out.append(render(rng, keeper, desc, scenario,
                               f"Tell me the whole story of the {name}.", chain))
        for _ in range(8):
            _, scenario, chain = lineage_chain(rng, shop)
            out.append(render(rng, keeper, desc, scenario,
                               "How long has your family kept this shop?", chain))
    for place in PLACES:
        for _ in range(12):
            guide = rng.choice(list(SHOPKEEPERS.values()))[0]
            _, scenario, chain = place_chain(rng, place)
            out.append(render(rng, guide, f"{guide}, who knows the roads and answers plainly.",
                               scenario,
                               f"Tell me everything about {place[0]} and where it leads.", chain))
    rng.shuffle(out)
    n_before = len(out)
    seen = set()
    deduped = []
    for convo in out:
        if convo not in seen:
            seen.add(convo)
            deduped.append(convo)
    with open(OUT, "w", encoding="utf-8") as f:
        for convo in deduped:
            f.write(json.dumps({"text": convo}) + "\n")
    print(f"wrote {len(deduped)} gold multi-sentence chains to {OUT} "
          f"({n_before - len(deduped)} exact-duplicate combinations dropped)")


if __name__ == "__main__":
    main()
