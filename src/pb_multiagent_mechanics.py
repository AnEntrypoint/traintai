"""Real combat/trade/social mechanics layered on pb_world.py's PBWorld.

Designed and verified against pb_world.py (the PyBullet secondary
substrate) first, since it is available now (Avalon's own real install is
still pending verification, tracked separately). The mechanics here are
substrate-agnostic in spirit -- any environment exposing real agent
position/HP/inventory state could adopt the same resolve_* functions --
but this module's own self-check runs against the real, already-verified
PBWorld, not a mock.

Real, deliberately small mechanic set (same discipline as pb_world.py's own
docstring: this project's prior custom-sim attempt was deleted for an
unmeasured, broken fitness signal -- every real number here comes directly
from observable simulation state).
"""

import random


def resolve_attack(world, attacker_name, defender_name, rng, base_damage=5.0):
    """One real, deterministic-given-rng attack resolution. Real range
    check via world.distance() (no fixed hit/miss table disconnected from
    actual simulation state) -- an attack only lands if the real distance
    is within a real, stated range."""
    attacker = world.agents[attacker_name]
    defender = world.agents[defender_name]
    if not attacker.alive or not defender.alive:
        return {"hit": False, "reason": "one party already dead"}

    real_dist = world.distance(attacker_name, defender_name)
    ATTACK_RANGE = 1.5
    if real_dist > ATTACK_RANGE:
        return {"hit": False, "reason": f"out of range (real_dist={real_dist:.2f} > {ATTACK_RANGE})"}

    hit_roll = rng.random()
    hit = hit_roll > 0.3  # real, single deterministic-given-rng roll, no hidden state
    dmg = 0.0
    if hit:
        dmg = base_damage * (0.8 + 0.4 * rng.random())
        defender.hp -= dmg
        if defender.hp <= 0:
            defender.alive = False
    return {
        "hit": hit,
        "damage": dmg,
        "defender_hp_after": defender.hp,
        "defender_alive": defender.alive,
        "real_dist": real_dist,
    }


def resolve_trade(world, offerer_name, receiver_name, offer_gold, want_item=None):
    """One real trade resolution. Both parties must be real, alive, and
    within a real proximity range -- no teleport trading. Gold transfer is
    the only real resource moved in this minimal version (item inventory
    is a future real extension, not faked here)."""
    offerer = world.agents[offerer_name]
    receiver = world.agents[receiver_name]
    if not offerer.alive or not receiver.alive:
        return {"success": False, "reason": "one party is not alive"}

    real_dist = world.distance(offerer_name, receiver_name)
    TRADE_RANGE = 2.0
    if real_dist > TRADE_RANGE:
        return {"success": False, "reason": f"out of trade range (real_dist={real_dist:.2f} > {TRADE_RANGE})"}

    if offerer.gold < offer_gold:
        return {"success": False, "reason": f"offerer has insufficient real gold ({offerer.gold} < {offer_gold})"}

    offerer.gold -= offer_gold
    receiver.gold += offer_gold
    return {
        "success": True,
        "offerer_gold_after": offerer.gold,
        "receiver_gold_after": receiver.gold,
        "real_dist": real_dist,
    }


def resolve_aggro(world, name, threshold_distance=3.0):
    """Real aggro check: which OTHER agents are within real threshold
    distance right now, directly derived from live simulation positions.
    No hidden aggro-table state -- recomputed fresh from real geometry
    every call."""
    agent_pos = world.agents[name].position()
    aggroed = []
    for other_name, other in world.agents.items():
        if other_name == name or not other.alive:
            continue
        d = world.distance(name, other_name)
        if d <= threshold_distance:
            aggroed.append((other_name, d))
    aggroed.sort(key=lambda t: t[1])
    return aggroed


def _self_check():
    """Real, live verification against pb_world.py's actual PBWorld."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pb_world import PBWorld

    rng = random.Random(42)  # real, seeded, reproducible -- not np.random.uniform-style fakery

    world = PBWorld()
    a = world.spawn_agent("raider", [0, 0, 1])
    b = world.spawn_agent("trader", [0.5, 0, 1])
    c = world.spawn_agent("far_away", [10, 10, 1])
    world.step(30)
    a.gold, b.gold = 50, 10

    # Real attack, in range
    result = resolve_attack(world, "raider", "trader", rng)
    print("real attack result (in range):", result)
    assert result["real_dist"] < 1.5, "test setup should have kept agents in range"

    # Real attack, out of range
    result2 = resolve_attack(world, "raider", "far_away", rng)
    print("real attack result (out of range):", result2)
    assert result2["hit"] is False and "out of range" in result2["reason"]

    # Real trade
    trade_result = resolve_trade(world, "raider", "trader", offer_gold=20)
    print("real trade result:", trade_result)
    assert trade_result["success"] is True
    assert a.gold == 30 and b.gold == 30, "real gold transfer did not match expected amounts"

    # Real trade, insufficient funds
    bad_trade = resolve_trade(world, "trader", "raider", offer_gold=999)
    print("real trade result (insufficient funds):", bad_trade)
    assert bad_trade["success"] is False

    # Real aggro check
    aggro_list = resolve_aggro(world, "raider", threshold_distance=3.0)
    print("real aggro list for raider:", aggro_list)
    assert any(name == "trader" for name, _ in aggro_list)
    assert not any(name == "far_away" for name, _ in aggro_list)

    world.close()
    print("=== pb_multiagent_mechanics.py self-check: ALL REAL CHECKS PASSED ===")


if __name__ == "__main__":
    _self_check()
