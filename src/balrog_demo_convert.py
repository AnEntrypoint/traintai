"""Converts BALROG's real expert-demonstration trajectories (records.zip,
see balrog-inspect/docs/few_shot_learning.md) into BALROG-shaped SFT rows
for this project's training mixture.

Why this exists: a 10-round BALROG evaluation campaign (BabyAI, Crafter,
BabaIsAI, TextWorld, MiniHack, NLE) landed 0% progression on every single
round, uniformly. The root cause is not a bug in balrog_server.py or the
model architecture -- it's that the model has never seen a single
BALROG-shaped "Observation: ... assistant: <action>" example during
training. This script builds that training data from BALROG's own real
expert demonstrations, so a future SFT round can actually teach the model
this prompt shape instead of only ever evaluating an untrained one.

Input: an already-extracted `records.zip` root (download+extract happens
on a Kaggle kernel with internet access -- this script only parses an
existing directory, it never downloads anything itself), with the
structure confirmed via balrog-inspect/balrog/dataset.py's
InContextDataset.icl_episodes(): `<records_root>/<env_name>/<task>/*.npz`.

Per-episode replay logic mirrors balrog-inspect/balrog/dataset.py's
InContextDataset.load_in_context_learning_episode() exactly: the first
array entry is a reset-only observation with no preceding action (popped
before pairing observations with actions), and iteration stops at the
first `done` (any(terminated, truncated)) rather than continuing past it.

Per-step message shape mirrors balrog-inspect/balrog/prompt_builder/
history.py's HistoryPromptBuilder.get_prompt(): a leading system/
instruction user-message (the real per-game instruction prompt, reproduced
below from each environment's own balrog/environments/<env>/__init__.py
get_instruction_prompt()/intruction_prompts, since none of them need a
live env instance beyond MiniHack's task-name goal-string branch and
BabyAI's mission text -- both reconstructable offline, see
_instruction_prompt() and _babyai_mission()), then alternating
user (Observation: <long_term_context>, or "Current Observation:" +
short_term_context for the final one) / assistant (action) messages.

Final prompt text is produced by REUSING balrog_server.py's own
build_prompt(messages) (imported directly, not reimplemented) so the
training rows are byte-for-byte the same flattening the model is actually
served with at inference time.

Output: data/npc/balrog_demos.jsonl, one {"text": "<prompt>assistant: <action>"}
row per (trajectory-prefix, next-action) step -- NOT one row per episode.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

sys.path.insert(0, HERE)
from balrog_server import build_prompt  # noqa: E402  (reused verbatim, not reimplemented)
from model import Config  # noqa: E402

DEFAULT_CAP = 20000  # same order of magnitude as st_prepare.py's *_CAP constants (FORGE_CAP=2500, KAGGLE_FANTASY_CAP=3000, etc -- this one caps a per-step row count across 6 games, so sits an order above the largest single-source cap there)

ENVS = ["babyai", "crafter", "babaisai", "textworld", "minihack", "nle"]


# ---- per-game instruction prompts -----------------------------------------
# Reproduced verbatim from each balrog/environments/<env>/__init__.py's
# get_instruction_prompt()/intruction_prompts (read directly this session,
# not re-derived/guessed). Each real get_instruction_prompt() takes a live
# `env` argument only to read its action space (identical across episodes
# of a given env -- MiniHack's env.actions subsetting is the one exception,
# approximated here with the full static ACTIONS dict since the live env
# object doesn't exist in an offline .npz replay) or, for BabyAI, the
# mission string (recovered per-episode from the demo's own observation
# text instead, see _babyai_mission()).

_BABAISAI_ACTIONS = {
    "idle": "wait for one step",
    "up": "take one step up",
    "right": "take one step to the right",
    "down": "take one step down",
    "left": "take one step to the left",
}

_BABYAI_ACTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object below you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object in front of you",
}

_CRAFTER_ACTION_DICT = {
    "Noop": "do nothing",
    "Move West": "move west on flat ground",
    "Move East": "move east on flat ground",
    "Move North": "move north on flat ground",
    "Move South": "move south on flat ground",
    "Do": "Multiuse action to collect material, drink from lake and hit creature in front",
    "Sleep": "sleep when energy level is below maximum",
    "Place Stone": "place a stone in front",
    "Place Table": "place a table",
    "Place Furnace": "place a furnace",
    "Place Plant": "place a plant",
    "Make Wood Pickaxe": "craft a wood pickaxe with a nearby table and wood in inventory",
    "Make Stone Pickaxe": "craft a stone pickaxe with a nearby table, wood, and stone in inventory",
    "Make Iron Pickaxe": "craft an iron pickaxe with a nearby table and furnace, wood, coal, and iron in inventory",
    "Make Wood Sword": "craft a wood sword with a nearby table and wood in inventory",
    "Make Stone Sword": "craft a stone sword with a nearby table, wood, and stone in inventory",
    "Make Iron Sword": "craft an iron sword with a nearby table and furnace, wood, coal, and iron in inventory",
}
_CRAFTER_ACTIONS_ORDER = list(_CRAFTER_ACTION_DICT.keys())

_MINIHACK_ACTIONS = {
    "north": "move north", "east": "move east", "south": "move south", "west": "move west",
    "northeast": "move northeast", "southeast": "move southeast", "southwest": "move southwest",
    "northwest": "move northwest", "far north": "move far north", "far east": "move far east",
    "far south": "move far south", "far west": "move far west", "far northeast": "move far northeast",
    "far southeast": "move far southeast", "far southwest": "move far southwest",
    "far northwest": "move far northwest", "up": "go up the stairs", "down": "go down the stairs",
    "wait": "rest one move while doing nothing", "more": "display more of the message",
    "apply": "apply (use) a tool", "close": "close an adjacent door", "open": "open an adjacent door",
    "eat": "eat something", "force": "force a lock", "kick": "kick an enemy or a locked door or chest",
    "loot": "loot a box on the floor", "pickup": "pick up things at the current location if there are any",
    "pray": "pray to the gods for help", "puton": "put on an accessory", "quaff": "quaff (drink) something",
    "search": "search for hidden doors and passages", "zap": "zap a wand",
}

_NLE_ACTIONS = {
    "north": "move north", "east": "move east", "south": "move south", "west": "move west",
    "northeast": "move northeast", "southeast": "move southeast", "southwest": "move southwest",
    "northwest": "move northwest", "far north": "move far north", "far east": "move far east",
    "far south": "move far south", "far west": "move far west", "far northeast": "move far northeast",
    "far southeast": "move far southeast", "far southwest": "move far southwest",
    "far northwest": "move far northwest", "up": "go up a staircase",
    "down": "go down a staircase (tip: you can only go down if you are standing on the stairs)",
    "wait": "rest one move while doing nothing",
    "more": "display more of the message (tip: ONLY ever use when current message ends with --More--)",
    "annotate": "leave a note about the level", "apply": "apply (use) a tool",
    "call": "name a monster or object, or add an annotation", "cast": "cast a spell",
    "close": "close an adjacent door", "open": "open an adjacent door",
    "dip": "dip an object into something", "drop": "drop an item",
    "droptype": "drop specific item types (specify in the next prompt)",
    "eat": "eat something (tip: replenish food when hungry)", "esc": "exit menu or message",
    "engrave": "engrave writing on the floor (tip: Elbereth)", "enhance": "advance or check weapons skills",
    "fire": "fire ammunition from quiver", "fight": "fight a monster (even if you only guess one is there)",
    "force": "force a lock", "inventory": "show your inventory", "invoke": "invoke ",
    "jump": "jump to a location", "kick": "kick an enemy or a locked door or chest",
    "look": "look at what is under you", "loot": "loot a box on the floor",
    "monster": "use a monster's special ability (when polymorphed)",
    "offer": "offer a sacrifice to the gods (tip: on an aligned altar)",
    "overview": "display an overview of the dungeon", "pay": "pay your shopping bill",
    "pickup": "pick up things at the current location", "pray": "pray to the gods for help",
    "puton": "put on an accessory", "quaff": "quaff (drink) something",
    "quiver": "select ammunition for quiver", "read": "read a scroll or spellbook",
    "remove": "remove an accessory", "rub": "rub a lamp or a stone",
    "search": "search for hidden doors and passages", "swap": "swap wielded and secondary weapons",
    "takeoff": "take off one piece of armor", "takeoffall": "take off all armor",
    "teleport": "teleport to another level (if you have the ability)",
    "throw": "throw something (e.g. a dagger or dart)",
    "travel": "travel to a specific location on the map (tip: in the next action, specify > or < for stairs, { for fountain, and _ for altar)",
    "twoweapon": "toggle two-weapon combat", "untrap": "untrap something", "wear": "wear a piece of armor",
    "wield": "wield a weapon", "wipe": "wipe off your face", "zap": "zap a wand",
    "minus": "-", "space": " ", "apos": "'",
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
}

_TEXTWORLD_PROMPTS = dict(
    treasure_hunter="""You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated
maze and must find a specific object. You need to explore different rooms to find the target object.
Here are the available commands: look: describe the current room. goal: print the goal of this game
inventory: print the player's inventory go <dir>: move the player north, east, south, or west. You can
only go in the direction indicated with an exit or a door. open ...: open a door or a container. You need to
open a closed door before you want to go through it. drop ...: drop an object on the floor take ...: take an
object that is visible. Make sure the object is visible to take. put ... on ...: place an object on a supporter
take ... from ...: take an object from a container or a supporter insert ... into ...: place an object into a
container unlock ... with ...: unlock a door or a container with a key. You need to unlock a locked door
with a matched key in your inventory before you want to open it.
- The target object might be located in a closed or locked container. - The adjective is useful for
determining whether the key is matched with the lock (e.g. non-euclidean keycard is matched with
non-euclidean safe). Make sure it is matched to unlock! - The key required to unlock the door may be in
another room or locked inside a container. - Take the key whenever you can. - After unlocking a locked
door or container, it will remain closed. You will then need to open it.
You have 40 steps to complete the task. Restarting is forbidden.""",
    the_cooking_game="""You are an agent playing TextWorld, a text-based adventure game where you navigate through different
rooms, interact with objects, and solve puzzles. Your goal is to first find the recipe, find and prepare food
according to the recipe, and finally prepare and eat the meal.
Here are the available commands: look: describe the current room goal: print the goal of this game
inventory: print player's inventory go <dir>: move the player north, east, south or west. You can only go
to directions indicated with an exit or a door. examine ...: examine something more closely eat ...: eat
edible food open ...: open a door or a container. You need to open a closed door before you can go through
it. drop ...: drop an object onto the floor take ...: take an object that is visible put ... on ...: place an object
on a supporter take ... from ...: take an object from a container or a supporter insert ... into ...: place an
object into a container lock ... with ...: lock a door or a container with a key unlock ... with ...: unlock a
door or a container with a key cook ... with ...: cook cookable food with something providing heat slice ...
with ...: slice cuttable food with something sharp chop ... with ...: chop cuttable food with something
sharp dice ... with ...: dice cuttable food with something sharp prepare meal: combine ingredients from
inventory into a meal. You can only prepare meals in the Kitchen.
- You can examine the cookbook to see the recipe when it is visible. - The BBQ is for grilling things,
the stove is for frying things, the oven is for roasting things. Cooking ingredients in the wrong way will
lead to a failure of the game. - Once you have got processed ingredients and the appropriate cooking tool
ready, cook all of them according to the recipe. - There are two conditions to correctly cook something
(grill/fry/roast): a) the ingredient you want to cook is in your inventory and b) there is a suitable cooking
tool in the room, and then use 'cook . . . with . . . ' command. - When you need to chop/slice/dice
ingredients, you need to take the knife and the ingredient in your inventory and then 'slice/chop/dice ...
with knife' - Make sure to first process the food (chop/slice/dice) before you try to cook them. - When
you have all the ingredients (that got processed or cooked according to the menu), you can 'prepare meal'
in the kitchen and then 'eat meal' to win the game. - The ingredients should EXACTLY match the color
in the recipe, but if the recipe doesn't specify color, any color would be fine. When you 'take ... with ...',
use the EXACT name you see. - You don't need to examine the container/supporter (e.g. toolbox) when
it says something like "there isn't a thing on it"/"has nothing on it"
You have 80 steps to complete the task. Restarting is forbidden.""",
    coin_collector="""You are an agent playing TextWorld, a text-based adventure game where you are in a randomly generated
maze and must find the coin. You need to explore different rooms to find the target object.
Here are the available commands: goal: print the goal of this game go <dir>: move the player north, east,
south, or west. You can only go in the direction indicated with something like an exit or a door. take coin:
2in the game by 'take coin' if you see the coin in the room
The only action you can do is go <dir> to explore the maze and 'take coin' when you see the coin in the
room.
You have 25 steps to complete the task. Restarting is forbidden.""",
)


def _action_strings(d):
    return ",\n".join(f"{a}: {desc}" for a, desc in d.items())


def _babaisai_prompt(task):
    return f"""Baba Is You is a puzzle game where you can manipulate the rules of each level. The following are the possible actions you can take in the game, followed by a short description of each action:

{_action_strings(_BABAISAI_ACTIONS)}.

Tips:
- Examine the level carefully, noting all objects and text blocks present.
- Identify the current rules, which are formed by text blocks in the format "[Subject] IS [Property]" (e.g. "BABA IS YOU").
- Consider how you can change or create new rules by moving text blocks around.
- Remember that you can only move objects or text that are not defined as "STOP" or similar immovable properties.
- Your goal is usually to reach an object defined as "WIN", but this can be changed.
- Think creatively about how changing rules can alter the properties and behaviors of objects in unexpected ways.
- If stuck, try breaking apart existing rules or forming completely new ones.
- Sometimes the solution involves making yourself a different object or changing what counts as the win condition.

PLAY!"""


def _babyai_mission(first_long_term_context):
    """BabyAI's real get_instruction_prompt(mission=...) is filled from
    obs["mission"] (balrog/environments/babyai_text/clean_lang_wrapper.py).
    That mission string isn't a separately stored .npz field, but the
    wrapper's get_prompt() renders infos["descriptions"] into
    long_term_context and the mission semantics ("go to the red ball" etc)
    show up there too in practice; since we don't have the raw obs["mission"]
    string in the replay, fall back to a generic phrasing rather than
    fabricate a mission we can't verify from the .npz contents alone."""
    return "complete the mission described in the observations below"


def _babyai_prompt(mission):
    return f"""You are an agent playing a simple navigation game. Your goal is to {mission}. The following are the possible actions you can take in the game, followed by a short description of each action:

{_action_strings(_BABYAI_ACTIONS)}.

In a moment I will present you an observation.

Tips:
- Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.
- It doesn't make sense to repeat the same action over and over if the observation doesn't change.

PLAY!"""


def _crafter_prompt():
    achievements = """1. Collect Wood
2. Place Table
3. Eat Cow
4. Collect Sapling
5. Collect Drink
6. Make Wood Pickaxe
7. Make Wood Sword
8. Place Plant
9. Defeat Zombie
10. Collect Stone
11. Place Stone
12. Eat Plant
13. Defeat Skeleton
14. Make Stone Pickaxe
15. Make Stone Sword
16. Wake Up
17. Place Furnace
18. Collect Coal
19. Collect Iron
20. Make Iron Pickaxe
21. Make Iron Sword
22. Collect Diamond"""
    action_strings = ",\n".join(f"{a}: {_CRAFTER_ACTION_DICT[a]}" for a in _CRAFTER_ACTIONS_ORDER)
    return f"""You are an agent playing Crafter. The following are the only valid actions you can take in the game, followed by a short description of each action:

{action_strings}.

These are the game achievements you can get:
{achievements}

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible by completing all the achievements.

PLAY!"""


def _minihack_prompt(task):
    t = (task or "").lower()
    if "corridor" in t:
        goal = "Your goal is to explore the level and reach the stairs down"
    elif "quest" in t:
        goal = "Your goal is to explore the level, fight monsters, and navigate rooms and mazes to ultimately reach the stairs down."
    elif "boxoban" in t:
        goal = "You are playing Boxoban, a box-pushing game inspired by Sokoban. Your goal is to push the boulders onto the fountains on the map. You can push the boulders by walking into them, as long as there are no obstacles behind them."
    else:
        goal = "Your goal is to get as far as possible in the game."
    return f"""You are an agent playing MiniHack. The following are the possible actions you can take in the game, followed by a short description of each action:

{_action_strings(_MINIHACK_ACTIONS)}.

In a moment I will present a history of actions and observations from the game.

Tip: there is no point in outputting the same action over and over if nothing changes.

{goal}

PLAY!"""


def _nle_prompt():
    return f"""You are an agent playing NetHack. The following are the possible actions you can take in the game, followed by a short description of each action:

{_action_strings(_NLE_ACTIONS)}.

Tips:
- When the message asks for a completion, such as: "What do you want to eat? [d or ?*]", you should respond with a single character corresponding to the item you want to eat/use.
    - For example, "What do you want to eat? [dgh or ?*]" -> Possible answers are "d", "g", or "h" to eat the associated food.
- When the message asks for a direction, such as: "In what direction?" you should respond with a direction.
- When the message has --More-- at the end, your next action should be "more" to see the rest of the message.
- Explore the environment to find the stairs down to the next level.
- Always carefully read the last message to understand the current state of the game and decide your next action accordingly.
- If you keep moving in the same direction, you will eventually hit a wall and stop moving. Your message might be: "It's solid stone", or "It's a wall". Change your action to move in another direction to continue exploring the environment.
- Read the language observation carefully and look at ascii map or image observation provided to decide the next action to take and where to move next.
- You can attack monsters by moving into them.

In a moment I will present a history of actions and observations from the game.
Your goal is to get as far as possible in the game.

PLAY!"""


def _textworld_prompt(task):
    prompt = _TEXTWORLD_PROMPTS.get(task)
    if prompt is None:
        # Unknown/unrecognized task name -- fall back to whichever prompt is
        # closest rather than crash the whole conversion run on one env's
        # naming drift; real per-task text still applies whenever task matches.
        prompt = next(iter(_TEXTWORLD_PROMPTS.values()))
    return prompt.strip()


def instruction_prompt_for(env_name, task, first_long_term_context):
    if env_name == "babaisai":
        return _babaisai_prompt(task)
    if env_name == "babyai":
        return _babyai_prompt(_babyai_mission(first_long_term_context))
    if env_name == "crafter":
        return _crafter_prompt()
    if env_name == "minihack":
        return _minihack_prompt(task)
    if env_name == "nle":
        return _nle_prompt()
    if env_name == "textworld":
        return _textworld_prompt(task)
    raise ValueError(f"unknown env_name {env_name!r}")


# ---- .npz episode replay, mirroring InContextDataset.load_in_context_learning_episode ----

def load_episode(filename):
    with np.load(filename, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def obs_text(observation):
    """Pull {"long_term_context", "short_term_context"} out of one
    per-step observation dict, matching history.py's update_observation()."""
    text = observation.get("text", {}) if isinstance(observation, dict) else {}
    if not isinstance(text, dict):
        text = {}
    long_ctx = text.get("long_term_context", "") or ""
    short_ctx = text.get("short_term_context", "") or ""
    return str(long_ctx), str(short_ctx)


def replay_steps(demo_path, env_name, task):
    """Yields (messages, action) pairs, one per real action taken in the
    trajectory, where `messages` is the OpenAI-style [{role, content}]
    list as it stood right before that action (mirroring what BALROG's
    HistoryPromptBuilder.get_prompt() would have produced at that turn --
    a leading instruction user-message, then alternating user/assistant
    turns for every observation/action pair so far, ending on the current
    observation as a "Current Observation:" user turn since that's what
    the real agent conditions on right before emitting its action).

    Mirrors dataset.py's load_in_context_learning_episode(): the first
    array entry is observation-only (env.reset(), no preceding action),
    popped before pairing observations with actions; iteration stops at
    the first `done` rather than continuing past it (dataset.py's own
    `if done: break`).
    """
    episode = load_episode(demo_path)

    actions = episode.pop("action").tolist()
    rewards = episode.pop("reward").tolist()
    terminated = episode.pop("terminated")
    truncated = episode.pop("truncated")
    dones = np.any([terminated, truncated], axis=0).tolist()
    keys = list(episode.keys())
    observations = [dict(zip(keys, values)) for values in zip(*episode.values())]

    if not observations or not actions:
        return

    # first transition only contains observation (env.reset()) -- same pop
    # order as dataset.py: observation, action, reward, done all popped
    # from index 0 together even though action/reward/done at index 0 are
    # placeholders never used (dataset.py itself discards them the same way).
    first_obs = observations.pop(0)
    actions.pop(0)
    rewards.pop(0)
    dones.pop(0)

    first_long, first_short = obs_text(first_obs)
    instruction = instruction_prompt_for(env_name, task, first_long)

    history = []  # list of (long_ctx, action_str) pairs already taken
    last_long, last_short = first_long, first_short

    for observation, action, reward, done in zip(observations, actions, rewards, dones):
        action_str = str(action)

        # Build the message list as it stood right before this action was
        # taken: instruction + all past observation/action turns +
        # "Current Observation:" for the most recent observation (the one
        # that was current when this action was emitted, i.e. `last_*`
        # from the previous loop iteration / the initial reset observation
        # on the first step). Mirrors history.py's get_prompt() shape.
        messages = _build_messages(instruction, history, last_long, last_short)

        yield messages, action_str

        history.append((last_long, action_str))
        last_long, last_short = obs_text(observation)

        if done:
            break


def _build_messages(instruction, history, current_long, current_short):
    messages = [{"role": "user", "content": instruction}]
    for hist_long, hist_action in history:
        messages.append({"role": "user", "content": f"Observation:\n{hist_long}"})
        messages.append({"role": "assistant", "content": hist_action})
    current_parts = ["Current Observation:"]
    if current_short:
        current_parts.append(current_short)
    if current_long:
        current_parts.append(current_long)
    messages.append({"role": "user", "content": "\n".join(current_parts)})
    return messages


def build_row(messages, action, tok, seq_len):
    """Returns (text, kept: bool). `text` is build_prompt(messages) with
    the target action appended after the trailing "assistant:" cue, so the
    row is a complete prompt+completion SFT example. Truncates the PROMPT
    portion from the left (same convention as balrog_server.py's own
    budget = max(seq_len - max_tokens, 1) handling) to fit seq_len once the
    action's own tokens are accounted for; if even the action alone plus a
    minimal prompt can't fit, the row is skipped rather than emitted broken."""
    prompt = build_prompt(messages)
    action_text = " " + action.strip()

    action_ids = tok.encode(action_text).ids
    if len(action_ids) >= seq_len:
        return None, False  # single action alone blows the whole context; skip

    budget = max(seq_len - len(action_ids), 1)
    prompt_ids = tok.encode(prompt).ids
    if len(prompt_ids) > budget:
        prompt_ids = prompt_ids[-budget:]
        prompt = tok.decode(prompt_ids)
        # decode() can lose the exact trailing "assistant:" cue formatting
        # after truncation; re-anchor it explicitly so the model always
        # sees a clean completion boundary regardless of where the left
        # truncation cut into the flattened text.
        if not prompt.rstrip().endswith("assistant:"):
            prompt = prompt.rstrip() + "\nassistant:"

    text = prompt + action_text
    total_ids = tok.encode(text).ids
    if len(total_ids) > seq_len:
        return None, False
    return text, True


def find_npz_files(records_dir, env_name):
    env_dir = os.path.join(records_dir, env_name)
    if not os.path.isdir(env_dir):
        return []
    return sorted(glob.glob(os.path.join(env_dir, "**", "*.npz"), recursive=True))


def task_of(records_dir, env_name, npz_path):
    """Task subdirectory name, mirroring dataset.py's
    `<records_root>/<env_name>/<task>/*.npz` structure -- the path
    component directly under env_dir."""
    env_dir = os.path.join(records_dir, env_name)
    rel = os.path.relpath(npz_path, env_dir)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", required=True,
                     help="root of an already-extracted records.zip "
                          "(<records-dir>/<env_name>/<task>/*.npz)")
    ap.add_argument("--out", default=os.path.join(DATA, "npc", "balrog_demos.jsonl"))
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP,
                     help="max total output rows across all games")
    ap.add_argument("--seq-len", type=int, default=None,
                     help="override model.py Config.seq_len (default: read from Config())")
    args = ap.parse_args()

    seq_len = args.seq_len if args.seq_len is not None else Config().seq_len

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    stats = {env: {"episodes": 0, "kept": 0, "skipped": 0} for env in ENVS}

    # Real bug found this session (round 11's Crafter episodes emitted
    # NetHack-vocabulary actions like "go west"/"open door" instead of
    # Crafter's real "Move West"/"Do"): this file is consumed downstream
    # by st_prepare.py's BALROG_DEMOS_CAP, which reads sequentially and
    # stops at the cap -- with games written in a fixed ENVS order, the
    # cap silently truncated after only 4 of 6 games, giving minihack and
    # nle (19% and 65% of the full converted data) ZERO real demo rows in
    # every round trained so far, directly explaining their persistent 0%
    # BALROG progression. Fixed by writing games ROUND-ROBIN (one row from
    # each game in turn) instead of sequential blocks, so any downstream
    # prefix-truncating cap gets a genuinely balanced sample across all
    # six games regardless of where it stops -- not just the true per-game
    # cap this converter's own --cap knew about.
    per_game_rows = {env: [] for env in ENVS}
    for env_name in ENVS:
        npz_files = find_npz_files(args.records_dir, env_name)
        if not npz_files:
            continue
        for npz_path in npz_files:
            task = task_of(args.records_dir, env_name, npz_path)
            try:
                steps = list(replay_steps(npz_path, env_name, task))
            except Exception as e:
                print(f"  [{env_name}] failed to replay {npz_path}: {e}", file=sys.stderr)
                continue

            stats[env_name]["episodes"] += 1
            for messages, action in steps:
                text, kept = build_row(messages, action, tok, seq_len)
                if kept:
                    per_game_rows[env_name].append(text)
                    stats[env_name]["kept"] += 1
                else:
                    stats[env_name]["skipped"] += 1

    # Real finding (2026-08-12): under strict round-robin, once a small
    # game's raw pool (e.g. babaisai's real available trajectories) is
    # exhausted, it simply stops contributing while games with larger
    # raw pools keep filling the cap -- so a game's real SHARE of the
    # final mixture is bounded by its raw pool size, not given an equal
    # target share. Direct evidence: babaisai ended up only 349/20000
    # rows (1.7%) in a real converted balrog_demos.jsonl, and BALROG's
    # own real eval logs show babaisai's failed_candidates are almost
    # entirely compass-direction/verb confusion with the OTHER games'
    # much-more-represented action vocabularies (north/south/east/west/
    # "go forward" instead of babaisai's real up/down/left/right) --
    # consistent with the model rarely seeing babaisai's distinct action
    # set during training. Fixed by wrapping (cycling) each game's rows
    # once its pool is exhausted, so every game with at least 1 real row
    # gets an equal target SHARE of the cap (repeating its own rows if
    # needed), not just an equal share of what's naturally available.
    total_rows = 0
    with open(args.out, "w", encoding="utf-8") as f:
        indices = {env: 0 for env in ENVS}
        active_envs = [env for env in ENVS if per_game_rows[env]]
        while total_rows < args.cap and active_envs:
            wrote_any = False
            for env_name in active_envs:
                if total_rows >= args.cap:
                    break
                rows = per_game_rows[env_name]
                i = indices[env_name] % len(rows)
                f.write(json.dumps({"text": rows[i]}) + "\n")
                indices[env_name] += 1
                total_rows += 1
                wrote_any = True
            if not wrote_any:
                break

    print()
    print(f"{'game':<12} {'episodes':>9} {'available':>9} {'written':>8} {'skipped(too-long)':>18}")
    for env_name in ENVS:
        s = stats[env_name]
        print(f"{env_name:<12} {s['episodes']:>9} {s['kept']:>9} {indices[env_name]:>8} {s['skipped']:>18}")
    print(f"\ntotal output rows: {total_rows} (cap={args.cap}, round-robin across all games) -> {args.out}")


if __name__ == "__main__":
    main()
