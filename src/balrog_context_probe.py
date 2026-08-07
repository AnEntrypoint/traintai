"""Real measurement: does each BALROG game's instruction prompt + a
representative single-turn observation fit inside Config.seq_len=512
tokens against our actual tokenizer? Uses the exact same
instruction_prompt_for()/build_prompt() functions balrog_demo_convert.py
and balrog_server.py use at both training-data-build and serving time --
no reimplementation, no assumed token counts.

Run: UV_NO_SYNC=1 uv run python src/balrog_context_probe.py [--seq-len N]
"""
import argparse
import os

from tokenizers import Tokenizer

from balrog_demo_convert import _build_messages, instruction_prompt_for
from balrog_server import build_prompt
from model import Config

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# One representative single-step observation per game, taken verbatim
# from each env's own real observation text shape (BabyAI: the existing
# data/balrog_probes.json samples, already real pulled probe text;
# others: the shortest real observation shape each env's own source
# produces for its short_term_context/long_term_context fields, per
# balrog/environments/<env>/__init__.py's get_text_observation()).
REPRESENTATIVE_OBS = {
    "babyai": "a wall 6 steps forward\na wall 3 steps right\na green key 2 steps left and 4 steps forward",
    "crafter": (
        "You see:\n- grass\n- tree\n- water\n- stone\n- coal\n"
        "You have nothing in your inventory."
    ),
    "babaisai": "ROCK IS PUSH\nBABA IS YOU\nFLAG IS WIN\nWALL IS STOP",
    "minihack": (
        "                                                                                \n"
        "                    ------                                                    \n"
        "                    |....|                                                    \n"
        "                    |.@..|                                                     \n"
        "                    |....|                                                    \n"
        "                    ------                                                    \n"
        "Hp:16(16) Pw:8(8) AC:7 Xp:1/0 T:1"
    ),
    "nle": (
        "                                                                                \n"
        "                    ------                                                    \n"
        "                    |....|                                                    \n"
        "                    |.@..|                                                     \n"
        "                    |....|                                                    \n"
        "                    ------                                                    \n"
        "Player the Rambler   St:16 Dx:12 Co:14 In:11 Wi:9 Ch:8  Neutral\n"
        "Dlvl:1 $:0 HP:14(14) Pw:2(2) AC:6 Xp:1/0 T:1"
    ),
    "textworld": (
        "-= Kitchen =-\nYou are in a kitchen.\nThere is an open fridge here.\n"
        "There is a red apple on the table.\nThere is an exit to the north."
    ),
}

TASK_PER_GAME = {
    "babyai": "BabyAI-MixedTrainLocal-v0/goto",
    "crafter": "default",
    "babaisai": "env/goto_win",
    "minihack": "MiniHack-MazeWalk-9x9-v0",
    "nle": "NetHackScore-v0",
    "textworld": "coin_collector",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=None,
                     help="override Config's default seq_len -- RoPE has no learned "
                          "position parameters, so this measures headroom at any target "
                          "length without needing an actual checkpoint at that length")
    args = ap.parse_args()

    tok = Tokenizer.from_file(os.path.join(DATA, "bpe32768.json"))
    seq_len = args.seq_len if args.seq_len is not None else Config().seq_len
    print(f"seq_len = {seq_len}\n")
    print(f"{'game':10s} {'instr_toks':>10s} {'turn1_toks':>10s} {'fits_1turn':>10s} {'fits_2turn_est':>14s}")

    for game, obs in REPRESENTATIVE_OBS.items():
        task = TASK_PER_GAME[game]
        instruction = instruction_prompt_for(game, task, obs)
        instr_toks = len(tok.encode(instruction).ids)

        messages = _build_messages(instruction, [], obs, "")
        prompt = build_prompt(messages)
        turn1_toks = len(tok.encode(prompt).ids)

        # Estimate a second turn by appending one more real history
        # entry (same obs repeated -- a real trajectory's obs length is
        # game-dependent but roughly stable turn-to-turn for a fixed
        # game/task, so this reuses the SAME measured text rather than a
        # guessed multiplier).
        messages2 = _build_messages(instruction, [(obs, "wait")], obs, "")
        prompt2 = build_prompt(messages2)
        turn2_toks = len(tok.encode(prompt2).ids)

        fits1 = turn1_toks <= seq_len
        fits2 = turn2_toks <= seq_len
        print(f"{game:10s} {instr_toks:10d} {turn1_toks:10d} {str(fits1):>10s} {str(fits2):>14s}")

    print(
        "\nfits_1turn = instruction + ONE observation turn fits in seq_len "
        "(the minimum needed for the model to act at all).\n"
        "fits_2turn_est = instruction + one history turn + current observation "
        "fits (estimates how many turns of real history survive before "
        "left-truncation kicks in, per balrog_demo_convert.py's own "
        "left-truncation policy)."
    )


if __name__ == "__main__":
    main()
