# traintai — training material for the tai NPC model

Everything that produces the SillyTavern NPC dialog model shipped by the
inference repo [AnEntrypoint/tai](https://github.com/AnEntrypoint/tai):
data engines, economy simulation, training loop, evaluation harnesses, and
the living lever log. Consumed by tai as a git submodule at `traintai/`.

Ship checkpoint: `runs/ple-st-r16-grpo.pt` — honest forge pass 74%,
action-beats 0% (dialog-only), economy actions emitting. The full measured
arc is in this repo's README history and in `AGENTS.md`.

## Layout

- `src/st_data.py` — combinatorial card->conversation generator (no raw
  scenario splicing; the mode-collapse attractor that rewrite removed is
  documented in AGENTS.md)
- `src/st_world.py` — world DB (items, prices, places, quests) grounded
  conversations
- `src/sim_econ.py` — economy simulation and decision oracle; emits
  DEAL/GOTO action conversations plus held-out oracle-labeled eval
  scenarios
- `src/st_prepare.py` — mixture builder (real RP + authored + world + sim +
  forge flywheel + template cap + TinyStories), decontamination and
  action-beat stripping
- `src/train.py` — PLE TinyLM trainer (WSD schedule, `--optimizer muon` A/B)
- `src/npc_grpo.py` — GRPO adherence RL: full-coverage reward (persona,
  template echo, intent, identity, grounding, stop, repetition, action
  correctness vs oracle), adaptive-difficulty prompt curriculum
- `src/npc_forge.py` — adversarial generate-grade-inject loop and flaw
  dashboard (the co-evolution flywheel; passing rows feed the next round)
- `src/sim_eval.py` — held-out simulation adherence: format, action-beats,
  invalid-action, oracle match (none/GOTO/DEAL separately)
- `src/round.py` — ONE ROUND end to end: prepare -> SFT -> GRPO -> forge
  -> sim_eval, with a summary block. All rounds run through this.
- `src/export.py` — checkpoint -> PLE1 int4 binary for the Rust runtime
- `AGENTS.md` — the discovered-lever log. Read it before changing anything.

## One round

```bash
uv sync
UV_NO_SYNC=1 uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch   # first time only: the lock pins CPU torch
mkdir -p runs && curl -sL -o runs/ple-st-r16-grpo.pt https://github.com/AnEntrypoint/traintai/releases/download/v0.1.0/ple-st-r16-grpo.pt
UV_NO_SYNC=1 uv run python src/round.py --prev runs/ple-st-r16-grpo.pt --tag st-rN
```

GPU note: `UV_NO_SYNC=1` only protects an already-provisioned venv -- on a
fresh clone you must sync AND reinstall torch from the cu128 index once
first, or every script dies with `ModuleNotFoundError: torch` (or worse,
silently trains on CPU). Run one torch job at a time. `round.py` builds
the TinyStories token bins itself when they are absent (downloads from HF,
no token needed). The Colab form of this recipe is the pinned gist
(notebook link at the top of the tai README's NPC section).

## Data regeneration

Everything in `data/npc/` is committed for reproducibility. Regenerating
from scratch: `st_data.py` (needs `character_codex.json`),
`st_world.py` (needs `dprashar-output.json`), `sim_econ.py --rows 3000`,
`sim_econ.py --scenarios 300` (held-out eval), then `st_prepare.py`.
The 300MB TinyStories slice (`data/tinystories_slice.txt`) is not
committed; it is a plain-text slice of the TinyStories dataset used by
`data/prepare.py` and `st_prepare.py`.
