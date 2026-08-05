# traintai — training material for the tai NPC model

Everything that produces the SillyTavern NPC dialog model shipped by the
inference repo [AnEntrypoint/tai](https://github.com/AnEntrypoint/tai):
data engines, economy simulation, survival-sim tournament self-play,
training loop, evaluation harnesses, and the living lever log. Consumed
by tai as a git submodule at `traintai/`.

Ship checkpoint: `runs/ple-st-r16-grpo.pt` — honest forge pass 74%,
action-beats 0% (dialog-only), economy actions emitting. Best measured
(not yet promoted to ship/export): `runs/ple-st-r23-grpo.pt`, forge pass
78%. The full measured arc, every round result, and the current hardware
decision (T4, not T4x2/P100 — see below) are in `AGENTS.md`.

## Layout

- `src/device.py` — single accelerator pick for every round-path script
  (XLA TPU > CUDA > MPS > CPU) plus XLA-correct optimizer_step/mark_step,
  plus real (hardware-unverified) TPU v5e-8 SPMD multi-chip support
- `src/st_data.py` — combinatorial card->conversation generator (no raw
  scenario splicing; the mode-collapse attractor that rewrite removed is
  documented in AGENTS.md)
- `src/st_world.py` — world DB (items, prices, places, quests) grounded
  conversations, plus a BFS-verified symmetric `TRAVEL_GRAPH`
- `src/sim_econ.py` — economy simulation and decision oracle; emits
  DEAL/GOTO action conversations plus held-out oracle-labeled eval
  scenarios
- `src/sim_world.py` — survival-sim `SurvivalWorld`/`Agent`: hunger/
  thirst/HP/skills/inventory/trade, wraps (not forks) `sim_econ.World`
- `src/sim_tournament.py` — population/tournament self-play generation
  loop (K rollout variants per agent-turn, forked world states, measured
  fitness selection); sparsely interleaves Kaggle public-data sources
  live during generation, not only via `st_prepare.py`'s mixture rebuild
- `src/branch_diversity.py` — measures real behavioral divergence between
  checkpoint lineages (greedy decode on a fixed probe set) so
  `round.py`'s multi-generation loop can flag lineage collapse
- `src/npc_action_forge.py` — oracle-correct GOTO/DEAL/BUY rejection
  sampling into the training mixture
- `src/st_chains.py` — gold multi-sentence chain generator (static,
  regenerate only if `st_world.py`'s tables change)
- `src/kaggle_*_convert.py` — convert scripts for each integrated public
  Kaggle dataset (names, fantasy corpus, wiki, werewolf, 13 Game Arena
  gameplay sets); shared shuffle/holdout-split/cap/write loop factored
  into `kaggle_source_convert.py`
- `src/st_prepare.py` — mixture builder (real RP + authored + world + sim
  + forge flywheel + template cap + TinyStories + Kaggle sources),
  decontamination and action-beat stripping
- `src/train.py` — PLE TinyLM trainer (WSD schedule, `--optimizer muon`
  A/B, fp16 AMP autocast+GradScaler on CUDA)
- `src/npc_grpo.py` — GRPO adherence RL: full-coverage reward (persona,
  template echo, intent, identity, grounding, stop, repetition, action
  correctness vs oracle), adaptive-difficulty prompt curriculum
- `src/npc_forge.py` — adversarial generate-grade-inject loop and flaw
  dashboard (the co-evolution flywheel; passing rows feed the next round)
- `src/sim_eval.py` — held-out simulation adherence: format, action-beats,
  invalid-action, oracle match (none/GOTO/DEAL separately)
- `src/holdout_eval.py` — teacher-forced ppl on every `*_holdout.jsonl`
  (the real-data generalization gate; run per round, compare vs ship)
- `src/round.py` — ONE GENERATION (or a full multi-generation autonomous
  loop via `--generations`/`--hours`) end to end: action-forge ->
  tournament -> prepare -> SFT -> GRPO -> forge -> sim_eval, with a
  summary block auto-appended to `AGENTS.md`. All rounds run through this.
- `src/export.py` — checkpoint -> PLE1 int4 binary for the Rust runtime
- `AGENTS.md` — the discovered-lever log. Read it before changing anything.

## One round

```bash
uv sync
UV_NO_SYNC=1 uv pip install torch --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch   # first time only: the lock pins CPU torch
mkdir -p runs && curl -sL -o runs/ple-st-r16-grpo.pt https://github.com/AnEntrypoint/traintai/releases/download/v0.1.0/ple-st-r16-grpo.pt
UV_NO_SYNC=1 uv run python src/round.py --prev runs/ple-st-r16-grpo.pt --tag st-rN
```

Autonomous multi-generation training (tournament self-play, diversity-
aware promotion, no human in the loop):

```bash
UV_NO_SYNC=1 uv run python src/round.py --prev runs/ple-st-r23-grpo.pt --tag st-tourney-01 \
  --branches 3 --generations 20 --hours 8
```

GPU/TPU note: `UV_NO_SYNC=1` only protects an already-provisioned venv -- on a
fresh clone you must sync AND install an accelerator torch once first, or
every script dies with `ModuleNotFoundError: torch` (or worse, silently
trains on CPU). For CUDA GPUs that is torch from the cu128 index (above);
on a TPU (Colab v5e) neither is needed: the runtime's system python already
ships the matched torch + torch_xla pair, and the notebook only adds the
missing light deps (`uv pip install --system numpy==1.26.4 requests
tokenizers tqdm`) -- `src/device.py`
auto-detects XLA TPU > CUDA > MPS > CPU, and the notebook below does the
full TPU setup. Run one torch job at a time. `round.py` builds
the TinyStories token bins itself when they are absent (downloads from HF,
no token needed). The Colab form of this recipe (TPU v5e-1) is
[`colab_round.ipynb`](colab_round.ipynb) in this repo —
[open it in Colab](https://colab.research.google.com/github/AnEntrypoint/traintai/blob/main/colab_round.ipynb).

On Kaggle, use a single T4 (`machine_shape: "NvidiaTeslaT4"`), not T4x2 or
P100: T4x2 has been directly confirmed to silently provision a single P100
instead of two T4s on this account, and that P100 (compute capability
sm_60) is explicitly incompatible with the Kaggle image's installed
PyTorch build (sm_70+ required) — see `AGENTS.md`'s GPU hardware decision
section for the full evidence.

## Data regeneration

Everything in `data/npc/` is committed for reproducibility. Regenerating
from scratch: `st_data.py` (needs `character_codex.json`),
`st_world.py` (needs `dprashar-output.json`), `sim_econ.py --rows 3000`,
`sim_econ.py --scenarios 300` (held-out eval), `st_chains.py`, then
`st_prepare.py`. The 300MB TinyStories slice
(`data/tinystories_slice.txt`) is not committed; it is a plain-text slice
of the TinyStories dataset used by `data/prepare.py` and `st_prepare.py`.
Kaggle source datasets are downloaded and converted on demand by their
respective `kaggle_*_convert.py` scripts, not committed as raw source.
