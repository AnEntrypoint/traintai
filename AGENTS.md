# tai — agent guide

Single-purpose SillyTavern NPC dialog model (28.9M params: 559K dense core +
25.2M PLE table) with a Rust desktop runtime. Ship: `runs/ple-st-r14-grpo.pt`
(honest forge pass 74%), exported to `firmware/model/model.bin`.

## Commands

```bash
# always this form, or uv reverts the venv to CPU torch
UV_NO_SYNC=1 uv run python src/<script>.py

# one training round (SFT top-up + GRPO + forge measurement)
UV_NO_SYNC=1 uv run python src/st_prepare.py
UV_NO_SYNC=1 uv run python src/train.py --arm ple --vocab 32768 --d-model 96 \
  --n-layers 6 --n-heads 4 --ple-dim 128 --fixed-ffn 66 --data-suffix _npc \
  --init-from <prev ckpt> --steps 300 --tag st-rN
UV_NO_SYNC=1 uv run python src/npc_grpo.py runs/ple-st-rN-s0.pt --st 150 --steps 200
UV_NO_SYNC=1 uv run python src/npc_forge.py runs/ple-st-rN-grpo.pt --cards 60 --k 6

# runtime
./target/release/tai generate --model firmware/model/model.bin \
  --tokenizer data/bpe32768.json --prompt "..." --tokens 80 --stop-string "Player:"
```

RAM discipline: ONE GPU/torch job at a time on this box; two concurrent
training jobs have crashed the machine twice.

## Discovered levers (the load-bearing knowledge)

Ordered by measured impact. Each was found by adversarial measurement, not
intuition -- the instrument always came before the fix.

1. **Grader visibility before training.** You cannot improve what the
   grader cannot see. The "55% pass" era was 3% honest + 81% template echo
   once a `template_echo` flaw existed. Every new behavior target needs a
   flaw class / reward term FIRST, then training.
2. **Data contamination beats capacity.** Mode collapse was ~16 hardcoded
   response templates x ~1,500 reps each in the bins, not a param limit.
   Fix: combinatorial generators (opener x grounding x closer), world-DB
   grounding, decontamination filters. Any repeated exact string in
   training data is an attractor the model WILL collapse onto.
3. **Reward coverage gaps hide as plateaus.** GRPO reward gave full intent
   credit to 4/6 question types for months (keys None -> free +1). Check
   every reward clause's coverage over the actual prompt distribution.
4. **RL objective scaling.** Policy-gradient loss must be per-token mean,
   advantage std-normalized. Summed logprobs x window doubling = collapse
   (no_stop 95%). Loss spikes >10x baseline mean the objective, not the lr.
5. **Data semantics leak into behavior.** Forge rejection rows stored
   without the turn marker taught response->eot (no_stop 43%). Any derived
   dataset must preserve the exact transition you want at runtime.
6. **GRPO fixes what SFT cannot.** Persona swap: 200M SFT tokens never
   below ~20%; 300 GRPO steps -> ~0%. Identity/adherence = reward problem.
   Form/structure = data problem. Know which lever class your flaw is in.
7. **Checkpoint owns its prompt convention.** `Name:` (no trailing space)
   outperforms the training-literal `Name: ` by 46% vs 32%; the GRPO
   policy adapted to it. Measure prompt variants; do not assume fidelity
   to training text is best.
8. **Muon < AdamW at this scale.** val 3.44 vs 2.90 at matched 800 steps.
   Revisit only at much longer horizons.
9. **Flywheel compounds.** Forge passing rows -> next round's bins:
   878 -> 3106 rows drove 42 -> 74% across five rounds. Rejection-sampled
   self-data + a flaw the reward can see = steady gains.
10. **Decoding is never the depth lever.** Temperature/top-k grids moved
    nothing structural in any era.

## Inferred next levers (from the pattern of what worked)

- Dialog-only constraint (strip action beats): model emits *action text*
  50% of the time (measured); same flaw+reward+data recipe applies.
- Machine-readable action triggers grounded in world DB: extends the
  grounding lever from items to decisions.
- Economy-sim decision oracle: replaces hand-authored gold answers with
  simulated utility-optimal choices -- the generalization of "world-DB
  grounded SFT" from facts to choices.
- Per-sentence chain coherence is still ~0.1 (measured): the deepest
  remaining quality gap; likely needs gold multi-sentence chains from the
  sim, not more rounds of the current recipe.

## Current lever ranking (max-progress-first protocol)

Always train the lever with the largest measured headroom; switch when a
round comes back flat; revisit flat levers after others move (they unlock).
Prepared-data recipe saturated at 74% (r14->r15 flat). Active frontier:
output-scope narrowing (dialog-only, action triggers) and sim-oracle data.

## Sim arc (round 16+)

- `src/sim_econ.py` -- economy oracle: stock/demand/markup/haggle-floor
  world; emits ST conversations with at most one bracket action
  (`[DEAL: item gold]`, `[GOTO: place]`) plus abstention/near-miss cases
  (out-of-stock -> no DEAL, nowhere -> no GOTO). `--scenarios N` writes
  held-out oracle-labeled eval scenarios (disjoint seed).
- `src/sim_eval.py` -- separate metrics per the research discipline:
  format rate, action-beats rate (dialog-only target), invalid-action
  rate, oracle match split none/GOTO/DEAL.
- Baseline (r14, pre-sim): beats 92%, GOTO 0%, DEAL 0%, abstain 100%,
  format 100%. Everything except abstention is open headroom.
- Research verdicts folded in: bracket verbs on their own line (ReAct/IF
  parser lineage), abstention + low command rate in data (SayCan/
  Toolformer), separate command-vs-dialog evals (Gorilla), engine-side
  tolerance for bad actions (Concordia: the engine is the referee).
  Future lever: verbs/entities as dedicated BPE tokens (Octopus v2) --
  needs a tokenizer retrain, parked.
- `src/round.py` -- one round end to end (prepare -> SFT -> GRPO -> forge
  -> sim_eval) with a summary block. All future rounds run through it.
