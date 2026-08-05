# tai — agent guide

Single-purpose SillyTavern NPC dialog model (28.9M params: 559K dense core +
25.2M PLE table) with a Rust desktop runtime. Ship: `runs/ple-st-r16-grpo.pt`
(honest forge pass 74%), exported to `firmware/model/model.bin`. Best
measured (not yet promoted to ship/export): `runs/ple-st-r23-grpo.pt`,
forge pass 78%, sim_eval oracle match 39% (2026-08-04). `st-tourney-01`
(2026-08-05, base=r23) is the first round to run the survival-sim
tournament self-play loop live in `round.py`, autonomous multi-generation
(`--branches 3 --generations 20 --hours 8`) on Kaggle T4 -- see the GPU
hardware section below for why T4 (not T4x2/P100).

## Commands

```bash
# always this form, or uv reverts the venv to CPU torch
UV_NO_SYNC=1 uv run python src/<script>.py

# one training round (action-forge + SFT top-up + GRPO + forge measurement)
UV_NO_SYNC=1 uv run python src/npc_action_forge.py <prev ckpt> --scenarios 800
UV_NO_SYNC=1 uv run python src/st_prepare.py
UV_NO_SYNC=1 uv run python src/train.py --arm ple --vocab 32768 --d-model 96 \
  --n-layers 6 --n-heads 4 --ple-dim 128 --fixed-ffn 66 --data-suffix _npc \
  --init-from <prev ckpt> --steps 300 --tag st-rN
UV_NO_SYNC=1 uv run python src/npc_grpo.py runs/ple-st-rN-s0.pt --st 150 --steps 200
UV_NO_SYNC=1 uv run python src/npc_forge.py runs/ple-st-rN-grpo.pt --cards 60 --k 6
# all of the above, wired together (also runs action-forge before prepare):
UV_NO_SYNC=1 uv run python src/round.py --prev <prev ckpt> --tag st-rN

# gold multi-sentence chains (static combinatorial data, regenerate only
# if st_world.py's item/place/event tables change -- fixed seed=23)
UV_NO_SYNC=1 uv run python src/st_chains.py

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

## Lessons adapted from the Colab notebook era

The old Colab notebook (gist c75ce7205d295d2f2399f6084987c558) attacked
the same system and produced the right IDEAS with none of the evidence.
Each idea now has a real counterpart in this repo:

- Engine-hook agency tags ([TRAVEL:]/[TRADE:]) -> REAL: [GOTO:]/[DEAL:]
  bracket actions with an economy oracle and held-out accuracy metrics.
- "Interpretability Quotient" -> REAL: sim_eval's action rate,
  invalid-action count, and oracle match split none/GOTO/DEAL.
- "Dialogue purity" -> REAL: action-beats rate, measured 92% -> 0%.
- 10x combinatorial lore diversity -> REAL: combinatorial template banks
  + trade/trait/place fills; no raw scenario splicing.
- HF regularization against overfit -> REAL: real roleplay sets
  (chimbiwide) + a TinyStories slice in every bin build.
- Live per-pulse dashboards -> REAL: round.py's per-round summary block
  plus the forge and sim_eval dashboards, computed from model outputs.
- AGENTS.md-as-progress-doc -> REAL: this file, updated every round.

Anti-patterns from that run, never to be repeated here:
1. Simulated metrics: "sprints" drew learning curves with random.uniform
   and declared "mastery 1.0" by construction. No number is recorded in
   this repo without a model run behind it.
2. Victory by declaration: five identical "final convergence" cells.
   A result lands once, with its measurement.
3. No held-out data: every "eval" scored the same distribution that
   produced it. Held-out sets here use disjoint seeds (sim_eval) or
   unseen cards (forge).
4. Evaluating constants: the 1M-sample "stress test" scored one fixed
   string per profession. Load tests are not correctness tests.
5. Plaintext credentials in notebooks: an HF token was committed to a
   gist (which keeps revision history -- rotating the token is the only
   fix). Secrets never enter this repo, gists, or notebooks.

## Gains projection (measured basis)

- Prepared-data recipe: saturated. r14 74% -> r15 73% (flat). More
  identical rounds buy nothing; do not run them.
- Dialog-only: DONE (92% -> 0% beats, held across r16-r17).
- Action accuracy: the open headroom, now with a measured reward-shaping
  arc (r16 baseline: GOTO 9%/DEAL 5%, emitting 53%, abstention 71%):
  - r17 (v1: exact +1.0 / miss -1.0 / unwarranted -0.5, additive):
    overshot to abstention (6% actions, forge 66%) -- rejected.
  - r19 (v2 partial credit: abstain-miss -0.8, wrong-arg -0.2, wrong-verb
    -0.6, unwarranted -0.3, exact +1.0, all additive): abstention softened
    (13% actions, none-acc 88%, forge 74% held) but GOTO/DEAL exact 0%.
    Root cause measured: the pass-zone sampler (pass = mean reward >= 3.0)
    excludes oracle prompts after 2 fresh visits -- the action gradient
    barely trained.
  - r20 (v2 + oracle-prompt sampling floor, every 3rd step): actions 0%.
    The floor worked (oracle prompts visibly trained) but additive shaping
    leaves clean abstention (~3.2 total) above sloppy attempts (~2.4), so
    more oracle-prompt training reinforced abstention harder.
  - r21 (v4: abstention on an oracle prompt early-returns flat -1.0, no
    dialog-term harvest): equilibrium shattered -- actions 96%, none-acc
    100% held, but 283/284 malformed (unclosed brackets, glued dialog),
    format rate 4%.
  - r22 (r21 + 300 more steps): FLAT -- invalid 272/276 (98.6%), format
    9%, GOTO/DEAL exact still 0%. All-garbage K=8 groups carry no syntax
    gradient: GRPO cannot amplify a well-formed action that never appears
    in the group, and the garbage attractor (2.2) sits closer to the
    policy than the exact form (4.0).
  VERDICT: action accuracy via GRPO reward shaping is a DEAD lever at
  this scale -- four measured variants, all worse than r16's SFT-only
  behavior (53% emission, 92% valid, GOTO 9%/DEAL 5%). Consistent with
  the standing lesson: form/syntax is a data problem, not a reward
  problem. The open path is data-side: rejection-sample the model's own
  VALID oracle-matching actions into the flywheel, and/or denser exact-
  action sim SFT rows. The variants are preserved behind
  `npc_grpo.py --action-reward off|v2|gate` (default off = r16 ship
  recipe; the oracle-prompt sampling floor activates with the flag).
- Chain depth (~0.1 grounded sentences) is the deepest remaining quality
  gap; likely needs sim-generated multi-sentence gold chains, the same
  shape of lever that fixed grounding (object_ungrounded 15% -> 3%).

## Round 17 result

Action reward v1: exact-match +1.0 / miss -1.0 / unwarranted -0.5 taught
near-total abstention (the safe local optimum under K=8 sampling) and
cost 8pp forge pass. Ship stays r16 (74%). This is the second recorded
instance of reward-shaping overshoot (first: template collapse era) --
shape rewards against the measured safe-optimum, not the ideal one.

## Rounds 18-22 results (action-reward arc, all non-ship; r16 stays)

- r18: standalone GRPO probe, never a round; sim_eval actions 1%,
  GOTO/DEAL/BUY 0% -- deepest abstention collapse. Measured and rejected.
- r19: first full round through round.py with v2 partial-credit reward +
  refreshed mixture (PIPPA 1457 real rows in bins, forge capped 2500, new
  sim data). SFT top-up flat (val 2.3702 init -> 2.3756; best=step0, and
  round.py's GRPO now consumes -best.pt when present). Forge 74% held;
  sim none-acc 71%->88% but exact GOTO/DEAL 0% -- sampler starvation
  measured (pass-zone cutoff excludes oracle prompts).
- r20: + oracle sampling floor. Actions 0% -- additive shaping leaves
  abstention (3.2) above sloppy attempts (2.4); floor amplified it.
- r21: + v4 abstain-gate (flat -1.0 early return). Actions 96%, none-acc
  100%, but 283/284 malformed, format 4%. Continuation: r22 for syntax.
- r22: +300 steps from r21. Flat: invalid 272/276, format 9%, exact
  GOTO/DEAL 0%. Syntax never entered K=8 groups, so no gradient; the
  reward-shaping arc on actions stops here (dead lever, see above).
  Third recorded reward-shaping overshoot: additive penalties leave the
  dialog harvest intact, and an early-return gate that removes it only
  trades abstention for garbage.
- PIPPA holdout gate is now runnable (src/holdout_eval.py): r19 357.11 vs
  r16 358.74 teacher-forced ppl -- the round did not overfit real data.

## Sibling-clone consolidation audit (2026-08-03)

Audited /c/dev/tai (older clone at 1b749b7 plus untracked notebook-era
files) for anything worth bringing across. Verdict: nearly nothing --
every artifact there is a mock-era placeholder that the real
implementations here supersede (npc_economy_sim 473B toy vs sim_econ.py,
iterative_loop fake-metric sprints vs round.py, output_filter vs
st_prepare beat stripping, runs/*.jsonl simulated curves, a 0-byte
best_npc_checkpoint.json). The ONE adopted idea: price fidelity as an
eval metric (from npc_final_eval.py) -- sim_eval now reports whether
quoted prices in DEAL scenarios land within 30% of the oracle price.

## World expansion (adopted from the accelerated-runs notebooks)

- Year-tagged EVENTS, ORIGIN_PLACE per trade, LINEAGE per keeper: lore now
  links item -> origin place -> historical event ("shimmering work out of
  Karhold, from before the Comet Year").
- EXPANDED_ITEMS: 4 combinatorial variants per canonical item (adjective x
  price multiplier x structured property/provenance), fixed seed. Filler
  descriptions from the notebooks were rejected; every expanded item keeps
  the property + provenance structure.
- Scarcity World in sim_econ: stock ticks deplete/regrow and every change
  leaves a narratable reason (526 conversations cite them), restock
  supplier chains (234: item -> origin place -> 1.3x reward -> [GOTO]).
- Rejected: fixed [TRADE:50] templates, random-choice oracles, fake sprint
  marathons, mana/combat scope creep. Deep spatial/territory sim is a
  future lever if dialog geography ever needs it.

## Evolutionary-sim research adoption (2026-08-03)

From the third notebook (skill-tree/crafting evolutionary sim): adopted
crafting-chain restocks (gather -> materials -> bench production, 161
conversations cite it), volatility price shocks that move the oracle price
AND the right choice (crash: sell fast; spike: hold firm), keeper levels
scaling markup and haggle floor (masters refuse harder, 84 level-flavored
declines), and apprentice/lineage texture. Rejected: the DNA-selection/
extinction framing -- fitness curves are sim-for-sim's-sake; this sim
exists to label good dialog decisions.

## Anti-overfit + gameplay expansion (2026-08-03)

Data audit: real rows were 3.7K vs 47K synthetic (real ~24% of bins) and
the forge flywheel had become the second-largest component -- a
self-distillation risk. Measures taken:

- PIPPA (PygmalionAI, apache-2.0) pulled and converted: 1457 real RP
  conversations (beat-stripped at prepare) + 74 held out in
  pippa_holdout.jsonl, which NEVER enters the bins -- it is the real-data
  generalization gate, runnable as `src/holdout_eval.py <ckpt>`
  (teacher-forced ppl; r19 357.11 vs r16 358.74, flat = no overfit).
- Forge rows capped at 2500 in st_prepare (self-distillation limit).
- Best-val checkpointing in train.py (ple-<tag>-s0-best.pt on every val
  improvement); round.py's GRPO stage consumes -best over latest when
  present (r19: top-up val rose, best=step0 shielded the GRPO start).
- Gameplay verbs now three: [DEAL] (NPC sells), [BUY] (NPC buys the
  player's item -- 454 convos), [GOTO] (travel). Quest turn-ins (188
  convos) close the restock loop: contract -> player returns with item ->
  reward. [BUY] validity requires the item in the card's "You carry:"
  context. parse_action/oracle_ok/sim_eval updated; eval splits BUY
  separately.
- HF token discipline: the token used for the pull is exposed (public
  gist history + chat) and must be rotated; it was never written to disk.

## r24 regression + empty-response bug (2026-08-04)

st-r24 (action-forge + gold chains added to r23's mixture) regressed
forge pass 78% -> 62%, `no_stop` flaws jumping ~8% -> 35%. Root cause,
found by direct inspection of the injected data, not assumed:
`npc_action_forge.py`'s rejection-sampling condition
(`len(action_lines) <= 1 and oracle_ok(oracle, action)`) let a
completely blank NPC response pass, since `oracle_ok(None, None)` is
True for both "correctly abstained with real dialog" and "generated
nothing." All 1028 rows injected that run had a blank response line --
the model was trained to trail off after emitting nothing. Fixed with a
`dialog_len >= 16` floor mirroring `npc_forge.py`'s existing
`too_short` flaw threshold. `PLACES`' travel-graph exits were also found
broken (4 of 6 places pointed to flavor-text strings with no matching
place entry) and rebuilt into a real symmetric, BFS-verified graph
(`TRAVEL_GRAPH`) ahead of the survival-sim work below.

## Survival-sim arc: sim_world.py, sim_tournament.py (2026-08-04)

User-directed expansion: agents with hunger/thirst/HP/skills/inventory
(`sim_world.py`, wraps `sim_econ.World` rather than forking it), new
verbs `TRAVEL|ATTACK|TALK|USE|WAIT` extending `ACTION_RE` in
`npc_score.py` (regression-verified the existing GOTO/DEAL/BUY parsing
is byte-for-byte unaffected), and `sim_tournament.py` -- a
population/tournament self-play generation loop (not GRPO reward
shaping, which AGENTS.md already records as a dead lever for action
accuracy): K rollout variants per agent-turn at varying temperature,
forked world states, a real measured fitness function (ticks survived +
trades + combats survived + places reached - died), top-fraction
selection into `data/npc/st_survival.jsonl`.

Baseline measured before any training on this: 0% emission of any new
verb against r23 (`sim_baseline.py`), matching a completely deterministic
tournament run (every branch dies at the identical tick regardless of
seed/temperature, since nothing the model does varies yet) -- the honest
starting point, not yet a result. `model.py` gained real attention
padding-mask support (`build_causal_padding_mask`, verified numerically
identical to the prior single-sequence path) so `sim_tournament.py` could
batch all branches in lockstep instead of one sequential call per
branch-turn: measured 2.86x real speedup (77.5s -> 27.1s, identical
config/seed), 51s at the plan's default 32-branch config.

## Kaggle dataset integration (2026-08-04)

Per user direction ("feed it whatever good public kaggle sets we can
find," later "all the sets integrated as the 5% of extra data to prevent
overfitting"): `kaggle_source_convert.py` extracts the shuffle/holdout-
split/cap/write loop `pippa_convert.py` already had (verified byte-
identical RNG output after the refactor) so a new source is a
`convert_row` function plus one `run_conversion` call.

- `kaggle_names_convert.py`: isaacbenge/fantasy-for-markov-generator
  (CC0), 608 decontaminated fantasy names (only the 2 unambiguously-
  synthetic categories used, not the 9 real-culture name lists; a
  blocklist catches the exact real-person contamination pattern found in
  `synthetic_names.jsonl` this session -- see the r24 section above).
- `kaggle_fantasy_convert.py`: mehhti/classic-fantasy-and-adventure-
  literature-corpus. The uploader's "all 124 titles pre-1931 public
  domain" claim was checked, not trusted -- found "A Farewell to Arms"
  (1929, historically estate-enforced) and "The Great Gatsby" (1925,
  US public domain only since 2021) in the list. Fixed with an explicit
  53-title allowlist (pre-1900 or long-settled public domain, genuinely
  fantasy/adventure/mythology genre), not a blanket accept.
- `kaggle_wiki_convert.py`: ffatty/plain-text-wikipedia-simpleenglish
  (MIT), sparse ~5%-of-mixture interleave (`KAGGLE_WIKI_CAP=900`) per the
  Distribution Smoothing / Emergent Misalignment Prevention literature --
  real encyclopedic text plays the same general-coherence role
  TinyStories does, deliberately capped small rather than scaled up.
- `kaggle_werewolf_convert.py` + `kaggle_gamearena_convert.py`: all 13
  user-named Kaggle Game Arena datasets (ultimate-tic-tac-toe,
  bargaining, lines-of-action, coin-game, checkers, clobber, dots-and-
  boxes, dark-hex, word-association, five-in-a-row, poker-heads-up,
  werewolf, chess -- CC BY 4.0, Claude/GPT/Gemini/Grok playing each
  other). Two content shapes found by direct inspection: language-heavy
  games have real reasoning/message text; pure move-log games (chess,
  checkers, etc) are OpenSpiel notation with zero natural-language
  content, rendered as compact state->move lines rather than a
  fabricated dialog card. Combined into one sparse `kaggle_gamearena.jsonl`
  (`KAGGLE_GAMEARENA_CAP=900`); Werewolf additionally keeps its own
  larger `KAGGLE_WEREWOLF_CAP=2000` allocation by explicit user choice.
  `sim_tournament.py` also sparsely interleaves rows from these same
  pools directly into its own generation output (`KAGGLE_INTERLEAVE_EVERY
  = 20`), not only via `st_prepare.py`'s per-round mixture rebuild.
- LLM-generated sources (gamearena, werewolf) are never counted toward
  the real-data ratio below, same discipline as every synthetic source.

**Real-vs-synthetic ratio re-audit (2026-08-04):** the 2026-08-03 audit
measured real rows at ~24% (3.7K/47K). Recomputed against the current
capped mixture (real: chimbiwide 3668 + pippa_st 1457 + kaggle_fantasy
3000 + kaggle_wiki 900 = 9025; synthetic: forge 2500 + action_forge 1028
+ chains 241 + st_world 900 + st_sim 2870 + template 6000 + gamearena 900
+ werewolf 693 + authored 26 = 15158) gives **37.3% real** (9025/24183),
UP from 24% -- adding kaggle_fantasy and kaggle_wiki as real sources
moved the ratio in the right direction even after also adding several
new synthetic (LLM-generated) Kaggle sources. Re-run this computation
after any future cap change; do not assume the ratio holds.

**GPU-hour quota (2026-08-04, WebSearch-confirmed):** Kaggle free tier is
30 GPU-hours/week (P100/T4/T4x2, shared pool) + 20 TPU-hours/week
(v3-8/v5e-8, separate pool), sessions capped ~9-12hrs. A single real T4
beat a P100 on a comparable PyTorch benchmark in cited literature, so T4
(not P100) is the right default; T4x2 only helps once code actually runs
two GPUs in parallel (not yet true here, and AGENTS.md already records
two concurrent torch jobs crashing this machine twice). TPU v5e-8 is an
8-chip pod but `device.py` only grabs one XLA device
(`xm.xla_device()`) -- using the full pod needs real `torch_xla`
SPMD/sharding work, scoped as a separate future item, not a config flag.

## GPU hardware decision: T4 confirmed, T4x2/P100 both ruled out (2026-08-05)

`device.py` gained real (still hardware-unverified) TPU v5e-8 SPMD support
this session (`tpu_chip_count`/`setup_spmd_mesh`/`shard_batch`) but three
real Kaggle TPU kernel pushes (`machine_shape` = `TpuV5e8`, `Tpu1VmV5e8`,
and no `machine_shape` with `enable_tpu: true` alone) all ended ERROR or
stuck QUEUED 20+ minutes, with no CLI mechanism to retrieve ERROR-status
notebook logs. A later minimal probe kernel (no repo/deps, just
`torch_xla.runtime.global_runtime_device_count()`) eventually completed
after the session moved on and confirmed `device_type=TPU device_count=8`
-- the pod itself is reachable, so the earlier ERROR pushes had some
other, never-diagnosed cause. Per user direction this is parked, not
pursued further this session.

Direct probes on real Kaggle GPU kernels (fast queue, unlike TPU) instead
answered the T4x2-vs-P100 question conclusively:
- `machine_shape: "NvidiaTeslaT4x2"` silently provisions a single P100
  (`torch.cuda.device_count() == 1`, `Tesla P100-PCIE-16GB`), not two T4s
  -- the shape string does not do what its name implies on this account/
  region.
- `machine_shape: "NvidiaTeslaP100"` (explicit) confirmed the same P100,
  and its log states directly: "Tesla P100-PCIE-16GB with CUDA capability
  sm_60 is not compatible with the current PyTorch installation" (the
  Kaggle image ships torch 2.10.0+cu128, which only supports sm_70-sm_120).
  `torch.cuda.is_available()` still reports `True`, but any real kernel
  launch would fail.
- Single T4 (sm_75) is therefore the only viable, already-proven GPU shape
  on this image -- unchanged from the existing recipe, now with the
  T4x2/P100 alternatives ruled out by direct evidence rather than cited
  benchmark alone.

`train.py` now runs fp16 AMP (`torch.amp.autocast` + `GradScaler`) on CUDA
to use T4's Tensor Cores (fp16, not bf16 -- T4 lacks fast bf16 Tensor Core
throughput). Confirmed stable on the real `st-tourney-01` data/round length
below: healthy loss curves, no inf/nan, ~65s/300 SFT steps.

## st-tourney-01 generation 0: SFT-checkpoint-path regression, fixed (2026-08-05)

First live run of the autonomous multi-generation tournament loop
(`--branches 3 --generations 20 --hours 8`, base r23) on the real Kaggle T4
kernel. All 3 branches of generation 0 completed SFT cleanly (val ppl
~11.1, wall ~65s/300 steps -- the AMP change confirmed working on real
data) but then crashed identically at the GRPO stage:
`FileNotFoundError: .../ple-st-tourney-01-g0-bN-s0.pt`. Root cause, found
by reading the real kernel logs directly (`kaggle kernels output
--file-pattern`): `run_one_round()` built `sft_ckpt`/`best_ckpt` with a
hardcoded `-s0` suffix, left over from before per-branch seeding existed.
`run_generation()` actually passes a distinct non-zero `sft_seed` per
branch (`1000 + gen_idx*100 + i`, for lineage diversity), and `train.py`'s
real output filename embeds that seed (`ple-<tag>-s<seed>.pt`) -- so the
hardcoded path never matched any file on disk, on any branch, ever. With
all 3 branches returning `None`, `rank_branches()` got an empty list and
the loop correctly halted after generation 0 (per its own
"ALL BRANCHES FAILED ... the run must stop" logic) rather than promote
nothing -- but this meant the entire 8-hour autonomous budget was wasted
after only ~47 minutes of real GPU time. Fixed by deriving the seed
suffix from the same `sft_seed` value already threaded through
`run_one_round` (falls back to `0` when `sft_seed=None`, matching
`train.py --seed`'s own default, so the un-branched single-round path is
byte-identical to before). This is exactly the class of bug this
project's data-mixture/checkpoint-naming discipline exists to catch:
real execution surfaced it, not code review -- the multi-generation
branching path was never actually run end-to-end on Kaggle before this.

## st-tourney-01, corrected run: 3 real generations, new best g1-b2 78%->73% (2026-08-05)

After the round.py fix (above), a fresh kernel run (v6) was manually
stopped at ~2h (user's own call, to surface interim progress rather than
wait the full 8-hour budget blind) after completing 3 real generations.
Confirmed via direct log read (`kaggle kernels output --file-pattern`,
not the notebook's own stdout, which was empty at cancel time):

- **g0** (base r23, 78% forge): branches 65%/70%/70% forge pass -- an
  expected dip from a fresh SFT/GRPO cycle on a mixture now including the
  new tournament-generated data for the first time.
- **g1** (promoted from g0): branches 66%/73%/72% -- real improvement
  over g0, best branch (`g1-b2`) recovering most of the way back toward
  r23's baseline.
- **g2** (promoted from g1): branches 49%/52%/58% -- a real, sharp
  regression, not noise (every branch dropped). Dominant flaw: `no_stop`
  at 33% (`st-tourney-01-g2-b0-forge.log`), the same flaw *class*
  root-caused for r24 above, but action-forge's own oracle-match rate
  stayed flat 0% across g1/g2 identically, so this is a different cause,
  not the same bug recurring -- not yet root-caused, tracked as an open
  PRD row (`generation-2-forge-regression`).

`g1-b2`'s checkpoint (`ple-st-tourney-01-g1-b2-grpo.pt`, 73% forge,
recovered from the kernel before its container tore down) uploaded to
the `heclgang/traintai-checkpoints` Kaggle dataset alongside r23, as a
real base for the next corrected run.

**Real, standing bug found in the same pass:** `sim_tournament.py`'s
`fitness_of()` produced **zero spread** (`min 8 max 8 mean 8.0`) across
every single one of the 3 generations checked -- `ticks_survived` is
constant at the `horizon=8` default since nobody ever dies within 8
ticks, and every other term (trades/combats/places_seen) is apparently
also identical this early in training, so every one of the 32 branches
per tournament run gets literally the same fitness score. The "survivors
kept: 8/32" selection is therefore not selecting anything meaningful yet
-- an arbitrary slice, not the intended top-fraction. Tracked as
`tournament-fitness-zero-spread-bug`/`-root-cause`; leading fix direction
is lengthening `horizon` (more ticks = more chance for real divergence)
before touching the fitness formula itself, but this needs a real
measured test, not a guess.
