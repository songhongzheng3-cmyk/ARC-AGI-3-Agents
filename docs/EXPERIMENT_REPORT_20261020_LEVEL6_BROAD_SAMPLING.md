# Level 6 Broad Sampling (2026-10-20)

## Scope

Sampled independent branches from the verified `row 620` checkpoint in
`sampling_runs/round16_atomic_navigation/checkpoint620_v2/result.json`. Each
branch replayed the 620-row recording prefix and all 61 saved checkpoint actions
before sampling. This was local offline exploration, not a Kaggle submission or
an Agent score evaluation.

## Results

- 20 valid branches, 740 sampled actions total (seeds `20261020` through
  `20261039`). Each branch stopped after 37 actions when its independent
  checkpoint copy exhausted its 74 resource points.
- All actions reduced the resource counter by 2. The four directional actions
  were sampled nearly evenly: ACTION1 186, ACTION2 186, ACTION3 181, ACTION4
  187.
- The player moved on 553 actions. On 167 actions the player did not move; the
  observation tracker detected dynamic-context changes on 151 of those, and no
  detected scene change on the other 16.
- 21 unique player coordinates were visited across the branches.
- No branch advanced beyond 5 completed levels or reached a terminal win state.

## State Changes

There were 20 detected task-state changes: 7 pattern changes and 13 palette
changes. All pattern changes occurred near the recorded coordinates `(4..6,
0..1)`:

| Transition | Count | Representative observed transition |
| --- | ---: | --- |
| `101/110/011` -> `111/010/010` | 4 | `(6,1)` -> `(5,1)`, ACTION3 (twice); also observed entering `(5,1)` and `(4,1)` |
| `111/010/010` -> `111/101/101` | 2 | `(4,0)` -> `(4,1)`, ACTION2; `(4,1)` -> `(5,1)`, ACTION4 |
| `111/101/101` -> `010/101/110` | 1 | `(4,1)` -> `(5,1)`, ACTION4 |

Palette changes were observed as `8 -> 12` (5 times), `12 -> 9` (4), `9 -> 14`
(3), and `14 -> 8` (1). These are repeated observations, not yet controlled
causal tests. The same coordinate and action can coincide with different
palette outcomes in a changing world, so player position alone is not a
sufficient state key.

## Interpretation and Next Experiment

This is the first useful broad-sampling signal for a candidate local interaction
mechanism: entering or traversing a compact region repeatedly coincides with
ordered pattern transitions. It does not establish whether the trigger is a
tile, a moving entity, timing, or an interaction between them, and it does not
identify the level's complete objective. Do not encode this as a deterministic
planner rule yet.

The next experiment should replay short, controlled prefixes to reproduce each
pattern transition while recording full observations and dynamic-entity context.
Hold the source state and action fixed where possible; vary only the visible
entity phase or preceding wait/movement sequence. Separately, test whether the
pattern chain advances the level when followed through the suspected region.

## Controlled Replay (round 18)

The first replay attempt reported a prefix divergence because the replay tool
restored row 620 and then started the sampled branch immediately. A sampled
branch is actually recorded after the 61 actions in
`round16_atomic_navigation/checkpoint620_v2`; those prerequisite actions must be
replayed before the sampled prefix. `scripts/replay_checkpoint_transition.py`
now infers that prerequisite from the branch result's `branch_result` field (or
accepts `--base-checkpoint-result` explicitly).

With the corrected order, three representative transitions from
`branch_v2_seed20261030` were each replayed three times:

| Check | Source -> final player | Action | Final pattern | Gameplay result |
| ---: | --- | --- | --- | --- |
| 26 | `(6,1)` -> `(5,1)` | ACTION3 | `111/010/010` | deterministic 3/3 |
| 35 | `(4,0)` -> `(4,1)` | ACTION2 | `111/101/101` | deterministic 3/3 |
| 36 | `(4,1)` -> `(5,1)` | ACTION4 | `010/101/110` | deterministic 3/3 |

The raw response includes nondeterministic recording metadata, but the gameplay
observation is identical in all three trials for each check. This confirms that
the three pattern transitions are reproducible under a fixed visible prefix;
it does not yet prove that the chain is the level objective or that it advances
level 6.

Outputs are retained under
`sampling_runs/round18_level6_controlled/seed20261030_check26/`,
`seed20261030_check35/`, and `seed20261030_check36/`.

## Counterfactual Action Controls

To separate action causality from state coincidence, the replay tool now
supports `--target-action`. Each of the three fixed prefixes above was replayed
with all four directional actions, three trials per action. Every case had a
deterministic gameplay observation across its three trials.

| Fixed source check | ACTION1 | ACTION2 | ACTION3 | ACTION4 |
| ---: | --- | --- | --- | --- |
| 26, source `(6,1)`, pattern `101/110/011` | `(6,0)`, unchanged | `(6,2)`, unchanged | `(5,1)`, `111/010/010` | `(6,1)`, unchanged |
| 35, source `(4,0)`, pattern `111/010/010` | `(4,0)`, unchanged | `(4,1)`, `111/101/101` | `(3,0)`, unchanged | `(5,0)`, unchanged |
| 36, source `(4,1)`, pattern `111/101/101` | `(4,0)`, unchanged | `(4,1)`, unchanged | `(3,1)`, unchanged | `(5,1)`, `010/101/110` |

The result is a strong local causal signal: each observed pattern transition is
triggered by entering one specific neighboring cell with one specific action,
while the other three actions preserve the pattern under the same prefix. This
is still a local interaction rule, not a complete level-6 solver; the next
experiment should test whether following the three triggers in sequence
changes the level or exposes a new goal-state signal before spending more
sampling budget.

Counterfactual outputs are retained under
`sampling_runs/round18_level6_controlled/check{26,35,36}_action{1,2,3,4}/`.

All valid per-step observations and summaries are retained under
`sampling_runs/round17_level6_broad_sampling/`. Two earlier harness startup
failures are also retained there with zero sampled actions; they are excluded
from the 20-branch totals above.

## Level 6 Navigation Collision Probe (round 19)

The checkpoint replay diagnostic now records each action boundary's available
actions, tracked moving components, visible object cells, and structured event
details. This keeps navigation anomalies attributable to an observed state
transition instead of inferring a cause from player coordinates alone.

At the reproducible source `(6,3)`, all four actions are available. ACTION1,
ACTION2, and ACTION4 move to `(6,2)`, `(6,4)`, and `(7,3)` respectively;
ACTION3 leaves the player at `(6,3)`. Thus the initial ACTION3 no-op is not an
API-level rejected action. A controlled three-trial comparison from `(7,3)`
then produced the same outcomes each time: ACTION1 -> `(7,2)`, ACTION2 stays
at `(7,3)`, ACTION3 -> `(6,3)`, ACTION4 relocates to `(6,3)`. The color-12,
area-10 component was at `(7,3)` before each action; it followed the player to
`(7,2)` / `(6,3)` when movement succeeded and stayed when ACTION2 was blocked.
That co-movement means it should not be treated as an independent moving
obstacle. ACTION4 additionally emits `player_relocated: (7,3) -> (6,3)`, which
is consistent with a blocked move or game-specific fallback, but does not
identify the rule that caused it.

A short route probe further showed that `(7,4)` blocks ACTION2 from `(7,3)`;
from `(7,2)`, ACTION4 reaches `(8,2)`, but a subsequent ACTION4 does not move,
and the following ACTION2 relocates the player from `(8,2)` to `(6,3)`. All
actions remained available and cost 2 resource points. The presumed ordinary
Manhattan route toward `(9,6)` is therefore invalid under the observed dynamic
state. No pattern, level-count, or terminal-win signal changed in these probes.

The checkpoint replay output now includes this diagnostic context in
`source_state`, `prelude_observations`, and `target_state`. Example retained
probe output: `/tmp/l6return_20261020/` and
`/tmp/l6targetroute_20261020/` (local diagnostics, not competition runs).

The four-action comparison is retained under
`sampling_runs/round18_level6_controlled/player_7_3_ACTION{1,2,3,4}/`. Next,
compare the same action from `(7,3)` under different preceding action phases,
and audit the destination cell's visible occupancy before ACTION4. Until the
relocation rule predicts held-out outcomes, do not encode it as a navigation
constraint. The level remains uncompleted.

## Phase-Controlled ACTION4 Return Probe (round 20)

To test whether the return from `(7,3)` depends on the moving-component phase,
the checkpoint was replayed with prefixes that all end at `(7,3)`, followed by
ACTION4:

| Prelude before target | Resource at source | Tracked moving-component positions | ACTION4 outcome |
| --- | ---: | --- | --- |
| `ACTION4` (baseline) | 72 | `(1,1)`, `(5,7)`, `(3,5)` | `(7,3) -> (8,3) -> (7,3) -> (6,3)` |
| `ACTION4 ACTION1 ACTION2` | 68 | `(3,1)`, `(3,7)`, `(2,4)` | `(7,3) -> (8,3) -> (7,3) -> (6,3)` |
| `ACTION4 ACTION1 ACTION2 ACTION1 ACTION2` | 64 | `(5,1)`, `(1,7)`, `(3,3)` | `(7,3) -> (8,3) -> (7,3) -> (6,3)` |
| `ACTION4 ACTION2` (ACTION2 is blocked) | 70 | `(1,1)`, `(5,7)`, `(3,5)` | `(7,3) -> (8,3) -> (7,3) -> (6,3)` |

Each phase was deterministic across three independent trials, with exactly one
player-image match in every animation subframe. The source gameplay
observations differ across all three phases, and the tracked components occupy
different positions, while the status pattern remains `101/110/011`. No level,
pattern, or terminal-state signal changed. This makes player-image ambiguity
and a return that occurs only in the baseline phase unlikely explanations.
It does not distinguish a fixed collision from a scripted route or another
untracked mechanic, so `player_relocated` remains a descriptive tracker event,
not a causal rule. Do not add a planner constraint based on it yet.

The blocked ACTION2 prefix costs two resource points but leaves all three
tracked component positions unchanged from the baseline phase. Thus counting
submitted actions is not a reliable proxy for dynamic phase; at least in this
case, a blocked action does not advance the observed components.

An additional source-frame audit found that the center pixel of destination
cell `(8,3)` is the inferred floor color, and no tracked static object occupies
that cell. In every animation audit the player visibly enters `(8,3)`
before returning through `(7,3)` to `(6,3)`. A simple stationary wall at the
destination is therefore not supported by the public image; a collision with
something not represented at the cell center or a game-specific movement
effect remains possible.

## Cross-Direction Entry Control

To separate a direction-specific ACTION4 response from an effect associated
with entering `(8,3)`, a second route reached `(8,2)` and used ACTION2. The
source was `(8,2)`, the commanded neighbor was `(8,3)`, and all three trials
again ended at `(6,3)` with the same 17-subframe path through `(8,3)`, `(7,3)`,
and `(6,3)`. This reproduces the return under a different input direction.

The color-12 area-10 and color-9 area-15 components move together at every
animation subframe: `(8,2)/(8,3)`, then `(8,3)/(8,4)`, `(7,3)/(7,4)`, and
`(6,3)/(6,4)`. The color-9 component is therefore part of the moving composite
appearance, not an independently moving obstacle. The cross-direction result
is consistent with a repeatable movement effect at this location, but does
not establish its trigger or explain why the composite continues left after
entering `(8,3)`; retain the trace rather than hard-coding a navigation rule.

Outputs are retained under
`sampling_runs/round18_level6_controlled/player_7_3_ACTION4/`,
`player_7_3_phase_action412_ACTION4/`, and
`player_7_3_phase_action41212_ACTION4/`. The blocked-action control is under
`player_7_3_phase_action42_ACTION4/`. The cross-direction control is under
`sampling_runs/round18_level6_controlled/entry_8_3_from_north_ACTION2/`. These
remain local replay diagnostics; level 6 is not completed.
