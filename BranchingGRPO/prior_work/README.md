# Prior work on branching / tree rollouts for GRPO

Compiled 2026-10-03. Full per-paper reports are in [reports/](reports/); this file is the synthesis.

## How much to trust this

- The papers were read by subagents, not by the author of this synthesis, and their claims were not spot-checked.
- Tree-GRPO, TreeRL, TreePO and ARPO were read in full together with their repos. No code was executed.
- PATR was read directly (pages 1–12). BPO and Crab were read through an automated summary of the arXiv HTML: mechanisms are reliable, exact numbers should be checked before citing.
- PATR, BPO and EPIG-Tree have no code to verify against.
- The async literature was seen mostly at abstract level.
- "No prior work found" on the async axis is a search result, not a proof. Several relevant papers are under three months old.
- **Round 3 (second session, same day) is weaker still.** That session's network policy blocked arxiv.org and every mirror, for both shell and WebFetch. BPO, EPIG-Tree, RTPO and the async papers were read only through web-search snippets. Nothing from round 3 is verbatim. See [reports/bpo_epig.md](reports/bpo_epig.md) and [reports/async_gap.md](reports/async_gap.md). The slime findings in round 3 are first-hand code reads of upstream slime, not the fork.

## Comparison

| Paper | arXiv | Fork unit | When the fork is decided | Signal | Stateful sandbox | Code |
|---|---|---|---|---|---|---|
| PATR | 2607.15610 | agent turn | online, every K turns | process scorer (heuristic, reward model, or LLM judge) | yes (SWE in Docker); fork mechanism not described | none found |
| ARPO | 2507.19849 | agent turn | online, after each tool call | token entropy | no | yes |
| BPO | 2607.14171 | agent step | post hoc, from one backbone trajectory | token entropy | yes, snapshot and restore | none |
| EPIG-Tree | 2609.20004 | step | post hoc, after pilot trajectories | cost-aware variance criterion | no (cloned simulators) | none seen |
| ATPO | 2603.02216 | dialogue turn | online, expand then prune | learned critic uncertainty | no | yes |
| Tree-GRPO | 2509.21240 | agent step | fixed schedule | uniform random | no | yes |
| TreeRL | 2506.11902 | token | fixed counts | token surprisal picks location | no (single-turn math) | yes |
| TreePO | 2508.17445 | 512-token segment | every segment boundary | none (log-prob variant was a null result) | no (single-turn math) | yes |

All of them train synchronously with one policy version for the whole tree.

Added in round 3 (snippet-level only):

| Paper | arXiv | Fork unit | When the fork is decided | Policy versions in a tree | Code |
|---|---|---|---|---|---|
| RTPO | 2608.18682 | agent turn k | reverse order: later turns trained first, then siblings forked at turn k | **mixed by design**: syncs weights before forking, so siblings run under a newer policy than their prefix; trajectory IS weight set to 1 | not checked |
| MBPO | 2608.07581 | vision-language decision point | not clear | replay buffer reuses old segments within an age window | not checked |

Other papers noted in the survey (see [reports/survey.md](reports/survey.md)): AT²PO (2601.04767), IGRPO (2607.06223), TSR (2602.11767), GRPO-MA (2509.24494), AEPO (2510.14545), VinePPO (2410.01679), FR3E (2507.07017), Reinforce-Ada (2510.04996), Selective Rollout (2605.05802), SARA (2607.26253), VIP (2602.01601), Crab (2604.28138), DeltaBox (2605.22781), VCPO (2602.17616), Single-Rollout Asynchronous Optimization (2607.07508), AReaL-DTA (2602.00482), PrefixRL (2601.18795), Prefix-Normalized Policy Optimization (2608.01418), TRPO vine (1502.05477).

## Findings from the code reads

- **ARPO's released code is much less adaptive than its paper.** The branch decision is a coin flip with probability about 0.5 + 0.2·ΔH, not a threshold. The entropy baseline dict is never reset between training steps. In practice it behaves close to random early forking with a slight entropy tilt.
- **Every tree method with released code (Tree-GRPO, TreeRL, TreePO, ARPO) trains shared prefix tokens once per descendant.** A prefix with more branches gets proportionally more gradient weight. Only TreeRL reweights (1/√n). None analyses it.
- **None analyses the selection bias from an adaptive branch rule.** PATR acknowledges it; BPO claims unbiasedness for its sibling baseline.
- **Evidence that branch placement matters is weak.** TreePO's log-prob-driven allocation was no better than uniform; EPIG-Tree found placement within noise on single-turn math; TreeRL's gain mostly disappears at equal training sequences (16 vs 16).
- **Tree-GRPO has no KV reuse and no environment forking** (stateless search tool; the prefix is re-submitted as a fresh prompt).

## Assessment against the idea

**(a) Adaptive online branching: already done.** ARPO and PATR both start with fewer than G and fork online at turn boundaries. ATPO does online expand-or-prune. PATR is the closest overall (4 initial branches, SWE-Bench). Still open: a rule driven by group-level outcome information, a cost-aware criterion applied online, and starting from a single trajectory with online forking.

**(b) Forking stateful environments mid-trajectory: done.** BPO on the algorithm side (Docker snapshot/restore, sibling baseline), Crab on the systems side (checkpoint/restore runtime pitched at tree RL). Still open: coupling the sandbox fork with KV-cache sharing, and fork latency as an input to the branch decision.

**(c) Interaction with async training and stale weights: nothing found.** BPO lists async tree training as future work; SARA lists tree rollouts as open. The async literature (VCPO, SAO, Prefix-Normalized PO) handles flat rollouts only, and SAO argues for dropping groups under async. Nearest building blocks: PrefixRL (mask gradients on an off-policy prefix, train the continuation) and AReaL-DTA (prefix-tree training in an async framework). Untouched problems:

- a fork sampled under weights θ' from a prefix and KV cache produced under θ;
- sibling baselines that mix policy versions;
- KV-cache invalidation at weight sync;
- fork now or wait, given a staleness bound.

A pure "adaptive branching GRPO" paper would be hard to position against PATR and ARPO. A paper about making tree rollouts work in an async colocated system would not have that problem.

### Round 3 revision of (c)

The gap still holds in its strong form: no paper found runs tree rollouts in a genuinely asynchronous trainer, without a sync barrier per fork, under a staleness bound. Three of the four "untouched problems" listed above are less untouched than they looked:

- **Continuing under θ' on KV produced under θ is existing practice for flat rollouts.** PipelineRL (2509.19128) keeps the KV cache through in-flight weight updates on purpose. Upstream slime implements it as `--flush-cache-interval ≠ 1` with in-place pause ([../notes/slime_async_path.md](../notes/slime_async_path.md)). torchtitan PR #4761 keeps per-group KV across policy updates and salts the cache by version so new groups cannot hit old KV. A fork that reuses its parent's stale KV is the same situation applied to a new branch. It is not new on its own.
- **Forking under newer weights from an older prefix is done by RTPO** (2608.18682), the closest threat found. It syncs weights before generating siblings at turn k and treats the old prefix as context with IS weight 1. That is a barrier, not lag, and it gives no correction for the stale prefix. Any claim has to be "no barrier, bounded lag", not "fork from a stale prefix".
- **Multi-version trajectories have a formalism.** Missing Old Logits (2605.12070) treats trajectories that span several policy versions under partial rollout. PNPO (2608.01418) gives a prefix-normalised IS weight, the geometric mean of token ratios along the prefix, which is the natural weight at a fork point. PrefixRL (2601.18795) masks the off-policy prefix and trains only the continuation. slime already has that masking for flat partial rollouts (`--mask-offpolicy-in-partial-rollout`).

What remains open, as far as round 3 found:

1. **Siblings that mix policy versions within one tree**, and what that does to the group baseline. VCPO (2602.17616) has a minimum-variance off-policy baseline for flat groups; nothing applies it to tree siblings.
2. **The fork decision as a scheduling problem coupled to weight sync.** With KV flushed at sync, a fork issued after a sync re-prefills the whole prefix; with in-place sync it is cheap but stale. A fork from an old prefix is born with that prefix's age and may already exceed the staleness bound. Nobody treats "fork now, before the next sync, or not at all" as a decision.
3. **SAO's objection must be answered.** Single-Rollout Asynchronous Optimization (2607.07508) argues that groups force waiting on the slowest member, which adds latency-driven off-policyness, and replaces the group with a value model. A tree makes this worse unless forks are dispatched and scored without waiting for every sibling.

### Round 3 notes on estimator correctness

- **BPO** (snippet level): leave-one-out sibling baseline, unbiased per state given s_t, variance bound K/(K−1)·E[Var(R|s_t)]. A snippet, possibly from a reviewer, says the entropy-chosen tree over-weights high-entropy states with no importance correction. Compute is matched by number of returns, 1 + M(K−1) = N, not by tokens or wall-clock.
- **EPIG-Tree** (snippet level): edge advantage V̂(child) − V̂(h) applied only to the tokens of that branch segment. Its own negative result: on math, placement is within single-seed noise, and masks and advantage construction dominate.
- **OPTS-TTPO** (2609.40035, not read) reportedly has a branch-aggregation lemma that weights transitions so heavily expanded regions are not over-counted. It is a further constraint on any estimator-correctness claim.
- **slime's agent path already dedups shared prefix turns.** `TrajectoryManager` trains a shared turn only on the first leaf that claims it and masks it elsewhere. The repos above all train the prefix once per descendant. slime then gives the prefix that one leaf's advantage, which is the wrong credit for an exploration tree. EPIG's edge advantage is the fix.

## Reports

| File | Covers | Read depth |
|---|---|---|
| [reports/tree-grpo.md](reports/tree-grpo.md) | Tree-GRPO | full paper (v3) + repo |
| [reports/treerl.md](reports/treerl.md) | TreeRL / EPTree | full paper + repo |
| [reports/treepo.md](reports/treepo.md) | TreePO | full paper + repo |
| [reports/arpo.md](reports/arpo.md) | ARPO (and pointers to AEPO, FR3E) | full paper (v1) + repo |
| [reports/patr.md](reports/patr.md) | PATR | automated summary of arXiv HTML |
| [reports/survey.md](reports/survey.md) | TRPO vine, PATR, BPO, EPIG-Tree, ATPO, and about 15 others | mixed; tagged per item |
| [reports/bpo_epig.md](reports/bpo_epig.md) | BPO and EPIG-Tree estimators, bias, fork cost | search snippets only (round 3) |
| [reports/async_gap.md](reports/async_gap.md) | Gap check for async trees; RTPO, SAO, PrefixRL, AReaL-DTA, PNPO, VCPO, Missing Old Logits, MBPO | search snippets only (round 3) |

## Papers

`papers/` holds the PDF and extracted text for Tree-GRPO, TreeRL, TreePO and ARPO (plus ARPO's arXiv HTML). The survey's papers were not downloaded.

## Repos

Cloned into `repos/` (git-ignored, about 210 MB). To re-create, clone the URL and check out the commit.

| Repo | URL | Commit | Commit date |
|---|---|---|---|
| ARPO (includes AEPO) | https://github.com/dongguanting/ARPO | 4d19a74 | 2026-09-12 |
| ATPO | https://github.com/Quark-Medical/ATPO | 25af836 | 2026-04-29 |
| Reinforce-Ada | https://github.com/RLHFlow/Reinforce-Ada | 05be605 | 2025-11-29 |
| Tree-GRPO | https://github.com/AMAP-ML/Tree-GRPO | 19bf3fa | 2026-01-27 |
| TreePO | https://github.com/multimodal-art-projection/TreePO | f055523 | 2025-10-08 |
| TreeRL | https://github.com/THUDM/TreeRL | 63ea99e | 2025-06-16 |
| selective-rollout | https://github.com/zhiyuanZhai20/selective-rollout | ba86cdc | 2026-05-02 |

The reports cite file paths under the original session scratchpad (`/private/tmp/claude-502/.../scratchpad/prior_work/<repo>/`). The same trees are now under `repos/<repo>/`.
