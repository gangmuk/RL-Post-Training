# Prior work on branching / tree rollouts for GRPO

Compiled 2026-10-03. Full per-paper reports are in [reports/](reports/); this file is the synthesis.

## How much to trust this

- The papers were read by subagents, not by the author of this synthesis, and their claims were not spot-checked.
- Tree-GRPO, TreeRL, TreePO and ARPO were read in full together with their repos. No code was executed.
- PATR was read directly (pages 1–12). BPO and Crab were read through an automated summary of the arXiv HTML: mechanisms are reliable, exact numbers should be checked before citing.
- PATR, BPO and EPIG-Tree have no code to verify against.
- The async literature was seen mostly at abstract level.
- "No prior work found" on the async axis is a search result, not a proof. Several relevant papers are under three months old.

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

## Reports

| File | Covers | Read depth |
|---|---|---|
| [reports/tree-grpo.md](reports/tree-grpo.md) | Tree-GRPO | full paper (v3) + repo |
| [reports/treerl.md](reports/treerl.md) | TreeRL / EPTree | full paper + repo |
| [reports/treepo.md](reports/treepo.md) | TreePO | full paper + repo |
| [reports/arpo.md](reports/arpo.md) | ARPO (and pointers to AEPO, FR3E) | full paper (v1) + repo |
| [reports/patr.md](reports/patr.md) | PATR | automated summary of arXiv HTML |
| [reports/survey.md](reports/survey.md) | TRPO vine, PATR, BPO, EPIG-Tree, ATPO, and about 15 others | mixed; tagged per item |

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
