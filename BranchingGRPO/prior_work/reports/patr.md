<!-- First written 2026-10-03 from an automated summary of the arXiv HTML; the "First-hand read" section at the end is from a direct read of the full paper text (v1, incl. appendices) the same day. -->

# PaTR: Process Reward Informed Tree Rollout for Effective Multi-Turn RL

arXiv 2607.15610 (17 Jul 2026). Xintong Li, Sha Li, Yuwei Zhang, Changlong Yu, Rongmei Lin, Hongye Jin, Shuyi Guan, Xin Liu, Linwei Li, Qingyu Yin, Jingbo Shang (UC San Diego, Amazon). No repo found.

## Problem and motivation
- GRPO/RLOO spend the rollout budget uniformly on independent full trajectories. In long-horizon agent tasks this wastes budget on uninformative trajectories (e.g. repeated tool-use loops) and under-explores promising intermediate states.
- Critique of prior tree methods: entropy/uncertainty-driven branching (ARPO-style) does "not directly measure whether a partial trajectory is making meaningful progress"; per-turn best-action search (TSR-style) "can bias the rollout distribution away from the current policy and discard informative negative examples".
- Framing: effective exploration = deciding where to branch.

## Method
- Start with B0 = 4 branches from the prompt.
- Advance active branches K steps (K = 5 FrozenLake, 13 SWE), then score each partial trajectory at checkpoints K, 2K, 3K, ...
- Scorers: (1) heuristic over state and accumulated step reward; (2) pretrained process reward model (Skywork-o1-Open-PRM) over a serialised partial trajectory, step scores aggregated (last / mean / max / weighted); (3) LLM judge with phase-specific prompts (diagnosis, editing, verification), blended with the heuristic: s = λ·s_judge + (1−λ)·s_heur.
- Expand: top-scoring branches each get M = 2 children sampled from the same intermediate state.
- Survive: moderate scorers continue another K steps.
- Prune: s < median − α (α = 0.5), or the same action repeated Θ = 6 turns.
- Pruned branches are kept in the training group as negatives.
- Advantage: plain GRPO mean/std normalisation of outcome reward over the whole group. Process scores only allocate budget; they never enter the reward.
- No explicit total-budget constraint stated; runs until no active branch remains.

## Not stated
- How sandbox/Docker state is duplicated at a fork.
- Cost of the scorer calls; total compute versus GRPO.
- Any formal bias analysis. A remark acknowledges "a controlled and localized bias" from non-uniform allocation and argues actions are still sampled from the current policy.

## Results
- FrozenLake: 75.8 vs GRPO 66.5 (Qwen2.5-0.5B); 74.3 vs 66.8 (Qwen2.5-3B).
- SWE-Bench Verified, Qwen3-4B trained on R2E-Gym: PaTR-PRM 27.2, ARPO 25.4, GRPO 22.2 (group size 8). Baselines also include DAPO and Tree-Random.
- Ablations: M=3 → 25.8; K=15 → 23.4, K=20 → 24.6; dropping pruned trajectories from the group → 25.8 (PRM) and 23.4 (judge, from 26.0).

## Stated limitations
- Depends on scorer quality; the PRM may favour longer responses.
- Only FrozenLake and SWE-Bench evaluated.

## First-hand read (2026-10-03): corrections, additions, judgement

Confirmed from the full text: B0 = 4, M = 2, K = 13 (SWE) / 5 (FrozenLake), α = 0.5, Θ = 6, pruned branches kept, plain GRPO advantage over the tree group (Eq. 9), loss is the standard per-trajectory token-mean clipped objective (Eq. 2).

Additions the summary missed:
- **Top-k = 2**: exactly the two highest-scoring active branches are expanded at each checkpoint (Table 3). The parent is replaced by its children.
- **Group size is variable and unreported.** With the defaults and a 50-turn cap, checkpoints fall at turns 13, 26, 39, each adding net +2 branches, so at most 10 leaves (my derivation). GRPO and DAPO baselines use 8 independent trajectories; ARPO and Tree-Random use 4 initial branches.
- **No budget accounting.** "Same training budget" is asserted; there is no table of tokens, turns, tool calls or wall-clock per method. Scorer cost is excluded.
- **The scorer is a larger model than the policy**: a 7B PRM or 7B judge scoring a 4B policy, on separate vLLM servers.
- **Framework**: rLLM; SWE-agent scaffold with Docker. Batch size 8, 300 steps, one H200 node. Synchronous ("resample with updated policy" each step).
- **Nothing on how a fork is implemented**: no mention of sandbox snapshot, container copy, action replay, or KV-cache reuse. "Reuses shared prefixes" is the only statement.
- **Reward of an early-stopped branch is not stated.** They are "retained as outcome-labeled trajectories".
- **PRM scorer uses only the last 10 steps and the last-step score.**
- No code, no seeds, no confidence intervals.

Judgement (mine):
- **Algorithmically this is the advisor's idea**: start with fewer than G, observe, fork from an intermediate turn sharing the prefix, decision driven by how the trajectory is going.
- **The evidence that the scorer matters is weak.** SWE-Bench Verified: GRPO 22.2, Tree-Random 24.8, ARPO 25.4, PATR-Judge 26.0, PATR-PRM 27.2. Half the gain over GRPO comes from the tree alone. On 500 instances the binomial standard error of one resolved rate near 25% is about 1.9 points, so PATR vs Tree-Random (2.4) and vs ARPO (1.8) are within noise for single runs. The K ablation is non-monotonic (K=13: 27.2, K=15: 23.4, K=20: 24.6), which also suggests run-to-run noise of a few points.
- **The score-based prune rule rarely fires.** Scores are in [0, 1] and the rule is s < median − 0.5, so it needs a median above 0.5 and a branch at least 0.5 below it. The loop detector (Θ = 6) is probably the main early stop.
- **An alternative explanation for the gain**: looping trajectories are stopped and kept as negatives, which teaches "do not loop / finish". Env-done rate rises from 79 (GRPO) to 92 (PATR), and Tree-Random also reaches 84. This is separate from where branches are placed.
- **Prefix over-weighting is untreated.** A trunk expanded at every checkpoint has its first 13 turns in up to 8 leaves, each a full row in the loss. Promising prefixes get proportionally more gradient.
- **Open after PATR**: fork implementation and cost, KV reuse, async training, equal-compute evaluation, bias/weighting, any signal that does not need a 7B external scorer.
