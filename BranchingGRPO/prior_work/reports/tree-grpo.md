<!-- Subagent report, saved verbatim on 2026-10-03. Not independently verified. Brief: agent_briefs/tree-grpo.txt -->

The paper exists as recalled, and its branching is uniformly random on a fixed schedule: there is no adaptive forking signal anywhere in the paper or the code. I read the full paper (arXiv v3, 27 pages including appendices) and the official repo's tree-search path; nothing was run. The clone is pinned at commit `19bf3fa` (2026-01-27).

## 1. Citation
- **Title:** "Tree Search for LLM Agent Reinforcement Learning" (method name: Tree-GRPO).
- **Authors:** Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, Liaoni Wu (Xiamen University; AMAP, Alibaba Group; SUSTech).
- **arXiv:** 2509.21240, submitted 2025-09-25; v3 dated 2026-03-18. Published at ICLR 2026.
- **Repo:** https://github.com/AMAP-ML/Tree-GRPO, built on Search-R1 and veRL; the README says the implementation is "inspired by TreeRL".

## 2. Tree construction
- **Node:** one complete agent step, i.e. a (thought, action, observation) tuple. A fork reuses context through the parent's observation, so siblings diverge from the next thought onward.
- **Procedure ("initialize-then-expand", §3.1):**
  1. Generate M independent full chains per prompt, giving M trees.
  2. Sample N non-leaf nodes from each tree at random.
  3. Roll out from each sampled node to completion and attach the result as a new branch.
  4. Repeat steps 2–3 L times.
- **Group size:** G = M × (L×N + 1) rollouts per prompt, evenly split across trees.
- **Schedule:** fixed and non-adaptive. M, N, L are static hyperparameters, and node choice is uniformly random, not driven by reward, uncertainty, value or entropy.
- **Defaults:** (M=2, N=2, L=1), giving 6 rollouts at an expected cost of about 4 chains. They prefer L=1 with larger N because expansion iterations run serially.
- **Tree depth:** bounded only by the tool-call cap (3 for QA, 5 for web).
- **Budget formula:** Eq. 4 prints E[B_tree] = M·B + L·N·B/2, which appears to drop a factor of M on the second term. The budgets labelled in Tables 3 and 7 match M·B·(1 + L·N/2).
- **In code:**
  - The candidate set is the root plus all non-leaf nodes, sampled with replacement (`random.choices`), so the root can be picked. A root pick is a fresh full chain inside the same tree, sharing only the prompt.
  - With L>1, later iterations can fork from branches added earlier.
  - A fourth parameter `ts_k` (not in the paper) keeps k random leaves per tree and prunes the rest; the scripts set k=3 = L·N+1, so nothing is pruned.

## 3. Advantage and baseline
- **Formula (Eq. 6–7):** for each leaf trajectory H_i with outcome reward R,
  - Â_intra(H_i) = (R(H_i) − mean over leaves of its own tree) / std over the same set
  - Â_inter(H_i) = (R(H_i) − mean over all leaves of all M trees for that prompt) / std over the same set
  - Â_tree = Â_intra + Â_inter, an unweighted sum.
- **Normalisation:** mean and std for both terms. In code, std is `torch.std` (unbiased) plus 1e-6; a group of size 1 gets mean 0, std 1.
- **Not a per-branch-point sibling baseline.** The intra group is the whole tree, not the sibling subtrees at each fork. Each trajectory gets one scalar advantage, tiled across all its LLM-generated tokens.
- **Where the "step-level signal" comes from:** it is implicit. A shared prefix appears in several leaf trajectories with different advantages, so its net gradient is the sum over its descendants.
- **Objective (Eq. 8):** standard GRPO clipped surrogate with token-level importance ratio and a KL term to the reference model (coefficient 0.001, K3 / `low_var_kl`).
- **Ablation (Table 4):** intra-only collapses at (2,2,1) and works at N=6; inter-only gives +6.3%; the sum gives +16%.

## 4. Shared prefix in the loss
- **Trained once per descendant leaf, no deduplication, no weighting.** Each leaf is materialised as a full root-to-leaf sequence and treated as an independent row in the batch.
- A prefix shared by c leaves contributes c times, each with that leaf's advantage.
- Aggregation is `token-mean` over the minibatch; observation tokens are masked out of the loss (`state_masking=true`).
- The paper does not discuss this multiplicity at all; this is verified from code only.

## 5. Bias and on-policy discussion
- **None.** The paper does not discuss whether tree sampling biases the policy gradient, the correlation between sibling samples, or prefix over-counting.
- **Only theory (§3.3, App. C):** under a binary-reward assumption, the intra-tree GRPO gradient has the same form as step-level DPO, w·[∇log p(H_win) − ∇log p(H_loss)]. The weight is p_win·p_loss for intra-tree GRPO versus p_loss for DPO. This is a structural-equivalence claim, not an unbiasedness or variance result.
- **My inference, not theirs:**
  - Fork points are chosen independently of reward and continuations are sampled from the current policy, so each leaf is marginally on-policy.
  - Leaves are correlated, and prefix tokens are reweighted by descendant count.
  - The intra-tree baseline depends on sibling rewards that share the prefix actions.
  - An adaptive (reward- or uncertainty-driven) fork rule would add a selection effect that this paper never has to face.

## 6. Environment state
- **No environment forking mechanism.** The only tool is stateless search, so "forking" is reusing the stored token context of the node (`TreeNode.input_ids`). There is no snapshot or replay.
- **Environments:** a local E5 retriever over a Wikipedia dump for single-hop and multi-hop QA; a real web search API (Bing in the repo, with Redis result caching) for web-agent QA.
- **Tasks (11 datasets):**
  - Single-hop: NQ, TriviaQA, PopQA (trained on NQ).
  - Multi-hop: HotpotQA, 2Wiki, Musique, Bamboogle (trained on HotpotQA).
  - Web-agent: SimpleQA, GAIA text-only, WebWalkerQA, BrowseComp (trained on 2.2k samples from ASearcher and WebDancer).
- Stateful environments (code sandboxes, filesystems) are not addressed.

## 7. Systems
- **Framework:** veRL (the older Search-R1 fork) with Ray, FSDP actor, and in-process vLLM SPMD rollout (README pins vllm 0.8.5.post1).
- **Fully synchronous:** each turn is a batch generate, then a batch search, then the next turn. The L expansion rounds run serially after the initial chains finish, then the actor updates.
- **No KV or prefix-cache reuse.** The expansion re-submits the full prefix token ids as a fresh prompt. No prefix-caching flag is set in the repo's own code; only `enable_chunked_prefill` is.
- The claimed saving is counted in generated tokens and tool calls, not prefill compute or wall-clock. The paper reports no wall-clock or throughput numbers.
- Old log-probs are recomputed by an actor forward pass over the full leaf sequences. vLLM log-probs are collected (`infer_log_probs`) but never used.

## 8. Results
- **Efficiency claim:** about 1.5× as many rollouts at equal token/tool-call budget (6 vs 4 at the default), and "superior performance over the chain-based method while using only a quarter of the rollout budget" on Qwen2.5-3b. The budget is an expectation, not enforced or measured.

Main comparison at budget ≈4 (Table 1, average EM, GRPO → Tree-GRPO):

| Model | Multi-hop | Single-hop |
|---|---|---|
| Qwen2.5-1.5b | 11.3 → 19.1 (+69%) | 43.4 → 47.5 |
| Qwen2.5-3b | 31.8 → 36.8 (+16%) | 48.1 → 50.0 |
| Llama3.2-3b | 26.7 → 36.8 (+38%) | 48.7 → 50.0 |
| Qwen2.5-7b | 36.4 → 37.8 (+3.9%) | 50.5 → 52.2 |
| Qwen2.5-14b | 41.8 → 45.3 (+8.4%) | 55.1 → 55.7 |

Budget sweep on Qwen2.5-3b, multi-hop average (Table 3):

| Budget per prompt | Chain | Tree (best config) |
|---|---|---|
| ≈2 | 14.9 | 31.6 (+112%) |
| ≈4 | 31.8 | 36.8 |
| ≈8 | 31.4 | 36.4 |
| ≈16 | 33.9 | 37.3 |

- **Quarter-budget claim:** tree at budget 4 (36.8) beats chain at budget 16 (33.9).
- **Web-agent (Table 2, F1):** GAIA average on 14b goes 16.4 → 21.0; BrowseComp gains are marginal (2.4 → 2.6).
- **Behaviour:** average tool calls per multi-hop question rise from 2.4 to 3.0.
- **Token-level trees (App. B.3):** forking at token/sentence level does worse than plain GRPO (22.2 vs 31.8 multi-hop), their argument for step-level nodes.
- **Tree shape (Table 7):** M controls diversity, N·L controls signal granularity; too few trees hurts (M=1, N=5 gives 33.6 vs 36.8 for M=2, N=2).

## 9. Code map
All under `/private/tmp/claude-502/-Users-gangmuk2-projects-rlopt/355117d8-8324-4513-bf82-748327fc62da/scratchpad/prior_work/Tree-GRPO/`.

- **`search_r1/llm_agent/tree_node.py`**
  - `TreeNode`: stores the full left context (`input_ids`) and the cumulative response (`responses`) per node.
  - `get_expand_node(n, mode)`: uniform `random.choices` over root plus non-leaf nodes.
  - `sample_leaf(k)` and `_prune_subtree`: keep k leaves per tree.
  - `get_token_level_score_from_leaf`: in `base` mode, puts the outcome reward on the last token.
  - `calculate_final_score_from_root`: a TreeRL-style subtree-difference reward (`tree_diff` mode); it is computed but unused by Tree-GRPO.
- **`search_r1/llm_agent/generation_ts.py`**
  - `LLMGenerationTreeSearchManager.run_llm_loop_tree_search`: creates roots, generates M chains, runs L expansion rounds, assembles leaf trajectories, calls the reward function, and tags each leaf with `tree_uid`.
  - `gen_action_chain`: the batched turn loop that creates a `TreeNode` per step.
  - `execute_predictions`: the environment step (batched HTTP search).
- **`verl/trainer/ppo/ray_trainer_ts.py`**
  - `compute_advantage`, branch `adv_estimator == 'tree'`: calls `compute_grpo_outcome_advantage` twice (grouped by prompt `uid`, then by `tree_uid`) and sums.
  - `fit`: the training loop, which repeats each prompt `ts_m` times.
- **`verl/trainer/ppo/core_algos.py`**
  - `compute_grpo_outcome_advantage`: the mean/std normaliser.
  - `compute_policy_loss_dual_clip`: the loss actually used when `policy_loss=grpo`.
- **`verl/workers/actor/dp_actor.py`**: `update_policy`.
- **`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd_ts.py`**: vLLM generation.
- **`train_multihopqa_tree_search.sh`**: the launch config.

**Paper vs code discrepancies:**
- `expand_mode` is passed through but ignored; selection is always uniform random.
- The root is expandable, which the paper's "except the leaf node" permits but its B/2 cost argument glosses over.
- `ts_k` leaf subsampling is not in the paper.
- The tree scripts pass `lr=$lr` with `$lr` never defined; the GRPO scripts hardcode 1e-6.
- The multi-hop scripts use LR warmup ratio 0.285 for tree and 0.5 for GRPO. Table 5 lists "0.285/0.5" and Fig. 6 ablates it, but the shipped configs are not identical across methods.
- The script sets `max_response_length=500` per turn and `max_prompt_length=4096`; the paper says "max response length 4096".
- Not released: the token/sentence-level tree baseline (B.3) and the web-agent train/test data ("will be released soon").
- Extra estimators exist that the paper does not describe: `tree_2norm`, `tree_per_token`, and `policy_loss=turn`. `tree_inner` corresponds to the intra-only ablation.

## 10. Limitations and future work
- **There is no dedicated limitations section.**
- **Stated caveats:**
  - Single-hop gains are small because trees are shallow (depth about 2).
  - Web-agent gains are limited by small, lower-quality training data and API cost.
  - Intra-tree-only advantages collapse with few branches.
  - L>1 hurts rollout throughput because rounds are serial.
  - Small M reduces exploration.
- **Only explicit future work (App. E):** integrate "reflective reasoning and richer exploration into the training loop".
- **Not mentioned at all:** adaptive or guided expansion, stateful environments, KV-cache reuse, asynchronous training, gradient bias from correlated samples.

## What this leaves open for the researcher's idea
- **Adaptive, signal-driven forking** — when, where and how many: this paper is static and uniform-random.
- **Starting small and growing the group on demand:** the paper always spends M full chains and then a fixed N·L forks.
- **Per-branch-point sibling baselines and explicit prefix weighting or deduplication in the loss.**
- **Any bias or variance analysis of tree-structured groups.**
- **Stateful environment forking** (snapshot or replay).
- **Prefix/KV-cache reuse and asynchronous rollout.**

The cited neighbours worth checking against the same list are TreeRL (2506.11902), TreePO (2508.17445), TreeRPO (2506.05183), SPO (2505.23564) and VinePPO (2410.01679). This paper describes them as token/sentence-level; I did not read them.
