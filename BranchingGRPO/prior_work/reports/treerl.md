<!-- Subagent report, saved verbatim on 2026-10-03. Not independently verified. Brief: agent_briefs/treerl.txt -->

The paper exists as you recalled it, and I read both the full text (PDF) and the released code. Short version for novelty: TreeRL forks at the token level inside a single-turn response, on a fixed schedule with fixed counts, chosen by token surprisal. It has no tools, no environment state, no turn-level nodes, no reward-adaptive forking, and no bias analysis.

Everything below is verified by direct reading unless marked "Inference". Nothing was run; the code findings come from reading the source. The repo is cloned at `/private/tmp/claude-502/-Users-gangmuk2-projects-rlopt/355117d8-8324-4513-bf82-748327fc62da/scratchpad/prior_work/TreeRL`, with the paper as `treerl.pdf` / `treerl.txt` beside it.

## 1. Citation
- **Title:** "TreeRL: LLM Reinforcement Learning with On-Policy Tree Search"
- **Authors:** Zhenyu Hou*, Ziniu Hu*, Yujiang Li*, Rui Lu*, Jie Tang, Yuxiao Dong (*equal contribution; Tsinghua, with Ziniu Hu at Caltech).
- **arXiv:** 2506.11902v1 [cs.LG], 13 Jun 2025.
- **Venue:** ACL 2025 main conference (per the arXiv comment and repo README).
- **Repo:** https://github.com/THUDM/TreeRL, a fork of OpenRLHF. I cloned HEAD `63ea99e` (16 Jun 2025).
- The search method is called EPTree, "entropy-guided tree search".

## 2. Tree construction
- **Node:** a token segment of one single-turn response. Forks happen at arbitrary token positions; a chain is split into segments at its fork points. There is no notion of a turn or step boundary.
- **Schedule:** fixed, described by four numbers (M, N, L, T).
  1. Sample M independent full responses from the prompt (M trees joined by a virtual root).
  2. In each of L iterations, pick the top-N fork tokens per tree, pooled over all existing nodes of that tree.
  3. From each fork point, sample T full continuations to EOS.
  - This gives M·(1 + N·L·T) leaves. Training used (6, 2, 1, 2) = 30 leaves, set against 16 flat chains.
- **Signal:** H(y_t) = −log π(y_t | x, y_<t), i.e. the surprisal of the sampled token, not the entropy of the distribution. The paper calls it entropy or cross-entropy; the code uses `-log_prob` of the sampled token.
- **Fork prefix:** tokens strictly before the selected token, so the surprising token itself is resampled.
- **Not adaptive in count or reward:** fork selection happens before any correctness evaluation. Only the location is data-dependent.
- **Masking:** the paper says tokens "near the end of sequences" are masked. In code this is: everything after the first token containing "conclusion" or "answer", tokens beyond the max length, and the first token of each branch.
- **Parameter choice:** (M, N, L, T) was grid-searched for PassRate on Omni-MATH-500 at roughly matched generation tokens (Table 4). A random-fork ablation (Table 2) gives 54.8 PassRate versus 56.9 for entropy at (6,2,1,2).
- **Possible quirk for L > 1:** masks are never updated after a fork, so I believe the same top tokens can be re-selected in a later iteration. This does not affect the L=1 training config.

## 3. Advantage and baseline (paper)
- **Value:** V(s) = fraction of correct leaves among the descendants L(s) of s; reward is 0/1 on the final answer.
- **Global advantage:** GA(s) = V(s) − V(root), where the virtual root covers all M trees.
- **Local advantage:** LA(s) = V(s) − V(parent(s)).
- **Final per-step signal:** R(s) = [GA(s) + LA(s)] / sqrt(|L(s)|).
- The paper frames this as a special case of GAE and mentions a general all-ancestor form, which it does not use.
- There is no critic and no std normalisation. R is applied to every token in the segment.

## 4. Shared prefix in the loss
- **Trained once per descendant leaf.** Each leaf is flattened into a full root-to-leaf sequence, and prefix tokens are recomputed in every one.
- **Only weighting is the 1/sqrt(|L(s)|) factor**, justified as "to prevent overfitting".
- **Loss:** a PPO-style clipped-ratio REINFORCE, masked-mean over tokens per sequence, then mean over sequences (`ReinforcePolicyLoss` in `openrlhf/models/loss.py`).
- **Inference:** a prefix with n leaves appears n times at weight 1/√n, so its net gradient weight is about √n × advantage, further distorted by the per-sequence length normalisation.

## 5. Bias and on-policy
- The paper has no bias analysis. "On-policy" only means the tree is sampled from the current policy, as opposed to offline MCTS data or a static process reward model.
- It does not discuss that surprisal-selected fork points, resampling low-probability tokens, or the √n reweighting change the sampling distribution or the gradient estimator.
- No importance correction appears in the paper or the tree code; the clipped ratio is the generic one inherited from OpenRLHF.
- **Inference:** this is an open gap you could address.

## 6. Environment state
- The tasks are single-turn math reasoning with no tools and no environment; training data is MATH-train and NuminaMath.
- LiveCodeBench is evaluation only.
- Forking is pure text-prefix continuation. Nothing in the paper or code addresses environment forking.

## 7. Systems
- **Stack:** OpenRLHF fork, Ray, DeepSpeed ZeRO-3, and vLLM offline `LLM.generate` as Ray actors.
- **Synchronous:** each prompt takes L+1 blocking batched generate calls (M initial chains, then M·N·T branches). Prompts are processed sequentially per actor rank.
- **No KV or prefix reuse:** branch prefixes are resubmitted as plain prompts. `enable_prefix_caching` exists in `create_vllm_engines` but `train_reinforce_ray.py` does not pass it, so it defaults to False.
- **Budget accounting:** the "budget" counts generation (output) tokens only, not prefix re-encoding.
- **Speed:** the paper's limitations section says the method is about 2× slower than chain sampling because inference engines are not optimised for tree search.

## 8. Results
- **Search only** (Omni-MATH-500, Qwen-2.5-14B-SFT, PassRate):

  | Setting | Leaves | PassRate | Generation tokens |
  |---|---|---|---|
  | 16 chains | 16 | 52.4 | 19,858 |
  | EPTree (6,2,1,2) | 30 | 56.9 | 22,268 |
  | 64 chains | 64 | 67.4 | 79,367 |
  | EPTree (8,4,2,2) | 136 | 71.0 | 77,768 |

  The paper claims about +3% PassRate, and Appendix B's theorem gives 4/3 to 12/5 times more leaves per token, assuming uniformly distributed fork positions.
- **RL training** (average over 6 benchmarks, TreeRL vs ChainRL):

  | Base model | ChainRL | TreeRL |
  |---|---|---|
  | Qwen-2.5-14B | 41.6 | 44.5 |
  | GLM4-9B | 27.2 | 29.3 |
  | R1-distilled-Qwen-7B | 64.5 | 65.8 |

- **The comparison is equal in inference tokens, not training compute:** TreeRL trains on 30 sequences per prompt (batch 480) versus 16 (batch 256).
- **Ablation (Table 3, Qwen-14B):**
  - TreeRL restricted to 16 responses is −3.2 on average, i.e. about 41.3, which is roughly ChainRL's 41.6.
  - Dropping the √n reweighting costs −1.9.
  - Dropping the local advantage costs −1.3.
  - So most of the gain comes from having more training sequences per generation token, not from the process signal alone.
- **General tasks (Table 5):** TreeRL and ChainRL are on par (64.9 vs 64.2).

## 9. Code
All paths are under `TreeRL/openrlhf/trainer/ppo_utils/` unless noted.

**Fork-point selection**
- `entropy_chain_local_manager.py`, `EntropyGuidedChainLocalManager.entropy_guided_chain`: the sample → select → expand loop, evaluation, and the hand-off to the advantage code.
- `tree_node.py`, `TreeNode.get_max_entropy_tokens`: top-N unmasked tokens by `-log_prob`.
- `TreeNode.__init__`: builds the masks. `get_prefix_ids`: builds the fork prefix.
- An optional `select_diverse_tokens` (max-min position spread) is not in the paper.

**Advantage computation**
- `tree_node.py`, `build_into_tree_format`: converts chains into a segment tree.
- `leaf_normalize` and `leaf_backpropagate`: leave-one-out baseline on leaf rewards, then subtree sums.
- `select_terminal`: chooses which leaves to train on.
- `compute_weighted_update`: divides by sqrt of the selected-leaf count.
- `parallel_mcts.py`, `path_from_root_to_node` and `gather_paths`: with `use_state_value_reward`, each segment's value becomes (V'(s) − V'(p)) + V'(s).
- `experience_maker.py`, `_generate_vllm_mcts` (line ~3105): broadcasts the segment value to tokens and builds one full sequence per leaf.
- `sample_responses_bymcts` (line ~2056) and `experience_maker_reinforce.py::make_experience`: wrap it into training experience.

**Paper versus code discrepancies**
1. **Baseline:** the code uses a leave-one-out mean over all leaves, so GA is scaled by N/(N−1) relative to V(s) − V(root).
2. **Reweighting order:** √n is applied to each node's value before differencing. LA is therefore V(s)/√n_s − V(p)/√n_p, not (V(s) − V(p))/√n_s. Here n counts selected leaves, while V uses all leaves.
3. **Zero-fill:** `fill_in_paths` replaces a segment's local advantage with the previous segment's when it is approximately zero. This is not in the paper.
4. **Released script trains on 16 leaves, not 30:** `scripts/treerl-qw14b.sh` sets `NUM_TRACE=16` without `--use_all_terminals`, so it subsamples 16 of 30 leaves (forcing at least one correct leaf if any exists). The paper's main result uses all 30.
5. **KL:** the script sets KL to 0.0; the paper says 1e-4.
6. **Truncation penalty:** `--mask_repeated_samples` overwrites the reward of an entire truncated sequence with −1, shared prefix included. This is not in the paper.
7. **Reward source:** correctness comes from LLM judge and extractor endpoints (`check_result`), while the paper says "rule-based".
8. **Dead flag:** the script passes `--normalize_reward_from_multi_traces_with_rloo`, but the tree path never applies it.

The repo also contains a full MCTS path and a VinePPO-style value path that are not the paper's main method.

## 10. Limitations and future work (stated)
- Inference engines lack tree-search optimisation, so EPTree needs 2+ iterations and is about 2× slower than chain sampling.
- Open questions they name: how to weight the importance of different steps, how to define more meaningful process signals from the tree structure, and how to do step-level reward normalisation.
- They say nothing about agents, tools, multi-turn tasks, or estimator bias.
