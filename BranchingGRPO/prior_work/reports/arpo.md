<!-- Subagent report, saved verbatim on 2026-10-03. Not independently verified. Brief: agent_briefs/arpo.txt -->

ARPO exists and is the closest prior work to the idea, but its released code is much less adaptive than the paper describes. I read arXiv v1 in full (main text, appendices A–E, Algorithm 1) and the ARPO half of the official repo; nothing was run. I did not read the ICLR camera-ready, so paper claims below are v1.

## 1. Citation
- **Title:** "Agentic Reinforced Policy Optimization".
- **Authors:** Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji-Rong Wen, Zhicheng Dou (Renmin University of China and Kuaishou).
- **arXiv:** 2507.19849, v1 dated 26 July 2025.
- **Venue:** ICLR 2026, according to the repo README (not checked on OpenReview).
- **Repo:** https://github.com/dongguanting/ARPO (README also links RUC-NLPIR/ARPO). Cloned to `/private/tmp/claude-502/-Users-gangmuk2-projects-rlopt/355117d8-8324-4513-bf82-748327fc62da/scratchpad/prior_work/ARPO`, HEAD `4d19a74`. It also contains the follow-up AEPO.

## 2. Rollout construction
**Paper (§3.1, Algorithm 1):**
- The group size M is fixed in advance. N trajectories start from the prompt; the remaining M−N slots are reserved for branches.
- The baseline is H_initial, the entropy of the first k tokens of each initial trajectory.
- After every tool call, the model generates k more tokens after the tool result, giving H_t. Then ΔH_t = Normalize(H_t − H_initial), where "normalize" is described as summing and dividing by vocab size V.
- The branch rule is P_t = α + β·ΔH_t; if P_t > τ, branch Z paths from that node, otherwise continue.
- Branching stops when the M−N budget is used. If all paths finish early, the remainder is filled with fresh from-prompt rollouts.
- Defaults (App. C.2): M=16, N=8, "entropy weight" β=0.2, "a"=0.5, threshold 0.5. Z and k are not given numerically.

**Code (`vLLMRolloutWithTools.generate_sequences`):**
- Scripts set `ROLLOUT_N=16`, `INITIAL_ROLLOUTS=8`, `BEAM_SIZE=2`, `BRANCH_PROBABILITY=0.5`, `Entropy_weight=0.2`. The 14B deep-search script uses 12 and 6.
- Generation is round-based: all active trajectories generate until a tool stop tag or EOS, tools execute, results are appended, then the branch step runs.
- Each active trajectory with slots remaining gets `beam_size − 1 = 1` branch attempt per round, so Z=1 and the parent continues.
- The decision is stochastic, not a threshold: `prob = random.random() − entropy_weight·ΔH`, clamped to [0,1], and the branch is skipped if `prob > branch_probability`. So P(branch) ≈ clip(0.5 + 0.2·ΔH).
- A branch copies the parent's token list including the just-appended tool result, its loss mask, and its call counter.
- Budget is allocated in index order within each prompt until `n` is reached. Prompts whose trajectories have all finished get one fresh from-prompt rollout per round. The output is always exactly `n` per prompt (the last is duplicated if short).
- "Active" also includes trajectories that stopped on a length limit, not only tool calls.

## 3. Advantage and baseline
- **Hard variant (paper only):** branch-specific tokens get Â_i = (r_i − mean(R))/std(R) over the whole group; shared tokens get the mean of Â over the d trajectories sharing that segment.
- **Soft variant (the default):** plain GRPO. Every trajectory, global or branched, is a full sequence, and all its tokens get the same group-normalised scalar. The paper argues shared-prefix tokens have identical importance ratios across descendants, so their summed gradient approximates the hard average (App. D.1 decomposes this). Figure 5 claims soft gives higher, more stable reward; no numbers in the text.
- **Normalisation:** a single mean/std over all M trajectories of the prompt. There is no per-branch-point or sibling baseline.
- **Code:** `compute_grpo_outcome_advantage` in `core_algos.py` is stock verl, grouped by one `uid` per prompt. No hard-advantage code exists in the repo (grep found nothing).
- **Reward:** −1 for bad format, 0 for good format but wrong, token-F1 if right, +0.1 if both search and python were used (`deep_research.py::compute_score`).

## 4. Shared prefix in the loss
- Prefix tokens are trained once per descendant. Each of the M rows is a full sequence, with no deduplication and no weighting.
- Loss aggregation is `token-mean`. Tool-result tokens are masked out with `loss_mask`.
- The effective prefix advantage is therefore the sum, not the mean, of descendant advantages. A prefix with more branches gets proportionally more gradient weight.

## 5. Bias and on-policy discussion
- **"Generalized Policy Gradient Theorem" (§3.3, App. D.2):** ∇J = E_τ[Σ_T ∇log π(MA_T | MS_T)·A_T]. This only restates that the log-likelihood factorises over arbitrary token segments ("macro actions"). It assumes τ ~ π_θ and says nothing about the tree-shaped sampling distribution.
- **App. D.1:** an algebraic rewrite of the GRPO objective into a pre-branch term plus a post-branch GRPO term. It is not a bias analysis.
- **Not addressed:** the selection effect of branching on an entropy signal, the over-representation of branched prefixes, and the correlation among siblings within the normalisation group.

## 6. Environment state
- **Tools:** search (Bing via Bright Data, top-10 snippets, on-disk JSON cache), a Python interpreter, and a browser agent that summarises pages. The paper uses the browser at evaluation and for deep-search tasks.
- **State:** everything is stateless. `PythonTool` runs a fresh `python -c <code>` subprocess per call, with no persistent kernel or filesystem.
- **Forking:** a branch is just a copy of the token list. No sandbox snapshot or fork exists or is discussed.

## 7. Systems
- **Framework:** verl with FSDP, in-process vLLM, rollout mode `sync_with_tool`. Training is strictly synchronous and on-policy: generate, reward, old log-probs, advantages, update.
- **Prefix cache:** vLLM is created with `enable_prefix_caching=True` in `vllm_rollout_spmd.py`. Each round resubmits the full token prefix, so reuse is whatever vLLM's automatic caching provides; there is no explicit KV forking.
- **Tool execution:** a 64-thread pool; each round blocks until all tool calls return.
- **Tool-call cost:** a `tools/total_calls` counter per batch. The per-trajectory limit is 3 (reasoning) or 6 (deep search), and branches inherit the parent's counter. Savings come from prefix tool calls being executed once and shared, plus the search cache.
- **Hardware and scale:** 8 H800 GPUs (16 for 14B), batch size 128, PPO mini-batch 16, KL coefficient 0, 2 epochs on 10K samples (5 epochs on 1K for deep search).
- The paper's rollout complexity claim, "O(n²) down to between O(n log n) and O(n²)", is asserted without derivation.

## 8. Results (paper tables)
Ten-benchmark average, ARPO versus the three flat baselines:

| Model | GRPO | Reinforce++ | DAPO | ARPO |
|---|---|---|---|---|
| Qwen2.5-3B | 50.4 | 49.7 | 50.6 | 52.8 |
| Llama3.1-8B | 51.1 | 51.1 | 50.4 | 55.3 |
| Qwen2.5-7B | 56.5 | 54.9 | 54.8 | 58.3 |

Deep search, GRPO → ARPO, trained on 1K samples:

| Model | GAIA | WebWalkerQA | HLE | xbench |
|---|---|---|---|---|
| Qwen3-8B | 32.0 → 38.8 | 29.0 → 30.5 | 7.8 → 8.8 | 20 → 25 |
| Qwen3-14B | 36.9 → 43.7 | 30.0 → 36.0 | 8.6 → 10.0 | 27 → 32 |

- **Tool budget:** "only half the number of tool calls" versus GRPO on Qwen2.5-7B (Figure 7). The text gives no absolute numbers.
- **Ablations (Figure 8):** the entropy weight peaks at 0.4 and declines at 1.0. N=8 of M=16 is best; N=16 (pure GRPO) is worse. Larger M helps.
- No seeds or variance are reported.

## 9. Code map and discrepancies
All in `ARPO/verl_arpo_entropy/verl/`:
- **Entropy:** `workers/rollout/vllm_rollout/vllm_rollout_with_tools.py`, `_calc_entropy` (l.172) and the monitoring block (l.283–306). It takes vLLM top-10 logprobs for the first up-to-20 tokens of the segment just generated, computes −Σ p·log p summed over all of them, and divides by log(vocab size).
- **Branch decision:** same file, l.434–526 (rule at l.468–477).
- **Advantage:** `trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` (l.113), called from `trainer/ppo/ray_trainer.py::compute_advantage` (l.255–268). Unmodified GRPO.
- **Legacy path:** `workers/agent/tool_agent.py` has an older branching path with a pure coin flip and no entropy. I only grepped it; it is not the path wired to `sync_with_tool`.

Discrepancies between paper and code:
1. **Stochastic, not thresholded.** The paper has a deterministic rule P_t > τ; the code flips a coin with probability about 0.5 + 0.2·ΔH.
2. **Entropy is not measured after the new tool result.** The code uses the start of the segment that ended in the tool call, i.e. the reaction to the previous tool result. In a trajectory's first round ΔH is exactly 0, so the first branch is a pure 50% coin flip.
3. **Stale baseline bug.** `self.initial_entropy_dict` is created in `__init__`, keyed by positional slot index, and never reset. After the first training step, "initial entropy" is whatever occupied that slot in the first batch. Branched slots would record their own first post-branch segment as baseline anyway.
4. **Truncated entropy.** It is top-10 only, summed (not averaged) over up to 20 tokens, so short segments score lower. Normalisation is by log V, not V as the paper says.
5. **Hard advantage estimation is not implemented.**
6. **Index-order budget allocation**, not ranked by entropy.

My inference, not measured: with a 0.2 weight on a small ΔH, the decision is close to a 50% coin flip, and the 8 branch slots are mostly used up at the first and second tool calls. In practice the code behaves much like Tree-GRPO-style random early forking with a slight entropy tilt.

## 10. Limitations, future work, related branching work
- **Stated limitations and future work:** v1 has no limitations section and none in the conclusion. The README says only that they will "continue to iterate".
- **AEPO** (Dong et al., arXiv 2510.14545, WWW 2026 per README; same group; I read only the README and a diff of its rollout file): sets the global/branch split per prompt from entropy pre-monitoring, penalises consecutive high-entropy branching, and adds entropy-aware clipping and advantage.
- **FR3E** ("First Return, Entropy-Eliciting Explore", Zheng et al., arXiv 2507.07017; cited by ARPO as an entropy-based RL study): from the title and citation context only, entropy-guided partial rollouts for single-turn reasoning.
- ARPO's related work does not discuss other tree or branching RL methods.

## Comparison with the researcher's idea
**Already done by ARPO:**
- Mixing from-prompt and mid-trajectory forked samples within one fixed-size GRPO group.
- Forking at turn boundaries after a tool result, sharing the prefix and its tool calls.
- An adaptive (nominally entropy-driven) fork signal.
- The claim of roughly half the tool calls.
- Reliance on engine prefix caching.

**Still open:**
- **Starting from very few and growing on demand.** ARPO fixes N=M/2 and reports N=8 of 16 as best, but Figure 8 is not readable from text, so I cannot say which small N were tested. AEPO varies N per prompt but still fixes M. Group size is never variable.
- **Outcome-aware branch signals.** ARPO uses only token entropy of the current trajectory. It never uses reward-so-far, sibling outcomes, or a learned success predictor, and never waits to observe how early trajectories end before deciding where to fork. Decisions are made in lockstep rounds.
- **Stateful sandbox forking.** Untouched; all tools are stateless.
- **Asynchronous training with stale weights.** Untouched; the system is fully synchronous verl, and a prefix and its branches always come from the same policy version.
- **Per-branch-point baselines.** Described only as the "hard" variant, which is not implemented and is still a global baseline averaged over descendants. Sibling-relative baselines at the fork node are open.
- **Prefix deduplication and weighting.** Not done; prefixes are trained once per descendant with implicit over-weighting. No correction for selection bias from the branch rule.
