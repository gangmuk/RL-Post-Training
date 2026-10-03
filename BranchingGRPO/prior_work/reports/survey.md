<!-- Subagent report, saved verbatim on 2026-10-03. Not independently verified. Brief: agent_briefs/survey.txt -->

Short version: adaptive online branching (a) and mid-trajectory sandbox forking (b) both have direct prior work; I found nothing that handles tree rollouts under asynchronous training with stale weights (c).

How I verified each item is tagged as follows:
- **[PDF]**: I read the paper pages directly.
- **[HTML]**: an automated summariser read the arXiv HTML full text and I asked it for quotes; reliable for mechanism, but check exact numbers before citing.
- **[ABS]**: abstract only.
- **[CODE]**: I read the repo code.

## 1. TRPO "vine" sampling (arXiv 1502.05477, Section 5.2) [PDF]

- **Mechanism:** Sample s0 and simulate the current policy to produce "trunk" trajectories. Choose a subset of N states along them, the "rollout set". From each state s_n, sample K actions from a proposal q, and run a short rollout after each to estimate Q(s_n, a_{n,k}).
- **Proposal:** Any q whose support covers the policy's gives a consistent estimator. They use the policy itself for continuous control and uniform for Atari.
- **Common random numbers:** The K rollouts from one state share the same noise sequence, which "greatly reduce[s] the variance of the Q-value differences between rollouts".
- **Estimator:** For small finite action spaces, every action is rolled out and L_n(θ) = Σ_k π_θ(a_k|s_n) Q̂(s_n, a_k). Otherwise a self-normalised importance-sampling estimator over the K actions is used (Eq. 16), which "removes the need to use a baseline for the Q-values".
- **Variance claim:** "our local estimate of the objective has much lower variance given the same number of Q-value samples in the surrogate objective. That is, the vine method gives much better estimates of the advantage values."
- **Costs:** "we must perform far more calls to the simulator for each of these advantage estimates", and vine "requires us to generate multiple trajectories from each state in the rollout set, which limits this algorithm to settings where the system can be reset to an arbitrary state". Single path "requires no state resets and can be directly implemented on a physical system."
- **Empirics:** Both variants solved all locomotion tasks. In the Atari table (Table 1) neither dominates: vine is higher on Breakout (34.2 vs 10.8) and Q*bert (7732.5 vs 1973.5), single path on B. Rider, Enduro, Seaquest and Space Invaders.
- **Relevance:** Vine picks branch states from completed trunks and uses a fixed K. It has no online decision rule and no notion of policy lag.

## 2. Relevant papers

### Closest to the idea

**1. Agentic Reinforced Policy Optimization (ARPO)** [HTML + CODE]
- Dong, Mao, Ma, Bao, Chen, Wang, et al.; arXiv 2507.19849, July 2025; repo https://github.com/dongguanting/ARPO (also contains the follow-up AEPO, arXiv 2510.14545).
- With a group budget M, it starts only N < M full trajectories and reserves M−N for partial branches.
- After each tool call it generates k tokens, computes their entropy, and compares it with the trajectory's initial entropy. The rule is P_t = α + β·ΔH_t, branch if P_t > τ, decided online while the trajectory is in progress.
- Advantages: the "hard" variant averages the advantage over shared prefix tokens; the default "soft" variant is plain GRPO in which shared tokens are handled through the ratio.
- Tools are search, browser and a Python interpreter. Environment-state forking is not discussed, and neither is async training or KV cache.
- Code: `ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py`, in `generate_sequences`. Config keys are `initial_rollouts`, `beam_size`, `branch_probability`, `entropy_weight`. Entropy is monitored around line 283 and the branch test is at lines 468–476 (`prob = random.random() - entropy_weight * entropy_delta; if prob > branch_probability`).
- Overlap: this is already "start with fewer than G, fork adaptively at turn boundaries". It differs in using stateless tools and a synchronous verl loop, and its only signal is token entropy.

**2. Process Reward Informed Tree Rollout for Effective Multi-Turn RL (PATR)** [PDF, pp. 1–12]
- Li, Li, Zhang, Yu, Lin, Jin, Guan, Liu, Li, Yin, Shang (UCSD/Amazon); arXiv 2607.15610, 17 Jul 2026; no repo found.
- Starts with B0 = 4 branches. Every K turns (K = 13 on SWE, 5 on FrozenLake) a process scorer scores each active partial trajectory; the scorer is a heuristic, a pretrained process-reward model, or an LLM judge.
- High scorers are expanded into M = 2 children from the same intermediate state, moderate ones survive, and those with s < median − α or repeated actions are stopped early.
- Pruned branches are kept in the group as negatives. The advantage is plain GRPO normalisation of outcome reward over the tree-generated group; the scorer is used only for allocation.
- Trained Qwen3-4B on R2E-Gym with SWE-agent in Docker and evaluated on SWE-Bench Verified (27.2 vs GRPO 22.2).
- The paper does not say how the Docker state is duplicated when a branch is expanded. Async training and KV cache are not mentioned. It acknowledges the "controlled and localized bias" from non-uniform allocation.
- Overlap: the closest match overall (G/4-style start, online observe-then-fork, stateful SWE sandbox). It differs in using an external scorer, having no stated snapshot mechanism, and no async or staleness treatment.

**3. Branching Policy Optimization: Sandbox-Native Language Agent Reinforcement Learning (BPO)** [HTML]
- He, Chen, Zhang, Liu; arXiv 2607.14171, 15 Jul 2026; no repo.
- Samples one backbone trajectory, then picks the top-M steps by token entropy post hoc (minimum spacing 64 tokens). It snapshots and restores the sandbox there, forks K siblings, and rolls each to termination.
- Snapshot technologies: Docker overlayfs, CRIU, interpreter pickling, browser session export. Reported snapshot cost is 42 ms on WebShop, 138 ms on ALFWorld and 1.9 s on SWE-bench.
- Advantage is a leave-one-out sibling baseline, A = G_k − mean of the other siblings, propagated to pre-branch steps with λ = 0.95. It claims unbiasedness and a variance reduction relative to GRPO.
- Reports +5.7 to +5.9 points over GRPO on WebShop, ALFWorld and SWE-bench Verified with Qwen2.5-7B.
- It does not cite Tree-GRPO, TreeRL, TreePO or TRPO vine (it does compare with VinePPO). Future work lists "asynchronous tree-distributed training over heterogeneous sandbox clusters" and "adaptive budget allocation".
- Overlap: explicit prior art for forking a stateful sandbox mid-trajectory with a sibling baseline. It differs in that branching is post hoc, not online, with no KV-cache or async treatment.

**4. EPIG-Tree: Compute-Optimal Branching for Gradient-Efficient Reinforcement Learning** [PDF, pp. 1–14]
- Khomich, Hermansson, Hakimi; arXiv 2609.20004, 17 Sep 2026; no repo seen.
- Frames tree construction as compute allocation for the policy-gradient estimator. A law-of-total-variance split separates decision uncertainty (reduced by new branches) from continuation uncertainty (reduced by repeated suffixes).
- Derives a Neyman-style suffix law n_e ∝ w_e‖∇log π‖σ_e/√c_e and a branch score S_EPIG(h) with cost in the denominator. Entropy only proposes candidates.
- The algorithm samples K0 root trajectories first, then scores candidates and greedily adds branches (post-hoc, pilot-based). Advantage is V̂(child) − V̂(h) with a branch-segment token mask.
- Experiments are on cloned-state MuJoCo, GSM8K and multi-turn Wordle (0.850 vs flat GRPO 0.790, two seeds). It reports a negative result: on single-turn math, branch placement is within noise and token masking matters more.
- Overlap: the most principled answer so far to "where to branch", with a cost-aware criterion. It uses no real sandbox and no async training.

**5. ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue** [HTML + CODE]
- Cao, Bai, Yao, Dong, Xu, Xiao; arXiv 2603.02216, Feb 2026; repo https://github.com/Quark-Medical/ATPO.
- At each turn it generates N candidate replies and computes U = α·|Bellman error| + (1−α)·Q-variance using a learned critic. If U > τ it keeps all N; otherwise it keeps one random child (with a 10% bypass), until a leaf budget is reached.
- Advantages are one-step TD from the critic, spread uniformly over the turn's tokens. The paper explicitly claims KV-cache reuse of shared prefixes.
- "Asynchronous" here means asyncio within the sampling phase, not asynchronous training. The environment is a user-simulator LLM, so state is just the message list.
- Code: `ATPO/codes/recipe/atpo/user_assistant_agent_loop.py`, the tree-search class's `run`, `_expand_leaves_with_assistant` and `_prune_and_update_active_leaves` (line 360: `0.3*nor_variance + 0.7*abs(diff_value) < variance_threshold and random.random() < 0.9`).
- Overlap: online expand-or-prune per turn with a value-uncertainty signal and KV reuse. It is expand-then-prune rather than start-small-then-fork, and needs a critic.

### Other tree extensions

**6. AT²PO: Agentic Turn-based Policy Optimization via Tree Search** [HTML]
- Zong, Chen, Li, Yi, Zhou, Li, Qian, Chen, Jiang; arXiv 2601.04767, Jan 2026; repo https://github.com/zzfoutofspace/ATPO.
- Runs M = 10 full chains, then L = 2 rounds of selecting the top-K = 6 nodes by s(n) = entropy − α·(sibling count) and regenerating the remainder from each.
- Credit is assigned per turn by entropy-weighted backup of leaf rewards, with a turn-level objective. Uses search QA tools, so no state to fork.
- Overlap: post-hoc entropy forking in the Tree-GRPO line; not online, not stateful.

**7. Information Gain-based Rollout Policy Optimization (IGRPO)** [HTML]
- Zhang, Xu, Ding, Xie, Gao, Ding, Zhang, Fu, Wang; arXiv 2607.06223, Jul 2026; no repo.
- Stage-wise expansion: nodes are sampled for expansion with probability softmax(γ·val(h)). val uses the increase in the policy's likelihood of the ground-truth answer at that node.
- Shows the induced trajectory distribution is ∝ π(o)·exp(γV(o)). Uses standard GRPO advantage on search QA.
- Overlap: adaptive online budget allocation across prefixes, but it needs a ground-truth answer string and uses stateless tools.

**8. TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents** [HTML]
- Djuhera, Kadhe, Ahmed, Boche (authors as on arXiv HTML; PATR's bibliography lists a slightly different author list); arXiv 2602.11767, Feb 2026.
- Per-turn best-of-N, beam, or shallow lookahead at rollout time with a task scoring function. Keeps the selected trajectories and leaves the optimiser unchanged. Tested on Sokoban, FrozenLake and WebShop.
- Overlap: it uses search to pick better data, not to estimate advantages. PATR criticises it for biasing the rollout distribution.

**9. Why Tree-Style Branching Matters for Thought Advantage Estimation in GRPO** [ABS]
- Wang, Huang, Wang, Ren, Dong; arXiv 2509.24494; repo https://github.com/whcpumpkin/GRPO-MA.
- Theory result: variance of the thought-level advantage persists as you add more thoughts but falls as 1/M with M continuations per thought.
- Overlap: useful support for the variance argument; single-turn only.

**Also seen, abstract or snippet only:**
- AEPO (2510.14545), the entropy-balanced successor to ARPO.
- AT-GRPO, "Dialogue Model Optimization via Agent Game and Adaptive Tree-based GRPO" (2602.08533).
- DATPO, "Difficulty-Adaptive Tree-Structured Policy Optimization…" (2609.08650), single-turn.
- VinePPO (2410.01679, ICML 2025), which uses Monte Carlo rollouts from intermediate text states as the value estimate.
- The survey "Generate, Filter, Control, Replay" (2605.02913), which has no section on environment state restoration or on tree rollouts combined with async training.

### Adaptive group size and early termination (flat, no branching)

**10. Reinforce-Ada** [HTML + CODE]
- Xiong, Ye, Liao, Dong, Xu, Monz, Bian, Jiang, Zhang; arXiv 2510.04996; repo https://github.com/RLHFlow/Reinforce-Ada.
- Samples in rounds and deactivates a prompt once it has at least one correct response ("pos") or n/2 correct and n/2 incorrect ("balance"). It then downsamples to n, with the baseline computed from the whole pool.
- Code: `verl/trainer/ppo/ray_trainer.py::_generate_multi_round_adaptive_downsampling` (line 929).
- Overlap: sequential "observe then sample more", but at prompt level and single-turn.

**11. Selective Rollout: Mid-Trajectory Termination for Multi-Sample Agent RL** [HTML + CODE]
- Zhai, Wang; arXiv 2605.05802; repo https://github.com/zhiyuanZhai20/selective-rollout.
- At turn K = 10 it computes the mean pairwise prefix edit distance of the G action sequences. If it is below 0.12, the whole group is stopped and dropped as a predicted zero-variance group. Tested on ALFWorld, about 10.7% wall-clock saving.
- Code: `src/divergence.py::divergence_at_K`; gate in `scripts/19_grpo_onpolicy.py::build_train_items`.
- Overlap: a mid-trajectory group-level signal used to kill groups, never to fork. The divergence signal could serve as a fork trigger.

**12. Sequential and predictive allocation**
- SARA, "Early Verdicts, Better Budgets" (2607.26253) [HTML]: Beta posterior with an SPRT-style commit/abandon rule. It lists "correlated tree rollouts" as open.
- VIP, "Adaptive Rollout Allocation for Online RL with Verifiable Rewards" (2602.01601, ICLR 2026) [ABS]: Gaussian-process prediction of success rate plus a convex allocation that minimises gradient variance.
- C-GRPO: seen only as a verl issue snippet (verl-project/verl #8077); unconfirmed as a paper.

### Infrastructure for (b) and async work for (c)

**13. Crab: A Semantics-Aware Checkpoint/Restore Runtime for Agent Sandboxes** [HTML]
- Wu, Chang, Cao, Gao, Wang; arXiv 2604.28138, Apr 2026.
- Uses an eBPF inspector to checkpoint at turn boundaries, skipping up to 87% of turns that change no state. Checkpointing overlaps with LLM inference time.
- Explicitly motivates tree RL: forking the sandbox from a checkpoint instead of re-executing the prefix cuts rollout tokens by 40–64% on SWE-agent at branching factors 1–5. This is a cost measurement only; no model is trained.
- DeltaBox (2605.22781; overlayfs/CRIU/copy-on-write, millisecond checkpoint/rollback) exists, but the fetch returned only PDF metadata, so its details are unverified.

**14. Async and off-policy work, none tree-aware**
- Stable Asynchrony / VCPO (2602.17616; repo https://github.com/mit-han-lab/vcpo) [ABS]: scales the learning rate by effective sample size and uses a minimum-variance off-policy baseline.
- Single-Rollout Asynchronous Optimization (2607.07508, Hou, Li, Tang, Dong) [ABS]: argues group-wise GRPO "does not naturally fit asynchronous agentic training" and replaces groups with one rollout plus a value model. This is a direct counter-position to keep in mind.
- AReaL-DTA (2602.00482, ICML 2026; https://github.com/areal-project/AReaL/tree/feat/dta) [ABS]: prefix-tree-aware training-side forward and backward passes inside an async RL framework. It covers trainer compute sharing, not rollout branching.
- PrefixRL, "Reuse your FLOPs" (2601.18795) [HTML]: on-policy continuation from very off-policy prefixes with gradients masked on the prefix. This is the natural precedent for forking from a prefix generated by older weights; it is single-turn math.
- Prefix-Normalized Policy Optimization (2608.01418) [HTML]: geometric-mean prefix importance weights under policy lag.
- Seen in snippets only: "Missing Old Logits in Asynchronous Agentic RL" (2605.12070), GAC (2603.01501), μ-GRPO (2605.17570).

## 3. Assessment

**(a) Adaptive, online decisions about when and where to branch: already done, several ways.**
- ARPO and PATR both start with fewer than G trajectories and fork online at turn boundaries; ATPO does online expand-or-prune per turn; IGRPO does stage-wise soft allocation.
- BPO, AT²PO and EPIG-Tree are post-hoc.
- PATR is the closest overall (4 initial branches, SWE-bench); ARPO is the earliest.
- Still open: a decision rule driven by group-level outcome information rather than entropy or an external scorer, and a cost-aware criterion (EPIG-style) applied online. An example of such a rule is forking because the initial trajectories are converging or diverging, as in Selective Rollout's signal. Starting from a single trajectory as the default is also unexplored; BPO does it, but post hoc.
- PATR admits allocation bias, and EPIG shows placement can be second-order to token masking. Both are risks for a novelty claim based on (a) alone.

**(b) Forking stateful agent environments mid-trajectory: done.**
- BPO is the closest on the algorithm side: snapshot/restore of Docker sandboxes on SWE-bench with a sibling baseline.
- Crab is the closest on the systems side: checkpoint-and-fork explicitly pitched at Tree-GRPO-style RL.
- PATR trains on SWE with branching but does not describe the mechanism.
- Still open: coupling the sandbox fork with KV-cache sharing of the prefix. ATPO claims KV reuse without stateful environments; BPO has stateful forks without KV discussion; no paper I found does both. Fork latency as an input to the branching decision is also untreated, apart from EPIG's abstract cost term.

**(c) Interaction with async training and stale weights: no prior work found.**
- Every tree method above assumes one fixed policy for the whole tree. BPO lists async tree training as future work, and SARA lists tree rollouts as open.
- The async literature (VCPO, SAO, Prefix-Normalized Policy Optimization, Missing Old Logits) handles only flat rollouts, and SAO argues for abandoning groups under async.
- The nearest building blocks are PrefixRL (mask gradients on an off-policy prefix, train on the continuation) and AReaL-DTA (prefix-tree training in an async system).
- The specific problems appear untouched: a fork sampled under weights θ' from a prefix and KV cache produced under θ; sibling baselines that mix policy versions; KV-cache invalidation at weight sync; and whether to fork now or wait given staleness bounds. This is the strongest novelty axis.

## Scratchpad

Everything is under `/private/tmp/claude-502/-Users-gangmuk2-projects-rlopt/355117d8-8324-4513-bf82-748327fc62da/scratchpad/prior_work/`. I cloned `ATPO/`, `Reinforce-Ada/` and `selective-rollout/`. `ARPO/` (which includes AEPO) was already there when I went to clone it, so I read it in place; `Tree-GRPO/`, `TreePO/` and `TreeRL/` were also already there and I did not touch them. Nothing outside the scratchpad was modified.
