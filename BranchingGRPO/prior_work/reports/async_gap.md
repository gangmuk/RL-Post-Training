<!-- Subagent report, saved verbatim on 2026-10-03. Sources were reachable only through web-search snippets (arxiv and mirrors blocked by the session network policy). Not independently verified. -->

# Async tree rollouts for GRPO: gap check and building blocks (as of 2026-10-03)

**Source caveat.** WebFetch was blocked for every host I tried: arxiv.org, alphaxiv, huggingface, semanticscholar, substack, 36kr and papers.cool. The GitHub API was also closed for repos not attached to this session. Everything below therefore comes from WebSearch result summaries, tagged **[HTML-summary]**; I could not check any of it against the papers. I could not pull **[verbatim]** quotes. Text in quotation marks is search-snippet wording, so treat it as UNVERIFIED-verbatim. I ran 25+ queries.

---

## (A) Gap check: is there a paper doing tree/branching rollouts under async training?

I found no paper that does GRPO-style tree or branching rollouts inside an asynchronous trainer with bounded staleness and an explicit correction for the stale prefix. Several pieces now sit very close, and two are new threats.

| Work | Branching? | Async/stale? | Notes |
|---|---|---|---|
| **RTPO**, 2608.18682 (Aug 19, 2026; v3 Aug 31) | **Yes.** Sparse reverse trees with sibling forks at turn *k* | **Yes, as a motivation.** "asynchronous policy drift when short and long trajectories are optimized under different policy versions" [HTML-summary] | **Closest threat**, detailed below |
| **MBPO**, 2608.07581 (Aug 2026) | Yes. Reasoning trees at vision-language decision points, branch-relative advantages | Partly. A "temporal replay buffer" reuses earlier-generated subsequences, limited to an age window "to avoid reuse of outdated subsequences after significant policy changes" [HTML-summary] | Multimodal, synchronous. Unclear whether new branches are forked from replayed old segments (UNVERIFIED) |
| **torchtitan PR #4761**, "Preserve per-group KV cache across policy updates" (date UNVERIFIED) | No (groups, not trees) | **Yes.** After a weight pull, "in-flight groups resume from their existing KV" built by the old policy. New groups get a new `cache_salt` = policy version so they "cannot reuse old-policy KV" [HTML-summary] | Systems precedent for continuing generation under new weights on top of an old-policy prefix |
| **STRIDE**, "Know When to Stop, Where to Restart", 2609.14636 (Sep 13, 2026) | Restarts from a cached prefix. A "prefix buffer caches high-quality prefixes and restarts generation at the weakest correct turn" [HTML-summary] | Implicit. Cached prefixes come from earlier student checkpoints | On-policy distillation, not GRPO. Async status UNVERIFIED |
| **Prefix-GRPO**, "From Trajectories to Prefixes", 2607.19395 (Jul 2026) | Forks online continuations from replayed teacher prefixes | No. It trains on the replayed prefix tokens with old log-probs taken from an SFT checkpoint | Same lab as Missing Old Logits |
| **Schedule-Level Shared-Prefix Reuse**, 2606.01143 (May 31, 2026, cs.DC) | Trainer-side only. Prefix forward pass once, each suffix runs as a microbatch reading stored prefix K/V, prefix backward pass once; "numerical equivalence", up to 4.395x | Not mentioned | Neither async nor rollout branching |
| **TRACE**, 2606.11119 (Jun 2026) | Yes. Spends rollout budget on turn-level prefixes using a success-probability predictor | Not mentioned | Synchronous tree allocation |
| **BPO, Branching Policy Optimization**, 2607.14171 (Jul 2026) | Yes. Snapshots the sandbox and forks K actions at M entropy-chosen points; sibling baseline | Not mentioned | Synchronous |
| **HARTS** 2608.28158; **AReaL-DTA** 2602.00482 | Trainer-side over arbitrary rollout trees | AReaL-DTA batches rollouts from an async generation phase | No rollout forking |
| **APRIL** 2509.18521; veRL fully_async; slime #1800 | No tree | Partial rollouts resumed under newer weights | A trajectory spans several policy versions. No branching found |
| **SPEC-RL** 2509.23232 | Reuses old rollouts as drafts. The current policy continues from the first rejected token | Prior-epoch prefix | A verify-then-continue precedent, not a tree |

**Further searches with no hit:** "tree rollout" in AReaL/slime/ROLL/veRL async modes; "mixed-version group"; "branch + staleness"; "partial rollout + tree"; KV reuse across weight updates in a tree setting.

### RTPO in detail (closest threat)
- It updates later turns first. Then: "Before generating siblings for turn k, RTPO synchronizes the training model with the inference engine. Siblings are then generated and continued under the current downstream policy… its trajectory-level importance weight is one." [HTML-summary]
- Mechanically, siblings are forked at an intermediate turn under newer weights, from a prefix (turns < k) that an earlier policy produced. That is exactly "fork under newer weights from an older prefix".
- **It does not pre-empt the async claim.** RTPO avoids staleness by forcing a weight sync before every fork. That is a synchronization barrier, not async training with a lagged generator. It treats the old prefix as conditioning context and gives no stated off-policy correction for it.
- **Verdict: partially pre-empts.** It takes the "fork from a stale prefix under fresh weights" idea. It does not take "under async, bounded-staleness training".
- Authors: Yugu Li, Zehong Cao, Jianglin Qiao, Siyi Hu.

---

## (B) Building blocks from the nearest async papers

### 1. SAO, Single-Rollout Asynchronous Optimization (2607.07508, Jul 2026; Tsinghua KEG / GLM team, used in GLM-5.2)
- **Mechanism [HTML-summary].** Each prompt gets one rollout, which goes to training as soon as it completes. The method adds:
  - an off-policy value model, updated more often than the actor and fine-tuned with frozen attention;
  - token-level IS using rollout-engine log-probs, with "stricter double-sided token-level clipping and masking".
- **Results [HTML-summary].** Vanilla GRPO collapses at about 160 async steps. SAO stays stable for about 1000 steps and beats GRPO on SWE-Bench Verified, BeyondAIME and IMOAnswerBench.
- **Why groups don't fit async (paraphrase; verbatim UNVERIFIED).** "Group-wise sampling induces latency-driven off-policy behavior because the group has to wait for the slower one to finish before fed into training." Waiting for every sibling brings back the straggler problem and widens the gap between rollout and update.
- **What replaces the group.** A learned value baseline (critic) instead of the group mean.
- **Relevance.** This is the main counter-argument the claim must answer. Tree forks are siblings, so they recreate the group-wait problem unless forks are scheduled asynchronously and scored with a stale baseline. SAO also shows that a critic baseline is the alternative design.
- **Verdict: no.** It pre-empts nothing on trees, but it is the strongest objection.

### 2. PrefixRL, "Reuse your FLOPs" (2601.18795, Jan 2026; Setlur, Wang, Cohen, Rashidinejad, Xie)
- **Mechanism [HTML-summary].** It conditions on prefixes of successful off-policy traces (rejection-sampled, from an earlier model) and runs on-policy RL to finish them.
- **Gradient masking: yes.** "gradients are not calculated for the prefix tokens".
- **IS weights: none.** It treats off-policy data "as a prompt rather than a target", which avoids IS gradient spikes. Shorter or longer prefixes adjust problem difficulty.
- **Results.** It reaches the same training reward 2x faster than SFT followed by RL, even counting the rejection-sampling compute, with 3x higher final reward. The authors report "back-generalization" to problems without a prefix.
- **Relevance.** Its justification for "condition on an old prefix, mask it, train only the fresh suffix" carries over directly to stale-prefix forks.
- **Verdict: partially.** It supplies the masking and consistency argument, but has no async training and no tree.

### 3. AReaL-DTA, Dynamic Tree Attention (2602.00482, Feb 2026; ICML 2026; Tsinghua IIIS / Ant AReaL)
- **Mechanism [HTML-summary].** It finds the prefix tree hidden in a batch of rollouts and traverses it depth-first in forward and backward passes, holding only one root-to-leaf path at a time. Batching across GPUs is load-balanced.
- **Async.** It batches rollouts "during an asynchronous rollout generation phase" into several prefix trees.
- **Results.** On τ²-bench, up to 8.31x over dense training and 1.70x over sparse.
- **Branching:** trainer-side deduplication only; the prefix sharing comes from the shared multi-turn history, and nothing is forked. The summaries mention no handling of different policy versions within one tree (UNVERIFIED).
- **Verdict: no.** It is useful as the training back-end for an async tree system.

### 4. PNPO, Prefix-Normalized Policy Optimization ("Reusing Rollouts under Policy Lag", 2608.01418, Aug 2026)
- **Weight [HTML-summary].**

  $w_{PN,t} = \exp\!\Big(\tfrac{1}{t}\sum_{k=1}^{t}\log\rho_k\Big)$, with $\rho_k = \pi_\theta/\pi_{\text{old}}$ at token $k$.

  This is the geometric mean of ratios along the causal prefix.
- **Motivation.** It keeps exact correction for the probability of reaching the prefix while avoiding the dynamic range of the full product.
- **Relevance.** This is the most natural correction weight for a fork whose prefix tokens came from an older policy. The fork-point weight is then $w_{PN,t_\text{fork}}$.
- **Related.** μ-GRPO (2605.17570) names a "prefix-support mismatch": the current policy gives little mass to stale prefixes, yet token-level GRPO still updates the suffixes that follow them.
- **Verdict: no.** It has policy lag but no tree.

### 5. VCPO, Stable Asynchrony (2602.17616, Feb 2026; ICML 2026; Huang, Zhang, Hu, Yang, Han, MIT HAN Lab; code at github.com/mit-han-lab/vcpo)
- **Mechanism [HTML-summary].** Stale rollouts produce heavy-tailed IS ratios. VCPO does two things:
  - scales the learning rate by effective sample size;
  - uses a closed-form minimum-variance off-policy baseline, with no value model.
- **Results.** Stable async training on math, reasoning and tool use at 1.5B–7B. In long-context multi-turn RL it gives a 2.5x end-to-end speedup while matching synchronous performance.
- **Relevance.** It offers a drop-in group baseline for siblings of mixed policy versions, and an ESS diagnostic for forked groups.
- **Verdict: no.**

### 6. Missing Old Logits in Asynchronous Agentic RL (2605.12070, May 2026; Guan, Guo, Sun, Huang, Di, Wu, Wu, Zhao)
- **Problem [HTML-summary].** "Under partial rollout collection, one trajectory can span multiple parameter versions." The training-side old logits for earlier tokens are lost. The decomposition into a training-inference discrepancy term and a staleness term then stops being semantically valid.
- **Fixes:**
  1. snapshot-based version tracking (exact, but memory- and I/O-heavy);
  2. a dedicated old-logit model;
  3. interrupt partial rollouts before each update and compute exact old logits with the still-resident version.

  It also proposes a revised PPO-EWMA.
- **Relevance.** A stale-prefix fork is the same situation: a multi-version trajectory, now shared by several siblings. Per-token version tags and option 3 can be reused directly. Branching is not mentioned.
- **Verdict: partially** for the multi-version-trajectory formalism; **no** for trees.

---

## Verdict
1. **The gap mostly holds.** No paper found combines tree/branching GRPO rollouts with genuinely async (bounded-staleness, non-barrier) training and stale-prefix correction.
2. **Closest threat: RTPO (2608.18682).** It already forks siblings at intermediate turns under freshly synced weights from earlier-policy prefixes. It does this with a sync barrier per fork and frames it as a fix for "asynchronous drift", so the claim has to be pitched as *no barrier, with lag*, not as forking from a stale prefix as such.
3. **Secondary threats:**
   - torchtitan PR #4761: continues old-policy prefix KV under new weights for in-flight groups.
   - MBPO: branching plus age-windowed replay of old segments.
   - STRIDE: restarts from cached prefixes, but for distillation.
4. **Building blocks to cite:**
   - PrefixRL: mask the prefix, train only the suffix.
   - PNPO: a prefix-normalized weight at the fork point.
   - Missing Old Logits: version-tagged old logits.
   - VCPO: ESS-scaled learning rate and a min-variance baseline.
5. **Must answer SAO's argument** that sibling groups add latency-driven off-policyness. For example: forks are dispatched asynchronously and scored against a value or VCPO baseline instead of waiting for every sibling.

Main sources: [RTPO](https://arxiv.org/abs/2608.18682), [SAO](https://arxiv.org/abs/2607.07508), [PrefixRL](https://arxiv.org/abs/2601.18795), [AReaL-DTA](https://arxiv.org/abs/2602.00482), [PNPO](https://arxiv.org/abs/2608.01418), [VCPO](https://arxiv.org/abs/2602.17616), [Missing Old Logits](https://arxiv.org/abs/2605.12070), [MBPO](https://arxiv.org/abs/2608.07581), [Schedule-Level Reuse](https://arxiv.org/abs/2606.01143), [TRACE](https://arxiv.org/pdf/2606.11119), [BPO](https://arxiv.org/abs/2607.14171), [STRIDE](https://arxiv.org/abs/2609.14636), [Prefix-GRPO](https://arxiv.org/abs/2607.19395), [HARTS](https://arxiv.org/abs/2608.28158), [torchtitan PR #4761](https://github.com/pytorch/torchtitan/pull/4761), [APRIL](https://arxiv.org/abs/2509.18521), [SPEC-RL](https://www.arxiv.org/pdf/2509.23232), [μ-GRPO](https://arxiv.org/pdf/2605.17570).
