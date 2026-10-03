# Branching in GRPO: where the compute goes

Written 2026-10-03. Scope: branching rollouts for learning efficiency and compute efficiency, synchronous training only. Async weight updates are out of scope here by request.

Sources:
- **Code, read first-hand this session:** Tree-GRPO, TreePO, TreeRL, ARPO and ATPO, at the commits listed in `prior_work/README.md`.
- **Papers:** the full-text reports in `prior_work/reports/`.
- **Snippet level only:** PATR, BPO and EPIG-Tree have no code. BPO and EPIG-Tree were seen through search snippets only.
- **Cost model:** `analysis/compute_model.py`, a synthetic model, not a measurement. Its assumptions are listed at the end.

## 1. The cost components

One GRPO step with G leaves per prompt spends compute in six places. Branching changes each one differently.

| Component | Bound by | Does sharing a prefix save it? |
|---|---|---|
| Decode of generated tokens | GPU memory bandwidth | Yes, always. The shared turns are decoded once. |
| Prefill of prompt and tool outputs | GPU compute | Only if the engine reuses the prefix KV cache. Without it, a fork re-prefills the whole prefix. |
| Prefill of history on later turns | GPU compute | Same: only with prefix caching. Most frameworks re-submit the full context every turn. |
| Tool or environment calls | CPU, network, sandbox | Yes, if the fork copies the parent after the tool result. |
| Training forward and backward | GPU compute | **No, in every released codebase.** Each leaf is a full row, so a prefix shared by k leaves is trained k times. |
| Idle time | Scheduling | Branching can add idle time: serial expansion phases, and per-round barriers that wait for the slowest path. |

Two facts frame everything below:
- **The saving ceiling is the shared-prefix fraction.** If forks happen at fraction f of the way through a trajectory, the shared tokens are at most f of each fork's tokens. Early forks save little.
- **Training compute is not reduced by any released implementation.** It usually goes up, because forked leaves are longer on average. This is explained in section 4.

## 2. Technique by technique

### TreePO (single-turn math; fixed 512-token segments)
- **Saving source.** Prefix decode, plus prefill through vLLM's automatic prefix cache (`enable_prefix_caching=True`, hard-coded at `vllm_rollout_spmd.py:200`).
- **Scheduling.** One blocking `generate` call per segment covers all active paths of all prompts on a worker. Every path waits at every 512-token boundary, which with a 7168-token budget is up to 14 barriers per rollout. The "bubble filling" option raises `NotImplementedError`.
- **What it measured.** +40% trajectories per second and +30% tokens per second on one H100, offline, no training. The token count includes prefill tokens, so prefix-cache hits count as processed tokens and inflate the token rate. Trajectories per second is the cleaner number.
  - The paper's Figure 4 shows a trade-off. Short segments mean many barriers and recomputation, while long segments mean little sharing. The optimum depth is model-dependent: 14 for one model, 28 for another.
  - Throughput falls past 16 rollouts per prompt on Qwen2.5-Math-7B, because divergent paths fragment the cache.
- **GPU-hours claim (−22% to −43%).** This is test-time sampling of trained checkpoints, not training cost. The −43% configuration loses 3.5 accuracy points.
- **Training side.** No dedup. Every leaf is a full 7k-token row.

### TreeRL / EPTree (single-turn math; token-level forks)
- **Saving source.** Decode only. Prefix caching is off: `--enable_prefix_caching` defaults to False and the training script does not set it. Every fork re-prefills prompt plus partial response.
- **Scheduling.** L+1 serial blocking `generate` calls: initial chains first, then all branches. Prompts are processed sequentially per actor rank.
- **What it measured.**
  - "Budget" counts output tokens only.
  - The authors' own limitations section says the method is about **2× slower** than chain sampling.
  - In RL training it trains on 30 sequences per prompt against 16 for chains. That is about 1.9× the training rows at matched generation tokens.
  - Restricted to 16 sequences, the gain over chains disappears (41.3 vs 41.6).
- **Net.** Fewer decode tokens per leaf, but more wall-clock and more training compute. The accuracy gain is mostly bought with that extra training compute.

### Tree-GRPO (multi-turn search agent; random forks at agent steps)
- **Saving source.** Decode and tool calls of the shared steps.
- **Prefix caching.** Not set explicitly (`vllm_rollout_spmd_ts.py:96`). It is whatever vLLM 0.8.5 does by default with `external_launcher`, which I have not verified.
- **Scheduling.**
  - M full chains run to completion first. Then L serial expansion rounds re-submit each chosen node's full context as a fresh prompt (`generation_ts.py:592-619`).
  - Expansion is a second serial phase whose batch is smaller than the first.
  - Inside each phase, every turn is a batch generate followed by a batch search, so there is a barrier per turn.
- **What it measured.**
  - Budget is the expected token count M·B·(1 + L·N/2), assuming uniform fork position. It was not measured.
  - The paper reports no wall-clock or throughput. The authors note that L>1 hurts throughput because rounds are serial.
  - The default (2,2,1) trains 6 rows at a cost of about 4 chains, so training compute per prompt rises about 1.5× at matched generation budget.
- **Uniform node choice is length-biased.** Choosing nodes uniformly over non-leaf nodes favours long trunks, so forked leaves are longer than average. See section 4.

### ARPO (multi-turn tool agent; online forks after tool calls)
- **Saving source.**
  - Decode and tool calls of the shared prefix.
  - Prefill through the prefix cache: `enable_prefix_caching=True` in both `vllm_rollout_spmd.py:167` and the async server.
  - A branch copies the parent's tokens including the tool result, so that tool call is not repeated.
- **Scheduling.** Rounds with two hard barriers (`vllm_rollout_with_tools.py:264-430`).
  1. One blocking `generate` over all active paths.
  2. All tool calls submitted to a 64-thread pool, waiting for every one to return.
  - Forks join the next round, so forking adds no serial phase.
- **What the code actually does to savings.**
  - The branch decision is a coin flip with probability about 0.5 + 0.2·ΔH, applied in index order.
  - With 8 branch slots and 8 initial paths, most slots are spent at the first or second tool call. Forks that early share almost nothing.
  - The cost model gives 3–11% savings, falling as trajectories get longer.
  - The paper's "half the tool calls" claim is plotted against GRPO with no absolute numbers.
- **Training side.** No dedup. "Soft" advantage is plain GRPO over full rows.

### ATPO (multi-turn dialogue; expand-then-prune with a critic)
- **What it generates.** Every turn, every active leaf generates N=2 candidates (`user_assistant_agent_loop.py:266-279`).
  - Each candidate then gets a user-simulator LLM call and a critic call.
  - When the uncertainty is low, the prune step keeps one random child with probability 0.9 (`:360`). The other child's decode, simulator call and critic call are thrown away.
  - Completed trajectories beyond M are also dropped (`run`, `[:M]`).
- **Extra models.** The critic is a second model doing a forward pass per node. The user simulator is a third.
- **Prefix caching.** Relies on the async server's cache. Each turn re-sends the full message history.
- **Scheduling.** Each prompt's tree runs its own asyncio loop with a barrier per turn inside that tree, but there is no barrier across prompts. This is the only released tree implementation without a batch-wide barrier.
- **Net.** It spends compute to buy information (expand-then-prune) rather than saving it. It is the opposite end from TreePO.

### PATR (multi-turn SWE; online, start with 4 and expand the top 2 every K turns)
- **Saving source.** Prefix decode and tool calls. Forks happen at turns 13, 26 and 39, so the shared prefix is long and the saving per leaf is the largest of any scheme here: about 37% in the cost model.
- **Costs not reported.**
  - The scorer is a 7B PRM or 7B LLM judge, larger than the 4B policy, called on every active branch at every checkpoint. That cost is excluded.
  - Group size is variable, up to about 10 leaves against GRPO's 8.
  - The paper gives no tokens, turns, tool calls or wall-clock.
- **Docker fork mechanism.** Not described, so the cost of copying a SWE sandbox is unknown.

### BPO (multi-turn, stateful sandboxes; post hoc on one backbone)
- **Saving source.** Prefix decode and tool calls, plus sandbox replay avoided by snapshot/restore.
  - Snapshot cost is about 42 ms on WebShop, 138 ms on ALFWorld and 1.9 s on SWE-bench. These figures are from the round-2 survey and not verified.
- **Scheduling.** The backbone must finish before branches start, so this is a serial phase, like Tree-GRPO.
- **Matching.** Compute is matched by number of returns, 1 + M(K−1) = N, not by tokens or wall-clock.

### EPIG-Tree (post hoc after pilot rollouts; cost-aware allocation)
- The only method with compute in its objective: a per-edge cost c_e in the allocation law.
- That cost is an abstract constant. It is not tied to decode versus prefill, cache state, or barrier time.
- Pilot then branch is a serial phase, like BPO and Tree-GRPO.

### Crab (systems; sandbox checkpoint/restore, no training)
- Forking from a checkpoint instead of re-executing the prefix cuts rollout tokens by 40–64% on SWE-agent at branching factors 1–5. This is round-2 survey material, not verified.
- It is the only measurement of the environment-side saving.

## 3. Cost-model results

`analysis/compute_model.py` builds about 8 leaves per prompt with each scheme's fork pattern, under two synthetic workloads, and averages 30 batches of 32 prompts.

- **Agentic workload:** 2–10 turns, about 300 decode and 600 tool-output tokens per turn, 1.5k-token prompt, lognormal tool latency with a 2 s mean.
- **SWE-like workload:** 10–50 turns, 4k-token prompt, 1.5k tool-output tokens per turn.

Per leaf, relative to flat GRPO (1.00 means the same cost per training sample):

| Scheme | Workload | Decode | Tool calls | Prefill, cache on | Prefill, cache off | Train, full rows | Train, dedup | Wall-clock, round barriers | Wall-clock, no barriers |
|---|---|---|---|---|---|---|---|---|---|
| tree_grpo | agentic | 0.80 | 0.80 | 0.81 | 0.92 | 1.09 | 0.81 | 1.42 | 1.37 |
| arpo | agentic | 0.89 | 0.90 | 0.90 | 0.97 | 1.02 | 0.90 | 0.95 | 1.00 |
| patr | agentic | 0.71 | 0.71 | 0.72 | 0.86 | 1.17 | 0.72 | 0.95 | 1.00 |
| bpo | agentic | 0.79 | 0.78 | 0.79 | 0.93 | 1.11 | 0.79 | 1.36 | 1.32 |
| mid_fork | agentic | 0.78 | 0.78 | 0.79 | 0.95 | 1.10 | 0.79 | 0.96 | 0.96 |
| tree_grpo | SWE | 0.73 | 0.73 | 0.73 | 0.88 | 1.08 | 0.73 | 1.31 | 1.55 |
| arpo | SWE | 0.97 | 0.97 | 0.97 | 0.98 | 0.99 | 0.97 | 1.00 | 1.01 |
| patr | SWE | 0.63 | 0.63 | 0.63 | 0.79 | 1.16 | 0.63 | 0.98 | 1.04 |
| bpo | SWE | 0.68 | 0.68 | 0.68 | 0.87 | 1.10 | 0.68 | 1.20 | 1.48 |
| mid_fork | SWE | 0.71 | 0.70 | 0.71 | 0.90 | 1.06 | 0.71 | 0.90 | 1.00 |

The wall-clock columns are ratios against flat under the same scheduling mode.

Flat GRPO under the two scheduling modes:

| Workload | Round-barrier time ÷ no-barrier time | Decode-slot occupancy with barriers |
|---|---|---|
| agentic | 3.7× | 0.15 |
| SWE | 6.7× | 0.15 |

## 4. What the numbers say

1. **Rollout savings from branching are real but bounded at roughly 20–37% per leaf.** They come from decode, tool calls and cached prefill, and track how deep the forks are. Late, spaced forks (PATR, mid_fork) save the most. Early forks (ARPO as released) save almost nothing, and less as trajectories get longer.
2. **Prefill savings need a prefix cache.** With the cache off, as in TreeRL and possibly Tree-GRPO, the prefill saving drops to 2–21%. This is because every turn already re-submits the whole history, and the fork only avoids the shared turns' share of that.
3. **Training compute does not go down in any released implementation: −1% to +17% per leaf.** Two causes:
   - **No dedup.** Each leaf is a full row, so shared prefixes are trained once per descendant.
   - **Forks are length-biased.** A fork needs a parent that is still running at turn j, so forked leaves are longer than independent chains. This is also a sampling bias in what the model learns from, independent of compute.
   - With prefix dedup, training compute would fall by the same 20–37% as rollout. slime's `TrajectoryManager` and AReaL-DTA already dedup; none of the tree papers do.
4. **Post-hoc forking costs wall-clock even when it saves tokens.** Tree-GRPO, BPO, EPIG-Tree and TreeRL must finish the first chains before forking. That adds a serial phase: +20% to +55% rollout time in the model, and TreeRL reports 2× in practice. Online forking (ARPO, PATR) keeps the critical path at about flat.
5. **Round barriers are a bigger lever than branching.** Under level-synchronous rounds, flat GRPO takes 3.7× (agentic) to 6.7× (SWE) as long as with no barriers, with decode-slot occupancy around 0.15. Branching's 20–37% token saving is small next to this.
   - TreePO's per-segment barrier and ARPO's per-round generate-then-all-tools barrier both sit on this cost.
   - This is the same long-tail problem as in `previous_works.md`, now inside a group. Siblings created by a fork inherit the parent's position in the tail.
6. **No paper reports a compute-honest comparison.** The units differ for each one:

   | Paper | What its compute saving is measured in |
   |---|---|
   | TreePO | Offline throughput with cache hits counted as processed tokens; GPU-hours at test time, not training |
   | TreeRL | Output tokens only, at 1.9× the training rows |
   | Tree-GRPO | Expected budget, not measured |
   | ARPO | Tool calls, relative only |
   | BPO | Number of returns |
   | PATR | Nothing; scorer cost excluded |
   | EPIG-Tree | Abstract cost constant |

## 5. What a compute-honest evaluation needs

Report all of these for flat GRPO and each branching scheme, at matched learning outcome or matched compute:

- Decode tokens, prefill tokens actually computed (from engine cache-hit statistics), and tool calls or sandbox seconds.
- Training tokens forwarded and backwarded, with and without prefix dedup.
- Rollout wall-clock and GPU-busy fraction, under the same scheduler.
- Extra models: scorer, critic, user simulator.
- Wasted work: pruned, truncated or discarded generations.
- Leaf length distribution against flat, to expose the fork length bias.

The cheapest first experiment is the four-way split from the earlier discussion, with these metrics:
- flat;
- tree with random online forks and no dedup;
- the same with prefix dedup;
- the same with an edge-level advantage.

It isolates how much of branching's benefit is prefix reuse, how much is training compute, and how much is credit assignment, before any smart branch rule is built.

## Assumptions in the cost model

- **Workload.**
  - Turns per trajectory are uniform in a range.
  - Decode length, tool-output length and tool latency are independent lognormals.
  - A fork resamples the remaining length from the same distribution, conditioned on reaching the fork.
- **Engine.**
  - Decode step time is flat in batch size (memory-bound). Prefill time is not modelled.
  - "Cache on" means an ideal prefix cache with no eviction.
- **Schemes.** Branch choices are random in every scheme, so this measures fork placement patterns, not branch signals.
- **Wall-clock modes.** The round-barrier mode lets every round last as long as its slowest decode plus its slowest tool call. The no-barrier mode lets each path run independently.
- **Not modelled:** KV memory pressure, cache eviction, sandbox snapshot cost, scorer or critic cost.

Changing the workload parameters at the top of the script changes the numbers. The direction of findings 3, 4 and 5 did not change between the two workloads tried.
