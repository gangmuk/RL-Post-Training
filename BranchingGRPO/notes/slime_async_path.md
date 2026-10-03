# slime's fully-async path, and where a fork would go

Read on 2026-10-03 from **upstream** slime (https://github.com/THUDM/slime, HEAD `8c17b67`, 2026-10-02), cloned fresh. The fork at `/Users/gangmuk2/projects/rlopt` (branch `gangmuk/buffer-filtering-port`) was not reachable from this session, so file names and line numbers here are upstream's. They differ from the fork: upstream has no `train_async.py`, and reward normalisation moved from `slime/ray/rollout.py` to `slime/data/batch_builder.py`. **Re-check each point against the fork before relying on it.**

## How fully-async works upstream

There is no separate async training loop. `train.py` is the same loop as the sync path. Async behaviour comes from swapping the rollout function:

```
--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
```

| Piece | Where | What it does |
|---|---|---|
| Background worker | `slime/rollout/fully_async_rollout.py:83` (`AsyncRolloutWorker`) | A thread with its own asyncio loop, created once and kept across rollout calls, so in-flight work survives a training step. |
| Admission | `fully_async_rollout.py:162-180` | Keeps at most `concurrency` group tasks in flight, where `concurrency = sglang_server_concurrency × engines ÷ n_samples_per_prompt`. Stops topping up while the completed-group queue already holds `concurrency` groups (backpressure). |
| Unit of work | `fully_async_rollout.py:171` | One asyncio task per **group**, running `generate_and_rm_group` (`sglang_rollout.py:384`), which starts all G samples at once and gathers them. |
| Consumption | `fully_async_rollout.py:222` | Each `generate_rollout` call drains completed groups until it has `rollout_batch_size`, applying the dynamic filter. The surplus stays queued for the next step. |
| Abort handling | `fully_async_rollout.py:210-216` | If any sample in a group comes back `ABORTED`, the whole group is pushed back to the data buffer. Completed samples in it are kept as is: `generate_and_rm` returns early for `COMPLETED`/`TRUNCATED` (`sglang_rollout.py:300`). Aborted ones continue from where they stopped, because `generate` subtracts `response_length` from `max_new_tokens` and re-submits prompt plus partial response (`sglang_rollout.py:172`). |
| Weight sync | `train.py:92-97`, `update_weight_from_distributed.py:103` | Pause admission, then pause generation on every engine, optionally flush the KV cache, push weights, resume. |

### The weight-sync knob that matters for this project

`--flush-cache-interval N` (`slime/utils/arguments.py:340`, policy in `slime/utils/weight_sync.py`):

- **N = 1 (default).** Every sync calls `pause_generation(mode="abort")` then `flush_cache`. In-flight requests are aborted and the radix/prefix cache is emptied. The partial sample is re-prefilled under the new weights when it resumes.
- **N ≠ 1.** Syncs that do not flush use `pause_generation(mode="in_place")`. Unfinished sequences keep their KV cache and **continue decoding under the new weights on KV computed by the old weights**. The help text names this PipelineRL. It requires separate rollout and training GPUs and switches the rollout function to fully-async automatically (`arguments.py:2209-2220`).

So slime already runs, for flat rollouts, the exact mixed-version situation the async-tree idea worried about: a continuation sampled under θ' on top of a prefix and KV cache produced under θ.

### Version tracking

- `Sample.weight_versions` (`slime/utils/types.py:133`) gets one entry per generate call that produced tokens, taken from SGLang's `meta_info["weight_version"]` (`types.py:488`).
- Staleness of a sample is `current_version − min(weight_versions)` (`slime/utils/staleness.py`), logged as `rollout/staleness/{mean,max}`.
- One entry per call, not per token. Under in-place sync, a single call that spans a sync records one version for tokens from two policies. I have not checked which version SGLang reports in that case.
- Per-token rollout log-probs are always stored (`return_logprob: True`, `sglang_rollout.py:184`). `--use-rollout-logprobs` and `--use-tis` use them for the behaviour-policy side, which is per token and therefore correct across a version boundary even when `weight_versions` is coarse.
- `--partial-rollout --mask-offpolicy-in-partial-rollout` sets `loss_mask = 0` on everything generated before a resume (`sglang_rollout.py:296`). This is the PrefixRL-style "train only the fresh continuation" option, already in the codebase for flat samples.

### Trajectory trees already exist, for a different reason

`slime/agent/trajectory.py` (`TrajectoryManager`) builds a per-session message tree from multi-turn agent traffic and linearises each root-to-leaf chain into one `Sample`. It forks when the prompt prefix diverges (sub-agent calls, context compaction, re-tokenisation drift), not to explore. Two details carry over directly:

- **Shared-prefix dedup is implemented.** "A generated turn shared by sibling leaves is trained only on the first leaf to claim it; later leaves re-emit it as loss_mask=0 context so the shared prefix isn't double-counted" (`trajectory.py:461`). Every tree-RL repo in `prior_work/` trains the prefix once per descendant; slime's agent path does not.
- **But the prefix then gets one arbitrary leaf's advantage.** With a scalar advantage per Sample (`get_grpo_returns`, `slime/utils/ppo_utils.py:533`), the first-claiming leaf's advantage is applied to the shared turns. For exploration trees that is the wrong credit. The prefix should get a node-level value, not a leaf's.

### Group-size assumptions that a tree breaks

- `AsyncRolloutWorker.concurrency` divides by `n_samples_per_prompt`, so admission assumes every group costs G samples.
- `generate_and_rm_group` launches all G at once. There is no hook to add samples to a running group.
- `_post_process_rewards` (`slime/data/batch_builder.py:202`) reshapes to `(-1, n_samples_per_prompt)` only when the total count is exactly `G × rollout_batch_size`. Otherwise it normalises the whole batch as one group, silently. Variable leaf counts per tree hit this fallback every time. The fork had the same trap at `slime/ray/rollout.py:916`.
- Escape hatches exist: `custom_reward_post_process_func` and `custom_convert_samples_to_train_data_func` in `batch_builder.py` take over normalisation and conversion entirely, so a tree estimator can live in a plug-in instead of a core patch.

### Sandbox side

The coding-agent example (`examples/coding_agent_rl/`) boots an E2B sandbox per sample and runs the agent harness (Claude Code or Codex CLI) **inside** it. slime only sees the model calls through `slime.agent.adapters.AnthropicAdapter`. Forking at turn t therefore means forking a running harness process plus filesystem, not just a message list. Options, none implemented upstream:

- sandbox snapshot/restore at a turn boundary (E2B pause/resume or snapshots; I have not verified what E2B supports);
- restart the harness from a saved session at turn t, if the harness can resume a conversation, and replay nothing;
- replay the tool calls of turns 0..t in a fresh sandbox, which is only safe for deterministic tools.

Simpler tasks with stateless tools (`examples/search-r1`, `examples/retool`) avoid this entirely and are the obvious first testbed.

## What a fork would look like

A sketch, not a design. The point is to find where the async interaction bites.

1. **Unit of work becomes a tree task.** Replace the per-group `generate_and_rm_group` call in the worker with a tree driver that starts k roots, observes turn boundaries, and spawns child tasks. A child is a deep copy of the parent `Sample` at turn t (tokens, `loss_mask`, `rollout_log_probs`, `weight_versions`) plus a sandbox fork. Admission must count leaves or tokens, not groups.
2. **The fork's first request is a prefix-cache hit or a full re-prefill, and the weight-sync schedule decides which.**
   - With `--flush-cache-interval 1`, any sync between the parent writing turn t and the fork being issued empties the cache. The fork re-prefills the whole prefix under θ'. In agentic tasks the prefix is long and prefill is most of the tokens (ROSE measured 77–86%), so this is the dominant fork cost.
   - With in-place sync, the fork reuses the parent's KV computed under θ. This is cheap but means the fork's suffix is conditioned on stale activations, exactly as PipelineRL continuations already are.
   - **Consequence:** "fork now or wait" is a real decision with a measurable cost on one side (re-prefill tokens after a flush) and a staleness cost on the other.
3. **A fork inherits its prefix's age.** Staleness is `current − min(weight_versions)`, so a fork from an old prefix is born already stale. Under a staleness bound (AReaL η, slime's dynamic filter), a late fork may be over the limit before it generates a token. Options: forbid forks from prefixes older than some age; or mask the prefix (`loss_mask = 0`, as `--mask-offpolicy-in-partial-rollout` does) and count only the suffix's age.
4. **Siblings may be sampled under different policies.** Two forks from the same node issued on either side of a sync differ in θ as well as in randomness. A sibling baseline then mixes versions. Per-token IS weights (`--use-tis`) correct each sample's own ratio but not the baseline. Whether that matters in practice is unmeasured.
5. **Training rows.** Simplest representation that fits slime's one-scalar-per-Sample advantage: one row per tree edge. Each row carries the full prefix as context with `loss_mask = 0` and the edge's own tokens with `loss_mask = 1`, and gets that edge's advantage (for example, child value minus parent value, with values from descendant leaf rewards). The prefix is then trained exactly once with a node-level credit. The cost is re-forwarding the prefix once per row in the trainer, which is what AReaL-DTA's prefix-tree training removes.
6. **Reward normalisation** goes in a `custom_reward_post_process_func` that groups by `group_index` and tree node, so the batch-wide fallback never fires.

## What to measure first (no algorithm change needed)

These use stock slime fully-async on a multi-turn task with stateless tools, and decide whether the async-tree angle has a systems problem worth solving:

1. **Prefix age at turn boundaries.** For each turn boundary of each trajectory, how many weight syncs happened since the prefix was produced? This bounds how often a fork would cross a sync.
2. **Fork cost under each sync mode.** Prefix length in tokens at each turn boundary, and SGLang prefix-cache hit rate for a re-submitted prefix (already recorded in `Sample.prefix_cache_info`), with `--flush-cache-interval 1` vs in-place.
3. **Version spread inside a group** under in-place sync: `max − min` of `weight_versions` across a group's samples. If flat groups already span several versions, a mixed-version sibling baseline is not a new problem.
