# How GRPO is implemented in the slime fork

Read on 2026-10-03 from `/Users/gangmuk2/projects/rlopt`, branch `gangmuk/buffer-filtering-port`, HEAD `08545943`. Line numbers are from that commit. Covers the synchronous loop only; `train_async.py`, `fully_async_rollout.py` and the rollout-buffer streaming code were not read.

## The algorithm

GRPO is PPO without a learned critic. For each prompt, sample G responses from the current policy, score each, and use the group's own reward statistics as the baseline:

```
A_i = (r_i − mean(r_group)) / (std(r_group) + ε)
```

Every token in response i gets that same scalar advantage. The loss is the PPO clipped surrogate over tokens, with an optional KL penalty to a frozen reference model:

```
ratio_t = π_θ(tok_t) / π_old(tok_t)
L = −min(ratio_t · A_i, clip(ratio_t, 1−ε, 1+ε) · A_i)  [+ β · KL(π_θ ‖ π_ref)]
```

A group where all rewards are equal gives zero advantage and no gradient, which is why the code tracks zero-std groups.

## Code path

Outer loop: `train.py:49`. Generate a rollout, train on it, push new weights to the SGLang engines, repeat.

| Step | Where | What happens |
|---|---|---|
| 1. Form groups | `slime/rollout/data_source.py:110` | Each prompt is deep-copied `n_samples_per_prompt` times, all sharing a `group_index`. |
| 2. Sample and score | `slime/rollout/sglang_rollout.py:1644` (`generate_and_rm_group`) | Each copy is generated via SGLang and scored. Early-filter and progressive variants are at lines 1031 and 1264. |
| 3. Group-normalise rewards | `slime/ray/rollout.py:905` (`_post_process_rewards`) | Reshape to `(-1, n_samples_per_prompt)`, subtract mean, divide by std. This is the core GRPO step and it runs on the rollout side, not in the trainer. |
| 4. Old log-probs | `slime/backends/megatron_utils/actor.py:482-509` | No-grad forward pass computes `log_probs` under the pre-update policy (and `ref_log_probs` if a ref model is loaded). |
| 5. Advantages | `slime/backends/megatron_utils/loss.py:720` → `slime/utils/ppo_utils.py:361` (`get_grpo_returns`) | The normalised scalar reward is broadcast across the response's tokens. |
| 6. Loss | `loss.py:881` (`policy_loss_function`) → `ppo_utils.py:124` (`compute_policy_loss`) | Clipped surrogate, minus entropy bonus, plus KL loss if enabled. |
| 7. Sync weights | `train.py:85` | `actor_model.update_weights()` pushes the updated policy to the rollout engines. |

## Details that matter

- **`--kl-coef` does nothing for GRPO here.** In `get_grpo_returns` the `kl` tensor is used only for its shape. The GRPO-paper KL term comes from `--use-kl-loss --kl-loss-coef` at `loss.py:1053`.
- **The fallback reshape is a trap.** At `rollout.py:916-920`, if the sample count is not exactly `n_samples_per_prompt × rollout_batch_size`, the whole batch is treated as one group: normalisation becomes batch-wide instead of per-prompt, with no error. This can bite when buffer filtering drops samples or groups. `slime/rollout/_fanout_test_helpers.py:76` has a version that groups by `group_index`.
- **Old log-probs come from one of two places.** By default the trainer recomputes them with Megatron. With `--use-rollout-logprobs` it uses the ones SGLang reported. `--use-tis` adds an importance-sampling correction for the mismatch (`loss.py:831`).
- **On the first step the ratio is 1.** The policy has not moved, so clipping is inactive and the update is plain REINFORCE with a group baseline. Clipping matters only when `num_steps_per_rollout > 1` or the data is off-policy.

## Variants on the same path

Selected with `--advantage-estimator`; all share the group normalisation unless noted.

- **`gspo`**: one sequence-level ratio (mean log-ratio over the response) instead of per-token ratios (`ppo_utils.py:95`).
- **`cispo`**: clips the ratio under stop-gradient and multiplies by `log_probs`, so clipped tokens still contribute gradient (`ppo_utils.py:152`).
- **Dr. GRPO**: `--disable-grpo-std-normalization` subtracts the mean only.
- **`reinforce_plus_plus_baseline`**: subtracts the group mean, never divides by std.
- **`ppo`**: learned critic with GAE; skips group normalisation.
