# The idea

## Original statement (2026-10-03, dictated, verbatim)

> I had a conversation with my advisor at the research meeting, and I was explaining how the RA push training works in terms of system architecture and workflow end to end. I was focusing on the asynchronous mode, like colocation mode, where the rollout cluster and training cluster are running at the same time, and the model weight is updated and synced in an async way. Rollout will not be stored by the trainer, and the trainer will not be stored by rollout. Still, there will be some stimulus control, like admission control or whatever.
>
> Anyway, that was the context, and now I am explaining how the GRPL works. You have a shared prompt, and then you have a G number of samples. GRPO samples them, and each trajectory will go to its own trajectory. They will diverge from the shared prompt and end up doing their own things. Some of them will find the answer, and some of them will not find the answer for the shared prompt for the task. Then you average and calculate the standard deviation, and you backpropagate. That's how GPT works.
>
> I was thinking, and my advisor suggests an interesting idea. You sample a trajectory, and we always assume this is a multi-turn agentic application, so there will be turn 0, 2col, turn 1, 2col, turn 2, 2col, and so on. In the middle, let's say you don't sample G trajectories in the first place, but you sample some number, let's say a quarter of G, or let's say 1 trajectory. You do a 2col, and based on how this trajectory goes, you see, "Okay, let's branch out a new trajectory, basically the second sample from turn 2," because of whatever the reason. In this way, do you know what I'm saying?

Transcription noise: "RA push training" = RL post-training, "2col" = tool call, "GRPL" / "GPT" = GRPO, "stored" is probably "stalled".

## Cleaned-up reading

- **Setting:** multi-turn agentic tasks (turn 0, tool call, turn 1, tool call, ...), trained with GRPO in asynchronous colocated mode. Rollout and training clusters run at the same time, weights sync asynchronously, neither side blocks the other, with some admission control.
- **Baseline:** G independent trajectories from a shared prompt, group mean/std normalisation of rewards, policy-gradient update.
- **Proposal:** do not launch all G up front. Launch a few (G/4, or 1). After observing how a trajectory goes, fork a new sample from an intermediate turn. The fork shares the prefix and resamples from that turn onward. The group becomes a tree, and the branch decision is adaptive.

## Why it is attractive

- **Cheaper rollouts:** the shared prefix is generated once and its KV cache reused, so the same token budget buys more leaves.
- **Sharper credit assignment:** siblings forked from the same turn-t state differ only in what happened after t, so their mean reward is a state-conditional baseline with lower variance than the prompt-level mean.
- **Budget goes where it matters:** skip branching on trajectories whose outcome is already decided; spend on uncertain ones.

## What has to be got right

- **Environment forking:** branching at turn t needs the sandbox state at t (filesystem, processes, tool side effects), by snapshot or by replaying tool calls.
- **Selection bias:** if the branch decision depends on how the trajectory turned out, the prefix distribution is no longer on-policy. Prefix tokens should be trained once, not once per descendant.
- **What "group" means:** normalise over all leaves under the prompt, over siblings at each branch node, or both. Prefix and suffix tokens then get different advantages.
- **Async staleness:** a branch may be sampled under newer weights than its prefix, so one trajectory mixes policy versions. Per-token rollout log-probs with importance-sampling correction (`--use-tis` in slime) can handle this, but it has to be deliberate.

## What would change in the slime fork

- `get_grpo_returns` (`slime/utils/ppo_utils.py`) broadcasts one scalar advantage over all tokens of a sample; a tree needs per-turn or per-segment advantages.
- `_post_process_rewards` (`slime/ray/rollout.py`) reshapes rewards to `(-1, n_samples_per_prompt)`; it would have to group by tree node.
- The progressive and early-filter code in `slime/rollout/sglang_rollout.py` (`_generate_group_progressive`, `_generate_group_with_early_filter`) already generates a fraction of the group first and then decides. Today the decision is continue-or-drop; this idea makes it where-to-spend-the-remaining-budget.
- `slime/rollout/filter_hub/sr_predictor.py` is a candidate branch signal (fork where predicted success is near 0.5).
