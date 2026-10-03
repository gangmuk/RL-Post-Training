# BranchingGRPO

Research notes on adaptive mid-trajectory branching for GRPO-style RL post-training of multi-turn LLM agents.

Started 2026-10-03. Codebase under study: the slime fork at `/Users/gangmuk2/projects/rlopt` (branch `gangmuk/buffer-filtering-port`).

## The idea in one paragraph

Standard GRPO samples G independent trajectories from a shared prompt. In a multi-turn agentic task (turns separated by tool calls), start with fewer (G/4, or even 1), watch how they unfold, and fork new samples from an intermediate turn. A fork shares the prefix (prompt plus earlier turns) and resamples from that turn onward, so the group becomes a tree, and where to branch is decided adaptively. The target setting is asynchronous colocated training, where rollout and training run concurrently and weights sync asynchronously. Full statement: [idea.md](idea.md).

## Status (2026-10-03)

- Understood how GRPO is implemented in the slime fork: [notes/slime_grpo_implementation.md](notes/slime_grpo_implementation.md).
- Prior-work pass done: 4 papers read in full with code, plus a wider survey. Synthesis: [prior_work/README.md](prior_work/README.md).
- No design, code, or experiments yet.

## Main finding so far

The idea as first stated (start with fewer than G, fork adaptively at tool-call boundaries) is already published in several forms: ARPO, PATR, BPO, ATPO, EPIG-Tree. What the search did not find is any tree-rollout method that works under asynchronous training. Every tree method assumes one policy version for the whole tree.

Candidate novelty axes, strongest first:

1. **Tree rollouts under async training.** A fork sampled under new weights from a prefix and KV cache produced under old weights; sibling baselines that mix policy versions; KV-cache invalidation at weight sync; fork-now-or-wait under a staleness bound.
2. **Sandbox fork and KV-cache fork as one system**, with fork latency as an input to the branch decision.
3. **An outcome-aware branch signal** (e.g. the success-rate predictor in `slime/rollout/filter_hub/sr_predictor.py`). Thin on its own: PATR's process scorer is close.
4. **Estimator correctness**: prefix deduplication, per-fork-node sibling baselines, bias from adaptive branching.

## Open questions

- Does branch placement matter at all on agent tasks? Evidence so far is weak (TreePO null result, EPIG-Tree within noise on single-turn math, TreeRL gain mostly from more training sequences).
- How does PATR fork Docker state? The paper does not say.
- Is the async gap real, or just not found? Several relevant papers are under three months old.
- Counter-position to address: Single-Rollout Asynchronous Optimization (2607.07508) argues group-wise GRPO does not fit async agentic training at all.

## Next steps

- Read PATR (2607.15610) and BPO (2607.14171) personally. They constrain the novelty claim most and got the shallowest reads.
- Check EPIG-Tree and BPO before claiming anything on estimator bias.
- Sketch what a fork looks like in slime's async path (`train_async.py`, `fully_async_rollout.py`, rollout buffer). That path has not been read yet.

## Layout

| Path | Contents |
|---|---|
| `idea.md` | Original statement of the idea and a cleaned-up reading of it |
| `notes/` | Working notes (slime GRPO implementation walkthrough) |
| `prior_work/README.md` | Comparison table, novelty assessment, trust caveats, repo list |
| `prior_work/reports/` | Per-paper reports from the reading agents, saved verbatim |
| `prior_work/papers/` | PDFs and extracted text of the four papers read in full |
| `prior_work/repos/` | Clones of seven prior-work repos (git-ignored; URLs and commits in `prior_work/README.md`) |
| `research_log.md` | Chronological log of what was done |
| `agent_briefs/` | The exact brief given to each reading agent |
| `agent_transcripts/` | Raw JSONL transcripts of each agent's session |
