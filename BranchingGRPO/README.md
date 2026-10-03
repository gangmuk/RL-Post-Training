# BranchingGRPO

Research notes on adaptive mid-trajectory branching for GRPO-style RL post-training of multi-turn LLM agents.

Started 2026-10-03. Codebase under study: the slime fork at `/Users/gangmuk2/projects/rlopt` (branch `gangmuk/buffer-filtering-port`).

## The idea in one paragraph

Standard GRPO samples G independent trajectories from a shared prompt. In a multi-turn agentic task (turns separated by tool calls), start with fewer (G/4, or even 1), watch how they unfold, and fork new samples from an intermediate turn. A fork shares the prefix (prompt plus earlier turns) and resamples from that turn onward, so the group becomes a tree, and where to branch is decided adaptively. The target setting is asynchronous colocated training, where rollout and training run concurrently and weights sync asynchronously. Full statement: [idea.md](idea.md).

## Status (2026-10-03, end of second session)

- Understood how GRPO is implemented in the slime fork: [notes/slime_grpo_implementation.md](notes/slime_grpo_implementation.md).
- Read upstream slime's fully-async path and sketched where a fork would go: [notes/slime_async_path.md](notes/slime_async_path.md). The fork itself was not reachable from that session; re-check against it.
- Prior-work passes: four papers read in full with code, a wider survey, PATR read first-hand, and a round-3 gap check on async trees (search snippets only). Synthesis: [prior_work/README.md](prior_work/README.md).
- Compute analysis of each branching technique (code-verified for Tree-GRPO, TreePO, TreeRL, ARPO, ATPO) with a synthetic cost model: [notes/compute_analysis.md](notes/compute_analysis.md), [analysis/compute_model.py](analysis/compute_model.py).
- No design or real experiments yet.

## Main finding so far

The idea as first stated (start with fewer than G, fork adaptively at tool-call boundaries) is already published in several forms: ARPO, PATR, BPO, ATPO, EPIG-Tree.

The async angle is still open, but narrower than it looked after round 2:

- Continuing generation under new weights on KV computed by old weights is existing practice for flat rollouts. PipelineRL does it on purpose, and upstream slime ships it (`--flush-cache-interval` other than 1 gives in-place weight sync).
- RTPO (2608.18682) forks siblings at turn k under newer weights from an older prefix, but puts a weight sync before every fork. That is a barrier, not lag.
- Prefix masking (PrefixRL; slime's `--mask-offpolicy-in-partial-rollout`) and prefix importance weights (PNPO) already exist as building blocks.

Candidate novelty axes, strongest first:

1. **Fork scheduling coupled to weight sync, without a barrier.** With KV flushed at sync, a fork issued after a sync re-prefills the whole prefix. With in-place sync it is cheap but stale. A fork inherits its prefix's age and may be born over the staleness bound. Treating "fork now, before the next sync, or not at all" as a scheduling decision is untouched.
2. **Sibling baselines that mix policy versions inside one tree**, and SAO's objection that groups force waiting on stragglers. VCPO's off-policy baseline exists for flat groups only.
3. **Sandbox fork and KV-cache fork as one system**, with fork latency as an input to the branch decision. In slime's coding-agent example the agent harness runs inside the E2B sandbox, so a fork must copy a live process.
4. **Estimator correctness**: per-edge advantages with segment masks (EPIG-style), prefix trained once. slime's agent path already dedups the prefix but gives it one leaf's advantage. BPO reportedly concedes an uncorrected over-weighting of high-entropy states. OPTS-TTPO (2609.40035) may already cover the weighting; unread.

## Open questions

- Does branch placement matter at all on agent tasks? Evidence so far is weak (TreePO null result, EPIG-Tree within noise on single-turn math, TreeRL gain mostly from more training sequences, PATR within noise of Tree-Random and ARPO).
- How does PATR fork Docker state? The paper does not say.
- How often would a fork actually cross a weight sync? If turn boundaries rarely straddle a sync, the async problems are second-order. This is measurable on stock slime; see "What to measure first" in [notes/slime_async_path.md](notes/slime_async_path.md).
- Do flat groups under in-place sync already span several policy versions? If so, a mixed-version sibling baseline is not a new problem.
- Counter-position to address: Single-Rollout Asynchronous Optimization (2607.07508) argues group-wise GRPO does not fit async agentic training at all.

## Next steps

- Re-read the slime fork's async path (`train_async.py`, `fully_async_rollout.py`, rollout buffer) and diff it against the upstream reading in `notes/slime_async_path.md`.
- Run the three measurements in `notes/slime_async_path.md` on stock slime fully-async with a stateless-tool multi-turn task (search-r1 or retool). They need no algorithm change and decide whether axis 1 is a real systems problem.
- Read RTPO (2608.18682) and SAO (2607.07508) first-hand from a machine with arXiv access. They bound the async claim most. Then BPO §4 and OPTS-TTPO for the estimator claim.
- Check what E2B offers for sandbox snapshot or fork, and whether the Claude Code or Codex harness can resume from a saved session.

## Layout

| Path | Contents |
|---|---|
| `idea.md` | Original statement of the idea and a cleaned-up reading of it |
| `notes/` | Working notes: slime GRPO walkthrough (fork), fully-async path and fork sketch (upstream) |
| `prior_work/README.md` | Comparison table, novelty assessment, trust caveats, repo list |
| `prior_work/reports/` | Per-paper reports from the reading agents, saved verbatim |
| `prior_work/papers/` | PDFs and extracted text of the four papers read in full |
| `prior_work/repos/` | Clones of seven prior-work repos (git-ignored; URLs and commits in `prior_work/README.md`) |
| `prior_work/reports/` (round 3) | `bpo_epig.md`, `async_gap.md`: search-snippet reads, weaker than the rest |
| `analysis/` | Synthetic cost model for branching schemes (`compute_model.py`) |
| `research_log.md` | Chronological log of what was done |
| `agent_briefs/` | The exact brief given to each reading agent |
| `agent_transcripts/` | Raw JSONL transcripts of each agent's session |
