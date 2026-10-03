# Research log

## 2026-10-03

**1. Read the GRPO path in the slime fork.**
Traced the synchronous loop from `train.py` through data source, rollout, reward normalisation, advantage computation and loss. Written up in `notes/slime_grpo_implementation.md`. Not read: the async path (`train_async.py`, `fully_async_rollout.py`, rollout buffer).

**2. Idea stated.**
From a meeting with the advisor: start with fewer than G trajectories and adaptively fork from intermediate turns. Recorded in `idea.md`. Initial guess at prior work from memory: Tree-GRPO, TreeRL, TreePO, TRPO vine.

**3. Prior-work reading, round 1.**
Launched four reading agents in parallel: Tree-GRPO, TreeRL, TreePO (each: confirm citation, read full paper, clone and read repo), and a wider survey (TRPO vine plus anything on adaptive branching, environment forking, adaptive group sizing, async). Briefs are in `agent_briefs/`.

- Tree-GRPO: confirmed (2509.21240, ICLR 2026). Random fixed-schedule forking at agent steps; stateless tool; no KV reuse; synchronous.
- TreePO: confirmed (2508.17445). Fixed 512-token segments, single-turn math. Named ARPO as doing entropy-guided branching after tool calls.
- TreeRL: confirmed (2506.11902, ACL 2025). Token-level forks chosen by surprisal, single-turn math.

**4. Prior-work reading, round 2.**
Added a fifth agent for ARPO after TreePO's related work flagged it as closest.

- Survey: found PATR, BPO, EPIG-Tree, ATPO and others. Conclusion: adaptive online branching and stateful sandbox forking both have prior work; tree rollouts under async training do not.
- ARPO: confirmed (2507.19849, ICLR 2026 per README). Paper describes online entropy-threshold branching after tool calls; released code is a near-coin-flip with a stale entropy baseline. Stateless tools, synchronous.

**5. Synthesis.**
Written up in `prior_work/README.md`. Outcome: the idea as first stated is not novel; the async-system angle is the open one.

**6. Saved everything here.**
Reports extracted verbatim from the agent transcripts; papers, repos and transcripts copied from the session scratchpad.

### Agent run stats

| Agent | Tool calls | Duration | Report |
|---|---|---|---|
| Tree-GRPO | 17 | 137 s | `prior_work/reports/tree-grpo.md` |
| TreePO | 16 | 148 s | `prior_work/reports/treepo.md` |
| TreeRL | 24 | 156 s | `prior_work/reports/treerl.md` |
| Survey | 51 | 289 s | `prior_work/reports/survey.md` |
| ARPO | 16 | 146 s | `prior_work/reports/arpo.md` |

### Caveats carried forward

- Nothing in any prior-work repo was executed.
- The agents' claims were not independently spot-checked.
- BPO and Crab numbers came through an automated summary; verify before citing.

**7. Second read of PATR.**
Fetched the arXiv HTML (via automated summary) to get the problem statement, scorer variants, prune rule and ablations. Saved as `prior_work/reports/patr.md`. Also re-read the abstracts and introductions of Tree-GRPO and ARPO from `prior_work/papers/` for their motivation. PATR reports ARPO as its strongest baseline on SWE-Bench Verified (25.4 vs 27.2).

**8. First-hand reads of PATR and ARPO.**
Read the full PATR text (pasted by the user) and the ARPO method section (§2–3.3) from `prior_work/papers/arpo.txt`. Both match the agent reports. New PATR details and a judgement are appended to `prior_work/reports/patr.md`. Note: the text the user pasted as "arpo paper" was a different paper, "BPO: Staying Close to the Behavior LLM Creates Better Online LLM Alignment" (arXiv 2406.12168, online DPO with the behaviour policy as reference model). It is unrelated to ARPO and also not the "Branching Policy Optimization" BPO (2607.14171) from the survey.

## 2026-10-03, second session (cloud, claude/task-a5uw74)

Network policy in this session blocked arxiv.org and its mirrors (alphaxiv, huggingface, ar5iv, semanticscholar, pith) for both shell and WebFetch. GitHub clones worked. The slime fork at `/Users/gangmuk2/projects/rlopt` was not reachable.

**9. Read upstream slime's fully-async path.**
Cloned https://github.com/THUDM/slime at `8c17b67` (2026-10-02) and read `fully_async_rollout.py`, `train.py`, the weight-update path, `sglang_rollout.py`, `slime/agent/trajectory.py`, staleness tracking and reward post-processing. Written up with a fork sketch in `notes/slime_async_path.md`. Main findings:

- Upstream has no `train_async.py`. Async is the normal `train.py` plus a background rollout worker that keeps groups in flight across steps.
- `--flush-cache-interval` other than 1 gives PipelineRL-style in-place weight sync: unfinished sequences keep decoding under new weights on old-weight KV. The default (1) aborts and flushes the cache at every sync.
- `Sample.weight_versions` records one version per generate call, and staleness is `current − min(versions)`. A fork would inherit its prefix's age.
- `TrajectoryManager` already trains shared prefix turns once across sibling leaves, but with the first leaf's advantage.
- The silent batch-wide normalisation fallback for uneven group sizes is still there upstream, in `slime/data/batch_builder.py:202`. Custom reward-post-process and convert hooks can replace it.
- In the coding-agent example, the agent harness runs inside an E2B sandbox, so forking a turn means forking a live process.

**10. Prior-work reading, round 3.**
Two agents, briefs in `agent_briefs/bpo_epig.txt` and `agent_briefs/async_gap.txt`. Both were limited to web-search snippets by the network policy.

- BPO and EPIG-Tree estimators (`prior_work/reports/bpo_epig.md`): BPO's unbiasedness is per state given s_t, and a snippet says the entropy-chosen tree over-weights high-entropy states without correction. Compute is matched by number of returns. EPIG-Tree uses an edge advantage with a segment mask, and finds on math that masks matter more than placement.
- Async gap check (`prior_work/reports/async_gap.md`): no paper found doing tree rollouts under barrier-free async training. Closest threat is RTPO (2608.18682), which forks siblings under newly synced weights from older prefixes with a sync barrier per fork. Building blocks: PrefixRL, PNPO, Missing Old Logits, VCPO. Counter-position: SAO.
- Confirmed through search that PipelineRL (2509.19128) deliberately keeps old-weight KV across in-flight weight updates.

**11. Synthesis updated.**
`prior_work/README.md` gained the round-3 table rows, a revision of axis (c), and estimator notes. `README.md` status, main finding, open questions and next steps were rewritten. The async axis is narrower: what remains is fork scheduling coupled to weight sync without a barrier, and mixed-version sibling baselines.

### Agent run stats (round 3)

| Agent | Tool calls | Duration | Report |
|---|---|---|---|
| BPO + EPIG-Tree | 35 | 185 s | `prior_work/reports/bpo_epig.md` |
| Async gap check | 51 | 695 s | `prior_work/reports/async_gap.md` |

### Caveats carried forward

- Nothing in round 3 is verbatim from a paper. Re-read RTPO, SAO, BPO and EPIG-Tree from a machine with arXiv access before citing.
- The slime reading is upstream, not the fork.
- torchtitan PR #4761 and OPTS-TTPO (2609.40035) were seen in search results only.

**12. Compute analysis of branching techniques (scope narrowed: sync only, async set aside by request).**
Re-cloned Tree-GRPO, TreePO, TreeRL, ARPO and ATPO at the same commits as round 1, and checked the compute-relevant code first-hand:
- Prefix caching: TreePO and ARPO hard-code it on; TreeRL leaves it off; Tree-GRPO leaves it at the vLLM default.
- Scheduling barriers: per segment in TreePO, generate-then-all-tools per round in ARPO, a serial expansion phase in Tree-GRPO and TreeRL, a per-tree turn loop in ATPO.
- ATPO discards pruned candidates after paying for decode, user-simulator and critic calls.
- None of them dedups shared prefixes in training.

Wrote `analysis/compute_model.py`, a synthetic token and wall-clock model of six fork patterns under an agentic and a SWE-like workload. Findings, in `notes/compute_analysis.md`:
- Rollout savings per leaf are roughly 20–37%, or 3–11% for ARPO's early forks.
- Training compute per leaf does not fall without dedup, because forked leaves are length-biased.
- Post-hoc forking adds 20–55% rollout wall-clock.
- Round barriers cost 3.7–6.7× wall-clock even for flat GRPO, which dwarfs the branching savings.
- No paper measures compute in a way that allows comparison across papers.
