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
