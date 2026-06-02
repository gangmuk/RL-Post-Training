# Previous Works: RL Post-Training Workload Scheduling and Systems

> Preliminary literature review for a research project on intelligent, application-aware
> scheduling of RL post-training workloads. This document (1) builds a general model of the
> RL post-training workload — its logical components, their mapping onto system components,
> the end-to-end workflow, the per-step bottlenecks, and how each step stresses the
> different hardware resources — and then (2) reviews each prior system, stating precisely
> *what problem it solves*, *at which layer of the stack it is implemented*, and *its approach*.

---

## 1. Background: the RL post-training workload

### 1.1 Why RL post-training, and what is distinctive about it as a *systems* workload

RL post-training (RLHF for preference alignment, and increasingly **RLVR** — RL with verifiable
rewards — for reasoning and agentic models) differs from pretraining/SFT in one decisive way:
**it is not a single dataflow over one model. It is a heterogeneous, multi-model, multi-phase
loop in which a generation (inference) phase and a training phase alternate (or overlap), and
the generation phase — not the gradient update — dominates wall-clock.** Across the systems
surveyed here, the rollout/generation phase accounts for **~70–91% of step time**. This single
fact, plus the **high variance** of generation length (a long-tail where the longest responses
are 25–32× the median), is the root cause of nearly every problem the prior works attack.

> **Assumed setting for this project.** GRPO (group-relative advantage) — **no PPO, no
> critic/value model, no reference model, no KL penalty** — with reward from a **verifier or
> heuristic scorer** (RLVR). The background below is framed around this setting; per-system
> entries still report each paper's *actual* algorithm faithfully (e.g. HybridFlow uses PPO),
> since the systems lessons transfer.

### 1.2 Logical components

Under the assumed GRPO + verifier setting there is **one trainable model** plus a non-learned
reward function (and, for agentic tasks, an environment):

| Component | Role | In this setting |
|---|---|---|
| **Actor / policy** | The only trainable model; generates rollouts *and* is the thing updated | ✓ |
| **Verifier / heuristic scorer** | Produces the scalar reward — math/answer check, unit tests, tool-success, or a rule-based score; *not* a learned reward model | ✓ |
| **Environment** (agentic) | Multi-turn state, tool execution, sandboxes, web/API calls | optional |
| **Critic / value model** | Per-token value baseline for GAE | ✗ — GRPO drops it |
| **Reference model + KL** | Frozen SFT anchor for a KL penalty | ✗ — dropped (no KL) |

**Why this matters for scheduling.** (1) Only *one* model occupies GPUs for training, so the
multi-model **placement** problem that PPO-style RLHF faces (co-placing actor + critic + reward +
reference, the design space HybridFlow searches) largely disappears — the resource story is
dominated by the single policy's train↔generate duality and by the rollout long-tail, not by
hosting 4–5 models. (2) GRPO samples a **group of G responses per prompt** and normalizes each
reward against the group mean/std; that **group structure is itself an exploitable scheduling
signal** (length prediction, speculative drafting, load balancing) used by GroupMind, RhymeRL,
slime, and others below. (3) The reward being a **verifier/heuristic** (often CPU-bound: run
tests, check answers, score a tool result) — not a GPU-resident learned RM — is what makes
reward a cheap, stateless, burstable workload that systems like RollArt push to serverless/CPU.

### 1.3 End-to-end workflow (one iteration)

```
            ┌─────────────────────────────────────────────────────────────┐
            │  1. ROLLOUT / GENERATION  (the bottleneck, ~70–91% of step)   │
  prompts → │  actor autoregressively decodes completions (vLLM/SGLang).    │
            │  Agentic: multi-turn loop — generate → tool/env step → repeat │
            └───────────────────────────────┬─────────────────────────────┘
                                             │ trajectories
            ┌───────────────────────────────▼─────────────────────────────┐
            │  2. REWARD + ADVANTAGE                                         │
            │  verifier / heuristic scorer assigns the scalar reward;       │
            │  advantage = group-normalized; no critic, no ref, no KL.      │
            └───────────────────────────────┬─────────────────────────────┘
                                             │ (prompt, response, advantage)
            ┌───────────────────────────────▼─────────────────────────────┐
            │  3. POLICY UPDATE / TRAINING                                   │
            │  forward+backward on the policy only; optimizer step;          │
            │  then NEW WEIGHTS must be synced back to the rollout engine    │
            └───────────────────────────────────────────────────────────────┘
                          weight sync ↺ back to step 1
```

The **weight-sync edge** (train → generate) is a first-class systems concern: the actor's
*training* tensor layout (e.g. 3D parallel: pipeline × tensor × data) differs from its
*inference* layout, so weights must be **resharded and transferred** every iteration.

### 1.4 Mapping logical components → system components

| Logical | System component | Typical implementation |
|---|---|---|
| Actor training | **Training backend** | Megatron-LM, FSDP, DeepSpeed (3D/ZeRO parallelism) |
| Actor generation | **Inference / rollout engine** | vLLM, SGLang, TensorRT-LLM (PagedAttention/RadixAttention, continuous batching) |
| Reward (verifier/heuristic) | **Reward service** | CPU verifier / rule-based scorer (run tests, check answers, score tool output) — no learned RM |
| Environment/tools | **Sandbox / env workers** | CPU containers (Docker/k8s), code exec, web/API calls |
| Coordination | **Controller** | Ray single-controller, or a custom dataflow driver |
| Trajectory staging | **Buffer / queue** | replay buffer / data buffer (decouples producer ↔ consumer) |
| Train→gen weights | **Weight-transfer path** | NCCL / RDMA / TCP; resharding engine (e.g. 3D-HybridEngine) |

A central design axis is **how the actor's two roles (train + generate) share GPUs**:

- **Colocated / time-shared** (HybridEngine style): training and generation reuse the *same*
  GPUs, switching layouts each iteration. Maximizes GPU ownership but forces resharding and is
  inherently **synchronous**.
- **Disaggregated**: generation GPUs and training GPUs are *distinct* pools. Enables
  **asynchronous** overlap and per-pool hardware specialization, at the cost of cross-pool
  weight transfer and staleness.

### 1.5 Where the problems live, step by step

| Step | Dominant problem | Why |
|---|---|---|
| **Rollout** | **Long-tail stragglers** | Length variance (25–32× median); synchronous rollout waits for the *slowest* trajectory → GPUs that finished short responses idle. ~70–91% of step time. |
| **Rollout (agentic)** | **Heterogeneity + flakiness** | Mixed CPU tool work, web/API latency, sandbox OOM/timeout; fixed-batch (engine-based) rollout blocks the whole DP group on one slow trajectory. |
| **Reward** | **Resource mismatch** | Stateless, bursty, often CPU-bound (verifiers/tests) — wasteful to pin to GPUs (utilization can be ~6%). |
| **Update** | **Pipeline bubbles** | Micro-batch pipeline stalls; in sync RL the training cluster idles entirely during rollout. |
| **Weight sync** | **Resharding + transfer cost** | Train vs inference layouts differ; peak memory + NVLink/PCIe/NIC traffic on every iteration. |
| **Whole loop (async)** | **Staleness / off-policy** | Decoupling rollout from training makes data off-policy → heavy-tailed importance weights, gradient-variance blowup. |

### 1.6 Resource taxonomy — how each step stresses the hardware

This is the lens for a scheduling/systems contribution: each phase has a *different* dominant
resource, which is exactly why one-size-fits-all placement underutilizes the cluster.

| Resource | Stressed most by | Notes / implications |
|---|---|---|
| **GPU compute (FLOPs/SM)** | Prefill (rollout), training fwd/bwd | Prefill is compute-bound; training step is compute- and memory-bound. Idle SMs during the decode long-tail are the harvestable resource (TLT). |
| **GPU memory (capacity)** | Policy weights + optimizer states + KV cache (single model under GRPO; 4–5 models under PPO-based systems) | Colocation contends for capacity; resharding has a peak-memory spike; KV cache size bounds rollout batch/concurrency. |
| **GPU memory bandwidth** | **Decode** (autoregressive generation) | Decode is memory-**bandwidth**-bound (one token = one forward pass, weight-streaming dominated). This is why speculative decoding (verify many tokens per pass) helps the long-tail. |
| **CPU compute** | Tool execution, verifiers, env simulation, tokenization, data filtering | Agentic envs are CPU-heavy and long-running; coupling them to GPU steps causes GPU idling. Stateless reward → serverless/CPU pools. |
| **CPU memory** | Replay/data buffers, trajectory staging, KV offload | The buffer that decouples producer↔consumer lives here; spill/offload targets. |
| **Network — NVLink** | Intra-node tensor/pipeline parallel collectives; resharding | Highest BW, intra-node; all-gather/all-reduce for parallel training and layout transitions. |
| **Network — PCIe** | Host↔device transfer, some weight movement, KV offload | Lower BW than NVLink; a bottleneck for offload-based designs. |
| **Network — NIC / InfiniBand** | Inter-node collectives, cross-pool/cross-cluster weight sync | Disaggregated/spot designs make this load-bearing (RDMA vs TCP); cross-cluster weight transfer (e.g. Mooncake over Ethernet) is a measured cost. |

### 1.7 The recurring design axes (a map of the solution space)

Every system below is a point in this space:

1. **Synchronous vs Asynchronous** — does the next training step wait for the whole batch's
   rollout? Async overlaps the two but introduces **staleness**, handled by (a) a max-staleness
   window + admission control, (b) importance-sampling / decoupled-PPO corrections, or
   (c) partial-rollout interruption.
2. **Colocated vs Disaggregated** placement of generation vs training GPUs.
3. **Engine-based vs Server-based rollout** — does the framework drive the inference engine as
   an in-process **library with fixed batches** (all requests finish before new ones submit), or
   does the engine run as a **persistent server** (OpenAI-style API) that the framework/agent
   calls as a client, enabling **continuous batching** and dynamic request submission? The latter
   is what lets short requests retire while a long one continues, and lets existing agent
   frameworks plug in unmodified. Note: a "rollout engine" is a *spectrum* — from pure token
   generation (the server's only job) up to a full agent runtime; in server-based designs the
   **tool/environment logic lives in client code, and the inference server only generates tokens.**
4. **Homogeneous vs Heterogeneous hardware** — map prefill-heavy work to compute-optimized GPUs,
   decode-heavy work to bandwidth-optimized GPUs, CPU env work to CPU/k8s, stateless reward to
   serverless.
5. **The long-tail lever** — async overlap, speculative decoding (model-free via history, or a
   continuously-trained drafter), or rollout scheduling/packing (tail batching, group rebalancing).

---

## 2. Previous Works Summary

The whole survey at a glance; each row is expanded in §3. Setting assumed throughout is GRPO +
verifier/heuristic reward (see §1.2) — the "Staleness handling" column therefore reads "on-policy"
for synchronous systems with no caveats about a critic/reference. One comprehensive table — the
index columns (venue · paper · one-line problem) come first, then the design-space columns.
*(Intentionally wide; venue/link mappings for OSDI'26 entries are inferred — see Appendix.)*

| System | Venue | Paper | Brief problem | Layer / stack | Sync? | Placement | Primary lever | Long-tail strategy | Staleness handling | Resource(s) targeted | Workload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **HybridFlow/veRL** | EuroSys 2025 | [arXiv 2409.19256](https://arxiv.org/abs/2409.19256) | Program & place the multi-model RLHF dataflow; reshard the actor between train/generate layouts (synchronous baseline). | framework (Ray + Megatron/FSDP/DeepSpeed + vLLM) | **sync** | colocate/split/disaggregate (auto-mapped) | programming model + placement search | — (the victim) | on-policy | GPU mem + NVLink/IB (resharding) | RLHF/reasoning |
| **RhymeRL** | ASPLOS 2026 | [arXiv 2508.18588](https://arxiv.org/abs/2508.18588) | Flatten + accelerate the synchronous rollout long-tail by exploiting prior-epoch rollout similarity. | on veRL (disaggregated) + custom HistoSpec engine | **sync** | per-length-group worker allocation | spec-decode (history) + predictive sched | predict tail from prior epoch → rebalance + draft | on-policy | GPU mem-BW (decode) + CPU/DRAM (suffix tree) | reasoning |
| **TLT/FastRL** | ASPLOS 2026 | [arXiv 2511.16665](https://arxiv.org/abs/2511.16665) | Harvest tail-idle GPUs to keep a drafter fresh; apply lossless speculative decoding only on the small-batch tail. | on veRL + SGLang + EAGLE | **sync** | colocated, idle-GPU harvesting | spec-decode (online drafter) + co-sched | harvest tail-idle GPUs to train drafter; SD only <32 reqs | on-policy (SD lossless) | GPU compute (idle) + mem-BW + mem (CUDAGraph) | reasoning |
| **GroupMind/Seer** | OSDI 2026 | [arXiv 2511.14617](https://arxiv.org/abs/2511.14617) | Predict rollout lengths from within-group siblings to load-balance + draft, using only within-step signal. | colocated (in-house vLLM + Megatron + Mooncake) | **sync** | chunked migratable rollout | group-aware online length prediction | probe-per-group → longest-first + grouped spec-decode | on-policy | GPU compute + mem-BW + KVCache | reasoning |
| **RollPacker** | NSDI 2026 | [arXiv 2509.21009](https://arxiv.org/abs/2509.21009) | Remove the synchronous long-tail bubble by repacking rollouts into short/long rounds, accuracy-preserving. | on ROLL (vLLM + Megatron + Ray) | **sync** | elastic TP per round | **tail batching** (short/long rounds) | oversample + abort tail → long rounds; stream trainer | on-policy | GPU compute + GPU mem (elastic TP) | reasoning |
| **AReaL** | arXiv 2025 (Ant/THU) | [arXiv 2505.24298](https://arxiv.org/abs/2505.24298) | Fully decouple rollout/train with bounded-staleness admission control + an off-policy-correct objective. | standalone (SGLang + Megatron/FSDP, SLURM) | **async** | static 3:1 inference:training | decoupling + admission control | overlap; interrupt mid-decode | hard max-staleness η + decoupled PPO | GPU compute (both pools) + GPU mem (KV) | reasoning |
| **StreamRL** | arXiv 2025 (PKU/StepFun) | [arXiv 2504.15930](https://arxiv.org/abs/2504.15930) | Kill the *pipeline* and *skewness* bubbles in disaggregated RL via stream generation + output-length-ranked dispatch; cross-datacenter. | standalone (in-house C++ engine + Megatron-style + RL-RPC) | **sync or 1-step async** | disaggregated gen/train pools (cross-DC, heterogeneous) | stream pipelining + skewness-aware dispatch | predict length → isolate long-tail on dedicated instances | one-step stale (async variant) | GPU compute (both bubbles) + **WAN/RDMA net** | reasoning |
| **RLBoost/PolyRL** | NSDI 2026 | [arXiv 2510.19225](https://arxiv.org/abs/2510.19225) | Offload stateless rollout onto cheap preemptible spot GPUs while keeping the trainer strictly on-policy. | veRL fork + Rust rollout-mgr + SGLang | **sync** | reserved train + elastic spot rollout | **cost** (spot harvesting) | token-level migration; JSQ load-balance | on-policy (latest-weight routing) | GPU compute (spot $) + **NIC/frontend net** | reasoning |
| **slime** | system, LMSYS '25 (THUDM) | [repo](https://github.com/THUDM/slime) · [APRIL 2509.18521](https://arxiv.org/abs/2509.18521) | Server-based rollout + a quality-gated buffer for flaky, long-tailed agentic rollout (the GLM line). | standalone (Megatron + SGLang servers + Ray) | **both/async** | colocate or disaggregate (±PD) | **server-based** rollout + quality buffer | continuous batching + APRIL partial-recycle | version-logged double-sided IS + clipping | GPU compute (cont. batch) + CPU (tools) | **agentic** |
| **RLinf/RLux** | OSDI 2026 | [arXiv 2509.15965](https://arxiv.org/abs/2509.15965) | Auto-choose the execution mode (colocate/disaggregate/hybrid) + placement for the RL dataflow. | standalone (Ray + Megatron/FSDP + SGLang/vLLM/HF) | **flex (both)** | auto colocate/disagg/**hybrid** (M2Flow) | execution-mode search | elastic pipelining overlap | mode-dependent | GPU compute + mem (onload/offload) + interconnect | reasoning + embodied |
| **DynaRL** | OSDI 2026 | [USENIX page](https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing) | Runtime dynamic resource/placement scheduling for RL training (paper not yet public). | unknown (OSDI'26, not public) | unknown | dynamic/runtime reallocation | dynamic scheduling | unknown | unknown | unknown | unknown |
| **ROSE** | arXiv 2026 | [arXiv 2605.06534](https://arxiv.org/abs/2605.06534) | Harvest idle online-serving GPUs for agentic rollout without breaking serving SLOs. | on ROLL (vLLM serving + Megatron + Mooncake) | elastic (±1-step off-policy) | reserved rollout + **borrowed serving GPUs** | **cost** (harvest *serving* GPUs) | turn-wise offload + long-tail overlap | one-step off-policy capable | serving GPU compute + HBM + **cross-DC net** | **agentic** |
| **RollArt/MARS** | OSDI 2026 | [arXiv 2512.22560](https://arxiv.org/abs/2512.22560) | Disaggregate heterogeneous agentic sub-workloads onto best-fit hardware + trajectory-level async. | ROLL lib (Ray + Megatron + vLLM/SGLang + k8s + serverless) | **async** | **4-pool heterogeneous disaggregation** | **hardware-affinity placement** | trajectory-level async pipelining | bounded staleness α (=1), abort over-stale | per-phase: H800 compute / H20 BW / CPU / serverless + **cross-cluster net (Mooncake)** | **agentic (multi-task)** |
| **Weave/ROLLMUX** | OSDI 2026 | [arXiv 2512.11306](https://arxiv.org/abs/2512.11306) | Multiplex complementary phases of *different* RL jobs to fill disaggregation idle bubbles. | scheduler on ROLL/Ray (+Redis) | **sync (per job)** | **multi-job packing** on disagg H20+H800 | **phase-level multiplexing across jobs** | straggler consolidation @~80% | on-policy (per job) | GPU compute (fill bubbles) + host DRAM + Ethernet | reasoning (multi-job) |
| **ThunderAgent** | arXiv 2026 | [arXiv 2602.13692](https://arxiv.org/abs/2602.13692) | Program-aware cross-layer scheduling (KV reuse + tool prep) for multi-turn agentic rollout/serving. | inference/rollout layer over vLLM/SGLang (under any trainer) | (rollout layer) | program-KV placement across nodes | **app-semantic ("LLM Program")** scheduling | phase-aware pause/restore + prefix reuse | n/a (serving/rollout layer) | GPU mem (KV) + CPU/IO (tool prep) + disk (GC) | **agentic** |
| **PivotRL** | arXiv 2026 (NVIDIA) | [arXiv 2603.21383](https://arxiv.org/abs/2603.21383) | Cut rollout *volume* by training single-turn from pivotal cached states (RL algorithm, not a systems contribution). | *algorithm* on NeMo-RL + NeMo-Gym | (algorithm) | n/a | **rollout-volume reduction** (algorithm) | single-turn-from-cached-state → no stragglers | n/a | GPU compute (rollout volume) | **agentic** |

**Reading the table as a design space:** the columns *Sync?*, *Placement*, and *Primary lever*
are the three orthogonal axes. veRL is the origin (sync, flexible-colocate, no lever). Moving
along *lever* with sync held fixed gives the §3.1 papers (predict + accelerate the tail). Flipping
*Sync?* gives §3.2 (decouple + manage staleness). Specializing *Placement* gives §3.4 (volatile/spot)
and §3.5 (heterogeneous hardware), with §3.3 making the mode itself a search decision. ThunderAgent
(§3.6) is orthogonal — it sits *below* the trainer and schedules the rollout substrate itself using
application semantics.

---

## 3. Prior systems

Organized by the **lever each system pulls** against the rollout-bottleneck / long-tail
problem established in §1. Each entry follows a fixed shape: a one-line **Problem (category)**;
a self-contained **Problem** paragraph (the precise sub-problem and the resource bottleneck it
arises from); an **Approach** (the mechanism, the key insight, and where it spends/saves each
resource); concise **Results** with the comparison baselines named; and finally **Stack** —
*where the contribution lives* in the training/inference/orchestration stack, which is the least
novel part and so comes last.

### 3.0 Foundational framework / baseline

### HybridFlow / veRL (EuroSys 2025, arXiv:2409.19256)

**Problem (category):** Programming model + actor train/generate resharding + model-to-GPU placement search for the multi-model RLHF dataflow (synchronous, on-policy).

**Problem:** RLHF is not a single model but a dataflow over four-to-five interacting models — actor, critic, reward, and reference (plus a cost model for Safe-RLHF) — that alternate between a generation phase and a training phase each iteration, and how to *program and place* this dataflow had no good answer. A pure single-controller that dispatches every operator from one driver is easy to express but its coordination overhead is prohibitive at billion-parameter scale; a pure multi-controller SPMD program (Megatron/FSDP) dispatches operators cheaply but bakes the cross-model dataflow into deeply nested, rigid code that is hard to reconfigure for a new algorithm. Compounding this, the actor alone must serve two incompatible roles on the *same* GPUs every step — a memory-heavy training layout (large TP/PP plus optimizer state) and a latency-optimal generation layout (smaller TP) — and naively switching between them re-materializes a second full copy of the weights and floods the interconnect with all-gather traffic, so the transition is bounded by GPU memory and NVLink/IB bandwidth. Finally the models must be mapped onto a fixed GPU budget, and the placement choice — colocate all models time-sharing one device set, fully disaggregate them onto disjoint sets, or split them into groups — trades GPU-memory pressure against stage parallelism and is a combinatorial search (15 partitions for four models) under a no-OOM constraint.

**Approach:** HybridFlow's answer is a *hybrid* control model: a single controller coordinates the inter-model dataflow (cheap, since only coarse control messages cross model boundaries) while each model runs as multi-controller SPMD internally (fast operator dispatch), reusing Megatron/FSDP/DeepSpeed for training and vLLM for generation. A `ResourcePool` abstraction makes placement a first-class knob — models sharing a pool are colocated (offload/onload, sequential execution, trading serialized GPU compute for lower memory pressure), models in different pools run in parallel — and an auto-mapping search (Algorithm 1) costs each placement×parallelism plan with a latency simulator (summing colocated stages, taking the max over parallel device sets) under a min-GPU no-OOM bound and picks the minimum-end-to-end plan. The actor resharding is solved by the **3D-HybridEngine**, which confines the train→generate all-gather to small "micro-DP" groups whose ranks are interval-selected so training and generation shards already overlap on each device — eliminating the redundant weight copy (GPU memory) and cutting transition traffic from (tp−1)/tp·M to (tp−t_g·p_g)/(t_g·p_g·tp)·M of NVLink/IB volume. The lasting contribution is the template itself: the colocation-oriented synchronous design and the colocate/split/disaggregate placement space that essentially every later system schedules against, and whose limitations — synchronous lockstep (generation stalls training), no off-policy staleness, colocation serialization — define the openings the async/disaggregated systems below exploit.

**Results:** 1.53×–20.57× end-to-end throughput over **DeepSpeed-Chat** (3.67× avg), **OpenRLHF** (3.25× avg), and **NeMo-Aligner** (12.52× avg) on Llama 7B–70B with PPO/ReMax/Safe-RLHF; the 3D-HybridEngine cuts the per-iteration train↔generate transition by 55.2% on average (89.1% at 70B). Up to 128 A100-80GB (600 GB/s NVLink, 200 Gbps inter-node).

**Stack (where the contribution lives):** An orchestration layer (~12k LoC Python) that owns the placement search and 3D-HybridEngine resharding: a Ray single-controller drives the dataflow above otherwise-unmodified SPMD backends — Megatron-LM / FSDP / DeepSpeed for training, vLLM for generation (its centralized KV-cache manager swapped for a distributed one so generation runs SPMD). Open-sourced as veRL, which has since grown SGLang, async, and multi-turn support beyond the paper. Repo: github.com/volcengine/verl ; paper: https://arxiv.org/abs/2409.19256.

---

### 3.1 Synchronous long-tail acceleration (rollout scheduling + speculative decoding)

*These keep RL strictly synchronous/on-policy (no accuracy change) and attack the long-tail
directly — by predicting it and rebalancing GPUs, and/or by accelerating bandwidth-bound decode
with speculative decoding.*

### RhymeRL — "History Rhymes: Accelerating LLM Reinforcement Learning with RhymeRL" (ASPLOS 2026; arXiv:2508.18588, SJTU IPADS + ByteDance)

**Problem (category):** Rollout GPU load-balancing across length-skewed groups + draft-model-free speculative decoding, exploiting cross-epoch rollout similarity (synchronous GRPO/DAPO).

**Problem:** In synchronous, on-policy reasoning RL the rollout phase consumes 84–91% of each step and leaves the trainer fully idle, and within a single rollout batch the per-prompt response lengths are heavily skewed, so every data-parallel rollout worker stalls waiting on the one longest response — veRL sits above 46% GPU idle as a result. This long tail has two distinct costs that must both be attacked without changing the algorithm: a *scheduling* cost (the few long sequences create GPU bubbles because work is not balanced against length, and length is unknown when the batch is launched) and a *per-token* cost (the long sequences are decoded autoregressively, which is memory-bandwidth-bound on HBM and dominates their wall-clock). Prior remedies sacrifice the RL paradigm — asynchrony introduces staleness (AReaL), truncation drops tokens (Kimi-K2) — so the open problem is to flatten the tail and accelerate long-sequence decode while remaining exactly synchronous and on-policy.

**Approach:** RhymeRL's insight is that under clipped GRPO/DAPO updates the rollouts of adjacent epochs "rhyme" — for the same prompt, ~93% of tokens recur and a prompt's length-rank within the batch is stable (only 2–4% churn) — so the *previous* epoch is a free, accurate predictor of *this* epoch's length distribution and token content. It turns this into two mechanisms. **HistoPipe** (the scheduler) sorts prompts by historical mean length into ranking groups and solves a small integer program (by binary search over a profiled cost table) that gives the few long groups extra data-parallel workers and the many short groups fewer, reshaping per-group completion times from exponential to linear and eliminating the bubble; outliers are handled by recompute-migration (cheaper than moving KV cache over the network). **HistoSpec** (the decoder) does draft-model-free speculative decoding: each prompt's prior-epoch responses are indexed in a reward-aware suffix tree (highest-reward suffix chosen as the draft to maximize acceptance) with an AIMD draft-length window, raising the arithmetic intensity of the bandwidth-bound decode at *zero* GPU cost because drafts come from CPU cores and host DRAM rather than a draft model. The reusable lesson is that iterative RL makes the otherwise-unknown rollout-length distribution *known from history*, so a synchronous trainer can both pre-balance GPUs and obtain draft tokens for free — and the right home for the draft index is CPU/host-DRAM, off the critical GPU path.

**Results:** up to **2.6× end-to-end vs veRL v0.4.1** (≈1.9× @8K, ≈2.3× @16K max length; e.g. Math-32B/128GPU 2.61×); also beats **AReaL v0.3.0** at off-policyness 1 and 8 while staying on-policy. Ablation: +HistoPipe 1.43×, +two-tier 1.56×, +HistoSpec 2.34×; HistoSpec holds 1.80–1.86× rollout speedup at extreme batch sizes where draft-model baselines (**Eagle3, Medusa**) degrade past batch 64. 128 GPUs; no accuracy loss. Baselines: veRL, AReaL, Eagle3/Medusa.

**Stack (where the contribution lives):** Built on **veRL v0.4.1** in its rollout/train-disaggregated mode (not the colocated 3D-HybridEngine). The two contributions are new components around veRL's existing Megatron-style trainer and TP rollout workers: HistoSpec is a custom speculative-decoding inference engine replacing the stock one, and HistoPipe is a scheduling layer; new "history workers" run on otherwise-idle CPU cores + multi-TB host DRAM to build the per-prompt suffix-tree indices asynchronously. Links: https://arxiv.org/abs/2508.18588 (code to be released).

### TLT / FastRL — "Taming the Long-Tail: Efficient Reasoning RL Training with Adaptive Drafter" (MIT Han Lab + NVIDIA, ASPLOS '26)

**Problem (category):** Harvesting long-tail-idle GPU compute to continuously co-train a drafter + adaptive lossless speculative decoding (synchronous reasoning RL).

**Problem:** In synchronous reasoning RL the rollout phase is ~85% of each step, and because a few very long responses dominate it, the GPUs of a 128-request batch idle progressively as most requests finish and a shrinking tail keeps decoding — so the bottleneck is simultaneously wasted GPU *compute* (idle SMs during the tail) and bandwidth-bound decode of the surviving long sequences. Speculative decoding (SD) is the natural cure for the latter, but applying it in RL is hard for three reasons that prior single-shot SD systems do not face: the target policy is updated every step so a fixed drafter goes stale and acceptance collapses; training a drafter costs compute that the RL budget cannot spare; and the rollout batch size shrinks continuously through the tail, so the SD configuration that pays off changes within a single step (SD helps only once the remaining batch is small enough to be memory-bandwidth-bound — empirically <32 requests — and hurts on large compute-bound batches).

**Approach:** TLT's insight is to treat the long tail as a *resource opportunity rather than a straggler to remove*: the very GPUs that the tail leaves idle are exactly what is needed to keep a drafter fresh. A **Worker Coordinator** runs a BUSY→IDLE→TRAINING state machine that promotes tail-idle workers to continuously train a tiny EAGLE-style one-layer drafter as preemptible low-priority "spot tasks" that halt the instant rollout needs the GPUs, so the drafter tracks the evolving policy at no extra cost (and is a free deployable byproduct). An **Adaptive Rollout Engine** then enables SD only below the <32-request threshold and selects its configuration (Draft_Depth, topK, Tokens_to_Verify) with an ε-greedy multi-armed bandit bucketed by batch size — converting the wasted HBM bandwidth of the small-batch tail into useful compute — backed by a bucketed pre-captured CUDAGraph pool that keeps the many SD configs cheap in GPU memory (10.69 GB vs a naive 30.39 GB on Llama-3-8B). SD is mathematically lossless, so RL behavior is unchanged. This is an intra-node compute/bandwidth/memory co-optimization, with no network/disaggregation component.

**Results:** >1.7× end-to-end vs **veRL-based SOTA**: Qwen2.5-7B 1.76–2.1×, Qwen2.5-32B 1.83–2.12× (4–8 nodes), Llama-70B ~2.1×. Rollout microbench (batch 1): 2.61× H100 / 2.79× A100 / 3.22× RTX 3090. SD per-batch speedup falls 3.62× (batch 1) → 1.70× (batch 32), which is what motivates the <32 threshold. Lossless (accuracy preserved). Baseline: veRL (+ SD baselines for the drafter).

**Stack (where the contribution lives):** A framework over three reused projects — **veRL** (training/controller loop), **SGLang** (rollout engine, extended with the adaptive-SD execution path + CUDAGraph capture), and **EAGLE** (drafter architecture + a separate `eagle-train` warm-up). The new contributions are the Worker Coordinator, the Adaptive Drafter trainer, and the Adaptive Rollout Engine layered on top. Repo: github.com/mit-han-lab/fastrl ; paper: https://arxiv.org/abs/2511.16665.

### GroupMind / Seer (OSDI 2026; arXiv:2511.14617 as "Seer", Moonshot AI + Tsinghua)

**Problem (category):** Group-aware output-length prediction for rollout load-balancing + grouped speculative decoding, using only within-step signal (synchronous GRPO).

**Problem:** Synchronous, on-policy RL is dominated by the rollout phase, and the difficulty is that per-request output lengths are unknown when the batch is issued yet span a few hundred to ~96k tokens, so the last ~10% of requests consume ~50% of rollout time and a length-blind scheduler cannot avoid the resulting GPU bubble. The same long tail also makes the surviving long requests expensive to decode (memory-bandwidth-bound on HBM). Unlike RhymeRL, which can mine the previous epoch, the goal here is to mitigate the tail using only information available *within the current step* — no cross-epoch history, no oracle lengths, and no relaxation of synchronization. The exploitable structure is that GRPO samples a *group* of G responses per prompt, and the responses within a group have correlated output-length profiles and recurring local token patterns.

**Approach:** GroupMind ("online context learning") turns the group into the scheduling signal. (1) *Context-aware scheduling*: it runs one cheap "probe" response per group first to estimate that group's length, converting the oracle-free unknown-length problem into an approximate longest-first ordering that drains the tail early. (2) *Divided rollout*: groups/requests are split into fine-grained chunks dynamically rebalanced across instances, and because the migrated chunk's prefix lives in a global KVCache pool, rebalancing is recompute-free — so dynamic load-balancing is cheap rather than paying a prefill penalty. (3) *Adaptive grouped speculative decoding*: a Distributed Grouped Draft Server aggregates tokens from all in-flight siblings of a group into per-group compressed suffix trees used as the draft source, with a marginal-benefit policy that prioritizes tail requests — attacking the bandwidth-bound decode while the group doubles as a free draft corpus. The reusable lesson is that *group-level statistical similarity is a first-class within-step scheduling signal*: one probe per group buys predict-then-longest-first scheduling and a draft corpus at once, where RhymeRL needs cross-epoch history to get the same.

**Results:** up to **2.04× rollout throughput** vs SOTA synchronous RL; long-tail latency **−72–94%**. Ablation: divided rollout +42%, context-aware scheduling +14% (≈89% less tail-phase time alone), grouped SD +26–48% (raises mean accepted draft length 1.70 → 2.85). Models incl. Moonlight, Kimi-K2. Baselines: **veRL**, **StreamRL-Oracle** (skewness-aware *with* oracle lengths — i.e. GroupMind matches an oracle-length scheduler without oracle lengths), and speculative-decode baselines (SuffixDecoding, dedicated draft model, MTP).

**Stack (where the contribution lives):** A standalone *colocated* RL system (veRL is the baseline). The contribution sits in a logically centralized **Context Manager** that coordinates rollout scheduling, a global **Mooncake**-backed KVCache pool (enables recompute-free chunk migration), and the grouped-SD draft server, above an in-house unified **vLLM** inference engine (note: *not* SGLang) and a **Megatron** trainer. Links: https://arxiv.org/abs/2511.14617 ; OSDI'26 https://www.usenix.org/conference/osdi26/presentation/qin

### RollPacker (NSDI 2026; arXiv:2509.21009)

**Problem (category):** Accuracy-preserving temporal repacking of long-tail rollouts into short/long rounds, co-designed with a streaming trainer (synchronous RL).

**Problem:** In synchronous RL a training step cannot begin until the rollout batch finishes, and because response lengths are heavily long-tailed the batch waits on a handful of stragglers — idling the rollout GPUs and stalling the trainer behind them. Asynchronous systems hide this bubble but pay for it with staleness and off-policy drift that can cost accuracy; the open problem RollPacker poses is to remove the *same* bubble while keeping training *strictly* synchronous and on-policy, i.e. without changing which samples the gradient sees or their distribution — a purely temporal-scheduling problem over the rollout→reward→train pipeline.

**Approach:** RollPacker's insight is that the long tail should be handled by *reorganizing when* work runs, not by relaxing synchronization. Its "tail batching" runs most prompts in balanced *short rounds* that speculatively over-sample (η = 1.25), admit the first P₀ to finish (race-to-completion), and **abort the slow tail** into a long-prompt queue; once that queue fills it runs as a dedicated *long round* (cadence ≈ four short : one long). Because this only changes the order in which a prompt's data is consumed — not the data or its distribution — accuracy and on-policy semantics are preserved. Two co-designs recover the freed resources: *elastic TP per round* raises tensor-parallelism in high-concurrency short rounds to shard weights and relieve KV-cache/preemption pressure (GPU memory), trading against interconnect cost, and lowers it in long rounds; and a *stream-based trainer* repurposes rollout GPUs for buffered gradient computation once 20–50% of responses complete, overlapping the train stage with rollout across periods to reclaim idle GPU compute. The result is a full-stack co-design rather than a drop-in scheduler, and it is the on-policy counterpoint to AReaL/slime — same bubble, opposite philosophy (RollPacker keeps strict synchronization; the async systems trade it for staleness).

**Results:** end-to-end **2.03×–2.56× vs veRL** (≈2.03× 7B / 2.22× 14B / 2.56× 32B) and **up to 2.24× vs RLHFuse**. Qwen2.5 7B/14B/32B (max len 8k/16k/32k) on up to **128 H800**. Baselines: veRL (synchronous), RLHFuse (overlap/fusion).

**Stack (where the contribution lives):** A standalone system (~6.6k LoC) on the **ROLL** framework (veRL is a baseline), with the round scheduler + elastic-TP planner + stream trainer as new components above **vLLM v0.8.4** rollout, **Ray** reward workers, and a **Megatron-LM v0.12.2** trainer whose optimizer-update boundary is integrated with rollout scheduling. Links: https://www.usenix.org/system/files/nsdi26-gao-wei.pdf ; https://arxiv.org/abs/2509.21009 (arXiv title: "Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training").

---

### 3.2 Asynchronous / decoupled training

*These decouple rollout from training onto separate (often disaggregated) pools and overlap them,
managing the resulting staleness algorithmically — AReaL goes fully async with a staleness budget,
StreamRL pipelines a disaggregated cross-datacenter setup (sync or one-step-async), and slime runs
server-based agentic rollout behind a quality-gated buffer.*

### AReaL: A Large-Scale Asynchronous RL System for Language Reasoning

**Problem (category):** Bounded-staleness admission control + off-policy-correct objective for fully-decoupled asynchronous reasoning RL.

**Problem:** Synchronous RL wastes GPUs because the slowest sequence in each batch stalls the rollout→train barrier; the obvious fix is to decouple generation from training so neither waits on the other, but unbounded decoupling makes the training data arbitrarily off-policy, which inflates importance-weight variance and breaks PPO. The problem AReaL must solve is therefore twofold and coupled: a *systems* problem of running rollout and training as a fully decoupled producer/consumer pipeline that keeps both GPU pools saturated (no sync barrier, no straggler idle), and an *algorithmic* problem of bounding and correcting the resulting staleness so convergence and final accuracy are preserved — all while a single long generation may need to span several weight versions rather than blocking its batch.

**Approach:** AReaL casts this as admission-controlled producer/consumer scheduling. A rollout controller tracks the generated-trajectory count and the current policy version and *rejects* any generation request that would exceed a hard max-staleness budget η (η=4 coding, η=8 math), which rate-matches the rollout producers to the trainer consumer; in-flight generations are made *interruptible*, so on a weight update a worker discards the KV cache computed under old weights (GPU memory), recomputes under the new weights, and resumes — eliminating the long-tail straggler by letting one sequence span versions. The residual off-policy bias is handled algorithmically by a **decoupled PPO** objective that separates the behavior policy (which generated the token) from a proximal policy used for the trust region, proven correct even for much older versions. Padding-free dynamic token batching (sequence packing) raises effective compute/bandwidth, and a deliberately *static* 75/25 inference:training device split is shown sufficient to keep both pools busy — the reusable lesson being that bounded-staleness asynchrony reduces RL to a classic admission-controlled pipeline where a version-budget rule (plus decoupled PPO) makes off-policy safe, without needing dynamic colocation. Network cost is the per-update trainer→rollout weight sync (transport unspecified in the paper).

**Results:** vs **veRL** end-to-end: 1.5B 2.27×, 7B 2.05×, 14B 2.03×, 32B 1.49× (veRL OOMs at 32k/32B); up to 2.77× vs synchronous overall; linear scaling to 512 GPUs. Ablations: dynamic batching ≈+30% throughput; interruptible generation +12–17%. Accuracy preserved: η≤8 ≈ matches the η=0 (synchronous oracle) — AIME24 42.0 vs 42.2, MATH500 89.2 vs 89.5. Baseline: veRL (synchronous).

**Stack (where the contribution lives):** A standalone asynchronous framework (veRL is the baseline it beats). The contribution is concentrated in a custom Rollout Controller (replay buffer + admission/version-tracking), an interruptible-generation hook into the inference engine, and the decoupled-PPO loss; the four decoupled component types (Interruptible Rollout Workers, Reward Service, Trainer Workers, Controller) run **SGLang** (or vLLM) for generation and **Megatron-Core / FSDP2** for training under SLURM, otherwise reused as-is. Links: https://arxiv.org/abs/2505.24298 ; https://github.com/inclusionAI/AReaL.

### StreamRL (arXiv:2504.15930, Peking University + StepFun)

**Problem (category):** Eliminating the two idle-bubble types of *disaggregated* RL — pipeline bubbles and skewness bubbles — via stream generation + output-length-ranked dispatch, including across datacenters (synchronous or one-step-async).

**Problem:** Disaggregating RL into a separate generation pool and a training pool is attractive (each can use its own parallelism and even its own hardware), but a naive disaggregated design leaves two distinct sources of idle GPU time. **Pipeline bubbles**: the generation stage sends its samples to the training stage only *after the whole batch is generated*, so the training GPUs sit idle for the entire generation phase (and vice versa) — a stage-dependency stall. **Skewness bubbles**: output lengths are long-tailed, so toward the end of a generation phase only a handful of long-tail samples are still decoding while the rest of the generation GPUs have gone idle — the same straggler problem, now *within* the generation pool. A further wrinkle is that the two pools may sit in *different datacenters* or on *heterogeneous* GPUs, so weight synchronization must cross a slow WAN. The problem is to remove both bubbles and keep both pools busy under disaggregation, without giving up convergence.

**Approach:** StreamRL attacks the two bubbles with two matched mechanisms. Against pipeline bubbles, **stream generation** forwards each completed sample to the training stage *immediately* rather than at end-of-batch, so training can begin sample-level work while generation is still running — overlapping the two stages across the disaggregated pools (in the one-step-async variant, training consumes the previous step's stream so the pools run fully concurrently). Against skewness bubbles, it predicts each prompt's output length with an **output-length ranker** — a *small LLM* supervised-fine-tuned to rank prompts by expected length — and uses **skewness-aware dispatching** (Algorithm 2) to isolate the predicted long-tail samples onto dedicated generation instances run at smaller batch size, so a few long generations no longer strand a whole instance's GPUs. The disaggregation is made cross-datacenter-capable by **RL-RPC**, a GPU-Direct-RDMA transfer layer, plus a network-aware broadcast tree for weight sync (only DP-rank-0 ships weights across the WAN to a remote generation instance, which then fans out locally). The reusable lesson is that disaggregation creates *two separable* bubbles — a cross-stage one cured by streaming/pipelining and an intra-generation one cured by length-prediction + straggler isolation — and that a cheap learned length predictor is enough to drive the latter. It offers both a synchronous (on-policy) and a one-step-asynchronous variant; the async reward curve is reported to closely match the synchronous one, so the staleness is benign. Resource-wise it targets GPU-compute utilization (both bubbles) and makes the cross-pool/cross-DC **network** (RDMA, WAN broadcast tree) a first-class concern.

**Results:** up to **2.66× throughput** over SOTA: **1.12×–2.12× vs veRL** and 1.06×–1.41× vs an in-house colocated baseline (**ColocationRL**) for the synchronous variant, and 1.30×–2.66× for the one-step-async variant; cross-datacenter cost-effectiveness 1.23×–1.31×. Qwen2.5 7B/32B/72B on 128 H800 (+32 H20 for the cross-DC experiments). Baselines: **veRL**, **ColocationRL** (colocated in-house).

**Stack (where the contribution lives):** A standalone disaggregated system (not built on veRL). The contribution — stream generation, the length ranker + skewness dispatcher, and the RL-RPC/WAN weight-broadcast layer — sits around an **in-house C++ generation engine (SGS)** (continuous batching + prefix sharing) and a Megatron-style 3D-parallel trainer. Same lineage as DistServe (lead author Yinmin Zhong). *Context:* this is the disaggregation point that RollArt/MARS beats (1.35×) and that RLBoost reimplements as its "Disagg.BAL" baseline. Links: https://arxiv.org/abs/2504.15930.

### slime — SGLang-native RL post-training framework (THUDM / Zhipu GLM team)

**Problem (category):** Server-based rollout architecture + quality-gated trajectory buffer for asynchronous, failure-prone agentic RL.

**Problem:** Agentic RL rollouts are heterogeneous, long-tailed, and *unreliable*: a trajectory is a multi-turn loop that interleaves generation with CPU-side tool calls, sandboxes, and web requests, so its latency varies wildly and it frequently fails midway (timeout, OOM, incomplete output). Two architectural mismatches make this expensive. First, the conventional *engine-based* rollout — where the trainer hands a fixed batch to a one-shot inference-engine call — stalls the whole batch on its slowest trajectory and forces the agent/tool control flow into the engine, which is the wrong place for it. Second, feeding these flaky, bursty, high-concurrency producers directly into a synchronous trainer means a hung sandbox or a degenerate trajectory (e.g. a zero-variance GRPO group) stalls or pollutes training. The problem is to drive heterogeneous agentic generation at high GPU utilization while decoupling the trainer from producer failures and admitting only good data.

**Approach:** slime's answer is to make rollout *server-based* and to interpose a quality-gated buffer. Persistent **SGLang servers** behind an OpenAI-compatible HTTP router serve generation under continuous batching, so arbitrary agent/environment code calls them as an ordinary client (the tool/agent logic stays in client code; the engine only emits tokens) and short requests retire while long ones continue — keeping the generation GPUs saturated and letting existing agent frameworks plug in unmodified. A **Data Buffer** sits between these producers and the Megatron trainer as admission control, filtering degenerate trajectories (timeout/OOM/incomplete/length/reward/zero-variance-group) so flaky CPU-bound tool execution never blocks the GPU trainer; the integrated **APRIL** partial-rollout scheme over-provisions requests, ends the step at a target completion count, and recycles in-flight partials (via SGLang `/abort_request`) into later steps so no work is discarded on the long tail. The reusable lesson: agentic rollout should be a continuously-batched, fully-async *producer* behind a server, and a quality-gated buffer is the right place to absorb both stragglers and failures. Deployable colocated (time-sliced) or disaggregated (±PD); async staleness is handled algorithmically (GLM-5 uses version-logged double-sided importance sampling with clipping).

**Results:** No single headline end-to-end number is published; the integrated **APRIL** (arXiv:2509.18521) reports rollout throughput +22.5% avg (up to +44%) and final accuracy +2.1% (up to +8%) over the non-partial baseline, validated on GRPO/DAPO/GSPO across NVIDIA and AMD GPUs. slime is the production RL infrastructure behind the full **GLM-4.5 → GLM-5** family. Baseline: synchronous/non-partial rollout (within slime).

**Stack (where the contribution lives):** A standalone framework (not built on veRL). The contribution is the rollout architecture + buffer: a RolloutManager + SlimeRouter (one OpenAI-compatible endpoint over many **SGLang** servers), the Data Buffer, and the APRIL controller, above an otherwise-standard **Megatron-LM** trainer (TP/PP/EP/CP) with **Ray** placement; weight sync is NCCL bucketed (CUDA-IPC in colocate). Links: https://github.com/THUDM/slime ; https://lmsys.org/blog/2025-07-09-slime/ ; APRIL https://arxiv.org/abs/2509.18521.

---

### 3.3 Flexible execution & dynamic-scheduling frameworks

*Where HybridFlow fixes one placement mode, these treat the execution mode itself
(colocate vs disaggregate vs hybrid) as a search/scheduling decision.*

### RLinf / RLux — Macro-to-Micro Flow transformation (OSDI 2026; arXiv:2509.15965)

**Problem (category):** Automatic execution-mode (colocate / disaggregate / hybrid) and placement planning for the multi-stage RL dataflow.

**Problem:** The RL dataflow is a heterogeneous multi-stage pipeline — generation-heavy inference and compute-heavy training, each with its own optimal parallelism and memory profile — and the single most consequential systems decision is the *execution mode*: whether generation and training time-share the same GPUs (colocated, à la HybridFlow), run on disjoint pools (disaggregated), or some hybrid. Existing systems hard-wire one mode, which is wrong for some workload/model-size points and leaves GPUs underutilized or pipelines stalled. The problem is to stop treating execution mode as a fixed design decision and instead *plan* it — jointly choosing colocate/disaggregate/hybrid placement, per-component parallelism, and stage scheduling — for a given job, ideally automatically.

**Approach:** RLinf's contribution is **Macro-to-Micro Flow (M2Flow)**: the user writes an easy-to-compose macro logical workflow, and the system decomposes it along temporal and spatial dimensions and recomposes it into an efficient micro-level physical execution. The decoupling rests on a fine-grained **Worker abstraction** (send/recv primitives + onload/offload resource functions + adaptive communication) so any component can be placed freely, plus two primitives that realize the modes: **context switching** — temporal GPU multiplexing via a distributed device lock, which gives the colocated mode (components co-reside by onload/offload, relieving GPU memory) — and **elastic pipelining** — variable-granularity output forwarding that overlaps rollout and training, which gives the disaggregated/hybrid modes (spatial overlap, eliminating inter-stage GPU-compute bubbles). A profiling-guided planner then uses recursive graph partitioning over profiled costs to auto-select the mode and placement, trading GPU compute, memory, and interconnect (NVLink intra-node / 400 Gbps RDMA inter-node) *per component* rather than globally. The reusable idea is exactly this: **execution mode becomes a search decision**, which is the key generalization beyond HybridFlow's single colocated paradigm and makes RLinf the most flexible framework here (it spans reasoning *and* embodied/agentic RL).

**Results:** on 256 H100-80GB: reasoning RL (Qwen2.5 1.5/7/32B) ≈**1.10×–1.58× vs veRL**; embodied RL (OpenVLA / OpenVLA-OFT) **1.25×–2.13×** (1.88× hybrid on ManiSkill). Headline 1.1×–2.13× (later revisions cite up to ~2.43× on embodied/hybrid). Baselines: veRL (reasoning), RL4VLA / SimpleVLA-RL (embodied).

**Stack (where the contribution lives):** A standalone framework (inspired by, not built on, veRL): the M2Flow planner + Worker runtime sit as an orchestration/scheduling layer (**Ray**) above interchangeable backends — **Megatron-LM + FSDP** training, **SGLang / vLLM / HF Transformers** inference — exposed through one Worker abstraction. Repo: github.com/RLinf/RLinf. *(OSDI'26 "RLux" rename not independently verified — arXiv title is "RLinf".)* Link: https://arxiv.org/abs/2509.15965.

### DynaRL — Flexible and Dynamic Scheduling of Large-scale RL Training (OSDI 2026) — *not yet public*

**Problem (category):** Runtime/online dynamic resource & placement scheduling for large-scale RL training (inferred from the title; paper unpublished).

**Status:** No preprint/abstract/PDF public as of mid-2026; the USENIX page exposes only title + authors (lead **Yuanqing Wang**, Peking University + Infinigence AI, with ICT/CAS, BeiHang, Tsinghua, SJTU). **Confirmed distinct from RLinf/RLux** (different title and headline term, different lead author, though overlapping author set — hence easy to confuse).

**Inferred problem (unconfirmed):** runtime, "flexible and **dynamic**" scheduling — online reallocation of device/placement/parallelism across RL stages at runtime (reacting to load / long-tail / stage transitions), in contrast to RLinf's largely plan-time M2Flow selection. Mechanism, stack, and results are **unknown** until release. Treat as a placeholder; re-check near the OSDI'26 conference.

Links: https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing

---

### 3.4 Cost-aware / elastic placement

### RLBoost / PolyRL (NSDI 2026, arXiv:2510.19225)

**Problem (category):** Cost-driven elastic offload of stateless rollout onto preemptible spot GPUs, while keeping the reserved trainer strictly on-policy (synchronous GRPO).

**Problem:** Rollout is ≈73% of step time and is stateless and embarrassingly parallel, whereas training is tightly-coupled and must stay on reserved GPUs — so the cost-efficient move is to push rollout onto cheap *preemptible* spot capacity (a 2×H100 spot instance is ~90% cheaper than reserved 8×H100) and keep only training on-demand. But spot capacity is volatile (instances appear and vanish mid-step), and naively offloading breaks three things that must be preserved simultaneously: on-policy correctness (a spot instance must never generate with stale weights), the synchronous trainer's progress (it cannot stall waiting for a slow or newly-joined spot peer), and work already done when an instance is preempted. Critically, the network path to spot instances is slow and asymmetric (a single ~50 Gbps frontend vNIC, no backend RDMA), so weight delivery cannot ride the fast backend interconnect — making cross-cluster weight transfer the core systems constraint.

**Approach:** RLBoost keeps RL strictly synchronous/on-policy and makes the *elasticity* the engineered part. Rather than partition work by request count, a feedback controller tunes a **time budget T_seed**: the reserved cluster "seeds" rollout itself only within a window at step start, then flips to training and overlaps with the spot instances finishing the rest (T_seed += η·(t_wait_train − t_wait_remote), with a per-instance-count memory to damp churn), and an N_prem cap keeps spot rollout just fast enough to match the trainer (the step-time floor). Weight provisioning is made **pull-based and per-instance**: the trainer stages weights to a CPU buffer and starts seeding immediately while each spot instance independently pulls the latest version over a decoupled multi-NIC sharded-TCP path (so slow peers never block, mid-step arrivals join the current step, and weight traffic never contends with the backend NVLink/RDMA used by training); on-policy is guaranteed by routing new requests only to instances already holding the latest version. Preemption and stragglers are handled by **token-granular migration**: a preempted request is re-prefilled (prompt + already-generated tokens) on a healthy instance — turning expensive checkpoint-restart fault-tolerance into a cheap one-extra-prefill stateless migration, which also drives JSQ load-balancing. *Note this stateless/token-granular premise is exactly what does not extend to stateful multi-turn agentic rollout.*

**Results:** vs **co-located veRL**: throughput 1.66×/1.97×/1.51× (Qwen3 8B/14B/32B), and 28–49% better cost-efficiency (tokens/$). vs **Disagg.BAL** (a StreamRL-style disaggregated reimplementation): matches throughput when spot is ample but at 45–75% lower per-token cost. Spot 2×H100 replayed from real cloud spot-availability traces; GRPO, OpenR1-Math, 14K max length. Baselines: veRL (co-located), Disagg.BAL (StreamRL-style).

**Stack (where the contribution lives):** A derived fork of veRL (2.7K LoC Python + 1.7K Rust). The contribution is a standalone **Rust rollout-manager** (instance allocation, T_seed, token-streaming, JSQ) plus decoupled per-node/per-instance **transfer agents** for pull-based weight movement; training reuses veRL/FSDP (Megatron also supported) on reserved 8×H100, rollout runs the **SGLang** server API on spot 2×H100, and the TCP transport is RDMA-pluggable (Mooncake/NIXL). Open-sourced as PolyRL. Links: https://arxiv.org/abs/2510.19225 ; https://github.com/Terra-Flux/PolyRL.

### ROSE — Rollout On Serving GPUs via Cooperative Elasticity (arXiv:2605.06534, HKUST + Alibaba)

**Problem (category):** SLO-safe harvesting of online-serving GPUs for agentic-RL rollout via cooperative co-location (cross-cluster, elastic).

**Problem:** An organization running RL training usually also runs an online inference-serving cluster that is heavily over-provisioned for tail traffic — measured at ~19% SM and ~14% HBM utilization on average — so its idle capacity is a tempting source of elastic rollout GPUs, especially since agentic-RL rollout demand swings wildly across steps (DAPO redundant sampling launches up to 5.7× the base batch). The difficulty is that serving has hard millisecond TTFT/TPOT SLOs under second-level bursty traffic, so the harvesting must be invisible to it. This raises three coupled problems: co-locating two *heterogeneous* LLMs (the rollout model and the serving model, with incompatible KV-cache layouts) on one GPU while sharing HBM and compute without breaking serving SLOs; synchronizing freshly trained weights from the training cluster to the serving cluster over a bandwidth-limited cross-datacenter Ethernet link (where a naive transfer takes tens of seconds to minutes); and scheduling long-tail multi-turn agentic trajectories across dedicated rollout GPUs *and* opportunistic, churning serving GPUs.

**Approach:** ROSE's insight is an **SLO asymmetry**: serving needs ms-level guarantees, but agentic rollouts tolerate second-level delay because long-tail trajectory overlap and cheap turn-granular rerouting absorb stalls — so serving can be *strictly prioritized* in both memory and compute while rollout still harvests the large idle slack. It keeps both models GPU-resident and shares dynamically, serving-first: a **co-serving executor** uses CUDA-VMM to remap 2 MB physical pages between rollout and serving on demand without changing attention kernels (rollout reactivation <5 s), reserves ~20% headroom with a one-time emergency KV-cache cut on bursts (preemptive memory reclaim), and a **dual-SLO admission controller** temporally shares compute using profiled prefill/decode latencies so rollout only runs in serving's slack. The cross-DC weight bottleneck is attacked by a **transfer engine** that ships only the weight delta ΔW — which RL's conservative updates make >95% sparse — shard-aware-routed to map heterogeneous train↔serve parallelism, cutting transfer 9–12×. An **elastic scheduler** does turn-wise (not trajectory-pinned) routing with prefix-KV cache-affinity and heartbeat rerouting under churn. The reusable lesson and the **contrast with RLBoost**: both are cost-driven rollout harvesting, but RLBoost steals external *cloud spot* GPUs (volatile, tens-of-seconds load, 6.8–26% churn tax) whereas ROSE steals *already-resident serving* GPUs in the same org (abundant, ≤5 s one-time swap, <0.5% overhead) — paying instead the price of solving SLO-safe co-location and cross-DC sync; they are complementary. (ROSE also notes multi-turn agentic rollout is prefill/compute-bound — prefill is 77–86% of tokens — unlike bandwidth-bound single-turn decode.)

**Results:** end-to-end **1.31×/1.46× (GRPO 8B/32B), 1.42×/3.31× (DAPO)** vs **ROLL**, and **1.44×/2.69× vs AReaL** (orthogonal to async). vs **RLBoost+** (their strengthened spot baseline): rollout −1.20×/−1.26× with <0.5% allocation overhead (vs 6.8–7.3% for spot, up to 26.1% for serverless λRL). **No P99 SLO violations**; memory preemption cuts P99 TPOT 9–14× vs static partition; weight transfer −9–12× (32B synced in ~21 s @200 Gbps). 16–48 training + 16–64 serving H800. Baselines: ROLL, AReaL, RLBoost+, serverless λRL.

**Stack (where the contribution lives):** A standalone framework (~5k LoC) on the **ROLL** agentic-RL framework (chosen over veRL for agentic support): the co-serving executor lives inside **vLLM 0.10.0** (serving + rollout engine), the elastic scheduler in ROLL, and the weight engine extends the **Mooncake Store** relay; training is **Megatron-LM**, CPU envs run on Kubernetes. Links: https://arxiv.org/abs/2605.06534 *(no public repo; post-cutoff, sourced from the PDF)*.

---

### 3.5 Disaggregated co-scheduling & heterogeneous placement (agentic RL)

### RollArt / MARS: Scaling Agentic RL Training via Disaggregated Infrastructure (Alibaba)

> **Note on identity:** the OSDI'26 entry **"MARS: Disaggregated Multi-Task Agentic RL Training at Scale"** (presentation *gao*) is, on strong but not page-verified evidence (same lead author Wei Gao, same Alibaba 3,000+ GPU system, same disaggregated multi-task framing), the **same work** as the arXiv preprint *RollArt* (2512.22560). Treated here as one entry; the "MARS" title foregrounds the **multi-task** angle (co-training one LLM across many heterogeneous environments — SWE-bench, WebShop, FrozenLake, GEM-math/game — each with different rollout lengths, turn counts, and reward latencies). Wei Gao is also lead author of RollPacker (NSDI'26) — distinct work.

**Problem (category):** Heterogeneous hardware-affinity disaggregation of agentic-RL sub-workloads + trajectory-level bounded-staleness async, at production scale.

**Problem:** A single step of agentic RL is internally a portfolio of sub-workloads with conflicting resource profiles — compute-bound prefill, memory-bandwidth-bound decode, CPU-heavy *stateful* environment simulation, stateless bursty reward evaluation, and throughput-bound training — and across *tasks* these profiles differ sharply (SWE-bench/FrozenLake are prefill-heavy with tens of turns; GEM-math is decode-heavy with long reasoning). Running all of this on one homogeneous GPU pool wastes resources at every turn: the slowest environment blocks the batch, decode-heavy work underuses compute-optimized GPUs, and dedicated reward GPUs sit at ~6% utilization. The problem is to map each sub-workload to the hardware that fits its bottleneck and to schedule the heterogeneous, stateful, straggler-prone trajectories so neither environment stragglers nor cross-cluster weight transfer stall the trainer — at a 3,000+-GPU production scale where a single failure must not take the job down.

**Approach:** RollArt treats agentic RL as a pipeline of resource-distinct sub-workloads and disaggregates them onto best-fit substrates, governed by three principles. P1 *hardware-affinity mapping*: route compute-bound prefill-heavy trajectories to compute-optimized H800 (989.5 TFLOPS) and memory-bandwidth-bound decode-heavy trajectories to bandwidth-optimized H20 (4 TB/s HBM). P2 *trajectory-level (not batch-level) asynchrony*: an LLMProxy gateway (command-driven ADD/ABORT, no batch stall) and per-trajectory EnvManagers overlap one trajectory's generation with another's environment interaction and a third's reward, hiding env stragglers under a bounded-staleness off-policy bound (α = max version gap per trajectory; α=1 default, over-stale trajectories aborted). P3 *statefulness-aware placement*: stateless reward goes to elastic multi-tenant serverless (Function Compute), lifting reward utilization 6%→88%, while stateful generation/env/training stay on dedicated clusters. The four resulting pools (H800 training / heterogeneous H800+H20 inference / k8s CPU env / serverless reward) are bridged by LLMProxy, and the cross-cluster weight channel rides an object store (Mooncake over 200 Gbps Ethernet, async) so cross-substrate bubbles and transfer latency are hidden. The reusable lesson is exactly P1–P3: match each sub-workload's bottleneck to a specialized substrate, then recover the cross-substrate bubbles with trajectory-granular async pipelining and an object-store weight channel.

**Results:** **2.05× end-to-end vs veRL+** (a strengthened veRL with async reward + async env + Reward-as-a-Service) and **1.35× vs StreamRL**; 2.65–4.58× throughput over a synchronous baseline. Serverless reward 6%→88% util (1.34× step-time); Mooncake 3.14× vs TCP for 32B weight sync. Bench cluster 96 H800 + 32 H20; production: hundreds-of-billions-param MoE on 3,000+ GPUs, 1 week continuous, single failure. Baselines: veRL / veRL+, StreamRL.

**Stack (where the contribution lives):** A production system (~60k LoC Python) evolving from ROLL + ROLL-Flash, shipped in github.com/alibaba/ROLL. The contribution is the disaggregation/scheduling layer — LLMProxy (trajectory-level generation gateway) and per-trajectory EnvManager — over **Ray** orchestration, a **Megatron v0.12.2** trainer (H800), **vLLM 0.8.4 + SGLang** inference (heterogeneous H800+H20), **Kubernetes** CPU env containers, and **Alibaba Function Compute** serverless reward; weight sync is NCCL intra-cluster, Mooncake cross-cluster. Links: https://arxiv.org/abs/2512.22560 ; ROLL-Flash https://arxiv.org/abs/2510.11345 ; repo https://github.com/alibaba/ROLL ; OSDI'26 (MARS) https://www.usenix.org/conference/osdi26/presentation/gao.

### Weave / ROLLMUX — Phase-level multiplexing across jobs (OSDI 2026; arXiv:2512.11306)

**Problem (category):** Cross-job phase multiplexing to fill rollout/training idle bubbles on shared disaggregated GPU pools (cluster-level, on-policy).

**Problem:** Disaggregating RL into separate generation and training GPU pools is attractive, but on-policy synchronization then forces a strict phase dependency: the training pool sits idle while rollout generates, and the rollout pool sits idle while training runs, so *each pool is idle for one full phase of every iteration* — a "dependency bubble" that a single job fundamentally cannot fill without breaking on-policy semantics. Almost all prior work optimizes a single job; the observation here is that a *cluster* runs many heterogeneous RL jobs at once, and one job's idle phase coincides with another's busy phase. The problem becomes a cluster-level packing/scheduling one: decide which jobs to co-locate on a shared (rollout-pool, training-pool) pair and how to interleave their phases so both pools stay busy, subject to per-job SLO and the host-DRAM capacity needed to keep co-located jobs' state resident.

**Approach:** Weave's insight is to make the within-job-unavoidable bubble a *cross-job schedulable resource* by **time-multiplexing** (not space-sharing) complementary phases of different jobs. A *co-execution group* shares one rollout pool and one training pool: for jobs {A,B} the rollout pool runs RolloutA→RolloutB while the training pool runs TrainA→TrainB, so A's training overlaps B's rollout and each job stays strictly synchronous/on-policy. A two-tier scheduler does an inter-group admission decision (pack jobs to minimize marginal GPU-provisioning cost under memory+SLO limits) and an intra-group round-robin meta-iteration (provably optimal for unsaturated groups); group size is bounded by host-DRAM caching of each job's model+optimizer state (~2–5 jobs per 8-GPU node), and keeping that state warm makes the phase context-switch ~1.7 s instead of ~80 s cold. It composes hardware-affinity placement (rollout on bandwidth-optimized H20, training on compute-optimized H800), long-tail straggler consolidation at ~80% completion, and a hierarchical broadcast for cross-cluster weight sync over slow Ethernet. The reusable lesson: raise the optimization scope from one job to multi-job orchestration — the bubble inside one synchronous on-policy job is recoverable by co-executing other jobs' complementary phases — which is orthogonal to, and composable with, the single-job long-tail and async techniques above; Weave reuses existing rollout/training engines rather than building new ones.

**Results:** **1.84× cost-efficiency vs solo 1:1 disaggregation** and **1.38× vs veRL co-located**, at **100% SLO attainment** ($510/h vs $936 solo-disagg vs $704 veRL over a 200-job, 2-week trace); per-job throughput overhead only 5–9%; dependency bubbles −24.4% (rollout) / −43.1% (training); 1.52× fewer H20 and 2.16× fewer H800 at peak. Testbed 328 H20 + 328 H800; Qwen 3B–32B. Baselines: solo disaggregation, veRL co-located, **Gavel+** (job-level — not phase-level — scheduler), and StreamRL/AReaL (which instead trade staleness; Weave stays on-policy).

**Stack (where the contribution lives):** A standalone scheduler (~5.2k LoC) on Alibaba's **ROLL** disaggregated-RL framework, modifying **Ray** for isolated per-job execution and adding **Redis** for scheduler↔job control; jobs expose phases via a transparent `@rollmux.phase` decorator + state load/offload hooks. It sits *above* the rollout/training engines as a pure scheduling layer (veRL co-located is a baseline, not the base). Links: https://arxiv.org/abs/2512.11306 ; OSDI'26 https://www.usenix.org/conference/osdi26/presentation/wu-tianyuan.

---

### 3.6 Program-aware agentic inference layer (the rollout substrate)

### ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System

**Problem (category):** Program-aware cross-layer (inference + tool-orchestration) scheduling for multi-turn agentic rollout and serving.

**Problem:** A multi-turn agent task is a sequence of model calls interleaved with tool/environment calls that together form *one* persistent workflow, but conventional agentic stacks bolt an isolated inference engine (vLLM/SGLang) onto an isolated tool orchestrator with no shared, end-to-end view of that workflow — and the inference engine sees only a stream of independent stateless requests. Without program identity the system mismanages exactly the things that matter for agentic efficiency: KV-cache context built up over earlier turns is evicted and recomputed even though the next turn will re-need it (and there is no cross-node locality); concurrent agents oscillate between GPU-bound *reasoning* and GPU-idle *acting* (tool-execution) phases, so GPU memory thrashes under some programs while others sit idle; tool-environment (Docker/sandbox) setup latency stalls the GPU; and finished or runaway tasks leak containers/sockets that bloat disk. This hurts both online agentic serving and synchronous RL *rollout* (the generation phase the trainer waits on).

**Approach:** ThunderAgent's insight is that the right scheduling unit for agentic workloads is the **program, not the request**. It introduces the "LLM Program" abstraction — a persistent, first-class schedulable object linking all of one agent task's model+tool requests and exposing the semantics a scheduler needs: live context length, bound tool environments, execution phase (Reasoning = GPU-bound vs Acting = tool-bound/GPU-idle), backend (which node holds its KV), and lifecycle state (Active/Paused/Terminated). A periodic program-aware scheduler then uses those semantics to pack GPU memory: it detects thrashing and pauses+evicts programs (preferring Acting programs, whose GPU is idle anyway, and shortest-context programs, since recompute cost is quadratic in tokens) to a waiting queue, and restores them onto idle nodes when capacity frees up — with location-aware routing that pins active programs to the node holding their KV but lets paused programs restore anywhere (their KV is already evicted), preserving locality without sacrificing load balance. Anticipatory tool-environment preparation pre-builds the next sandbox before GPU memory is committed, overlapping tool-init latency with GPU compute, and lifecycle-aware GC reclaims sandboxes/sockets/disk on termination. The reusable lesson is that elevating the whole workflow to a program gives the scheduler the application semantics (KV footprint, phase, bound resources, location) to do cross-layer scheduling that a request-level engine cannot. It is reward/algorithm-agnostic — a rollout/serving substrate beneath any trainer.

**Results:** 1.5–3.6× serving throughput over **vLLM** (SWE-Agent, OpenHands, ToolOrchestra); **1.8–3.9× RL-rollout throughput** on dual 8×H100, demonstrated as the rollout backend under **slime** (Search-R1) and **SkyRL** (mini-swe-agent); up to 4.2× disk savings via lifecycle-aware GC; ~2 LoC to integrate. Baseline: vLLM/SGLang-based agentic stacks.

**Stack (where the contribution lives):** An inference/rollout serving *layer* sitting beneath RL trainers and between agent clients and the engines — not a trainer (no Megatron/FSDP). The program-aware scheduler + tool resource manager are layered atop **vLLM/SGLang**'s existing paging/KV machinery and exposed via an OpenAI-compatible passthrough API that needs only a `Program_id` field. Repo: https://github.com/ThunderAgent-org/ThunderAgent ; site: https://thunderagent.ai ; paper: https://arxiv.org/abs/2602.13692.

---

### 3.7 Algorithm-level rollout reduction (adjacent — not a systems paper)

### PivotRL (NVIDIA + UC Berkeley; arXiv:2603.21383)

**Problem (category):** Algorithmic reduction of multi-turn agentic rollout *volume* via pivot-state single-turn RL (an RL algorithm, not a systems contribution — included for its effect on the rollout workload).

**Problem:** Long-horizon agentic post-training faces a tension between cost and generalization: SFT is compute-cheap but causes out-of-domain catastrophic forgetting, whereas end-to-end (E2E) RL preserves out-of-domain ability but requires many turns of full multi-turn on-policy rollout per update — and that rollout is the dominant cost. Worse, the rollout is largely wasted: ~71% of randomly sampled turns are uniformly solved or failed, yielding zero group-normalized advantage and therefore zero gradient. The problem PivotRL poses is one of rollout-budget allocation and local credit assignment — how to spend the rollout budget only on the turns that actually carry learning signal — which, although framed algorithmically, directly reshapes the workload a systems scheduler would see.

**Approach:** PivotRL runs a three-step pipeline over existing SFT trajectories: offline turn profiling (under a frozen reference policy, sample K local rollouts per candidate turn, score with a verifier, estimate reward mean/variance), pivot filtering (keep only turns with nonzero variance — and thus nonzero advantage — that remain difficult), and local single-turn rollout with GRPO (condition the policy on the intermediate *expert* state from the SFT data and roll out a single turn, rewarded by a functional-equivalence verifier rather than exact string match). The net effect — and the reason it matters here — is that each training sample becomes one *single-turn* rollout from a pre-fixed expert prefix rather than a full multi-turn E2E unroll. For a scheduler this is a profound workload reshape: variable-length, stateful, straggler-prone multi-turn trajectories become a large set of *uniform, embarrassingly parallel, fixed-prefix single-turn* generations with no long-tail trajectory stragglers, and the expensive sampling moves into a one-time offline profiling pass decoupled from the online training loop. In other words, it attacks the rollout long-tail *algorithmically* — by not generating it — where every other system here attacks it via scheduling. (Caveat: the paper does not net the one-time offline-profiling cost against the savings.)

**Results:** +4.17% avg in-domain accuracy vs same-data **SFT**; out-of-domain retention +0.21 vs SFT's −9.83; ~4× fewer rollout turns and ~5.5× less rollout wall-clock vs **E2E RL** at comparable SWE-Bench accuracy. Baselines: same-data SFT, end-to-end agentic RL.

**Stack (where the contribution lives):** Purely an algorithm/data-pipeline layer on **NVIDIA NeMo-RL** (policy optimization) + **NeMo-Gym** (env rollouts); from Qwen3-30B-A3B-Thinking, deployed as the agentic-vertical RL stage of Nemotron-3-Super-120B. Generation/training backends inside NeMo-RL are not named. Links: https://arxiv.org/abs/2603.21383 ; NeMo-RL https://github.com/NVIDIA-NeMo/RL.

---

## 4. Neighboring work (not on the core list, relevant for related-work framing)

**Asynchronous / decoupled RL.** *(StreamRL is now a full entry in §3.2.)*
**AsyncFlow** (arXiv:2507.01663) — TransferQueue data
store, bounded-staleness producer/consumer, ~1.59×. **Trinity-RFT** (arXiv:2505.17826) — decoupled
Explorer/Trainer/Buffer; unifies sync/async + on/off-policy + agentic. **AReaL-Hex**
(arXiv:2511.00796) — AReaL over *heterogeneous* GPUs (placement/allocation across mixed hardware).

**Rollout scheduling / long-tail.** *(RollPacker and GroupMind/Seer are now full entries in §3.1.)*
**Laminar** (arXiv:2510.12633) — fully-async, decouples per-trajectory generation to mitigate
long-tail. **RLHFuse** (arXiv:2409.13221, NSDI'25) — inter-stage (sample-level subtask overlap) +
intra-stage (micro-batch pipeline) fusion, up to 3.7×. **SPEC-RL** (arXiv:2509.23232) / **SRT**
(arXiv:2601.09083) — additional spec-decode-for-RL using prior-epoch segments (same family as
RhymeRL/TLT/GroupMind — dedup when citing). **OrchestrRL** (arXiv:2601.01209) — another OSDI-era
disaggregation/orchestration point surfaced while disambiguating DynaRL.

**Disaggregated / heterogeneous placement for RL.** **OpenRLHF** (arXiv:2405.11143) — Ray + vLLM;
placement groups + 3 colocation modes + Hybrid Engine. **ReaLHF** (arXiv:2406.14088) — dynamic
**parameter reallocation** + parallelism search per stage, 2–10.6×. **NeMo-Aligner**
(arXiv:2405.01481) — Megatron train + TensorRT-LLM generate.

**Agentic RL infrastructure (rollout-as-a-service).** **ProRL Agent** (arXiv:2603.18815, NVIDIA) —
full agentic rollout lifecycle (env init, tool exec, reward) as an independent HTTP service.
**SkyRL-Agent** (arXiv:2511.16108) — decomposes each trajectory into runtime-init / agent-run /
reward jobs, async dispatch, backend-interoperable. Survey: ["When LLMs Grow Hands and
Feet…"](https://amberljc.github.io/blog/2025-09-05-agentic-rl-systems.html) — taxonomy of
rollout-as-a-service designs.

**Serving substrate (what rollout is built on).** **vLLM** (PagedAttention + continuous batching),
**SGLang** (RadixAttention prefix reuse + PD-disaggregation), **DistServe** (arXiv:2401.09670, the
canonical prefill/decode disaggregation reference these RL systems borrow from), **Mooncake**
(KV-cache-centric disaggregated serving / cross-cluster object store, used by RollArt, GroupMind, ROSE).

---

## 5. Open gaps and angles for this project

Framed against the *application-aware / semantic scheduling* thesis:

1. **Semantic/predictive rollout scheduling beyond length.** RhymeRL (prior-epoch history) and
   GroupMind/Seer (within-group siblings) exploit essentially *one* signal — length predictability.
   Richer **application semantics** — task type, expected difficulty, tool-call pattern, GRPO-group
   reward structure — are largely untapped as *scheduling* signals (which trajectory to prioritize,
   preempt, migrate, or speculatively draft). ThunderAgent's "LLM Program" is the closest and is
   brand new. This is the most direct fit for the application-aware-scheduling thesis.

2. **Dynamic, semantics-aware heterogeneous co-scheduling.** RollArt opened heterogeneous placement
   but its routing is largely *static* hardware-affinity tagging (task → H800/H20/CPU/serverless).
   A scheduler that reacts to *per-trajectory phase* in real time (this trajectory is about to do a
   long tool call / a prefill-heavy planning step) is open.

3. **Unifying the three long-tail levers.** Async (AReaL), spec-decode (RhymeRL/TLT), and
   scheduling/packing (RollPacker) are studied in isolation. Their *interactions* are unexplored —
   e.g. does speculative drafting change the optimal staleness budget? does tail-batching break
   drafter freshness or history-rhyme acceptance?

4. **Staleness as a tunable scheduling objective, not just a constraint.** Everyone *bounds*
   staleness. Treating it as a per-trajectory knob the scheduler optimizes against throughput and
   accuracy (cheap-to-redo vs expensive trajectories tolerate different staleness) is open.

5. **Elasticity for stateful agentic rollout.** RLBoost harvests spot capacity (stateless,
   token-granular migration, synchronous); ROSE harvests serving GPUs (turn-granular, agentic) but
   migrates only *prefix KV*, not arbitrary tool/environment state. **Elastic capacity for stateful,
   long-running agentic rollouts** — migrating live tool/env state mid-trajectory, not just tokens or
   KV — is still unsolved.

6. **Cross-job / multi-tenant RL scheduling.** Weave/ROLLMUX is the first to schedule *across* RL
   jobs (phase multiplexing on shared disaggregated pools). Almost everything else optimizes a single
   job. Cluster-level questions — fair sharing, gang scheduling of heterogeneous RL jobs, semantics-
   aware co-placement of complementary phases, multi-tenant SLOs — are wide open.

7. **Execution mode as a learned/online decision.** RLinf chooses colocate/disaggregate/hybrid via
   offline profiling + graph partitioning; DynaRL (unreleased) hints at *runtime* dynamic
   reallocation. Whether the optimal execution mode and inference:training split should adapt *online*
   to the evolving length distribution, drafter acceptance, or staleness budget is unexplored.

---

## Appendix: caveats on sourcing

A few figures are illustrative or from secondary/very-recent sources and should be confirmed
against camera-ready before citing: slime's "engine-vs-server" framing is its own design narrative
(well-corroborated but the LMSYS blog did not surface the literal quantitative claim); AReaL's
weight-sync transport is not named in the paper (network mapping inferred); RollArt's per-trajectory
H800-vs-H20 routing may be static-tag rather than dynamic (confirm in PDF); ThunderAgent
(arXiv:2602.13692, Feb 2026) and several neighbors (ProRL Agent, SRT, AReaL-Hex) carry 2026 arXiv
IDs — verify publication status. SPEC-RL/SRT/TLT/RhymeRL/GroupMind overlap as a "spec-decode-for-RL"
family; RollArt ⊃ ROLL-Flash (arXiv:2510.11345).

**OSDI'26 ↔ arXiv title mappings (the USENIX pages returned HTTP 403, so renames are inferred from
matching authors/abstracts, not page-verified):**
- **RLinf** (arXiv:2509.15965) ↔ **RLux** (OSDI'26) — *unverified* rename; arXiv title is "RLinf".
- **Seer** (arXiv:2511.14617) ↔ **GroupMind** (OSDI'26, Qin) — high-confidence (same lead author/group/mechanism), not page-verified; arXiv title is still "Seer". Note the inference engine is in-house **vLLM**, not SGLang.
- **ROLLMUX** (arXiv:2512.11306) ↔ **Weave** (OSDI'26, Wu) — strongly supported (the paper itself uses "weave"/"woven" for its core idea).
- **RollArt** (arXiv:2512.22560) ↔ **MARS** (OSDI'26, Gao) — *merged here* on strong evidence (same author/system/scale), not page-verified; there may instead be a distinct MARS artifact. Wei Gao also leads RollPacker (distinct).
- **DynaRL** (OSDI'26, Wang) — **no preprint public**; everything beyond title/authors is inferred. Confirmed *distinct* from RLinf.
- **RollPacker** arXiv title is "Mitigating Long-Tail Rollouts for Fast, Synchronous RL Post-Training" (= NSDI'26 "RollPacker: …Tail Batching").

**Post-cutoff / PDF-only (no independent corroboration beyond the fetched paper):** ROSE
(arXiv:2605.06534, May 2026), PivotRL (arXiv:2603.21383, Mar 2026), Weave/ROLLMUX, MARS/RollArt,
GroupMind/Seer v3 numbers. **PivotRL** is an RL *algorithm* paper, not a systems contribution — its
"~4×/5.5× rollout reduction" is relative (Fig 1 read-offs) and does not net the offline-profiling cost.
