# What can we predict in RL post training workload?

## Intro

This document describes what we can predict in RL post training workload, specifically in rollout phase. The purpose of potential predictions is to make better scheduling decisions in rollout phase. The scheduling decisions that we are interested in are four fold. (1) request scheduling at the router; which llm inferene instance sends a request to, (2) request ordering (priority) for each rollout trajectory within a training batch; which request should be scheduled to be processed at each time t (priority). This is more subtle than the other schedulings in terms of where it should be enforced. It can be enforced by the router or the router does not control admission but each llm inference instance can enforce it. I am not sure which one is better design. The reason that we might want admission control is asychronous training and we want the GRPO group completion to be nicely distributed across the end-to-end rollout time for the sake of trainer. (3) iteration scheduling inside the llm inference instances; which request should be sent at each iteration batch, and (4) KV cache management (eviction policy specifically) in each llm inference instance; what KV cache should the instance keep and what should be evicted when KV cache reaches its capacity limit.

## What would be **useful** to predict for scheduling

### We are not interested in per-trajectory latency. A smalleset job unit is GRPO group. 

### The max length of trajectories in a GRPO group.

What's the definition of length of trajectory? It can be total number of llm's decoded token across all turns. This can be done through approximation. 

**Which scheduler can use this information?**
- request ordering for uniform group completion rate

But I am not sure by how big margin the optimal policy is better than random. If there isn't much, then we should stop thinking about it.

### Request ordering scheduling example

There are four GRPO groups. Each character represents different GRPO groups. The length of the line means the max trajectory length of each GRPO group.

In this example, I argue that random ordering is close enough to the optimal ordering and it is theoretically better than the longest job first and the shortest job first.

```
A --
B ----
C --------
D ----------------
```

Max concurrency = 2

**Longest job first**
```
slot 1: ---------------- 
slot 2: -------- ---- --
                1    1  2
```

**Shortest job first**
```
slot 1: -- --------
slot 2: ---- ----------------
          1 1      1         1
```

**Random**
```
slot 1: -- ---------------- 
slot 2: -------- ----
          1     1    1     1
```

### Next $i$th turn tool call time right after llm finishes $i$th decoding


## What seems predictable

## How to predict?

I think we need to be careful about this. There are a couple of very important decisions here. (1) What to predict which is explored in the previous section. It is obvious question we want to answer. We don't need to be bothered. (2) when to predict. This is subtle and one might miss the significance. Different prediction and different scheduler can leverage predictions made in different 


