"""Compute-cost model for branching rollout schemes in multi-turn agentic GRPO.

Not a simulator of any real system. It counts tokens, tool calls and an
idealised wall-clock for several ways of building a group of ~8 leaves per
prompt, under one synthetic multi-turn workload. Every parameter is an
assumption; change them at the top and re-run.

    python3 analysis/compute_model.py            # default agentic workload
    python3 analysis/compute_model.py --swe      # long-context SWE-like workload

Schemes (named after the paper whose fork pattern they imitate; only the
compute-relevant part is imitated, never the branch signal):
  flat       G independent chains (plain GRPO).
  tree_grpo  M chains, then N random non-leaf nodes per tree expanded AFTER the
             chains finish (post hoc, serial phase). Root is a valid pick.
  arpo       N0 initial chains; after each tool call every active path forks
             with prob 0.5 until the branch budget is used (online, early).
  patr       B0 chains; every C turns the top-2 active paths (random here) are
             replaced by 2 children each (online, spaced).
  bpo        one backbone, then M random turns get K siblings each, after the
             backbone finishes (post hoc, serial).
  mid_fork   2 trunks, each forks 3 extra children at mid-trajectory (online).

Cost columns:
  decode     unique generated tokens (shared prefix decoded once).
  tools      unique tool calls.
  pf_apc     prefill tokens with an ideal automatic prefix cache: every token
             is prefilled once (prompt once per prompt, each tool output once).
  pf_noapc   prefill tokens when every turn re-submits the full context and
             nothing is cached (what happens with prefix caching off, or after
             a cache flush).
  train      tokens forwarded+backwarded in training when every leaf is a full
             row (what all released tree codebases do).
  train_dd   same with each unique token trained once (prefix dedup).
  T_sync     rollout wall-clock with level-synchronous rounds (every round:
             all active paths decode one turn, the round lasts as long as the
             longest decode; then all tool calls, lasting as long as the
             slowest). Post-hoc schemes add a second serial phase.
  T_cont     wall-clock with no barriers (each path proceeds independently).
  occ        decode-slot occupancy in sync mode: decoded tokens divided by
             (active paths x longest decode) summed over rounds. 1.0 means no
             padding to the slowest member.
Decode step time is assumed flat in batch size (memory-bound decode).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Workload:
    prompt: int = 1500            # prompt tokens
    min_turns: int = 2
    max_turns: int = 10           # turns per trajectory ~ uniform[min, max]
    dec_mean: float = 300.0       # decode tokens per turn, lognormal
    dec_sigma: float = 0.8
    obs_mean: float = 600.0       # tool-output tokens per turn, lognormal
    obs_sigma: float = 0.6
    tool_mean_s: float = 2.0      # tool latency seconds, lognormal
    tool_sigma: float = 1.0
    step_s: float = 0.025         # seconds per decode step

    def lognorm(self, rng, mean, sigma):
        mu = np.log(mean) - sigma**2 / 2
        return float(rng.lognormal(mu, sigma))


SWE = Workload(prompt=4000, min_turns=10, max_turns=50, dec_mean=200, obs_mean=1500,
               tool_mean_s=4.0, tool_sigma=1.2)


@dataclass
class Node:
    dec: int
    obs: int
    lat: float
    parent: int | None
    depth: int                    # number of turns up to and including this one
    ctx_before: int               # context tokens before this turn's decode
    t_end_cont: float = 0.0       # finish time of this turn in no-barrier mode
    phase: int = 0                # 0 = online phase, 1 = post-hoc phase
    rnd: int = 0                  # round index within its phase (sync scheduling)


@dataclass
class Tree:
    nodes: list[Node] = field(default_factory=list)
    leaves: list[int] = field(default_factory=list)


class Builder:
    def __init__(self, wl: Workload, rng: np.random.Generator):
        self.wl, self.rng = wl, rng

    def _turn(self, parent, tree, start_t, phase):
        wl = self.wl
        p = tree.nodes[parent] if parent is not None else None
        ctx = wl.prompt if p is None else p.ctx_before + p.dec + p.obs
        dec = max(1, int(wl.lognorm(self.rng, wl.dec_mean, wl.dec_sigma)))
        obs = max(1, int(wl.lognorm(self.rng, wl.obs_mean, wl.obs_sigma)))
        lat = wl.lognorm(self.rng, wl.tool_mean_s, wl.tool_sigma)
        n = Node(dec, obs, lat, parent, 1 if p is None else p.depth + 1, ctx, phase=phase)
        n.rnd = p.rnd + 1 if (p is not None and p.phase == phase) else (n.depth if phase == 0 else 1)
        n.t_end_cont = start_t + dec * wl.step_s + lat
        tree.nodes.append(n)
        return len(tree.nodes) - 1

    def total_turns(self, at_least):
        lo = max(self.wl.min_turns, at_least)
        return int(self.rng.integers(lo, self.wl.max_turns + 1))

    def grow(self, tree, parent, phase=0, start_t=None):
        """Grow a path from `parent` (None = from prompt) to termination. Returns leaf id."""
        depth = 0 if parent is None else tree.nodes[parent].depth
        t = 0.0 if parent is None else tree.nodes[parent].t_end_cont
        if start_t is not None:
            t = start_t
        target = self.total_turns(depth + 1)
        cur = parent
        while depth < target:
            cur = self._turn(cur, tree, t, phase)
            t = tree.nodes[cur].t_end_cont
            depth += 1
        tree.leaves.append(cur)
        return cur

    def path(self, tree, leaf):
        out, cur = [], leaf
        while cur is not None:
            out.append(cur)
            cur = tree.nodes[cur].parent
        return out[::-1]


# --- schemes -----------------------------------------------------------------

def flat(b: Builder, G=8):
    t = Tree()
    for _ in range(G):
        b.grow(t, None)
    return t


def tree_grpo(b: Builder, M=2, N=3):
    t = Tree()
    for _ in range(M):
        b.grow(t, None)
    finish = max(t.nodes[l].t_end_cont for l in t.leaves)     # batch barrier
    trunks = list(t.leaves)
    for leaf in trunks:
        path = b.path(t, leaf)
        cands = [None] + path[:-1]                               # root + non-leaf nodes
        for pick in random.choices(cands, k=N):
            b.grow(t, pick, phase=1, start_t=finish)
    return t


def arpo(b: Builder, N0=4, budget=4, p=0.5):
    """Online: forks copy the parent after a tool call and run concurrently."""
    t = Tree()
    # grow turn by turn so forks can be inserted online
    active = []
    for _ in range(N0):
        target = b.total_turns(1)
        active.append((None, target))
    left = budget
    while active:
        nxt = []
        for parent, target in active:
            cur = b._turn(parent, t, 0.0 if parent is None else t.nodes[parent].t_end_cont, 0)
            if t.nodes[cur].depth >= target:
                t.leaves.append(cur)
                continue
            nxt.append((cur, target))
            if left > 0 and random.random() < p:
                left -= 1
                nxt.append((cur, b.total_turns(t.nodes[cur].depth + 1)))
        active = nxt
    return t


def patr(b: Builder, B0=4, C=None, expand=2, children=2):
    # PATR uses K=13 turns on SWE (50-turn cap) and 5 on FrozenLake; scale with length
    C = C or max(3, b.wl.max_turns // 4)
    t = Tree()
    active = [(None, b.total_turns(1)) for _ in range(B0)]
    rnd = 0
    while active:
        rnd += 1
        nxt = []
        for parent, target in active:
            cur = b._turn(parent, t, 0.0 if parent is None else t.nodes[parent].t_end_cont, 0)
            if t.nodes[cur].depth >= target:
                t.leaves.append(cur)
            else:
                nxt.append((cur, target))
        if rnd % C == 0 and nxt:
            random.shuffle(nxt)
            chosen, rest = nxt[:expand], nxt[expand:]
            for cur, _ in chosen:
                for _ in range(children):
                    rest.append((cur, b.total_turns(t.nodes[cur].depth + 1)))
            nxt = rest
        active = nxt
    return t


def bpo(b: Builder, M=7, K=2):
    t = Tree()
    leaf = b.grow(t, None)
    path = b.path(t, leaf)
    finish = t.nodes[leaf].t_end_cont
    cands = [None] + path[:-1]
    for pick in random.choices(cands, k=M):
        for _ in range(K - 1):
            b.grow(t, pick, phase=1, start_t=finish)
    return t


def mid_fork(b: Builder, trunks=2, extra=3):
    t = Tree()
    for _ in range(trunks):
        target = b.total_turns(2)
        cur, d = None, 0
        fork_at = max(1, target // 2)
        while d < target:
            cur = b._turn(cur, t, 0.0 if cur is None else t.nodes[cur].t_end_cont, 0)
            d += 1
            if d == fork_at:
                for _ in range(extra):
                    b.grow(t, cur)
        t.leaves.append(cur)
    return t


SCHEMES = {"flat": flat, "tree_grpo": tree_grpo, "arpo": arpo, "patr": patr, "bpo": bpo, "mid_fork": mid_fork}


# --- accounting --------------------------------------------------------------

def costs(b: Builder, t: Tree):
    wl = b.wl
    nodes = t.nodes
    decode = sum(n.dec for n in nodes)
    tools = len(nodes)
    pf_apc = wl.prompt + sum(n.obs for n in nodes)
    pf_noapc = sum(n.ctx_before for n in nodes) + sum(n.obs for n in nodes)
    train = 0
    for leaf in t.leaves:
        n = nodes[leaf]
        train += n.ctx_before + n.dec + n.obs
    train_dd = wl.prompt + sum(n.dec + n.obs for n in nodes)
    return dict(leaves=len(t.leaves), decode=decode, tools=tools, pf_apc=pf_apc,
                pf_noapc=pf_noapc, train=train, train_dd=train_dd)


def sync_time(b: Builder, trees: list[Tree]):
    """Level-synchronous rounds over a whole batch of trees, per phase."""
    wl = b.wl
    total, used, slots = 0.0, 0, 0
    for phase in (0, 1):
        by_round: dict[int, list[Node]] = {}
        for t in trees:
            for n in t.nodes:
                if n.phase != phase:
                    continue
                # post-hoc branches all start together: round counts from the fork
                by_round.setdefault(n.rnd, []).append(n)
        for d in sorted(by_round):
            ns = by_round[d]
            mx = max(n.dec for n in ns)
            total += mx * wl.step_s + max(n.lat for n in ns)
            used += sum(n.dec for n in ns)
            slots += mx * len(ns)
    return total, (used / slots if slots else 1.0)


def cont_time(trees: list[Tree]):
    return max(t.nodes[l].t_end_cont for t in trees for l in t.leaves)


def run(wl: Workload, batch=32, reps=30, seed=0):
    rows = {}
    for name, fn in SCHEMES.items():
        acc: dict[str, list[float]] = {}
        for r in range(reps):
            rng = np.random.default_rng(seed + r)
            random.seed(seed + r)
            b = Builder(wl, rng)
            trees = [fn(b) for _ in range(batch)]
            cs = [costs(b, t) for t in trees]
            for k in cs[0]:
                acc.setdefault(k, []).append(np.mean([c[k] for c in cs]))
            ts, occ = sync_time(b, trees)
            acc.setdefault("T_sync", []).append(ts)
            acc.setdefault("T_cont", []).append(cont_time(trees))
            acc.setdefault("occ", []).append(occ)
        rows[name] = {k: float(np.mean(v)) for k, v in acc.items()}
    return rows


def show(rows):
    base = rows["flat"]
    cols = ["leaves", "decode", "tools", "pf_apc", "pf_noapc", "train", "train_dd", "T_sync", "T_cont", "occ"]
    print("absolute, per prompt (times per batch, seconds)")
    print(f"{'scheme':10s}" + "".join(f"{c:>10s}" for c in cols))
    for name, r in rows.items():
        print(f"{name:10s}" + "".join(f"{r[c]:10.1f}" if c != "occ" else f"{r[c]:10.2f}" for c in cols))
    print("\nper leaf, relative to flat (1.00 = same cost per training sample)")
    rel = ["decode", "tools", "pf_apc", "pf_noapc", "train", "train_dd"]
    print(f"{'scheme':10s}" + "".join(f"{c:>10s}" for c in rel) + f"{'T_sync':>10s}{'T_cont':>10s}")
    for name, r in rows.items():
        per = [r[c] / r["leaves"] / (base[c] / base["leaves"]) for c in rel]
        print(f"{name:10s}" + "".join(f"{x:10.2f}" for x in per)
              + f"{r['T_sync'] / base['T_sync']:10.2f}{r['T_cont'] / base['T_cont']:10.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--swe", action="store_true")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    show(run(SWE if args.swe else Workload(), reps=args.reps))
