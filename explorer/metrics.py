"""Reliability metrics for multi-iteration evaluation runs.

Each per-sample score is binary: 1.0 if the record's `execution_accuracy` is
truthy, else 0.0. For one instruction with sample scores g = [S_1 .. S_n]:

  - Average       P̄       = mean over instances of mean(g)
  - Aptitude      A^90     = mean over instances of percentile(g, 90)
  - Unreliability U_10^90  = mean over instances of [percentile(g,90) - percentile(g,10)]
  - Reliability   R        = 1 - U                 (scores are 0..1, not 0..100)
  - pass@k (Chen et al. 2021, unbiased): for an instance with n samples and c
    passes, 1 - C(n-c, k)/C(n, k) (= 1 when n-c < k), averaged across instances
    with n >= k. Returned as a sweep {k: value} for k = 1 .. max(n_i).

Percentiles use numpy's default (linear interpolation). NOTE: with binary 0/1
scores and small n, a per-instance 90th/10th percentile collapses to 0 or 1, so
Aptitude/Unreliability are coarse (near-degenerate) for small n. This is inherent
to applying the paper's graded-score formulas to a binary score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityStats:
    avg_performance: float        # P̄
    aptitude: float               # A^90
    unreliability: float          # U_10^90
    reliability: float            # 1 - U
    passk: dict[int, float]       # {k: pass@k}, k = 1..max_n


def _scores(group: list[dict]) -> list[float]:
    return [1.0 if r.get("execution_accuracy") else 0.0 for r in group]


def _passk_instance(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one instance: 1 - C(n-c, k)/C(n, k).

    Computed as 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i) to avoid overflow.
    """
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def passk_curve(groups: dict[str, list[dict]]) -> dict[int, float]:
    if not groups:
        return {}
    max_n = max(len(g) for g in groups.values())
    curve: dict[int, float] = {}
    for k in range(1, max_n + 1):
        vals: list[float] = []
        for g in groups.values():
            n = len(g)
            if n < k:
                continue
            c = int(sum(_scores(g)))
            vals.append(_passk_instance(n, c, k))
        if vals:
            curve[k] = sum(vals) / len(vals)
    return curve


def reliability_metrics(groups: dict[str, list[dict]]) -> ReliabilityStats:
    if not groups:
        return ReliabilityStats(0.0, 0.0, 0.0, 0.0, {})
    means: list[float] = []
    p90s: list[float] = []
    ranges: list[float] = []
    for g in groups.values():
        if not g:
            continue
        s = _scores(g)
        means.append(float(np.mean(s)))
        p90 = float(np.percentile(s, 90))
        p10 = float(np.percentile(s, 10))
        p90s.append(p90)
        ranges.append(p90 - p10)
    if not means:
        return ReliabilityStats(0.0, 0.0, 0.0, 0.0, passk_curve(groups))
    unreliability = sum(ranges) / len(ranges)
    return ReliabilityStats(
        avg_performance=sum(means) / len(means),
        aptitude=sum(p90s) / len(p90s),
        unreliability=unreliability,
        reliability=1.0 - unreliability,
        passk=passk_curve(groups),
    )
