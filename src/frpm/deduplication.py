"""Near-duplicate removal.

Combines four signals so that only genuinely different frames survive:

* perceptual hash (phash) of the face crop - overall visual similarity
* face embedding cosine similarity - same identity + very similar look
* blendshape-vector cosine similarity - a compact, always-available proxy
  for "facial landmark configuration" similarity (blendshapes are derived
  directly from the landmarks)
* timestamp proximity - near-duplicates almost always come from the same
  short video segment

Two candidates are only merged into the same "near duplicate" group if ALL
signals agree AND they are close in time; within each group only the
top ``max_per_window`` (by quality) survive.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import DedupThresholds
from .embeddings import cosine_similarity
from .models import Candidate


def _blendshape_vector(blendshapes: dict[str, float], keys: list[str]) -> np.ndarray:
    return np.array([blendshapes.get(k, 0.0) for k in keys], dtype=np.float32)


def _phash_distance(a, b) -> int:
    if a is None or b is None:
        return 999
    return int(a - b)  # imagehash overloads subtraction as Hamming distance


def deduplicate(candidates: list[Candidate], thresholds: Optional[DedupThresholds] = None) -> list[Candidate]:
    """Returns the surviving (non-duplicate) candidates. Dropped candidates
    have ``duplicate_of`` set to the candidate_id of the kept representative."""
    t = thresholds or DedupThresholds()
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: c.timestamp)
    all_keys = sorted({k for c in ordered for k in (c.blendshapes or {})})

    parent = {c.candidate_id: c.candidate_id for c in ordered}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n = len(ordered)
    for i in range(n):
        ci = ordered[i]
        for j in range(i + 1, n):
            cj = ordered[j]
            if cj.timestamp - ci.timestamp > t.time_window_sec:
                break  # sorted by time: nothing further can be in-window either
            if ci.identity_cluster != cj.identity_cluster:
                continue
            if _phash_distance(ci.phash, cj.phash) > t.phash_max_distance:
                continue
            if ci.embedding is None or cj.embedding is None:
                continue
            if cosine_similarity(ci.embedding, cj.embedding) < t.embedding_min_similarity:
                continue
            if all_keys:
                vi = _blendshape_vector(ci.blendshapes, all_keys)
                vj = _blendshape_vector(cj.blendshapes, all_keys)
                if cosine_similarity(vi, vj) < t.blendshape_min_similarity:
                    continue
            union(ci.candidate_id, cj.candidate_id)

    groups: dict[str, list[Candidate]] = {}
    for c in ordered:
        groups.setdefault(find(c.candidate_id), []).append(c)

    survivors: list[Candidate] = []
    for members in groups.values():
        members.sort(key=lambda c: c.quality.quality if c.quality else 0.0, reverse=True)
        keep = members[: t.max_per_window]
        drop = members[t.max_per_window :]
        survivors.extend(keep)
        rep_id = keep[0].candidate_id if keep else None
        for c in drop:
            c.duplicate_of = rep_id

    return survivors
