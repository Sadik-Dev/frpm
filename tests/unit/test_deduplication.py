"""Duplicate detection - synthetic candidates, no real images/models needed."""
import numpy as np
import imagehash

from frpm.config import DedupThresholds
from frpm.deduplication import deduplicate

SAME_HASH = imagehash.ImageHash(np.zeros((8, 8), dtype=bool))
DIFF_HASH = imagehash.ImageHash(np.ones((8, 8), dtype=bool))


def _vec(seed=0, dim=128):
    return np.random.RandomState(seed).randn(dim).astype(np.float32)


def test_near_duplicates_within_time_window_are_collapsed(candidate_factory):
    base_emb = _vec()
    a = candidate_factory("a", timestamp=0.0, embedding=base_emb, quality=0.9, phash=SAME_HASH)
    b = candidate_factory("b", timestamp=0.3, embedding=base_emb, quality=0.7, phash=SAME_HASH)
    c = candidate_factory("c", timestamp=0.6, embedding=base_emb, quality=0.5, phash=SAME_HASH)
    survivors = deduplicate([a, b, c], DedupThresholds(max_per_window=1))
    assert len(survivors) == 1
    assert survivors[0].candidate_id == "a"  # highest quality kept


def test_max_per_window_keeps_more_than_one(candidate_factory):
    base_emb = _vec()
    cands = [
        candidate_factory(f"c{i}", timestamp=i * 0.2, embedding=base_emb, quality=1.0 - i * 0.1, phash=SAME_HASH)
        for i in range(5)
    ]
    survivors = deduplicate(cands, DedupThresholds(max_per_window=2))
    assert len(survivors) == 2


def test_different_timestamps_far_apart_both_survive(candidate_factory):
    base_emb = _vec()
    a = candidate_factory("a", timestamp=0.0, embedding=base_emb, quality=0.9, phash=SAME_HASH)
    b = candidate_factory("b", timestamp=30.0, embedding=base_emb, quality=0.8, phash=SAME_HASH)
    survivors = deduplicate([a, b], DedupThresholds())
    assert len(survivors) == 2


def test_different_phash_both_survive_even_if_close_in_time(candidate_factory):
    base_emb = _vec()
    a = candidate_factory("a", timestamp=0.0, embedding=base_emb, quality=0.9, phash=SAME_HASH)
    b = candidate_factory("b", timestamp=0.3, embedding=base_emb, quality=0.8, phash=DIFF_HASH)
    survivors = deduplicate([a, b], DedupThresholds())
    assert len(survivors) == 2


def test_different_embeddings_both_survive_even_if_close_in_time(candidate_factory):
    a = candidate_factory("a", timestamp=0.0, embedding=_vec(seed=1), quality=0.9, phash=SAME_HASH)
    b = candidate_factory("b", timestamp=0.3, embedding=_vec(seed=2), quality=0.8, phash=SAME_HASH)
    survivors = deduplicate([a, b], DedupThresholds())
    assert len(survivors) == 2


def test_different_identity_clusters_never_merge(candidate_factory):
    base_emb = _vec()
    a = candidate_factory("a", timestamp=0.0, embedding=base_emb, quality=0.9, phash=SAME_HASH, identity_cluster=0)
    b = candidate_factory("b", timestamp=0.1, embedding=base_emb, quality=0.8, phash=SAME_HASH, identity_cluster=1)
    survivors = deduplicate([a, b], DedupThresholds())
    assert len(survivors) == 2


def test_empty_input():
    assert deduplicate([]) == []


def test_dropped_candidates_record_duplicate_of(candidate_factory):
    base_emb = _vec()
    a = candidate_factory("a", timestamp=0.0, embedding=base_emb, quality=0.9, phash=SAME_HASH)
    b = candidate_factory("b", timestamp=0.1, embedding=base_emb, quality=0.5, phash=SAME_HASH)
    deduplicate([a, b], DedupThresholds(max_per_window=1))
    assert b.duplicate_of == "a"
