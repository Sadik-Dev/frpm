"""Identity clustering and target-person selection - synthetic embeddings,
no real face images/models needed."""
import numpy as np
import pytest

from frpm.config import AppConfig
from frpm.errors import NoIdentitySelectedError
from frpm.identity import DEFAULT_EPS, cluster_identities, match_reference_embedding, select_target_identity


def _cluster_around(center, n, seed, spread=0.02, dim=128):
    rng = np.random.RandomState(seed)
    return [center + rng.randn(dim).astype(np.float32) * spread for _ in range(n)]


def test_cluster_identities_separates_two_people(candidate_factory):
    rng = np.random.RandomState(42)
    person_a_center = rng.randn(128).astype(np.float32) * 3
    person_b_center = rng.randn(128).astype(np.float32) * 3

    candidates = [
        candidate_factory(f"a{i}", embedding=e) for i, e in enumerate(_cluster_around(person_a_center, 8, seed=1))
    ] + [
        candidate_factory(f"b{i}", embedding=e) for i, e in enumerate(_cluster_around(person_b_center, 8, seed=2))
    ]

    clusters = cluster_identities(candidates, eps=DEFAULT_EPS, min_samples=4)
    assert len(clusters) == 2
    assert sorted(len(v) for v in clusters.values()) == [8, 8]


def test_cluster_identities_groups_tight_cluster_together(candidate_factory):
    rng = np.random.RandomState(0)
    center = rng.randn(128).astype(np.float32) * 3
    candidates = [
        candidate_factory(f"c{i}", embedding=e)
        for i, e in enumerate(_cluster_around(center, 10, seed=5, spread=0.01))
    ]
    clusters = cluster_identities(candidates)
    assert len(clusters) == 1
    assert len(next(iter(clusters.values()))) == 10


def test_cluster_identities_empty_input_returns_empty():
    assert cluster_identities([]) == {}


def test_select_target_identity_dominant_auto_selects(candidate_factory):
    cfg = AppConfig(url="https://youtu.be/xxxxxxxxxxx", output_dir="out")
    clusters = {
        0: [candidate_factory(f"m{i}") for i in range(90)],
        1: [candidate_factory(f"n{i}") for i in range(10)],
    }
    assert select_target_identity(clusters, cfg) == 0


def test_select_target_identity_person_override(candidate_factory):
    cfg = AppConfig(url="https://youtu.be/xxxxxxxxxxx", output_dir="out", person=2)
    clusters = {
        0: [candidate_factory(f"m{i}") for i in range(60)],
        1: [candidate_factory(f"n{i}") for i in range(40)],
    }
    assert select_target_identity(clusters, cfg) == 1


def test_select_target_identity_person_out_of_range_raises(candidate_factory):
    cfg = AppConfig(url="https://youtu.be/xxxxxxxxxxx", output_dir="out", person=5)
    clusters = {0: [candidate_factory("m0")]}
    with pytest.raises(NoIdentitySelectedError):
        select_target_identity(clusters, cfg)


def test_select_target_identity_no_prompt_falls_back_to_largest(candidate_factory):
    cfg = AppConfig(url="https://youtu.be/xxxxxxxxxxx", output_dir="out")
    clusters = {
        0: [candidate_factory(f"m{i}") for i in range(55)],
        1: [candidate_factory(f"n{i}") for i in range(45)],
    }
    assert select_target_identity(clusters, cfg, prompt_fn=None) == 0


def test_select_target_identity_uses_prompt_when_ambiguous(candidate_factory):
    cfg = AppConfig(url="https://youtu.be/xxxxxxxxxxx", output_dir="out")
    clusters = {
        0: [candidate_factory(f"m{i}") for i in range(55)],
        1: [candidate_factory(f"n{i}") for i in range(45)],
    }
    chosen = select_target_identity(clusters, cfg, prompt_fn=lambda infos: infos[1].cluster_id)
    assert chosen == 1


def test_select_target_identity_empty_raises():
    cfg = AppConfig(url="https://youtu.be/xxxxxxxxxxx", output_dir="out")
    with pytest.raises(NoIdentitySelectedError):
        select_target_identity({}, cfg)


def test_match_reference_embedding_picks_closest_cluster(candidate_factory):
    center_a = np.array([1.0, 0.0, 0.0] + [0.0] * 125, dtype=np.float32)
    center_b = np.array([0.0, 1.0, 0.0] + [0.0] * 125, dtype=np.float32)
    clusters = {
        0: [candidate_factory("a0", embedding=center_a), candidate_factory("a1", embedding=center_a)],
        1: [candidate_factory("b0", embedding=center_b), candidate_factory("b1", embedding=center_b)],
    }
    best_id, sim = match_reference_embedding(clusters, reference_embedding=center_b.copy())
    assert best_id == 1
    assert sim == pytest.approx(1.0, abs=1e-4)


def test_match_reference_embedding_handles_noisy_reference(candidate_factory):
    rng = np.random.RandomState(1)
    center_a = rng.randn(128).astype(np.float32) * 3
    center_b = rng.randn(128).astype(np.float32) * 3
    clusters = {
        0: [candidate_factory(f"a{i}", embedding=center_a + rng.randn(128).astype(np.float32) * 0.01) for i in range(5)],
        1: [candidate_factory(f"b{i}", embedding=center_b + rng.randn(128).astype(np.float32) * 0.01) for i in range(5)],
    }
    noisy_reference = center_a + rng.randn(128).astype(np.float32) * 0.05
    best_id, sim = match_reference_embedding(clusters, noisy_reference)
    assert best_id == 0
    assert sim > 0.9


def test_select_target_identity_uses_reference_embedding_over_dominant(candidate_factory):
    """Even when one cluster is numerically dominant, an explicit reference
    image should override the "pick the biggest cluster" heuristic."""
    cfg = AppConfig(url="https://youtu.be/xxxxxxxxxxx", output_dir="out")
    center_dominant = np.array([1.0, 0.0] + [0.0] * 126, dtype=np.float32)
    center_minor = np.array([0.0, 1.0] + [0.0] * 126, dtype=np.float32)
    clusters = {
        0: [candidate_factory(f"d{i}", embedding=center_dominant) for i in range(90)],
        1: [candidate_factory(f"m{i}", embedding=center_minor) for i in range(10)],
    }
    chosen = select_target_identity(clusters, cfg, reference_embedding=center_minor.copy())
    assert chosen == 1


def test_select_target_identity_person_override_wins_over_reference(candidate_factory):
    cfg = AppConfig(url="https://youtu.be/xxxxxxxxxxx", output_dir="out", person=1)
    center_a = np.array([1.0, 0.0] + [0.0] * 126, dtype=np.float32)
    center_b = np.array([0.0, 1.0] + [0.0] * 126, dtype=np.float32)
    clusters = {
        0: [candidate_factory(f"a{i}", embedding=center_a) for i in range(5)],
        1: [candidate_factory(f"b{i}", embedding=center_b) for i in range(5)],
    }
    chosen = select_target_identity(clusters, cfg, reference_embedding=center_b.copy())
    assert chosen == 0  # --person 1 -> first (largest-count-ordered) cluster, ignoring the reference match
