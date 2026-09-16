"""Synthesis stage - pure logic (missing-category detection, base-image
selection, prompt building) - no ML models needed for these tests."""
import pytest

from frpm.synthesis import (
    CATEGORY_SPEC,
    POSE_ADJACENCY,
    build_synthesis_prompt,
    identify_missing_categories,
    pick_base_candidate,
)


def test_identify_missing_categories_empty_input_returns_all():
    missing = identify_missing_categories([])
    assert len(missing) == len(CATEGORY_SPEC)


def test_identify_missing_categories_finds_gap(candidate_factory):
    # Only front_neutral present -> every other category should be "missing".
    candidates = [candidate_factory("a", pose="front", expressions=["neutral"], quality=0.9)]
    missing = identify_missing_categories(candidates)
    assert "front_neutral" not in missing
    assert "left_profile" in missing
    assert "laughing" in missing


def test_identify_missing_categories_low_quality_still_counts_as_missing(candidate_factory):
    candidates = [candidate_factory("a", pose="front", expressions=["neutral"], quality=0.05)]
    missing = identify_missing_categories(candidates)
    assert "front_neutral" in missing


def test_identify_missing_categories_all_present_returns_empty(candidate_factory):
    candidates = []
    for name, (pose_key, expr_tags) in CATEGORY_SPEC.items():
        candidates.append(candidate_factory(
            f"c_{name}", pose=pose_key or "front", expressions=expr_tags or ["neutral"], quality=0.9,
        ))
    missing = identify_missing_categories(candidates)
    assert missing == []


def test_pick_base_candidate_prefers_matching_pose(candidate_factory):
    front = candidate_factory("front", pose="front", quality=0.9)
    three_q_left = candidate_factory("3ql", pose="three_quarter_left", quality=0.7)
    chosen = pick_base_candidate([front, three_q_left], "left_profile")
    # left_profile's adjacency prefers profile_left, then three_quarter_left, then front.
    assert chosen.candidate_id == "3ql"


def test_pick_base_candidate_falls_back_to_best_quality_overall(candidate_factory):
    only_right = candidate_factory("right", pose="profile_right", quality=0.6)
    chosen = pick_base_candidate([only_right], "left_profile")
    assert chosen.candidate_id == "right"


def test_pick_base_candidate_expression_only_category_prefers_front(candidate_factory):
    front = candidate_factory("front", pose="front", quality=0.8)
    profile = candidate_factory("profile", pose="profile_left", quality=0.95)
    chosen = pick_base_candidate([front, profile], "laughing")
    assert chosen.candidate_id == "front"


def test_pick_base_candidate_empty_input_returns_none():
    assert pick_base_candidate([], "front_neutral") is None


def test_pick_base_candidate_no_quality_or_pose_returns_none(candidate_factory):
    c = candidate_factory("a")
    c.quality = None
    assert pick_base_candidate([c], "front_neutral") is None


def test_pick_base_candidate_picks_highest_quality_within_preferred_pose(candidate_factory):
    low = candidate_factory("low", pose="front", quality=0.4)
    high = candidate_factory("high", pose="front", quality=0.9)
    chosen = pick_base_candidate([low, high], "front_neutral")
    assert chosen.candidate_id == "high"


def test_build_synthesis_prompt_includes_pose_phrase():
    prompt = build_synthesis_prompt("left_profile")
    assert "profile" in prompt.lower()


def test_build_synthesis_prompt_includes_expression_phrase():
    prompt = build_synthesis_prompt("laughing")
    assert "laughing" in prompt.lower()


def test_build_synthesis_prompt_combines_pose_and_expression():
    prompt = build_synthesis_prompt("front_slight_smile")
    assert "forward" in prompt.lower() or "front" in prompt.lower()
    assert "smile" in prompt.lower()


def test_build_synthesis_prompt_unknown_category_still_returns_string():
    prompt = build_synthesis_prompt("totally_unknown_category")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_category_spec_and_pose_adjacency_cover_all_pose_keys():
    pose_keys = {pose_key for pose_key, _ in CATEGORY_SPEC.values() if pose_key is not None}
    for key in pose_keys:
        assert key in POSE_ADJACENCY, f"missing POSE_ADJACENCY entry for {key}"


def test_effective_strength_boosts_profile_more_than_expression_only():
    from frpm.synthesis import effective_synthesis_strength

    profile_strength = effective_synthesis_strength("left_profile", 0.6)
    expr_only_strength = effective_synthesis_strength("laughing", 0.6)
    front_strength = effective_synthesis_strength("front_neutral", 0.6)
    assert profile_strength > expr_only_strength
    assert profile_strength > front_strength
    assert expr_only_strength == pytest.approx(0.6)
    assert front_strength == pytest.approx(0.6)


def test_effective_strength_is_capped():
    from frpm.synthesis import effective_synthesis_strength

    assert effective_synthesis_strength("left_profile", 0.95) <= 0.75


def test_effective_strength_has_a_floor():
    from frpm.synthesis import effective_synthesis_strength

    assert effective_synthesis_strength("front_neutral", 0.0) >= 0.05
