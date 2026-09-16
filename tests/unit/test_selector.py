"""Diversity-aware selection - synthetic candidates, no ML models needed."""
from frpm.selector import categories_for, select_diverse_pack, select_reference_pack


def test_categories_for_front_neutral(candidate_factory):
    c = candidate_factory("a", pose="front", expressions=["neutral"])
    assert "front_neutral" in categories_for(c)


def test_categories_for_profile_has_no_expression_requirement(candidate_factory):
    c = candidate_factory("a", pose="profile_left", expressions=["big_smile"])
    assert "left_profile" in categories_for(c)


def test_categories_for_can_match_multiple(candidate_factory):
    c = candidate_factory("a", pose="three_quarter_left", expressions=["confident_smile"])
    cats = categories_for(c)
    assert "left_three_quarter_smile" in cats
    assert "confident" in cats


def test_select_diverse_pack_covers_distinct_categories_first(candidate_factory):
    candidates = [
        candidate_factory("cand_front_neutral", pose="front", expressions=["neutral"], quality=0.9, timestamp=0.0),
        candidate_factory("cand_front_smile", pose="front", expressions=["slight_smile"], quality=0.9, timestamp=10.0),
        candidate_factory("cand_profile", pose="profile_left", expressions=["neutral"], quality=0.9, timestamp=20.0),
    ]
    selected = select_diverse_pack(candidates, pack_size=30)
    assert set(selected.keys()) == {"cand_front_neutral", "cand_front_smile", "cand_profile"}
    assert "front_neutral" in selected["cand_front_neutral"]
    assert "left_profile" in selected["cand_profile"]


def test_select_diverse_pack_never_picks_low_quality_to_fill_category(candidate_factory):
    candidates = [candidate_factory("bad_front", pose="front", expressions=["neutral"], quality=0.05, timestamp=0.0)]
    selected = select_diverse_pack(candidates, pack_size=30)
    assert len(selected) == 0


def test_select_diverse_pack_respects_pack_size_budget(candidate_factory):
    candidates = [
        candidate_factory(f"c{i}", pose="front", expressions=["neutral"], quality=0.9, timestamp=i * 5.0)
        for i in range(50)
    ]
    selected = select_diverse_pack(candidates, pack_size=10)
    assert len(selected) == 10


def test_select_diverse_pack_prefers_higher_quality_within_same_category(candidate_factory):
    candidates = [
        candidate_factory("front_neutral", pose="front", expressions=["neutral"], quality=0.95, timestamp=0.0),
        candidate_factory("close_dup", pose="front", expressions=["neutral"], quality=0.94, timestamp=0.1),
        candidate_factory("far_diverse", pose="three_quarter_left", expressions=["neutral"], quality=0.80, timestamp=50.0),
    ]
    selected = select_diverse_pack(candidates, pack_size=2)
    ids = set(selected.keys())
    assert ids == {"front_neutral", "far_diverse"}
    assert "close_dup" not in ids


def test_select_diverse_pack_only_includes_categories_actually_present(candidate_factory):
    candidates = [candidate_factory("only_front", pose="front", expressions=["neutral"], quality=0.9)]
    selected = select_diverse_pack(candidates, pack_size=30)
    all_categories = {cat for cats in selected.values() for cat in cats}
    assert all_categories == {"front_neutral"}


def test_select_reference_pack_prefers_frontal_and_sharp(candidate_factory):
    candidates = [
        candidate_factory("front", pose="front", quality=0.9, timestamp=0.0),
        candidate_factory("profile", pose="profile_left", quality=0.9, timestamp=10.0),
    ]
    ref = select_reference_pack(candidates, size=1)
    assert ref == ["front"]


def test_select_reference_pack_respects_size(candidate_factory):
    poses = ["front", "three_quarter_left", "three_quarter_right", "profile_left", "profile_right"]
    candidates = [
        candidate_factory(f"c{i}", pose=poses[i % len(poses)], expressions=["neutral"], quality=0.9 - i * 0.01, timestamp=i * 5.0)
        for i in range(20)
    ]
    ref = select_reference_pack(candidates, size=5)
    assert len(ref) == 5


def test_select_reference_pack_empty_input():
    assert select_reference_pack([], size=5) == []
