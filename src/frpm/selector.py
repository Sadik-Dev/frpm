"""Diversity-aware selection.

Builds the final curated pack, the smaller premium reference pack, and (the
candidate pool for) the LoRA dataset from the deduplicated, fully-scored
candidates belonging to the target identity.

Strategy: cover every pose x expression category actually present in the
footage with one strong example first (never a sub-par one just to fill a
box), then top up the remaining pack-size budget with the next best-quality,
time-diverse candidates.
"""
from __future__ import annotations

import logging
from typing import Callable

from .models import Candidate

logger = logging.getLogger("frpm.selector")

MIN_CATEGORY_QUALITY = 0.35
MIN_FILL_QUALITY = 0.30
FILL_TIME_SPACING_SEC = 2.0
MAX_PER_CATEGORY_IN_FILL = 4


def _pose(name: str) -> Callable[[Candidate], bool]:
    return lambda c: c.pose is not None and c.pose.pose.value == name


def _expr(*names: str) -> Callable[[Candidate], bool]:
    return lambda c: c.expressions is not None and any(n in c.expressions.tags for n in names)


def _both(pose_name: str, *expr_names: str) -> Callable[[Candidate], bool]:
    p, e = _pose(pose_name), _expr(*expr_names)
    return lambda c: p(c) and e(c)


TARGET_CATEGORIES: list[tuple[str, Callable[[Candidate], bool]]] = [
    ("front_neutral", _both("front", "neutral")),
    ("front_slight_smile", _both("front", "slight_smile")),
    ("front_serious", _both("front", "serious")),
    ("left_three_quarter_neutral", _both("three_quarter_left", "neutral")),
    ("left_three_quarter_smile", _both("three_quarter_left", "slight_smile", "confident_smile", "big_smile")),
    ("left_three_quarter_serious", _both("three_quarter_left", "serious")),
    ("right_three_quarter_neutral", _both("three_quarter_right", "neutral")),
    ("right_three_quarter_smile", _both("three_quarter_right", "slight_smile", "confident_smile", "big_smile")),
    ("right_three_quarter_serious", _both("three_quarter_right", "serious")),
    ("left_profile", _pose("profile_left")),
    ("right_profile", _pose("profile_right")),
    ("looking_down", _pose("looking_down")),
    ("looking_up", _pose("looking_up")),
    ("laughing", _expr("laughing")),
    ("talking", _expr("talking")),
    ("focused", _expr("focused")),
    ("confident", _expr("confident_smile")),
]


def _quality_of(c: Candidate) -> float:
    return c.quality.quality if c.quality else 0.0


def categories_for(c: Candidate) -> list[str]:
    return [name for name, matcher in TARGET_CATEGORIES if matcher(c)]


def select_diverse_pack(candidates: list[Candidate], pack_size: int) -> dict[str, list[str]]:
    """Returns ``{candidate_id: [category_name, ...]}`` for the selected pack."""
    pool = [c for c in candidates if _quality_of(c) > 0]
    used: set[str] = set()
    selected: dict[str, list[str]] = {}
    per_category_count: dict[str, int] = {name: 0 for name, _ in TARGET_CATEGORIES}

    # Phase 1: one strong example per category actually present.
    for name, matcher in TARGET_CATEGORIES:
        matches = [
            c for c in pool
            if c.candidate_id not in used and matcher(c) and _quality_of(c) >= MIN_CATEGORY_QUALITY
        ]
        if not matches:
            continue
        best = max(matches, key=_quality_of)
        used.add(best.candidate_id)
        selected[best.candidate_id] = categories_for(best)
        per_category_count[name] += 1
        if len(selected) >= pack_size:
            return selected

    # Phase 2: fill remaining budget, favouring quality + time diversity.
    remaining_sorted = sorted((c for c in pool if c.candidate_id not in used), key=_quality_of, reverse=True)
    selected_timestamps = [c.timestamp for c in candidates if c.candidate_id in selected]
    for c in remaining_sorted:
        if len(selected) >= pack_size:
            break
        if _quality_of(c) < MIN_FILL_QUALITY:
            continue
        cats = categories_for(c)
        if cats and all(per_category_count.get(cat, 0) >= MAX_PER_CATEGORY_IN_FILL for cat in cats):
            continue
        if any(abs(c.timestamp - ts) < FILL_TIME_SPACING_SEC for ts in selected_timestamps):
            continue
        used.add(c.candidate_id)
        selected[c.candidate_id] = cats
        selected_timestamps.append(c.timestamp)
        for cat in cats:
            per_category_count[cat] = per_category_count.get(cat, 0) + 1

    # Phase 3: relax time-spacing (never quality) if budget remains and
    # decent material exists, rather than leaving the pack unnecessarily short.
    if len(selected) < pack_size:
        for c in remaining_sorted:
            if len(selected) >= pack_size:
                break
            if c.candidate_id in used or _quality_of(c) < MIN_FILL_QUALITY:
                continue
            used.add(c.candidate_id)
            selected[c.candidate_id] = categories_for(c)

    return selected


_POSE_FRONTALITY_BONUS = {
    "front": 1.0,
    "three_quarter_left": 0.85,
    "three_quarter_right": 0.85,
    "looking_up": 0.6,
    "looking_down": 0.6,
    "profile_left": 0.5,
    "profile_right": 0.5,
}


def select_reference_pack(candidates: list[Candidate], size: int) -> list[str]:
    """Premium identity-reference subset: sharp, well-exposed, mostly
    frontal/near-frontal, diverse angles, minimal blur - the set you'd hand
    directly to another image model as "use this person"."""

    def ref_score(c: Candidate) -> float:
        if not c.quality or not c.pose:
            return 0.0
        pose_bonus = _POSE_FRONTALITY_BONUS.get(c.pose.pose.value, 0.5)
        return 0.55 * c.quality.quality + 0.25 * c.quality.sharpness + 0.20 * pose_bonus

    pool = sorted((c for c in candidates if c.quality), key=ref_score, reverse=True)
    selected: list[Candidate] = []
    used_categories: dict[str, int] = {}
    for c in pool:
        if len(selected) >= size:
            break
        cats = categories_for(c) or ["uncategorized"]
        if all(used_categories.get(cat, 0) >= 3 for cat in cats) and len(selected) >= max(1, size // 2):
            continue
        selected.append(c)
        for cat in cats:
            used_categories[cat] = used_categories.get(cat, 0) + 1
    return [c.candidate_id for c in selected]
