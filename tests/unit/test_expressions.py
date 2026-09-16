"""Expression tagging - rule-based over synthetic blendshape dicts, no ML
models needed for these tests."""
from frpm.expressions import classify_expressions


def test_neutral_when_all_zero():
    assert classify_expressions({}).tags == ["neutral"]


def test_big_smile():
    tags = classify_expressions({"mouthSmileLeft": 0.8, "mouthSmileRight": 0.8}).tags
    assert "big_smile" in tags


def test_slight_smile_not_big():
    tags = classify_expressions({"mouthSmileLeft": 0.3, "mouthSmileRight": 0.3}).tags
    assert "slight_smile" in tags
    assert "big_smile" not in tags
    assert "confident_smile" not in tags


def test_laughing_combines_smile_and_open_mouth():
    tags = classify_expressions({"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9, "jawOpen": 0.5}).tags
    assert "laughing" in tags
    assert "big_smile" not in tags  # mutually exclusive tier - laughing wins


def test_mouth_open_without_smile():
    tags = classify_expressions({"jawOpen": 0.6}).tags
    assert "mouth_open" in tags


def test_talking_mid_range_jaw():
    tags = classify_expressions({"jawOpen": 0.25}).tags
    assert "talking" in tags
    assert "mouth_open" not in tags


def test_eyes_closed():
    tags = classify_expressions({"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9}).tags
    assert "eyes_closed" in tags


def test_gaze_direction_left():
    tags = classify_expressions({"eyeLookOutLeft": 0.6, "eyeLookInRight": 0.6}).tags
    assert "eyes_left" in tags
    assert "eyes_right" not in tags


def test_gaze_direction_right():
    tags = classify_expressions({"eyeLookOutRight": 0.6, "eyeLookInLeft": 0.6}).tags
    assert "eyes_right" in tags
    assert "eyes_left" not in tags


def test_raised_eyebrows():
    tags = classify_expressions({"browOuterUpLeft": 0.6, "browOuterUpRight": 0.6, "browInnerUp": 0.6}).tags
    assert "raised_eyebrows" in tags


def test_frowning():
    tags = classify_expressions({"mouthFrownLeft": 0.5, "mouthFrownRight": 0.5}).tags
    assert "frowning" in tags


def test_surprised_requires_no_smile():
    tags = classify_expressions({
        "browOuterUpLeft": 0.6, "browOuterUpRight": 0.6, "browInnerUp": 0.6,
        "jawOpen": 0.5, "eyeWideLeft": 0.6, "eyeWideRight": 0.6,
    }).tags
    assert "surprised" in tags


def test_multiple_tags_can_coexist():
    tags = classify_expressions({
        "mouthSmileLeft": 0.3, "mouthSmileRight": 0.3,
        "eyeLookOutRight": 0.6, "eyeLookInLeft": 0.6,
    }).tags
    assert "slight_smile" in tags
    assert "eyes_right" in tags


def test_blendshapes_are_preserved_on_record():
    record = classify_expressions({"jawOpen": 0.6})
    assert record.blendshapes.get("jawOpen") == 0.6
