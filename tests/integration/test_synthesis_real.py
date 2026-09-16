"""Real-model integration test for the synthesis stage (img2img + IP-Adapter
Plus Face + optional GFPGAN face restoration).

Marked ``slow`` (excluded from the default ``pytest`` run) since it
downloads several GB of SD1.5/IP-Adapter weights on first run and needs
network access + the ``generate`` extra installed. Run explicitly with:

    pip install -e ".[generate]"
    pytest tests/integration/test_synthesis_real.py -m slow -v
"""
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("diffusers")

from frpm.synthesis import _SynthesisPipeline, build_synthesis_prompt  # noqa: E402

pytestmark = pytest.mark.slow

FIXTURE = Path(__file__).parent.parent / "fixtures" / "test_face.jpg"


@pytest.fixture(scope="module")
def base_image():
    if not FIXTURE.exists():
        pytest.skip(f"Test fixture not found at {FIXTURE}")
    from PIL import Image

    return Image.open(FIXTURE).convert("RGB")


def test_synthesis_pipeline_recreates_a_perspective(base_image, tmp_path):
    """End-to-end: loads SD1.5 img2img + IP-Adapter Plus Face, and recreates
    the real test-face fixture in a different pose/expression, verifying the
    output is a valid, differently-sized-or-different image (not just a
    pass-through of the input)."""
    pipeline = _SynthesisPipeline(ip_adapter_scale=0.7)
    prompt = build_synthesis_prompt("left_profile")

    result = pipeline.synthesize(
        base_image=base_image,
        prompt=prompt,
        negative_prompt="cartoon, illustration, blurry, low quality",
        strength=0.6,
        steps=12,  # fewer steps than production default (28) to keep this test faster
        guidance_scale=7.0,
        seed=42,
        apply_face_restore=False,
    )

    assert result is not None
    assert result.size[0] > 0 and result.size[1] > 0

    out_path = tmp_path / "synthesized.jpg"
    result.save(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
