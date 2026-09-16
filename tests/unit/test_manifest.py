"""manifest.json construction, distribution stats, and JSON round-trip
(the "output metadata" test target)."""
import json

from frpm.models import HeadPoseRecord, Manifest, PoseBucket, SelectedImageRecord, VideoMetadata
from frpm.report import build_manifest


def _make_image(name="a.jpg", quality=0.8, pose=PoseBucket.FRONT, expressions=None, body="upper_body"):
    return SelectedImageRecord(
        file=f"faces/{name}",
        source_timestamp=1.23,
        frame_number=10,
        identity_score=0.95,
        quality_score=quality,
        face=HeadPoseRecord(yaw=1.0, pitch=2.0, roll=3.0, pose=pose),
        expressions=expressions or ["neutral"],
        body_visibility=body,
        categories=["front_neutral"],
    )


def test_build_manifest_computes_distributions():
    images = [
        _make_image("a.jpg", quality=0.95, pose=PoseBucket.FRONT, expressions=["neutral"]),
        _make_image("b.jpg", quality=0.55, pose=PoseBucket.THREE_QUARTER_LEFT, expressions=["slight_smile", "eyes_left"]),
    ]
    video = VideoMetadata(video_id="abc123XYZ00", title="Test Video", url="https://youtu.be/abc123XYZ00", duration=120.0)
    manifest = build_manifest(
        video=video, identity_cluster_id=0, frames_examined=500, faces_found=480,
        identities_found=2, candidates_after_dedup=120, pack_size_requested=45,
        images=images, reference_pack=["selected/faces/a.jpg"], lora_dataset=[],
    )
    assert manifest.identities_found == 2
    assert manifest.faces_found == 480
    assert manifest.pose_distribution["front"] == 1
    assert manifest.pose_distribution["three_quarter_left"] == 1
    assert manifest.expression_distribution["neutral"] == 1
    assert manifest.expression_distribution["slight_smile"] == 1
    assert manifest.expression_distribution["eyes_left"] == 1
    assert sum(manifest.quality_distribution.values()) == 2
    assert len(manifest.images) == 2
    assert manifest.reference_pack == ["selected/faces/a.jpg"]


def test_manifest_json_round_trip():
    video = VideoMetadata(video_id="abc123XYZ00", title="T", url="u", duration=1.0)
    manifest = build_manifest(
        video=video, identity_cluster_id=0, frames_examined=1, faces_found=1, identities_found=1,
        candidates_after_dedup=1, pack_size_requested=30, images=[_make_image()],
        reference_pack=[], lora_dataset=[],
    )
    raw = manifest.model_dump()
    encoded = json.dumps(raw)  # must be plain-JSON serializable: no numpy/enum leakage
    decoded = json.loads(encoded)
    restored = Manifest(**decoded)
    assert restored.video.video_id == "abc123XYZ00"
    assert restored.images[0].face.pose == PoseBucket.FRONT
    assert restored.images[0].file == "faces/a.jpg"


def test_manifest_with_no_images_has_empty_distributions():
    video = VideoMetadata(video_id="v", title="", url="", duration=0.0)
    manifest = build_manifest(
        video=video, identity_cluster_id=0, frames_examined=0, faces_found=0, identities_found=0,
        candidates_after_dedup=0, pack_size_requested=45, images=[], reference_pack=[], lora_dataset=[],
    )
    assert manifest.images == []
    assert manifest.quality_distribution == {}
    assert manifest.expression_distribution == {}
    assert manifest.pose_distribution == {}


def test_selected_image_record_pose_bucket_value():
    img = _make_image(pose=PoseBucket.PROFILE_LEFT)
    assert img.face.pose == PoseBucket.PROFILE_LEFT
    assert img.face.pose.value == "profile_left"
