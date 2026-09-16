"""End-to-end pipeline orchestration with coarse-grained resume support.

Four checkpoints (matching the spec's resume requirement), each cached under
``output/<VIDEO_ID>/.cache/``:

1. ``download``            - source video obtained + probed
2. ``frame_analysis``       - frames sampled, faces detected, embeddings
                              computed (identity clustering input only)
3. ``identity_clustering``  - clusters computed, target person chosen, and
                              landmarks/quality/pose/expression/body/dedup
                              computed *for that person's faces only*
4. selection/output         - always recomputed fresh (it's fast and
                              deterministic, and users legitimately want to
                              retry with a different --pack-size without
                              redoing detection)
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from . import expressions as expr_mod
from . import identity as identity_mod
from . import pose as pose_mod
from . import selector as selector_mod
from . import synthesis as synthesis_mod
from .body import BodyPoseDetector, classify_body_visibility
from .captions import build_caption
from .config import AppConfig
from .contact_sheet import build_grid
from .crops import make_body_crop, make_face_crop, make_portrait_crop
from .deduplication import deduplicate
from .downloader import download_video, extract_video_id
from .embeddings import SFaceEmbedder
from .errors import NoFacesFoundError
from .face_detection import YuNetDetector, filter_detections
from .landmarks import FaceLandmarker
from .models import BBox, Candidate, HeadPoseRecord, Manifest, SelectedImageRecord, VideoMetadata
from .quality import compute_quality
from .report import build_manifest, render_report, write_lora_dataset, write_manifest
from .utils import ensure_dir, ensure_model, imread_unicode, imwrite_unicode, link_or_copy, read_json, write_json
from .video import VideoInfo, probe_video, sample_frames

logger = logging.getLogger("frpm.pipeline")

# Bumped whenever clustering/quality/pose/expression logic changes materially
# enough that a previously-cached frame_analysis/identity_clustering result
# could be silently wrong under the new code (e.g. the identity-clustering
# algorithm switch from DBSCAN to complete-linkage). Included in each of
# those caches' validity keys so upgrading the tool never resumes with
# stale results computed under old logic.
PIPELINE_CACHE_VERSION = 2

POSE_BUCKETS = [
    "front", "three_quarter_left", "three_quarter_right", "profile_left",
    "profile_right", "looking_up", "looking_down",
]
EXPRESSION_SHEET_ORDER = [
    "neutral", "slight_smile", "serious", "laughing", "focused", "talking", "looking_down", "looking_up",
]


def _progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )


def _candidate_id(frame_number: int, bbox: BBox) -> str:
    return f"f{frame_number:08d}_{bbox.x}_{bbox.y}_{bbox.w}_{bbox.h}"


class Pipeline:
    def __init__(self, cfg: AppConfig, prompt_fn: Optional[Callable] = None):
        self.cfg = cfg
        self.prompt_fn = prompt_fn

    def run(self) -> tuple[Manifest, Path]:
        cfg = self.cfg
        video_path, video_meta = self._stage_download()
        out_dir = ensure_dir(cfg.video_output_dir(video_meta.video_id))
        cache_dir = cfg.cache_dir(video_meta.video_id)

        candidates, _video_info = self._stage_frame_analysis(video_path, out_dir, cache_dir)
        if not candidates:
            raise NoFacesFoundError(
                "No faces were detected in the sampled frames. Try a lower --fps floor, a wider "
                "--start/--end range, or verify the video actually shows a face."
            )

        deduped, cluster_infos, chosen_cluster = self._stage_identity_and_enrich(
            candidates, out_dir, cache_dir
        )

        manifest, report_path = self._stage_selection_and_output(
            deduped, cluster_infos, chosen_cluster, video_meta, out_dir, len(candidates)
        )

        if not cfg.keep_video:
            self._cleanup_download(video_path, cache_dir)
        if not cfg.keep_candidates:
            self._cleanup_candidates(out_dir, cache_dir)

        return manifest, report_path

    # ------------------------------------------------------------------
    # Stage 1: download
    # ------------------------------------------------------------------
    def _stage_download(self) -> tuple[Path, VideoMetadata]:
        cfg = self.cfg
        video_id_hint = extract_video_id(cfg.url) or "video"
        cache_dir = cfg.cache_dir(video_id_hint)
        cached = read_json(cache_dir / "download.json") if cfg.resume else None
        if cached:
            video_path = Path(cached["video_path"])
            if video_path.exists():
                logger.info("[1/4] Reusing cached download: %s", video_path)
                return video_path, VideoMetadata(**cached["metadata"])
            logger.info("Cached download record found but the file is missing; re-downloading.")

        logger.info("[1/4] Downloading video via yt-dlp...")
        video_path, meta = download_video(cfg, cfg.url)
        write_json(cache_dir / "download.json", {"video_path": str(video_path), "metadata": meta.model_dump()})
        return video_path, meta

    def _cleanup_download(self, video_path: Path, cache_dir: Path) -> None:
        try:
            if video_path.exists():
                video_path.unlink()
            record = cache_dir / "download.json"
            if record.exists():
                record.unlink()
        except OSError as exc:
            logger.debug("Could not clean up downloaded video: %s", exc)

    def _cleanup_candidates(self, out_dir: Path, cache_dir: Path) -> None:
        candidates_dir = out_dir / "candidates"
        if candidates_dir.exists():
            shutil.rmtree(candidates_dir, ignore_errors=True)
            # frame_analysis.json / identity_clustering.json both store
            # frame_path pointers into the now-deleted candidates/ dir, so
            # they can no longer be used to resume - remove them rather
            # than leave a resume path that crashes on a future run.
            # (Resume is for *interrupted* runs; a completed+cleaned run
            # legitimately needs to re-sample/re-detect from scratch.)
            for stale in ("frame_analysis.json", "identity_clustering.json"):
                (cache_dir / stale).unlink(missing_ok=True)

    @staticmethod
    def _cached_frames_still_exist(cached_candidates: list[dict], sample_size: int = 5) -> bool:
        """Defensive check: a cache is only usable if the frame files it
        points at are still on disk (they may have been removed by a prior
        run's cleanup, or manually deleted). Checking a small sample keeps
        this cheap even for caches with thousands of entries."""
        if not cached_candidates:
            return True
        sample = cached_candidates[:: max(1, len(cached_candidates) // sample_size)][:sample_size]
        return all(Path(d["frame_path"]).exists() for d in sample)

    # ------------------------------------------------------------------
    # Stage 2: sample frames + detect faces + embed (for clustering only)
    # ------------------------------------------------------------------
    def _stage_frame_analysis(self, video_path: Path, out_dir: Path, cache_dir: Path):
        cfg = self.cfg
        cache_file = cache_dir / "frame_analysis.json"
        sampling_key = {
            "fps": cfg.fps, "max_frames": cfg.max_frames, "max_duration": cfg.max_duration,
            "start": cfg.start, "end": cfg.end, "cache_version": PIPELINE_CACHE_VERSION,
        }
        cached = read_json(cache_file) if cfg.resume else None
        if cached and cached.get("sampling_key") == sampling_key and self._cached_frames_still_exist(cached["candidates"]):
            logger.info("[2/4] Reusing cached frame analysis (%d face detections).", len(cached["candidates"]))
            info = VideoInfo(**cached["video_info"])
            candidates = [Candidate.from_cache_dict(d) for d in cached["candidates"]]
            return candidates, info

        logger.info("[2/4] Sampling frames and detecting faces...")
        candidates_dir = out_dir / "candidates"
        detector = YuNetDetector(ensure_model("yunet", cfg.models_cache_dir))
        embedder = SFaceEmbedder(ensure_model("sface", cfg.models_cache_dir))

        sampled, info = sample_frames(video_path, cfg, candidates_dir)
        candidates: list[Candidate] = []
        with _progress() as progress:
            task = progress.add_task("Detecting faces", total=len(sampled))
            for sf in sampled:
                progress.advance(task)
                img = imread_unicode(sf.path)
                if img is None:
                    continue
                for bbox, conf, row in filter_detections(detector.detect(img)):
                    try:
                        emb = embedder.embed(img, row)
                    except Exception as exc:
                        logger.debug("Embedding failed at %.2fs: %s", sf.timestamp, exc)
                        continue
                    candidates.append(Candidate(
                        candidate_id=_candidate_id(sf.frame_number, bbox),
                        frame_path=sf.path, timestamp=sf.timestamp, frame_number=sf.frame_number,
                        bbox=bbox, detection_confidence=conf, yunet_row=row, embedding=emb,
                    ))

        write_json(cache_file, {
            "sampling_key": sampling_key,
            "video_info": vars(info),
            "candidates": [c.to_cache_dict() for c in candidates],
        })
        logger.info("Found %d face detections across %d sampled frames.", len(candidates), len(sampled))
        return candidates, info

    # ------------------------------------------------------------------
    # Stage 3: identity clustering + selection + per-person enrichment
    # ------------------------------------------------------------------
    def _stage_identity_and_enrich(self, candidates: list[Candidate], out_dir: Path, cache_dir: Path):
        cfg = self.cfg
        cache_file = cache_dir / "identity_clustering.json"

        logger.info("[3/4] Clustering identities...")
        clusters = identity_mod.cluster_identities(candidates)
        if not clusters:
            raise NoFacesFoundError("Faces were detected but none could be confidently clustered into an identity.")
        cluster_infos = identity_mod.summarize_clusters(clusters)
        logger.info("Found %d distinct identities.", len(cluster_infos))

        sheets_dir = ensure_dir(out_dir / "contact_sheets")
        for info in cluster_infos:
            members = clusters[info.cluster_id][:12]
            items = [(m.frame_path, "") for m in members]
            build_grid(
                items, sheets_dir / f"identity_candidate_{info.cluster_id}.jpg", columns=4,
                title=f"Identity {info.cluster_id} - {info.count} faces",
            )

        reference_embedding = None
        if cfg.reference_image is not None:
            reference_embedding = identity_mod.embed_reference_image(cfg.reference_image, cfg.models_cache_dir)

        chosen_cluster = identity_mod.select_target_identity(
            clusters, cfg, prompt_fn=self.prompt_fn, reference_embedding=reference_embedding
        )
        chosen_members = clusters[chosen_cluster]
        logger.info("Selected identity cluster %d (%d face detections).", chosen_cluster, len(chosen_members))

        cached = read_json(cache_file) if cfg.resume else None
        if (
            cached
            and cached.get("cache_version") == PIPELINE_CACHE_VERSION
            and cached.get("chosen_cluster") == chosen_cluster
            and cached.get("source_count") == len(candidates)
            and self._cached_frames_still_exist(cached["candidates"])
        ):
            logger.info("Reusing cached landmarks/quality/pose/expression/body/dedup results.")
            enriched = [Candidate.from_cache_dict(d) for d in cached["candidates"]]
            return enriched, cluster_infos, chosen_cluster

        deduped = self._enrich_and_dedup(chosen_members, out_dir)
        write_json(cache_file, {
            "cache_version": PIPELINE_CACHE_VERSION,
            "chosen_cluster": chosen_cluster,
            "source_count": len(candidates),
            "candidates": [c.to_cache_dict() for c in deduped],
        })
        return deduped, cluster_infos, chosen_cluster

    def _enrich_and_dedup(self, members: list[Candidate], out_dir: Path) -> list[Candidate]:
        cfg = self.cfg
        frame_cache: dict[str, Optional[np.ndarray]] = {}

        def get_frame(path: str) -> Optional[np.ndarray]:
            if path not in frame_cache:
                frame_cache[path] = imread_unicode(path)
            return frame_cache[path]

        logger.info("Analyzing landmarks, quality, pose and expressions for the selected identity...")
        landmarker = FaceLandmarker(ensure_model("face_landmarker", cfg.models_cache_dir))
        try:
            with _progress() as progress:
                task = progress.add_task("Scoring faces", total=len(members))
                for c in members:
                    progress.advance(task)
                    img = get_frame(c.frame_path)
                    if img is None:
                        continue
                    result = landmarker.analyze(img, c.bbox, c.detection_confidence)
                    if result is None:
                        continue
                    c.blendshapes = result["blendshapes"]
                    c.landmark_confidence = result["landmark_confidence"]
                    c.face_transform_matrix = result["transform"]
                    c.quality = compute_quality(img, c.bbox, c.landmark_confidence, cfg.quality_weights)
                    c.pose = pose_mod.compute_head_pose(result["transform"], cfg.pose_thresholds)
                    c.expressions = expr_mod.classify_expressions(c.blendshapes, cfg.expression_thresholds)
                    c.phash = self._compute_phash(img, c.bbox)
        finally:
            landmarker.close()

        logger.info("Detecting body visibility...")
        body_detector = BodyPoseDetector(ensure_model("pose_landmarker", cfg.models_cache_dir))
        try:
            scored = [c for c in members if c.quality is not None]
            with _progress() as progress:
                task = progress.add_task("Body visibility", total=len(scored))
                for c in scored:
                    progress.advance(task)
                    img = get_frame(c.frame_path)
                    if img is None:
                        continue
                    try:
                        poses = body_detector.detect(img)
                        c.body_visibility = classify_body_visibility(poses, c.bbox, img.shape[1], img.shape[0])
                    except Exception as exc:
                        logger.debug("Body detection failed at %.2fs: %s", c.timestamp, exc)
        finally:
            body_detector.close()

        logger.info("Deduplicating %d scored candidates...", len(scored))
        deduped = deduplicate(scored, cfg.dedup_thresholds)
        logger.info("%d candidates remain after deduplication.", len(deduped))
        return deduped

    @staticmethod
    def _compute_phash(img: np.ndarray, bbox: BBox):
        try:
            import cv2
            import imagehash
            from PIL import Image as PILImage

            h, w = img.shape[:2]
            x1, y1 = max(0, bbox.x), max(0, bbox.y)
            x2, y2 = min(w, bbox.x + bbox.w), min(h, bbox.y + bbox.h)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                return None
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            return imagehash.phash(PILImage.fromarray(face_rgb))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Stage 4: diversity selection, crops, contact sheets, manifest, report
    # ------------------------------------------------------------------
    def _stage_selection_and_output(
        self,
        deduped: list[Candidate],
        cluster_infos,
        chosen_cluster: int,
        video_meta: VideoMetadata,
        out_dir: Path,
        faces_found: int,
    ) -> tuple[Manifest, Optional[Path]]:
        cfg = self.cfg
        logger.info("[4/4] Selecting diverse pack, reference pack%s...", " and LoRA dataset" if cfg.lora_mode else "")

        by_id = {c.candidate_id: c for c in deduped}
        pack_map = selector_mod.select_diverse_pack(deduped, cfg.pack_size)
        reference_ids = selector_mod.select_reference_pack(deduped, cfg.reference_pack_size)

        selected_dir = ensure_dir(out_dir / "selected")
        originals_dir, faces_dir, portraits_dir = (
            ensure_dir(selected_dir / "originals"), ensure_dir(selected_dir / "faces"), ensure_dir(selected_dir / "portraits"),
        )
        poses_dir, expr_dir, body_dir = ensure_dir(out_dir / "poses"), ensure_dir(out_dir / "expressions"), ensure_dir(out_dir / "body")

        images: list[SelectedImageRecord] = []
        source_paths_by_file: dict[str, str] = {}
        union_ids = list(dict.fromkeys(list(pack_map.keys()) + reference_ids))

        generated: dict[str, dict[str, str]] = {}
        for cid in union_ids:
            c = by_id.get(cid)
            if c is None:
                continue
            img = imread_unicode(c.frame_path)
            if img is None:
                continue
            stem = f"frame_{c.frame_number:08d}_{cid[-8:]}"

            original_path = originals_dir / f"{stem}.jpg"
            imwrite_unicode(original_path, img)

            face_crop = make_face_crop(img, c.bbox)
            face_path = faces_dir / f"{stem}.jpg"
            imwrite_unicode(face_path, face_crop)

            portrait_path = None
            portrait_crop = make_portrait_crop(img, c.bbox)
            if portrait_crop is not None:
                portrait_path = portraits_dir / f"{stem}.jpg"
                imwrite_unicode(portrait_path, portrait_crop)

            rel_face = str(face_path.relative_to(out_dir)).replace("\\", "/")
            rel_original = str(original_path.relative_to(out_dir)).replace("\\", "/")
            rel_portrait = str(portrait_path.relative_to(out_dir)).replace("\\", "/") if portrait_path else None
            generated[cid] = {"face": rel_face, "original": rel_original, "portrait": rel_portrait, "abs_original": str(original_path)}

            if c.pose and c.pose.pose.value in POSE_BUCKETS:
                link_or_copy(face_path, poses_dir / c.pose.pose.value / face_path.name)
            if c.expressions:
                for tag in c.expressions.tags:
                    link_or_copy(face_path, expr_dir / tag / face_path.name)
            if c.body_visibility.value != "none":
                body_crop = make_body_crop(img, c.bbox)
                body_path = ensure_dir(body_dir / c.body_visibility.value) / f"{stem}.jpg"
                imwrite_unicode(body_path, body_crop)

        for cid, cats in pack_map.items():
            c = by_id.get(cid)
            g = generated.get(cid)
            if c is None or g is None or not c.pose or not c.expressions:
                continue
            file_field = f"faces/{Path(g['face']).name}"
            source_paths_by_file[file_field] = g["abs_original"]
            images.append(SelectedImageRecord(
                file=file_field,
                source_timestamp=c.timestamp,
                frame_number=c.frame_number,
                identity_score=float(c.detection_confidence),
                quality_score=c.quality.quality if c.quality else 0.0,
                face=c.pose,
                expressions=c.expressions.tags,
                body_visibility=c.body_visibility.value,
                categories=cats,
                face_crop=g["face"],
                portrait_crop=g["portrait"],
                original=g["original"],
            ))

        reference_pack_files = [generated[cid]["face"] for cid in reference_ids if cid in generated]

        lora_files: list[str] = []
        if cfg.lora_mode:
            captions: dict[str, str] = {}
            for img_rec in images:
                cid = self._file_to_candidate_id(img_rec.file, union_ids, generated)
                cand = by_id.get(cid) if cid else None
                if cand is None:
                    continue
                src_img = imread_unicode(source_paths_by_file.get(img_rec.file, ""))
                captions[img_rec.file] = build_caption(cand, cfg.trigger_token, src_img)
            lora_files = write_lora_dataset(images, captions, out_dir, source_paths_by_file)

        synthesized_records: list = []
        if cfg.synthesize_missing:
            try:
                synthesized_records = synthesis_mod.run_synthesis_stage(
                    deduped, cfg, out_dir,
                    ensure_base_image_path=lambda c: self._ensure_face_crop_path(c, faces_dir, out_dir, generated),
                )
            except ImportError as exc:
                logger.warning(
                    "--synthesize-missing requires the 'generate' extra (pip install -e \".[generate]\") - %s", exc
                )
            except Exception as exc:
                logger.warning("Synthesis stage failed (%s) - continuing without synthesized images.", exc)

        manifest = build_manifest(
            video=video_meta,
            identity_cluster_id=chosen_cluster,
            frames_examined=faces_found,
            faces_found=faces_found,
            identities_found=len(cluster_infos),
            candidates_after_dedup=len(deduped),
            pack_size_requested=cfg.pack_size,
            images=images,
            # face_crop/portrait_crop/original/reference_pack entries are already
            # out_dir-relative (e.g. "selected/faces/xxx.jpg") - do not re-prefix.
            reference_pack=reference_pack_files,
            lora_dataset=[str(Path(f).relative_to(out_dir)).replace("\\", "/") for f in lora_files],
            synthesized_images=synthesized_records,
        )
        write_manifest(manifest, out_dir)

        contact_sheets = self._build_final_contact_sheets(images, generated, by_id, union_ids, out_dir)

        report_path = None
        if cfg.generate_report:
            report_path = render_report(manifest, contact_sheets, out_dir)

        write_json(cfg.cache_dir(video_meta.video_id) / "selection.json", {
            "pack_size": len(images), "reference_pack_size": len(reference_pack_files), "lora_size": len(lora_files),
        })

        logger.info(
            "Done: %d final images, %d reference images%s.",
            len(images), len(reference_pack_files), f", {len(lora_files)} LoRA images" if lora_files else "",
        )
        return manifest, report_path

    @staticmethod
    def _file_to_candidate_id(file_field: str, union_ids: list[str], generated: dict) -> Optional[str]:
        name = Path(file_field).name
        for cid in union_ids:
            g = generated.get(cid)
            if g and Path(g["face"]).name == name:
                return cid
        return None

    @staticmethod
    def _ensure_face_crop_path(c: Candidate, faces_dir: Path, out_dir: Path, generated: dict) -> Optional[str]:
        """Returns the absolute on-disk path to ``c``'s face crop, reusing
        one already generated for the real selected/reference pack, or
        creating it on demand otherwise (used by the synthesis stage, which
        may need a base image for a candidate that isn't part of the real
        pack itself)."""
        g = generated.get(c.candidate_id)
        if g and g.get("face"):
            return str(out_dir / g["face"])
        img = imread_unicode(c.frame_path)
        if img is None:
            return None
        stem = f"frame_{c.frame_number:08d}_{c.candidate_id[-8:]}"
        face_crop = make_face_crop(img, c.bbox)
        face_path = faces_dir / f"{stem}.jpg"
        imwrite_unicode(face_path, face_crop)
        rel_face = str(face_path.relative_to(out_dir)).replace("\\", "/")
        generated.setdefault(c.candidate_id, {})["face"] = rel_face
        return str(face_path)

    def _build_final_contact_sheets(self, images, generated, by_id, union_ids, out_dir: Path) -> dict[str, str]:
        sheets_dir = ensure_dir(out_dir / "contact_sheets")
        result: dict[str, str] = {}

        by_pose: dict[str, SelectedImageRecord] = {}
        by_expr: dict[str, SelectedImageRecord] = {}
        for img in sorted(images, key=lambda i: i.quality_score, reverse=True):
            if img.face.pose.value not in by_pose:
                by_pose[img.face.pose.value] = img
            for tag in img.expressions:
                by_expr.setdefault(tag, img)

        identity_order = ["front", "three_quarter_left", "three_quarter_right", "profile_left", "profile_right"]
        identity_items = [
            (str(out_dir / img.face_crop), pose.replace("_", " "))
            for pose in identity_order if (img := by_pose.get(pose)) is not None
        ]
        path = build_grid(identity_items, sheets_dir / "identity.jpg", columns=min(5, max(1, len(identity_items))), title="Identity")
        if path:
            result["identity"] = "contact_sheets/identity.jpg"

        expr_items = [
            (str(out_dir / img.face_crop), tag.replace("_", " ").title())
            for tag in EXPRESSION_SHEET_ORDER if (img := by_expr.get(tag)) is not None
        ]
        path = build_grid(expr_items, sheets_dir / "expressions.jpg", columns=4, title="Expressions")
        if path:
            result["expressions"] = "contact_sheets/expressions.jpg"

        pose_items = [
            (str(out_dir / img.face_crop), pose.replace("_", " "))
            for pose in POSE_BUCKETS if (img := by_pose.get(pose)) is not None
        ]
        path = build_grid(pose_items, sheets_dir / "poses.jpg", columns=4, title="Head Poses")
        if path:
            result["poses"] = "contact_sheets/poses.jpg"

        return result
