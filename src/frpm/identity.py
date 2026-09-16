"""Identity clustering and target-person selection.

Clusters L2-normalized SFace embeddings using **complete-linkage**
agglomerative clustering (euclidean distance on unit vectors is a monotonic
function of cosine similarity, so a euclidean ``distance_threshold`` is
equivalent to a cosine-similarity threshold).

Complete linkage (not DBSCAN) is used deliberately: DBSCAN's density-based
"chaining" can transitively link two genuinely different people together
through a path of intermediate, moderately-similar frames (e.g. motion blur,
poor lighting), silently merging distinct identities into one. This was
verified empirically - on real footage, DBSCAN collapsed 3512 face
detections spanning several different people into a single 3508-member
"cluster" where 75% of pairs fell below the same-person similarity
guideline. Complete linkage requires the *worst-case* pair between two
groups to be within threshold before merging them, which avoids this
chaining at the cost of initially fragmenting a single real person's
appearances (across very different poses/lighting/scenes) into several raw
clusters. The centroid-based merge pass below then recombines those
same-person fragments using a stricter, noise-averaged similarity check.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .config import AppConfig, DOMINANT_IDENTITY_RATIO
from .errors import NoIdentitySelectedError
from .models import Candidate, IdentityClusterInfo

logger = logging.getLogger("frpm.identity")

# OpenCV Zoo's SFace sample recommends a cosine-similarity threshold of
# ~0.363 for "same identity". For L2-normalized vectors,
# euclidean_dist**2 = 2 - 2*cosine_sim, so this is the equivalent eps.
SFACE_COSINE_SAME_PERSON = 0.363
DEFAULT_EPS = float(np.sqrt(max(0.0, 2 - 2 * SFACE_COSINE_SAME_PERSON)))
DEFAULT_MIN_SAMPLES = 4
# Higher bar than SFACE_COSINE_SAME_PERSON for merging two *cluster
# centroids* (as opposed to two individual embeddings): centroids already
# average out per-frame noise, so requiring a stronger match here resists
# accidentally re-merging two different people's fragments while still
# reuniting one real person's fragments split across poses/lighting/scenes.
# Validated empirically against real multi-person footage (see module docstring).
DEFAULT_MERGE_COSINE = 0.75


def cluster_identities(
    candidates: list[Candidate], eps: float = DEFAULT_EPS, min_samples: int = DEFAULT_MIN_SAMPLES
) -> dict[int, list[Candidate]]:
    valid = [c for c in candidates if c.embedding is not None]
    if not valid:
        return {}

    X = np.stack([c.embedding / (np.linalg.norm(c.embedding) + 1e-8) for c in valid]).astype(np.float64)

    if len(X) == 1:
        labels = np.array([0])
    else:
        agg = AgglomerativeClustering(n_clusters=None, distance_threshold=eps, metric="euclidean", linkage="complete")
        labels = agg.fit_predict(X)

    raw_clusters: dict[int, list[Candidate]] = {}
    for c, label in zip(valid, labels):
        raw_clusters.setdefault(int(label), []).append(c)

    merged = _merge_similar_clusters(raw_clusters)

    # Drop clusters too small to be a meaningful, independently-identifiable
    # person (mirrors DBSCAN's old "noise" concept) rather than cluttering
    # identity selection with singleton/near-singleton spurious detections.
    significant = {cid: members for cid, members in merged.items() if len(members) >= min_samples}
    for members in merged.values():
        if len(members) < min_samples:
            for c in members:
                c.identity_cluster = None

    # Renumber 0..N-1 by descending size for stable, predictable --person indexing.
    renumbered = {
        i: members for i, (_cid, members) in enumerate(sorted(significant.items(), key=lambda kv: -len(kv[1])))
    }
    for new_id, members in renumbered.items():
        for c in members:
            c.identity_cluster = new_id
    return renumbered


def _merge_similar_clusters(
    clusters: dict[int, list[Candidate]], merge_cosine: float = DEFAULT_MERGE_COSINE
) -> dict[int, list[Candidate]]:
    """Merges clusters whose centroid embeddings are highly similar (see
    module docstring). Returned dict keys are arbitrary/non-contiguous -
    ``cluster_identities`` does the final size-filtering and renumbering."""
    if len(clusters) <= 1:
        return clusters

    ids = list(clusters.keys())
    centroids = {}
    for cid in ids:
        embs = np.stack([c.embedding for c in clusters[cid]])
        centroid = embs.mean(axis=0)
        centroids[cid] = centroid / (np.linalg.norm(centroid) + 1e-8)

    parent = {cid: cid for cid in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = float(np.dot(centroids[ids[i]], centroids[ids[j]]))
            if sim >= merge_cosine:
                union(ids[i], ids[j])

    merged: dict[int, list[Candidate]] = {}
    for cid in ids:
        merged.setdefault(find(cid), []).extend(clusters[cid])
    return merged


def summarize_clusters(clusters: dict[int, list[Candidate]]) -> list[IdentityClusterInfo]:
    infos = []
    for cid, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        rep = max(members, key=lambda c: c.detection_confidence)
        infos.append(IdentityClusterInfo(cluster_id=cid, count=len(members), representative_file=rep.frame_path))
    return infos


def _cluster_centroid(members: list[Candidate]) -> np.ndarray:
    embs = np.stack([c.embedding for c in members])
    centroid = embs.mean(axis=0)
    return centroid / (np.linalg.norm(centroid) + 1e-8)


def match_reference_embedding(
    clusters: dict[int, list[Candidate]], reference_embedding: np.ndarray
) -> tuple[int, float]:
    """Returns ``(best_cluster_id, cosine_similarity)`` - the identity
    cluster whose centroid embedding is most similar to a user-supplied
    reference face embedding (e.g. from ``--reference-image``)."""
    ref = reference_embedding / (np.linalg.norm(reference_embedding) + 1e-8)
    best_id, best_sim = None, -1.0
    for cid, members in clusters.items():
        sim = float(np.dot(ref, _cluster_centroid(members)))
        if sim > best_sim:
            best_sim, best_id = sim, cid
    return best_id, best_sim


def embed_reference_image(image_path, models_cache_dir) -> np.ndarray:
    """Detects the largest face in a standalone reference photo and returns
    its SFace embedding, for matching against video identity clusters.

    Uses a more lenient detection threshold than the main video pipeline:
    the video can afford to be selective since it has thousands of frames to
    choose from, but a single user-supplied reference photo (often a tight,
    slightly blurry crop/screenshot) has no fallback if it's rejected.
    """
    from .embeddings import SFaceEmbedder
    from .face_detection import YuNetDetector, filter_detections
    from .utils import ensure_model, imread_unicode

    img = imread_unicode(image_path)
    if img is None:
        raise NoIdentitySelectedError(f"Could not read reference image: {image_path}")
    detector = YuNetDetector(ensure_model("yunet", models_cache_dir), score_threshold=0.3)
    dets = filter_detections(detector.detect(img))
    if not dets:
        raise NoIdentitySelectedError(f"No face could be detected in the reference image: {image_path}")
    bbox, _conf, row = max(dets, key=lambda d: d[0].area)
    embedder = SFaceEmbedder(ensure_model("sface", models_cache_dir))
    return embedder.embed(img, row)


def select_target_identity(
    clusters: dict[int, list[Candidate]], cfg: AppConfig, prompt_fn=None, reference_embedding: np.ndarray | None = None
) -> int:
    """Decide which cluster is "the" target person.

    Priority: ``--person`` override > ``--reference-image`` match > one
    overwhelmingly dominant cluster > interactive prompt > (no prompt
    available) largest cluster + warning.
    """
    if not clusters:
        raise NoIdentitySelectedError("No identity clusters were found - no faces were confidently clustered.")

    infos = summarize_clusters(clusters)
    ordered_ids = [info.cluster_id for info in infos]

    if cfg.person is not None:
        idx = cfg.person - 1
        if idx < 0 or idx >= len(ordered_ids):
            raise NoIdentitySelectedError(
                f"--person {cfg.person} is out of range; found {len(ordered_ids)} identities."
            )
        chosen = ordered_ids[idx]
        logger.info("Using --person %d -> identity cluster %d (%d faces).", cfg.person, chosen, len(clusters[chosen]))
        return chosen

    if reference_embedding is not None:
        best_id, sim = match_reference_embedding(clusters, reference_embedding)
        if best_id is None:
            raise NoIdentitySelectedError("Could not match the reference image to any detected identity cluster.")
        logger.info(
            "Reference image best matches identity cluster %d (cosine similarity %.3f, %d faces).",
            best_id, sim, len(clusters[best_id]),
        )
        if sim < SFACE_COSINE_SAME_PERSON:
            logger.warning(
                "Reference-image match confidence is only %.3f (below the %.3f same-person guideline). "
                "Check contact_sheets/identity_candidate_%d.jpg to confirm this is the right person.",
                sim, SFACE_COSINE_SAME_PERSON, best_id,
            )
        return best_id

    total = sum(info.count for info in infos)
    top = infos[0]
    if total > 0 and top.count / total >= DOMINANT_IDENTITY_RATIO:
        logger.info(
            "One dominant identity found (%d/%d = %.0f%% of faces) - auto-selecting it.",
            top.count, total, 100 * top.count / total,
        )
        return top.cluster_id

    if prompt_fn is not None:
        return prompt_fn(infos)

    logger.warning(
        "Multiple identities found and no interactive selection is available; "
        "defaulting to the largest one (identity cluster %d, %d faces). Pass --person N to choose explicitly.",
        top.cluster_id, top.count,
    )
    return top.cluster_id
