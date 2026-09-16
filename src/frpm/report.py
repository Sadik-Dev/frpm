"""``manifest.json`` + ``report.html`` generation."""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from jinja2 import BaseLoader, Environment, select_autoescape

from .models import Manifest, SelectedImageRecord, SyntheticImageRecord, VideoMetadata
from .utils import ensure_dir, write_json

logger = logging.getLogger("frpm.report")


def build_manifest(
    video: VideoMetadata,
    identity_cluster_id: int,
    frames_examined: int,
    faces_found: int,
    identities_found: int,
    candidates_after_dedup: int,
    pack_size_requested: int,
    images: list[SelectedImageRecord],
    reference_pack: list[str],
    lora_dataset: list[str],
    synthesized_images: Optional[list[SyntheticImageRecord]] = None,
) -> Manifest:
    quality_dist: Counter = Counter()
    for img in images:
        lo = min(90, int(img.quality_score * 10) * 10)
        quality_dist[f"{lo}-{lo + 10}%"] += 1
    expr_dist = Counter(tag for img in images for tag in img.expressions)
    pose_dist = Counter(img.face.pose.value for img in images)

    return Manifest(
        video=video,
        identity_cluster_id=identity_cluster_id,
        frames_examined=frames_examined,
        faces_found=faces_found,
        identities_found=identities_found,
        candidates_after_dedup=candidates_after_dedup,
        pack_size_requested=pack_size_requested,
        images=images,
        reference_pack=reference_pack,
        lora_dataset=lora_dataset,
        synthesized_images=synthesized_images or [],
        quality_distribution=dict(sorted(quality_dist.items())),
        expression_distribution=dict(expr_dist.most_common()),
        pose_distribution=dict(pose_dist.most_common()),
    )


def write_manifest(manifest: Manifest, out_dir: Path) -> Path:
    path = out_dir / "manifest.json"
    write_json(path, manifest.model_dump())
    return path


_REPORT_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ m.video.title or m.video.video_id }} - Identity Reference Pack</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { background:#111317; color:#e8e8e8; font-family: 'Segoe UI', Arial, sans-serif; margin:0; padding:0 28px 56px; }
  h1 { font-weight:600; margin-top:32px; }
  .meta { color:#9aa0a6; margin-bottom: 24px; font-size:14px; }
  .meta a { color:#6cb4ff; }
  .stats { display:flex; flex-wrap:wrap; gap:14px; margin: 24px 0; }
  .stat { background:#1b1e24; border-radius:10px; padding:14px 18px; min-width:130px; }
  .stat .num { font-size:26px; font-weight:700; }
  .stat .label { color:#9aa0a6; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
  .dist { display:flex; flex-wrap:wrap; gap:8px; margin: 6px 0 22px; }
  .pill { background:#22262e; border-radius:999px; padding:5px 13px; font-size:12.5px; }
  .pill b { color:#fff; }
  section { margin: 44px 0; }
  h2 { border-bottom: 1px solid #2a2e35; padding-bottom:10px; font-weight:600; }
  .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap:14px; }
  figure { margin:0; background:#1b1e24; border-radius:10px; overflow:hidden; }
  figure img { width:100%; display:block; aspect-ratio:1/1; object-fit:cover; background:#000; }
  figcaption { padding:8px 10px; font-size:11.5px; color:#b7bcc4; }
  .sheet img { max-width:100%; border-radius:10px; margin: 10px 0; display:block; }
  .empty { color:#777; font-style: italic; }
  .synthetic-badge { display:inline-block; background:#5b3aa8; color:#fff; font-size:10.5px; font-weight:700; padding:2px 8px; border-radius:999px; margin-bottom:6px; letter-spacing:.03em; }
  .synthetic-note { color:#b39ddb; font-size:13px; margin-top:-8px; margin-bottom:18px; }
  footer { color:#666; margin-top:56px; font-size:12px; }
</style>
</head>
<body>
  <h1>{{ m.video.title or m.video.video_id }}</h1>
  <div class="meta">
    Source: <a href="{{ m.video.url }}">{{ m.video.url }}</a><br>
    Uploader: {{ m.video.uploader or "unknown" }} &middot;
    Duration: {{ "%.0f"|format(m.video.duration) }}s &middot;
    {{ m.video.width }}x{{ m.video.height }}
  </div>

  <div class="stats">
    <div class="stat"><div class="num">{{ m.frames_examined }}</div><div class="label">Frames examined</div></div>
    <div class="stat"><div class="num">{{ m.faces_found }}</div><div class="label">Faces found</div></div>
    <div class="stat"><div class="num">{{ m.identities_found }}</div><div class="label">Identities found</div></div>
    <div class="stat"><div class="num">#{{ m.identity_cluster_id }}</div><div class="label">Selected identity</div></div>
    <div class="stat"><div class="num">{{ m.candidates_after_dedup }}</div><div class="label">After dedup</div></div>
    <div class="stat"><div class="num">{{ m.images|length }}</div><div class="label">Final images</div></div>
    <div class="stat"><div class="num">{{ m.reference_pack|length }}</div><div class="label">Reference pack</div></div>
    {% if m.synthesized_images %}<div class="stat"><div class="num">{{ m.synthesized_images|length }}</div><div class="label">Synthesized</div></div>{% endif %}
  </div>

  <section>
    <h2>Distributions</h2>
    <div><strong>Quality</strong>
      <div class="dist">{% for k, v in m.quality_distribution.items() %}<span class="pill">{{ k }}: <b>{{ v }}</b></span>{% endfor %}</div>
    </div>
    <div><strong>Expressions</strong>
      <div class="dist">{% for k, v in m.expression_distribution.items() %}<span class="pill">{{ k }}: <b>{{ v }}</b></span>{% endfor %}</div>
    </div>
    <div><strong>Head angles</strong>
      <div class="dist">{% for k, v in m.pose_distribution.items() %}<span class="pill">{{ k }}: <b>{{ v }}</b></span>{% endfor %}</div>
    </div>
  </section>

  {% if contact_sheets.identity %}
  <section class="sheet"><h2>Identity Contact Sheet</h2><img src="{{ contact_sheets.identity }}" alt="Identity contact sheet"></section>
  {% endif %}
  {% if contact_sheets.expressions %}
  <section class="sheet"><h2>Expression Contact Sheet</h2><img src="{{ contact_sheets.expressions }}" alt="Expression contact sheet"></section>
  {% endif %}
  {% if contact_sheets.poses %}
  <section class="sheet"><h2>Pose Contact Sheet</h2><img src="{{ contact_sheets.poses }}" alt="Pose contact sheet"></section>
  {% endif %}

  <section>
    <h2>Best Identity References</h2>
    <div class="grid">
      {% for f in m.reference_pack %}<figure><img src="{{ f }}" loading="lazy"></figure>{% else %}<p class="empty">No reference pack images.</p>{% endfor %}
    </div>
  </section>

  <section>
    <h2>Expressions</h2>
    <div class="grid">
      {% for img in m.images %}<figure><img src="{{ img.face_crop or img.original }}" loading="lazy"><figcaption>{{ img.expressions|join(", ") }}</figcaption></figure>{% endfor %}
    </div>
  </section>

  <section>
    <h2>Head Angles</h2>
    <div class="grid">
      {% for img in m.images %}<figure><img src="{{ img.face_crop or img.original }}" loading="lazy"><figcaption>{{ img.face.pose.value if img.face.pose.value else img.face.pose }} (yaw {{ "%.0f"|format(img.face.yaw) }}&deg;)</figcaption></figure>{% endfor %}
    </div>
  </section>

  <section>
    <h2>Body References</h2>
    <div class="grid">
      {% for img in m.images if img.body_visibility != "none" %}<figure><img src="{{ img.original }}" loading="lazy"><figcaption>{{ img.body_visibility }}</figcaption></figure>{% else %}<p class="empty">No usable body references were found in this footage (close-up only, or framing too tight).</p>{% endfor %}
    </div>
  </section>

  {% if m.synthesized_images %}
  <section>
    <h2>Synthesized Perspectives</h2>
    <div class="synthetic-note">AI-recreated to fill pose/expression gaps the source video didn't capture - not extracted from the original footage. Each image is seeded from a real photo of this identity (shown as its source).</div>
    <div class="grid">
      {% for img in m.synthesized_images %}<figure><span class="synthetic-badge" style="margin:8px 8px 0">AI-GENERATED</span><img src="{{ img.file }}" loading="lazy"><figcaption>{{ img.category.replace("_", " ") }} &middot; based on {{ img.source_real_image }}</figcaption></figure>{% endfor %}
    </div>
  </section>
  {% endif %}

  <footer>Generated entirely locally by FRPM - no images or data left this machine.</footer>
</body>
</html>
"""


def render_report(manifest: Manifest, contact_sheets: dict[str, str], out_dir: Path) -> Path:
    env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
    template = env.from_string(_REPORT_TEMPLATE)
    html = template.render(m=manifest, contact_sheets=contact_sheets)
    out_path = out_dir / "report.html"
    ensure_dir(out_dir)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_lora_dataset(
    images: list[SelectedImageRecord], captions: dict[str, str], out_dir: Path, source_paths: dict[str, str]
) -> list[str]:
    import shutil

    lora_dir = ensure_dir(out_dir / "lora_dataset")
    written = []
    for img in images:
        src = source_paths.get(img.file)
        if not src:
            continue
        src_path = Path(src)
        if not src_path.exists():
            continue
        dest_img = lora_dir / src_path.name
        shutil.copy2(src_path, dest_img)
        caption = captions.get(img.file, "")
        (lora_dir / (dest_img.stem + ".txt")).write_text(caption, encoding="utf-8")
        written.append(str(dest_img))
    return written
