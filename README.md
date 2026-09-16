# FRPM — Facial Identity Reference Pack Maker

[![Tests](https://github.com/Sadik-Dev/frpm/actions/workflows/tests.yml/badge.svg)](https://github.com/Sadik-Dev/frpm/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Turn a YouTube video of yourself into a curated, locally-processed **facial
identity and expression reference pack** for AI image-generation workflows
(FLUX, PuLID, InstantID, character sheets, LoRA training).

This is not a "grab random frames" tool. It downloads the video, samples
frames intelligently, detects and clusters faces by identity, lets you pick
*which* person is you, scores every face for quality, removes near-duplicates,
classifies head pose/expression/eye-direction, finds body references where
available, and finally selects a **diverse, high-quality** subset — then
packages everything into an organized folder with contact sheets, an HTML
report, and full JSON metadata.

Everything runs **100% locally**. No cloud APIs, no telemetry.

## Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Basic usage](#basic-usage)
- [Advanced usage](#advanced-usage)
- [Output structure](#output-structure)
- [Models & licensing](#models--licensing)
- [Resume / caching](#resume--caching)
- [Optional: local image generation](#optional-local-image-generation-frpmgenerate)
- [Synthesizing missing perspectives](#synthesizing-missing-perspectives---synthesize-missing)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## How it works

```text
YouTube URL → download (yt-dlp) → sample frames → detect faces (YuNet)
  → embed faces (SFace) → cluster identities → choose target person
  → landmarks/blendshapes (MediaPipe) → quality scoring → deduplicate
  → head pose + expression + eye direction → body visibility (MediaPipe Pose)
  → diversity-aware selection → crops → contact sheets → manifest + HTML report
```

Expensive per-face analysis (landmarks, quality, pose, expression, body,
deduplication) only runs on faces belonging to the **selected identity**, not
every bystander in the video, which keeps long/crowded videos practical.

## Installation

### 1. Python

Requires **Python 3.11+**.

- Windows: `winget install Python.Python.3.11` (or from [python.org](https://www.python.org/downloads/))
- Linux: `sudo apt install python3.11 python3.11-venv` (or your distro's equivalent)
- macOS: `brew install python@3.11`

### 2. FFmpeg

Required by both `yt-dlp` (muxing/remuxing) and OpenCV's video decoding.

- Windows: `winget install Gyan.FFmpeg` (then **restart your terminal** so PATH updates)
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

If FFmpeg genuinely can't be put on PATH, FRPM will fall back to the
`imageio-ffmpeg` package's bundled binary automatically — but a real FFmpeg
install is still recommended for best format/codec support.

### 3. Project dependencies

```bash
cd FRPM
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

This installs: `yt-dlp`, `opencv-contrib-python`, `mediapipe`, `scikit-learn`,
`ImageHash`, `pydantic`, `rich`, `tqdm`, `Jinja2`, `imageio-ffmpeg`, `pytest`.

The first pipeline run also downloads ~10MB of model weights (YuNet, SFace,
MediaPipe Face/Pose Landmarker) into `~/.cache/frpm/models/` — a one-time
download, cached for all future runs.

**No GPU/CUDA is required.** Everything runs on CPU by default; see
[Hardware / CPU vs GPU](#hardware--cpu-vs-gpu) below.

## Basic usage

```bash
python main.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
```

This downloads the video, processes it end-to-end, and writes an identity
pack to `./output/<VIDEO_ID>/`. If more than one person appears and no one
person overwhelmingly dominates the footage, you'll be prompted to pick:

```text
Found 3 people.

  #   Identity     Appearances
  1   Person 0     784
  2   Person 1     152
  3   Person 2     47

Select target identity (number):
```

Contact sheets for each candidate identity are saved to
`<output>/<video_id>/contact_sheets/identity_candidate_*.jpg` so you can open
them and see who's who before answering. If you have a clear photo of your
own face handy, `--reference-image` skips this prompt entirely by matching
identity clusters against it automatically (see [Advanced usage](#advanced-usage)).

## Advanced usage

Custom output directory:

```bash
python main.py "URL" --output ./my_packs
```

Sampling rate and range:

```bash
python main.py "URL" --fps 4 --start 30 --end 300
```

Limit how much of a very long video is processed:

```bash
python main.py "URL" --max-duration 600 --max-frames 2000
```

Skip the interactive prompt and pick a specific identity directly:

```bash
python main.py "URL" --person 1
```

Automatically pick the identity that best matches a reference photo of your
face (no prompt, no guessing which number is you):

```bash
python main.py "URL" --reference-image ./me.jpg
```

Control the size of the final curated pack (30-60) and the premium reference
pack (12-20):

```bash
python main.py "URL" --pack-size 60 --reference-pack-size 20
```

Also generate a captioned LoRA training dataset:

```bash
python main.py "URL" --lora-mode --trigger-token my_character_v1
```

Keep the downloaded source video and all sampled candidate frames instead of
cleaning them up:

```bash
python main.py "URL" --keep-video --keep-candidates
```

Skip HTML report generation, or force a clean re-run ignoring any cache:

```bash
python main.py "URL" --no-report
python main.py "URL" --force-restart
```

Run `python main.py --help` for the full flag list.

## Output structure

```text
output/
└── VIDEO_ID/
    ├── source/metadata.json        # title, uploader, duration, resolution...
    ├── candidates/                 # sampled frames (pruned after a run unless --keep-candidates)
    ├── selected/
    │   ├── originals/               # untouched source frames for each selected image
    │   ├── faces/                   # 1024x1024 head+hair+neck crops
    │   └── portraits/                # 4:5 head+shoulders+chest crops (where framing allows)
    ├── poses/                        # same face crops, grouped by head angle
    │   ├── front/ three_quarter_left/ three_quarter_right/
    │   ├── profile_left/ profile_right/ looking_up/ looking_down/
    ├── expressions/                  # same face crops, grouped by expression tag(s)
    │   ├── neutral/ slight_smile/ confident_smile/ big_smile/ laughing/
    │   ├── serious/ focused/ talking/ surprised/ ...
    ├── body/                          # body references, where visible
    │   ├── upper_body/ half_body/ full_body/
    ├── contact_sheets/
    │   ├── identity.jpg               # front / 3Q-left / 3Q-right / profile-left / profile-right
    │   ├── expressions.jpg            # expression grid
    │   ├── poses.jpg                  # all head-angle buckets
    │   └── identity_candidate_*.jpg   # one per detected identity, for the selection prompt
    ├── lora_dataset/                  # only with --lora-mode: images + matching .txt captions
    ├── synthesized/                    # only with --synthesize-missing: AI-recreated gap-filling images
    │   └── <category>/synth_00.jpg     # e.g. left_profile/, laughing/ - grouped by the missing category filled
    ├── manifest.json                  # full metadata for every selected image
    └── report.html                    # open this in a browser
```

`poses/` and `expressions/` are populated via symlinks/hardlinks back to the
files in `selected/faces/` where the OS allows it (falling back to a plain
copy otherwise), so the same image can belong to several categories without
duplicating disk space. `manifest.json` always has the authoritative paths.

Each entry in `manifest.json` looks like:

```json
{
  "file": "faces/frame_00231_a1b2c3d4.jpg",
  "source_timestamp": 48.23,
  "identity_score": 0.97,
  "quality_score": 0.93,
  "face": { "yaw": -18.2, "pitch": 3.8, "roll": 1.2, "pose": "three_quarter_left" },
  "expressions": ["slight_smile", "eyes_right"],
  "body_visibility": "upper_body"
}
```

## Models & licensing

Chosen specifically to avoid restrictive licenses while staying CPU-friendly
and easy to install on Windows:

| Component | Model | License |
|---|---|---|
| YouTube download | yt-dlp | Unlicense (public domain) |
| Face detection | OpenCV **YuNet** (opencv_zoo) | MIT |
| Face recognition / identity embeddings | OpenCV **SFace** (opencv_zoo) | Apache-2.0 |
| Landmarks, blendshapes, head-pose matrix | **MediaPipe Face Landmarker** | Apache-2.0 |
| Body pose | **MediaPipe Pose Landmarker** | Apache-2.0 |
| Clustering | scikit-learn | BSD-3-Clause |
| Perceptual hashing | ImageHash | BSD-2-Clause |
| Image generation (optional, `generate` extra) | Stable Diffusion 1.5 | CreativeML OpenRAIL-M |
| Identity conditioning (optional, `generate` extra) | IP-Adapter Plus Face | Apache-2.0 |
| Face restoration (optional, `--face-restore`) | GFPGAN | Apache-2.0 |

**InsightFace was deliberately not used**: its pretrained model-zoo weights
are licensed "non-commercial research purposes only", which conflicts with
avoiding unnecessarily restrictive licenses for personal use. **dlib** was
also avoided — not for licensing (it's fine, Boost-1.0), but because it has
no official prebuilt wheels for recent Python on Windows and requires a C++
build toolchain. All models above install from prebuilt PyPI wheels and are
downloaded automatically (from their official sources) on first run.
Likewise, **InstantID / PuLID / IP-Adapter-FaceID and CodeFormer were
considered and not used** for the same reason - see
[Optional: local image generation](#optional-local-image-generation-frpmgenerate)
and [Synthesizing missing perspectives](#synthesizing-missing-perspectives---synthesize-missing)
for details.

## Hardware / CPU vs GPU

FRPM detects available acceleration at startup and logs what it finds
(`frpm.hardware`), but the OpenCV/MediaPipe wheels used here are CPU-only
prebuilt binaries on PyPI, so the pipeline **always runs on CPU** regardless
of what's detected. This is intentional per the project's requirements — no
CUDA/GPU is assumed or required. Processing time scales primarily with
`--fps` × video duration (i.e., how many frames get sampled), not the video's
raw resolution, so use `--fps`, `--max-duration`, and `--max-frames` to keep
long videos practical.

## Resume / caching

Four checkpoints are cached under `output/<VIDEO_ID>/.cache/`:
`download` → `frame_analysis` → `identity_clustering` (includes per-face
quality/pose/expression/body/dedup for the chosen person) → final selection.

Re-running the exact same command reuses everything still valid — e.g. if
you only change `--pack-size` or `--lora-mode`, detection/clustering/scoring
is **not** redone, only the final selection/crops/report step. Pass
`--force-restart` to ignore the cache and start clean.

## Optional: local image generation (`frpm.generate`)

FRPM's main pipeline only *builds* the reference pack - the intended
workflow is to hand it to an external AI image model. If you'd rather
generate images locally, FRPM also includes an optional module using
Stable Diffusion 1.5 + IP-Adapter Plus Face, conditioned on one of your
`selected/faces/` crops:

```bash
pip install -e ".[generate]"
python -m frpm.generate --face "output/VIDEO_ID/selected/faces/frame_....jpg" \
    --prompt "photo of a person wearing a black beanie and grey hoodie, leaning against a car at night, realistic photo" \
    --output "output/VIDEO_ID/generated/result.png"
```

Notes:
* **CPU-only by design**, like the rest of FRPM - no GPU required, but a
  512x768 image takes roughly 3-5 minutes on a modern multi-core CPU (vs.
  seconds on a GPU). A GPU-enabled `torch` install is used automatically if present.
* **Why not InstantID/PuLID/IP-Adapter-FaceID?** Those give a tighter
  identity lock, but all three depend on InsightFace's face-recognition
  backbone, whose pretrained weights are licensed "non-commercial research
  only" - the same restriction FRPM's core face-embedding model (SFace) was
  deliberately chosen to avoid. IP-Adapter Plus Face uses a CLIP image
  encoder instead (Apache-2.0, no such restriction), at the cost of a looser
  identity match.
* Keep prompts under ~77 tokens (CLIP's limit for SD1.5) and put the most
  important scene details first - text past that limit is silently dropped.
* Tune `--ip-adapter-scale` (default 0.7): higher locks identity more
  tightly but can suppress prompt adherence for scene/clothing/lighting
  details not present in the face crop; lower gives the text prompt more
  control at some cost to identity fidelity.

## Synthesizing missing perspectives (`--synthesize-missing`)

Real footage doesn't always capture every useful angle - a talking-head
video might never show a clean profile shot, or never catch the person
laughing. `--synthesize-missing` fills those specific gaps by recreating a
realistic image for each pose/expression category (from the same set the
diversity selector targets - see [Diversity selection](#how-it-works))
that has no good real match:

```bash
pip install -e ".[generate]"
python main.py "URL" --synthesize-missing --synthesis-count 1 --face-restore
```

How it works: for each missing category, the best *real* photo of the
identified person is picked as an img2img base (preferring the real pose
closest to the target - e.g. a real three-quarter shot as the base for a
missing profile shot) and processed with Stable Diffusion 1.5 img2img +
IP-Adapter Plus Face, seeded from that same photo for identity. Starting
from a real photo (rather than pure noise) keeps skin tone, hair, and
lighting far more consistent than plain text-to-image. An optional GFPGAN
(Apache-2.0) face-restoration pass (`--face-restore`) sharpens the result
further.

Output goes to `synthesized/<category>/` and a separate
`synthesized_images` list in `manifest.json` - **never mixed into**
`images`/`reference_pack`/`lora_dataset`, and clearly labeled "AI-recreated"
in `report.html`. Real and synthesized content should never be
indistinguishable in the output. Base images are resized to 512px on their
longer side before synthesis, matching SD1.5's native training resolution
(see the note below on why).

Flags:
* `--synthesis-count N` - images per missing category (default: 1)
* `--synthesis-strength F` - base img2img denoising strength, 0-1 (default:
  0.6; automatically boosted per-category for larger pose changes - see
  limitation note below - and capped at 0.75)
* `--synthesis-max-categories N` - cap on how many missing categories to
  synthesize per run, to bound CPU time on videos with many gaps (default: 6)
* `--face-restore` - apply the optional GFPGAN pass

**Working resolution note.** FRPM's face crops are 1024x1024, but SD1.5
was trained at 512x512. Running img2img directly at 1024x1024 was tested
on a real identity pack and verified to produce a visible duplicate/ghost
face artifact at the frame edge (a well-known SD1.5 failure mode above its
native resolution) *and* ~5x slower CPU inference for no quality benefit.
Base images are therefore always downscaled to 512px (longer side) before
synthesis - fixing both the artifact and the slowdown in that same test.

**Known limitation - pose changes are a nudge, not a guarantee.** This
uses text-prompt guidance only, with no pose/skeleton conditioning
(ControlNet). During development this was verified empirically: a large
pose change (e.g. recreating a profile view from a front-facing photo)
only partially rotates the face at safe strength settings, and pushing
strength higher to force it further was also verified to make the *base
SD1.5 model's own training biases* dominate over both the source photo and
the identity conditioning - in one test this produced content completely
unrelated to the input photo. `_STRENGTH_ADJUSTMENT` and
`MAX_SYNTHESIS_STRENGTH` in `src/frpm/synthesis.py` are deliberately capped
conservatively as a result. Precise pose control would require ControlNet
with pose/landmark conditioning, which is not currently implemented (see
`src/frpm/synthesis.py`'s module docstring for why, and consider it a
natural next contribution).

**GFPGAN compatibility note:** GFPGAN's dependency `basicsr` imports a
module that modern `torchvision` versions removed. FRPM applies a small,
inert-if-unneeded compatibility shim automatically (see
`_patch_torchvision_functional_tensor` in `src/frpm/synthesis.py`) so
`--face-restore` works without needing to pin an old torchvision.
**CodeFormer** was considered as an alternative restoration model but is
licensed "non-commercial purposes only" (S-Lab License 1.0) and was
therefore not used, consistent with FRPM's licensing stance throughout.

## Testing

```bash
python -m pytest tests/unit          # fast, synthetic data, no models/network - seconds
python -m pytest tests/integration   # slow, real models + a tiny real video - opt-in only
```

`tests/integration` is excluded by default (see `pyproject.toml`'s
`addopts = "-m 'not slow'"`) since it downloads real model weights and needs
network access. Unit tests cover quality scoring, duplicate detection,
head-pose classification/round-trip, diversity selection, manifest/report
metadata, and YouTube URL validation.

## Troubleshooting

**"FFmpeg was not found on PATH"**
Install FFmpeg (see [Installation](#installation)) and open a *new* terminal
— PATH changes from an installer don't apply to already-open shells. On
Windows, `winget install Gyan.FFmpeg` explicitly warns about this.

**yt-dlp errors: "Video unavailable" / "Private video" / age-restricted**
FRPM surfaces yt-dlp's error with a clear message. Private, deleted, or
region-locked videos can't be downloaded by any tool; age-restricted videos
generally require being logged in, which this tool intentionally does not do
(no credential handling). Try a different video, or download it yourself and
point a future version at a local file.

**CUDA / GPU**
Not required. See [Hardware / CPU vs GPU](#hardware--cpu-vs-gpu) above — the
pipeline is CPU-only by design and this is expected, not an error.

**Very long videos (1h+)**
Lower `--fps` (e.g. `--fps 1`), set `--max-duration`/`--max-frames`, or use
`--start`/`--end` to focus on the segment that actually shows you clearly.
The sampler automatically increases its stride to respect `--max-frames`
regardless of video length, so it will never explode into millions of frames.

**Multiple people in the video**
This is handled automatically via identity clustering — see
[Basic usage](#basic-usage). Use `--person N` to skip the prompt once you
know which number is you (numbers are stable for a given run based on
descending face count).

**"No faces were detected" / "none could be confidently clustered"**
Try a lower `--fps` floor is unlikely to help here since faces are checked on
every sampled frame already; instead verify the video actually shows a clear,
reasonably-sized face, or widen `--start`/`--end` to cover more of the video.

**SSL/certificate errors when downloading models (`CERTIFICATE_VERIFY_FAILED`)**
Some managed/corporate Windows machines intercept HTTPS with a custom root
CA that's trusted by the OS but not by Python's bundled `certifi` store.
FRPM already depends on `truststore` and enables it automatically (making
Python verify against the OS trust store instead), which fixes this in the
vast majority of cases with no action needed. If it still happens, make sure
`truststore` installed correctly (`pip show truststore`).

**Output looks like it's missing portrait/body crops for some images**
This is intentional: FRPM never fabricates missing body area. If the source
framing is a tight close-up, only the face crop is produced for that image.

## Privacy

All face detection, embedding, clustering, and analysis happens locally on
your machine. The only network calls made are yt-dlp downloading the video
you asked for, and a one-time download of the (Apache-2.0) model weights
listed above. No images, embeddings, or metadata are ever sent anywhere else.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup, project structure, and guidelines (including our
license-discipline rule for any new face-recognition model). Please also
review our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

FRPM's own code is licensed under the [MIT License](LICENSE). This covers
the tool itself only — the pretrained models it downloads and uses at
runtime (YuNet, SFace, MediaPipe Face/Pose Landmarker, and optionally
Stable Diffusion 1.5 + IP-Adapter) have their own separate licenses; see
[Models & licensing](#models--licensing) above.
