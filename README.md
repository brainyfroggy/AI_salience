# AI_salience

## Purpose

Code for testing whether vision-language models (VLMs) can predict human
visual saliency / fixation patterns. Models (Qwen3-VL, InternVL3.5,
Llama-3.2-11B/90B-Vision, and GPT-5.x) are prompted to directly output a
fixation-density model for a given image — either as a mixture of Gaussian
components or as a coarse probability grid — which is then rendered as a
grayscale saliency map, color heatmap, and image overlay for comparison
against ground-truth human eye-tracking data. Stimuli are drawn from the
EMOd image set (a copyright-restricted research stimulus set derived in
part from IAPS); the images themselves are not distributed here, only the
code that operates on them. The repo also includes HPC (HiPerGator/Slurm)
job-submission tooling for running the open-weight models on GPU nodes.

This is a research-code snapshot (not a packaged library) — scripts assume a
particular local/HPC directory layout (see path defaults inside each script)
and are shared for transparency and methods reference rather than one-command
reuse.

## Contents

- `scripts/run_vlm_prompt_saliency.py` — Prompts an open-weight VLM (served
  locally or on HPC) to output a saliency map as a mixture of 3-10 elliptical
  Gaussian fixation components grounded in named image features; renders the
  numeric density, grayscale map, heatmap, and overlay. Explicitly checks
  requested model IDs against a list of restricted-license model families
  (Qwen, InternVL, DeepSeek, etc.) before running.
- `scripts/query_chatgpt_saliency.py` — Same idea via the OpenAI API
  (GPT-5.x): asks for a 16x16 saliency grid per image, upsamples it, and
  overlays it on the original image.
- `scripts/prepare_emod_complete_raw.py` — Builds a derived, complete raw-image
  folder for the EMOd stimulus set (EMOd is split across author-collected and
  IAPS-sourced images, and a few IAPS raw files are backfilled from a separate
  IAPS copy) without modifying the original source folders.
- `hpg/setup_vlm_env.sbatch`, `hpg/vlm_prompt_saliency.sbatch`,
  `hpg/submit_vlm_prompt_saliency.sh` — Slurm environment-setup and array-job
  submission scripts for running the open-weight VLM saliency pipeline on
  HiPerGator GPU nodes.
- `high_resolution_fmri_beta_tools_meta_analysis.md` — A short rapid
  meta-analysis (literature write-up, public citations only) of which
  beta-estimation software (SPM, FSL, AFNI, etc.) high-resolution task-fMRI
  studies commonly use. This is an unrelated side-note, not part of the
  saliency pipeline; it is included only because it originated in the same
  working folder.

## How to Use

**Expected input.** None of the scripts ship with stimulus images. You need
your own copy of the EMOd image set (and, for the IAPS-derived subset, an
IAPS copy) laid out under a local `data/` folder — see the `--emod-root` /
`--kim-root` defaults in `prepare_emod_complete_raw.py` and the
`--image-dir` default in `query_chatgpt_saliency.py` for the expected
directory shape. No data, model weights, or API keys are included in this
repo; you must supply your own.

**Run order:**

1. *(Optional, only for `run_vlm_prompt_saliency.py`)* Run
   `scripts/prepare_emod_complete_raw.py` to assemble a complete, unmodified
   EMOd raw-image folder if your local EMOd copy is missing a few
   IAPS-sourced files.
2. Build a manifest listing the images to process. `run_vlm_prompt_saliency.py`
   requires a tab-separated `--manifest` file with `item_id` and `image_path`
   columns (there is no manifest-building script in this repo — the original
   one-off HPC manifest helper was removed as unfinished/non-portable; write
   your own small script or spreadsheet export to produce this TSV).
   `query_chatgpt_saliency.py` does not need a manifest — it walks
   `--image-dir` directly.
3. Run one of:
   - `python scripts/run_vlm_prompt_saliency.py --manifest <manifest.tsv> --output-dir <out> --model-id <hf-model-id>`
     for an open-weight VLM (loaded locally via `transformers`, or on an HPC
     GPU node). Refuses restricted-license model families by name (Qwen,
     InternVL, DeepSeek, etc.) — see `RESTRICTED_MODEL_PATTERNS` in the
     script.
   - `python scripts/query_chatgpt_saliency.py --image-dir <dir> --out-dir <out> --models gpt-5.4`
     for GPT-5.x via the OpenAI API (or a NaviGator-style proxy).
   - On HiPerGator: submit `hpg/setup_vlm_env.sbatch` once to build the
     conda/venv environment, then use `hpg/submit_vlm_prompt_saliency.sh`
     (which submits `hpg/vlm_prompt_saliency.sbatch` as an array job) to run
     `run_vlm_prompt_saliency.py` at scale across GPU nodes.
4. Each script writes, per image: the raw model response/parameters, the
   numeric saliency density, a grayscale saliency map, a color heatmap, and
   an image overlay, to the given `--output-dir` / `--out-dir`.

`high_resolution_fmri_beta_tools_meta_analysis.md` is a standalone literature
note — just open and read it; it has no code to run.

## Dependencies

- Python 3.10+
- `numpy`, `pillow` (used by all three scripts)
- `torch`, `transformers` (imported lazily inside
  `run_vlm_prompt_saliency.py` to load open-weight VLMs — not needed for the
  other two scripts)
- `query_chatgpt_saliency.py` uses only the Python standard library
  (`urllib`) to call the OpenAI-compatible API, so it does not require the
  `openai` package; it does need an `OPENAI_API_KEY` (and, for a proxy such
  as UF's NaviGator, `OPENAI_BASE_URL`) environment variable, or the
  equivalent `GPT_NAVIGATOR_API_KEY` / `GPT_NAVIGATOR_BASE_URL` pair
- `prepare_emod_complete_raw.py` uses only the standard library
- For the HPC scripts: a Slurm cluster (developed against HiPerGator) with a
  GPU-enabled Python environment set up per `hpg/setup_vlm_env.sbatch`

## Notes / caveats

- No stimulus images, model outputs, manifests, or HPC log files are included
  in this repo — only the code. The EMOd/IAPS image set referenced by these
  scripts is copyright-restricted and must be obtained separately by anyone
  wishing to reproduce the experiments.
- Several scripts hard-code local/HPC paths (e.g. `N:\Experimental_Data\...`,
  `/blue/mzding/...`) as defaults — these reflect the original compute
  environment and will need to be adapted to run elsewhere.
