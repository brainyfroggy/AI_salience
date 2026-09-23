"""Run open-weight VLM prompt-elicited human saliency prediction.

The model directly specifies a continuous fixation-density map as a mixture of
image-grounded Gaussian components. This is a behavioral VLM prediction, not
an internal attention map. The script saves the model response, map parameters,
numeric density, grayscale saliency map, color heatmap, and image overlay.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RESTRICTED_MODEL_PATTERNS = (
    "qwen",
    "internvl",
    "opengvlab",
    "deepseek",
    "chatglm",
    "glm-",
    "minicpm",
    "yi-vl",
    "cogvlm",
    "internlm",
    "baichuan",
    "sensetime",
)

SYSTEM_PROMPT = """You are a computational vision researcher who predicts
human eye-fixation probability during the first 3 seconds of task-free viewing.
You must inspect the supplied image and return an image-grounded spatial
probability-density model. Return valid JSON only."""

USER_PROMPT = """Predict the human visual saliency map for the supplied image.

Definition: a human visual saliency map is a continuous two-dimensional
probability density over image locations. A high value means that many human
observers are likely to fixate there during the first 3 seconds of free viewing;
a low value means that the location is likely to be ignored. It is not an object
segmentation map, aesthetic score, scene description, or list of every object.

Represent the saliency map directly as a mixture of 3 to 10 elliptical Gaussian
fixation components. Each component is one localized peak in the final density:
- x and y are the peak center in normalized image coordinates. (0,0) is the
  top-left and (1,1) is the bottom-right.
- weight is the component's share of total fixation probability. All weights
  must sum to approximately 1.0.
- sigma_x and sigma_y are the horizontal and vertical standard deviations in
  normalized image units. Use 0.025 to 0.20. Small details such as eyes or text
  need smaller values; larger objects or diffuse regions need larger values.
- label names the visible image feature at the peak so grounding can be checked.

Base the peaks on the actual image. Faces and eyes usually dominate, followed by
people, animals, readable text, emotionally meaningful or unusual objects, and
strongly contrasting object centers. Use scene context and composition only when
they plausibly guide fixation. Do not create a linear gradient, corner preference,
regular lattice, repeated template, or generic center-only map. Suppress sky,
walls, foliage, floor, and other uniform background unless visually distinctive.

Return only JSON with this exact structure:
{
  "map_type": "human_free_viewing_fixation_density",
  "fixation_components": [
    {
      "label": "visible feature at this location",
      "x": 0.50,
      "y": 0.50,
      "weight": 0.40,
      "sigma_x": 0.07,
      "sigma_y": 0.07
    }
  ]
}

Do not return a numeric grid, markdown, explanation, or text outside the JSON.
"""

LLAMA_SYSTEM_PROMPT = """You are a computer-vision research model that returns
JSON for a benign academic visual-attention localization task. Do not describe,
classify, judge, or discuss image content. Return valid JSON only."""

LLAMA_USER_PROMPT = """Return an abstract visual saliency map for the supplied
image as a continuous two-dimensional attention probability density. This is a
neutral computer-vision coordinate-localization task: mark where gaze would most
likely fall in the image, not what the image means. Do not describe sensitive
content, do not make judgments, and do not include prose.

Represent the map as 3 to 8 elliptical Gaussian attention peaks:
- x and y are normalized image coordinates, with (0,0) at top-left and (1,1)
  at bottom-right.
- weight is the peak's share of total attention probability.
- sigma_x and sigma_y are normalized standard deviations from 0.025 to 0.20.
- label should be a neutral identifier such as "peak_1", "peak_2", etc.

Choose peaks from the actual visible structure of the image. Avoid linear
gradients, corner templates, regular lattices, and uniform background peaks.

Return only JSON with this exact structure:
{
  "map_type": "human_free_viewing_fixation_density",
  "fixation_components": [
    {
      "label": "peak_1",
      "x": 0.50,
      "y": 0.50,
      "weight": 0.40,
      "sigma_x": 0.07,
      "sigma_y": 0.07
    }
  ]
}
"""

LLAMA_MINIMAL_USER_PROMPT = """Return JSON only. This is a neutral image
coordinate task. Pick 3 to 8 localized gaze-density peaks from visible image
structure. Do not describe the image. Use neutral labels peak_1, peak_2, etc.
Use normalized coordinates: x=0 left, x=1 right, y=0 top, y=1 bottom. Weights
are positive and sum to 1. sigma_x and sigma_y are between 0.025 and 0.20.

Required JSON:
{
  "map_type": "human_free_viewing_fixation_density",
  "fixation_components": [
    {
      "label": "peak_1",
      "x": 0.50,
      "y": 0.50,
      "weight": 0.40,
      "sigma_x": 0.07,
      "sigma_y": 0.07
    }
  ]
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16"])
    return parser.parse_args()


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def reject_restricted_model(model_id: str) -> None:
    lowered = model_id.lower()
    for pattern in RESTRICTED_MODEL_PATTERNS:
        if pattern in lowered:
            raise ValueError(f"Refusing restricted model identifier: {model_id}")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    for row in rows:
        if "item_id" not in row or "image_path" not in row:
            raise ValueError("Manifest must contain item_id and image_path columns.")
    return rows


def shard_rows(rows: list[dict[str, str]], shard_index: int, num_shards: int) -> list[dict[str, str]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]


def turbo_like_colormap(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return np.dstack([red, green, blue])


def save_grid_csv(path: Path, grid: np.ndarray) -> None:
    rows = [",".join(f"{float(value):.8f}" for value in row) for row in grid]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def render_outputs(
    image_path: Path,
    grid: np.ndarray,
    components: list[dict[str, Any]],
    out_dir: Path,
    overlay_alpha: float = 0.55,
) -> None:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    max_value = float(grid.max()) if grid.size else 0.0
    display_grid = grid / max_value if max_value > 0 else grid

    grid_img = Image.fromarray(np.uint8(np.clip(display_grid, 0, 1) * 255), mode="L")
    heat_l = grid_img.resize((width, height), resample=Image.Resampling.BICUBIC)
    heat = np.asarray(heat_l, dtype=np.float32) / 255.0

    heat_img = Image.fromarray(np.uint8(turbo_like_colormap(heat) * 255), mode="RGB")
    alpha = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255 * overlay_alpha), mode="L")
    overlay = Image.composite(heat_img, image, alpha)
    overlay = ImageEnhance.Contrast(overlay).enhance(1.05)

    heat_l.save(out_dir / "saliency_map.png")
    heat_img.save(out_dir / "heatmap.png")
    overlay.save(out_dir / "overlay.png")
    np.save(out_dir / "saliency_density.npy", grid)
    save_grid_csv(out_dir / "saliency_density.csv", grid)

    component_vis = image.copy()
    component_draw = ImageDraw.Draw(component_vis)
    for component in components:
        x = float(component["x"]) * width
        y = float(component["y"]) * height
        rx = float(component["sigma_x"]) * width * 2.0
        ry = float(component["sigma_y"]) * height * 2.0
        component_draw.ellipse(
            [x - rx, y - ry, x + rx, y + ry],
            outline=(255, 40, 40),
            width=max(2, round(min(width, height) / 250)),
        )
        component_draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 255))
    component_vis.save(out_dir / "component_locations.png")

    cell = 28
    colors = np.uint8(turbo_like_colormap(display_grid) * 255)
    grid_vis = Image.new("RGB", (grid.shape[1] * cell, grid.shape[0] * cell), "white")
    draw = ImageDraw.Draw(grid_vis)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            x0 = col * cell
            y0 = row * cell
            draw.rectangle(
                [x0, y0, x0 + cell - 1, y0 + cell - 1],
                fill=tuple(int(c) for c in colors[row, col]),
            )
    grid_vis.save(out_dir / "density_grid.png")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def parse_components(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = extract_json_object(text)
    if data.get("map_type") != "human_free_viewing_fixation_density":
        raise ValueError("Response did not identify a human free-viewing fixation density.")
    raw_components = data.get("fixation_components")
    if not isinstance(raw_components, list) or not 3 <= len(raw_components) <= 10:
        raise ValueError("Expected 3 to 10 fixation_components.")

    components: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_components):
        if not isinstance(raw, dict):
            raise ValueError(f"Component {index} is not an object.")
        label = str(raw.get("label", "")).strip()
        if not label:
            raise ValueError(f"Component {index} has no visible-feature label.")
        try:
            x = float(raw["x"])
            y = float(raw["y"])
            weight = float(raw["weight"])
            sigma_x = float(raw["sigma_x"])
            sigma_y = float(raw["sigma_y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Component {index} has invalid numeric fields.") from exc
        values = np.asarray([x, y, weight, sigma_x, sigma_y], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Component {index} contains a non-finite value.")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"Component {index} center is outside the image.")
        if weight <= 0.0:
            raise ValueError(f"Component {index} weight must be positive.")
        if not (0.025 <= sigma_x <= 0.20 and 0.025 <= sigma_y <= 0.20):
            raise ValueError(f"Component {index} sigma must be between 0.025 and 0.20.")
        components.append(
            {
                "label": label,
                "x": x,
                "y": y,
                "weight": weight,
                "sigma_x": sigma_x,
                "sigma_y": sigma_y,
            }
        )

    weight_sum = sum(component["weight"] for component in components)
    if weight_sum <= 0.0:
        raise ValueError("Component weights sum to zero.")
    for component in components:
        component["weight"] /= weight_sum
    return components, data


def components_to_density(components: list[dict[str, Any]], grid_size: int) -> np.ndarray:
    if grid_size < 32:
        raise ValueError("--grid-size must be at least 32 for component rendering.")
    axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    density = np.zeros((grid_size, grid_size), dtype=np.float64)
    for component in components:
        dx = (xx - component["x"]) / component["sigma_x"]
        dy = (yy - component["y"]) / component["sigma_y"]
        gaussian = np.exp(-0.5 * (dx * dx + dy * dy))
        gaussian_sum = float(gaussian.sum())
        if gaussian_sum <= 0.0:
            raise ValueError(f"Degenerate component: {component['label']}")
        density += component["weight"] * gaussian / gaussian_sum
    density_sum = float(density.sum())
    if density_sum <= 0.0 or not np.isfinite(density).all():
        raise ValueError("Generated density is invalid.")
    return (density / density_sum).astype(np.float32)


def torch_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    return torch.bfloat16


def load_model(model_id: str, dtype_name: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    reject_restricted_model(model_id)
    dtype = torch_dtype(dtype_name)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)

    model.eval()
    torch.set_grad_enabled(False)
    return processor, model


def move_inputs(inputs: Any, model: Any) -> Any:
    try:
        import torch

        device = getattr(model, "device", None)
        if device is None or str(device) == "meta":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return inputs.to(device)
    except Exception:
        return inputs


def prompt_for_model(model_id: str, retry_index: int = 0) -> tuple[str, str]:
    if "Llama-3.2" in model_id or "Mllama" in model_id:
        if retry_index > 0:
            return LLAMA_SYSTEM_PROMPT, LLAMA_MINIMAL_USER_PROMPT
        return LLAMA_SYSTEM_PROMPT, LLAMA_USER_PROMPT
    return SYSTEM_PROMPT, USER_PROMPT


def build_messages(
    image_path: Path,
    variant: int,
    model_id: str,
    retry_index: int = 0,
) -> list[dict[str, Any]]:
    system_prompt, user_prompt = prompt_for_model(model_id, retry_index)
    image_text = f"image_id: {image_path.stem}\n\n{user_prompt}"
    if variant == 0:
        image_content: dict[str, Any] = {"type": "image", "image": str(image_path)}
    elif variant == 1:
        image_content = {"type": "image", "url": str(image_path)}
    else:
        image_content = {"type": "image", "image": Image.open(image_path).convert("RGB")}
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [image_content, {"type": "text", "text": image_text}]},
    ]


def generate_text(
    processor: Any,
    model: Any,
    image_path: Path,
    max_new_tokens: int,
    retry_index: int = 0,
) -> str:
    last_error: Exception | None = None
    for variant in range(3):
        try:
            messages = build_messages(
                image_path,
                variant,
                getattr(model, "name_or_path", ""),
                retry_index=retry_index,
            )
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = move_inputs(inputs, model)
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            input_len = inputs["input_ids"].shape[1]
            trimmed = generated[:, input_len:]
            decoded = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return decoded[0].strip()
        except Exception as exc:
            last_error = exc
            print(f"[warn] generation variant {variant} failed for {image_path}: {exc}", flush=True)
    raise RuntimeError(f"All generation variants failed for {image_path}: {last_error}")


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_label = safe_label(args.model_id)
    model_out = args.output_dir / model_label
    model_out.mkdir(parents=True, exist_ok=True)

    rows = shard_rows(read_manifest(args.manifest), args.shard_index, args.num_shards)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print("[done] no rows for this shard", flush=True)
        return 0

    print(f"[load] {args.model_id}", flush=True)
    processor, model = load_model(args.model_id, args.torch_dtype)
    print(f"[run] {len(rows)} images; shard {args.shard_index}/{args.num_shards}", flush=True)

    summary: list[dict[str, Any]] = []
    for row in rows:
        item_id = row["item_id"]
        image_path = Path(row["image_path"])
        out_dir = model_out / item_id
        done_file = out_dir / "done.json"
        if done_file.exists() and not args.overwrite:
            print(f"[skip] {item_id}", flush=True)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        status: dict[str, Any] = {
            "item_id": item_id,
            "image_path": str(image_path),
            "model_id": args.model_id,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "status": "started",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_status(out_dir / "status.json", status)
        try:
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            text = generate_text(processor, model, image_path, args.max_new_tokens)
            (out_dir / "response_text.txt").write_text(text, encoding="utf-8")
            try:
                components, parsed = parse_components(text)
                used_retry_prompt = False
            except Exception as first_parse_exc:
                if "Llama-3.2" not in args.model_id and "Mllama" not in args.model_id:
                    raise
                (out_dir / "response_text_first_attempt.txt").write_text(text, encoding="utf-8")
                text = generate_text(
                    processor,
                    model,
                    image_path,
                    args.max_new_tokens,
                    retry_index=1,
                )
                (out_dir / "response_text.txt").write_text(text, encoding="utf-8")
                try:
                    components, parsed = parse_components(text)
                    used_retry_prompt = True
                except Exception as second_parse_exc:
                    raise ValueError(
                        f"First parse failed: {first_parse_exc!r}; "
                        f"retry parse failed: {second_parse_exc!r}"
                    ) from second_parse_exc
            grid = components_to_density(components, args.grid_size)
            (out_dir / "response_parsed.json").write_text(
                json.dumps(parsed, indent=2), encoding="utf-8"
            )
            (out_dir / "fixation_components.json").write_text(
                json.dumps(
                    {
                        "map_type": "human_free_viewing_fixation_density",
                        "fixation_components": components,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            render_outputs(image_path, grid, components, out_dir)
            status.update(
                {
                    "status": "ok",
                    "elapsed_seconds": round(time.time() - started, 2),
                    "density_sum": float(grid.sum()),
                    "density_max": float(grid.max()),
                    "component_count": len(components),
                    "component_labels": [component["label"] for component in components],
                    "prediction_format": "gaussian_fixation_density_v1",
                    "used_retry_prompt": used_retry_prompt,
                }
            )
            write_status(done_file, status)
            print(f"[ok] {item_id}", flush=True)
        except Exception as exc:
            status.update(
                {
                    "status": "failed",
                    "elapsed_seconds": round(time.time() - started, 2),
                    "error": repr(exc),
                }
            )
            write_status(out_dir / "error.json", status)
            print(f"[failed] {item_id}: {exc}", flush=True)
        summary.append(status)
        write_status(model_out / f"summary_shard_{args.shard_index:03d}.json", {"runs": summary})

    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Fatal: {exc}", file=sys.stderr)
        raise SystemExit(1)
