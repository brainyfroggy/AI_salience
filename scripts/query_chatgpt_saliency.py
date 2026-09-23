"""Ask GPT vision models for image saliency grids and save heatmap overlays.

This is a small API experiment, not a replacement for supervised saliency
models such as SalFBNet or TranSalNet. The model is asked to predict a
human-viewer fixation/salience probability map as a coarse grid; this script
upsamples that grid and overlays it on the original image.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = Path(
    r"N:\Experimental_Data\yujunchen\projects\data\EMOd\EMOdImages1019_RawImage"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "chatgpt_saliency_test"
DEFAULT_ENV_FILES = [
    Path(r"N:\Experimental_Data\yujunchen\projects\AI_ratings\codes\.env"),
    Path(
        r"N:\Experimental_Data\yujunchen\projects\AI_ratings"
        r"\ck_whole_video_script\ck_whole_video_standalone\.env"
    ),
]
DEFAULT_MODELS = "gpt-5.2,gpt-5.3,gpt-5.4,gpt-5.5,gpt-5.6"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


SYSTEM_PROMPT = (
    "You are a computational vision researcher estimating human visual "
    "salience. Predict where typical human observers would look during the "
    "first few seconds of free viewing. Return structured numeric output only."
)


USER_PROMPT = """Predict a human saliency map for this image.

Use the behavior of saliency models such as ChatGPT IIE, SalFBNet, and
TranSalNet as inspiration: emphasize likely human fixations from faces, people,
animals, text, high-contrast objects, semantically important objects, emotional
content, unusual objects, and central composition. Suppress uniform background.

Return a 16 by 16 saliency_grid. Each cell must be a number from 0.0 to 1.0.
0.0 means no expected attention and 1.0 means strongest expected attention.
The grid is row-major from top-left to bottom-right and should form a smooth
spatial saliency probability field, not bounding boxes.

Return only JSON matching the schema."""


@dataclass
class ImageItem:
    image_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query GPT-5.x vision models for coarse saliency maps."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--image-ids",
        default="0001.jpg,0002.jpg,0003.jpg",
        help="Comma-separated filenames to test. Defaults to first 3 EMOd images.",
    )
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument(
        "--detail",
        choices=["low", "high", "auto", "original"],
        default="original",
        help="Image detail level sent to the Responses API.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature. Omitted by default for GPT-5.x compatibility.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=5000)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create request previews and fallback overlays without API calls.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_default_env() -> None:
    for path in DEFAULT_ENV_FILES:
        load_env_file(path)

    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("GPT_NAVIGATOR_BASE_URL", "")
    use_navigator_key = bool(
        os.environ.get("GPT_NAVIGATOR_API_KEY")
        and re.search(r"navigator|litellm|proxy", base_url, re.IGNORECASE)
    )
    if use_navigator_key:
        os.environ["OPENAI_API_KEY"] = os.environ["GPT_NAVIGATOR_API_KEY"]
    elif "OPENAI_API_KEY" not in os.environ and os.environ.get("GPT_NAVIGATOR_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["GPT_NAVIGATOR_API_KEY"]

    if "OPENAI_BASE_URL" not in os.environ and os.environ.get("GPT_NAVIGATOR_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["GPT_NAVIGATOR_BASE_URL"]


def normalize_base_url(base_url: str | None) -> str:
    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def id_to_sort_key(value: str) -> tuple[float, int, str]:
    try:
        return (float(Path(value).stem), 0, value)
    except ValueError:
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return (float(match.group(0)), 0, value)
        return (float("inf"), 1, value)


def list_images(image_dir: Path, image_ids: list[str]) -> list[ImageItem]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if image_ids:
        items = []
        for image_id in image_ids:
            path = image_dir / image_id
            if not path.exists():
                raise FileNotFoundError(f"Requested image not found: {path}")
            items.append(ImageItem(path.name, path))
        return items

    paths = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    paths.sort(key=lambda path: id_to_sort_key(path.name))
    return [ImageItem(path.name, path) for path in paths[:3]]


def image_to_data_url(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type is None:
        media_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"


def saliency_schema(grid_size: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "saliency_grid": {
                "type": "array",
                "minItems": grid_size,
                "maxItems": grid_size,
                "items": {
                    "type": "array",
                    "minItems": grid_size,
                    "maxItems": grid_size,
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                },
            }
        },
        "required": ["saliency_grid"],
        "additionalProperties": False,
    }


def build_request_body(
    model: str,
    item: ImageItem,
    grid_size: int,
    detail: str,
    temperature: float | None,
    max_output_tokens: int,
    include_image: bool = True,
) -> dict[str, Any]:
    data_url = image_to_data_url(item.path) if include_image else "<omitted>"
    body: dict[str, Any] = {
        "model": model,
        "instructions": SYSTEM_PROMPT,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "saliency_grid_response",
                "strict": True,
                "schema": saliency_schema(grid_size),
            }
        },
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"image_id: {item.image_id}\n"
                            f"grid_size: {grid_size}x{grid_size}\n\n{USER_PROMPT}"
                        ),
                    },
                    {"type": "input_image", "image_url": data_url, "detail": detail},
                ],
            }
        ],
        "max_output_tokens": max_output_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    return body


def extract_output_text(response_body: dict[str, Any]) -> str:
    if isinstance(response_body.get("output_text"), str):
        return response_body["output_text"]
    chunks: list[str] = []
    for item in response_body.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and "text" in content:
                chunks.append(content["text"])
    return "\n".join(chunks)


def call_responses_api(
    body: dict[str, Any],
    api_key: str,
    base_url: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    endpoint = f"{normalize_base_url(base_url)}/responses"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def parse_grid(text: str, grid_size: int) -> np.ndarray:
    text = text.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    data = json.loads(text)
    grid = np.asarray(data["saliency_grid"], dtype=np.float32)
    if grid.shape != (grid_size, grid_size):
        raise ValueError(f"Expected grid {(grid_size, grid_size)}, got {grid.shape}")
    grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(grid, 0.0, 1.0)


def fallback_center_bias(grid_size: int) -> np.ndarray:
    coords = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    grid = np.exp(-2.8 * (xx * xx + yy * yy))
    return (grid / grid.max()).astype(np.float32)


def turbo_like_colormap(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return np.dstack([red, green, blue])


def save_grid_csv(path: Path, grid: np.ndarray) -> None:
    rows = [",".join(f"{float(value):.4f}" for value in row) for row in grid]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def render_outputs(
    image_path: Path,
    grid: np.ndarray,
    out_prefix: Path,
    overlay_alpha: float,
) -> dict[str, str]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    grid_img = Image.fromarray(np.uint8(np.clip(grid, 0, 1) * 255), mode="L")
    heat_l = grid_img.resize((width, height), resample=Image.Resampling.BICUBIC)
    heat_l = heat_l.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 100.0))
    heat = np.asarray(heat_l, dtype=np.float32) / 255.0

    heat_rgb = np.uint8(turbo_like_colormap(heat) * 255)
    heat_img = Image.fromarray(heat_rgb, mode="RGB")
    alpha = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255 * overlay_alpha), mode="L")
    overlay = Image.composite(heat_img, image, alpha)
    overlay = ImageEnhance.Contrast(overlay).enhance(1.05)

    raw_heatmap_path = out_prefix.with_name(out_prefix.name + "_heatmap.png")
    overlay_path = out_prefix.with_name(out_prefix.name + "_overlay.png")
    grid_csv_path = out_prefix.with_name(out_prefix.name + "_grid.csv")
    grid_png_path = out_prefix.with_name(out_prefix.name + "_grid.png")

    heat_img.save(raw_heatmap_path)
    overlay.save(overlay_path)
    save_grid_csv(grid_csv_path, grid)

    cell = 28
    grid_vis = Image.new("RGB", (grid.shape[1] * cell, grid.shape[0] * cell), "white")
    draw = ImageDraw.Draw(grid_vis)
    colors = np.uint8(turbo_like_colormap(grid) * 255)
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            x0 = col * cell
            y0 = row * cell
            draw.rectangle(
                [x0, y0, x0 + cell - 1, y0 + cell - 1],
                fill=tuple(int(c) for c in colors[row, col]),
            )
    grid_vis.save(grid_png_path)

    return {
        "heatmap": str(raw_heatmap_path),
        "overlay": str(overlay_path),
        "grid_csv": str(grid_csv_path),
        "grid_png": str(grid_png_path),
    }


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> int:
    args = parse_args()
    load_default_env()

    models = [part.strip() for part in args.models.split(",") if part.strip()]
    image_ids = [part.strip() for part in args.image_ids.split(",") if part.strip()]
    items = list_images(args.image_dir, image_ids)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not args.dry_run and not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Set it or put it in the AI_ratings .env file."
        )

    manifest: dict[str, Any] = {
        "image_dir": str(args.image_dir),
        "out_dir": str(args.out_dir),
        "models": models,
        "images": [item.image_id for item in items],
        "grid_size": args.grid_size,
        "detail": args.detail,
        "dry_run": args.dry_run,
        "runs": [],
    }

    for model in models:
        for item in items:
            label = f"{safe_label(Path(item.image_id).stem)}_{safe_label(model)}"
            print(f"[run] model={model} image={item.image_id}", flush=True)
            body = build_request_body(
                model=model,
                item=item,
                grid_size=args.grid_size,
                detail=args.detail,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                include_image=not args.dry_run,
            )
            preview = build_request_body(
                model=model,
                item=item,
                grid_size=args.grid_size,
                detail=args.detail,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                include_image=False,
            )
            request_path = args.out_dir / f"{label}_request_preview.json"
            request_path.write_text(json.dumps(preview, indent=2), encoding="utf-8")

            raw_path = args.out_dir / f"{label}_response_raw.json"
            text_path = args.out_dir / f"{label}_response_text.json"
            status = "ok"
            error = None
            started = time.time()
            try:
                if args.dry_run:
                    text = json.dumps({"saliency_grid": fallback_center_bias(args.grid_size).tolist()})
                    response_body = {"dry_run": True, "output_text": text}
                else:
                    response_body = call_responses_api(
                        body=body,
                        api_key=api_key or "",
                        base_url=base_url,
                        timeout_seconds=args.timeout_seconds,
                    )
                    text = extract_output_text(response_body)
                raw_path.write_text(json.dumps(response_body, indent=2), encoding="utf-8")
                text_path.write_text(text, encoding="utf-8")
                grid = parse_grid(text, args.grid_size)
                output_paths = render_outputs(
                    image_path=item.path,
                    grid=grid,
                    out_prefix=args.out_dir / label,
                    overlay_alpha=args.overlay_alpha,
                )
            except Exception as exc:
                status = "failed"
                error = str(exc)
                output_paths = {}
                (args.out_dir / f"{label}_error.txt").write_text(error, encoding="utf-8")
                print(f"[error] model={model} image={item.image_id}: {error}", flush=True)

            manifest["runs"].append(
                {
                    "model": model,
                    "image_id": item.image_id,
                    "status": status,
                    "elapsed_seconds": round(time.time() - started, 2),
                    "request_preview": str(request_path),
                    "response_raw": str(raw_path),
                    "response_text": str(text_path),
                    "error": error,
                    **output_paths,
                }
            )
            (args.out_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

    print(f"[done] wrote manifest: {args.out_dir / 'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Fatal: {exc}", file=sys.stderr)
        raise SystemExit(1)
