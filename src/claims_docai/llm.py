"""Model-based extraction with a record/replay cassette.

- text: a small instruct model (Qwen2.5-1.5B, 4-bit, MLX) reads the OCR rows.
- vision: a small vision-language model (Qwen2.5-VL-3B, 4-bit, MLX) reads the receipt image directly.

Greedy decoding. Every raw response is stored in cassettes/<name>.jsonl keyed by document id, so evaluation, tests
and CI replay the exact outputs without a GPU.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

from .data import ROOT

CASSETTES = ROOT / "cassettes"
TEXT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
VISION_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
# Long-side sizes tried in order. Full-size CORD pages (up to 2304x4096) make the 4-bit model emit only "!!!!", and
# some pages do it at one size and not another, so a degenerate answer is retried smaller.
SIDES = (1024, 896, 768, 640)

INSTRUCTIONS = """Extract these fields from the receipt for an insurance claim.
Return only JSON: {"total": "...", "subtotal": "...", "tax": "...", "items": [{"name": "...", "price": "..."}]}
- total: the final amount paid (not cash given, not change).
- subtotal: the amount before tax and service, only if printed.
- tax: the tax amount (TAX, PB1, PPN), only if printed.
- items: each purchased item with its line price.
Copy amounts exactly as printed. Use null for a field that is not on the receipt."""


def parse(raw: str) -> dict:
    """First JSON object in the response; missing or malformed parts become null/empty."""
    out: dict = {"total": None, "subtotal": None, "tax": None, "items": []}
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    if not isinstance(obj, dict):
        return out
    for k in ("total", "subtotal", "tax"):
        v = obj.get(k)
        out[k] = None if v in (None, "", "null") else str(v)
    for it in obj.get("items") or []:
        if isinstance(it, dict) and (it.get("name") or it.get("price")):
            out["items"].append({"name": it.get("name"), "price": None if it.get("price") is None else str(it["price"])})
    return out


def degenerate(raw: str) -> bool:
    """The fp16 failure mode: a response made only of '!' (or nothing)."""
    return not raw.strip().strip("!")


class Cassette:
    def __init__(self, name: str):
        self.path = CASSETTES / f"{name}.jsonl"
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.rows[r["id"]] = r

    def get(self, doc_id: str) -> str | None:
        r = self.rows.get(doc_id)
        return r["raw"] if r else None

    def put(self, doc_id: str, raw: str, seconds: float):
        self.rows[doc_id] = {"id": doc_id, "raw": raw, "seconds": round(seconds, 2)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(self.rows[doc_id]) + "\n")

    def seconds(self) -> list[float]:
        return [r["seconds"] for r in self.rows.values()]


class TextModel:
    def __init__(self, model: str = TEXT_MODEL):
        from mlx_lm import load

        self.model, self.tok = load(model)

    def __call__(self, ocr_text: str) -> str:
        from mlx_lm import generate

        msgs = [{"role": "user", "content": INSTRUCTIONS + "\n\nReceipt (OCR rows):\n" + ocr_text}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return generate(self.model, self.tok, prompt=prompt, max_tokens=500)


class VisionModel:
    def __init__(self, model: str = VISION_MODEL):
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        self.model, self.processor = load(model)
        self.config = load_config(model)

    def __call__(self, image: bytes) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from PIL import Image

        prompt = apply_chat_template(self.processor, self.config, INSTRUCTIONS, num_images=1)
        path = Path("/tmp/claims-docai-page.png")
        text = ""
        for side in SIDES:
            page = Image.open(io.BytesIO(image)).convert("RGB")
            page.thumbnail((side, side))
            page.save(path)
            res = generate(self.model, self.processor, prompt, image=[str(path)], max_tokens=500, temperature=0.0, verbose=False)
            text = res.text if hasattr(res, "text") else str(res)
            if not degenerate(text):
                break
        return text
