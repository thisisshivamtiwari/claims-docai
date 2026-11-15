"""CORD v2 receipts (CC-BY-4.0, naver-clova-ix/cord-v2) as claim supporting documents.

Each document has an image, the fields a claims handler keys in (total, subtotal, tax, line items) and the dataset's
own word boxes ("gold OCR"). Gold OCR keeps only text and position, never the dataset's field labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
HEADER_FIELDS = ("total", "subtotal", "tax")


@dataclass
class Word:
    text: str
    x: float  # left, 0..1 of page width
    y: float  # top, 0..1 of page height
    w: float
    h: float
    conf: float = 1.0


@dataclass
class Doc:
    id: str
    image: bytes
    truth: dict  # {"total": str|None, "subtotal": ..., "tax": ..., "items": [{"name", "price"}]}
    gold_tokens: list[Word] = field(default_factory=list)


def _first(v):
    while isinstance(v, list):
        v = v[0] if v else None
    return v


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def parse_truth(gt: dict) -> dict:
    p = gt["gt_parse"]
    total = _first(p.get("total")) or {}
    sub = _first(p.get("sub_total")) or {}
    items = []
    for m in _as_list(p.get("menu")):
        name, price = _first(m.get("nm")), _first(m.get("price"))
        if name or price:
            items.append({"name": name, "price": price})
    return {
        "total": _first(total.get("total_price")),
        "subtotal": _first(sub.get("subtotal_price")),
        "tax": _first(sub.get("tax_price")),
        "items": items,
    }


def gold_tokens(gt: dict) -> list[Word]:
    meta = gt["meta"]["image_size"]
    width, height = meta["width"], meta["height"]
    out = []
    for line in gt.get("valid_line", []):
        for w in line["words"]:
            q = w["quad"]
            xs, ys = [q["x1"], q["x2"], q["x3"], q["x4"]], [q["y1"], q["y2"], q["y3"], q["y4"]]
            out.append(
                Word(
                    w["text"],
                    min(xs) / width,
                    min(ys) / height,
                    (max(xs) - min(xs)) / width,
                    (max(ys) - min(ys)) / height,
                )
            )
    return out


def load(split: str, limit: int | None = None) -> list[Doc]:
    import pyarrow.parquet as pq

    table = pq.read_table(RAW / f"{split}.parquet")
    docs = []
    for i, row in enumerate(table.to_pylist()):
        if limit is not None and i >= limit:
            break
        gt = json.loads(row["ground_truth"])
        docs.append(Doc(f"{split}-{i:03d}", row["image"]["bytes"], parse_truth(gt), gold_tokens(gt)))
    return docs


def truths(split: str) -> dict[str, dict]:
    """Ground truth only (no images), from the committed copy so CI needs no download."""
    path = ROOT / "data" / f"truth-{split}.json"
    return json.loads(path.read_text())
