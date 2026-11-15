"""OCR: Apple Vision (macOS, 'fast' level) or the dataset's gold word boxes, cached as JSON so CI can replay.

Rows: tokens whose vertical centres are within half a line height are one row, read left to right. The extractors
see the receipt as these rows, which keeps label and amount ("TOTAL ... 60.000") on one line.
"""

from __future__ import annotations

import io
import json
import statistics
from dataclasses import asdict

from .data import ROOT, Doc, Word

CACHE = ROOT / "data" / "ocr"


def vision_tokens(image: bytes) -> list[Word]:
    from ocrmac import ocrmac  # macOS only
    from PIL import Image

    img = Image.open(io.BytesIO(image))
    out = []
    for text, conf, (x, y, w, h) in ocrmac.OCR(img, recognition_level="fast").recognize():
        out.append(Word(text, x, 1 - y - h, w, h, float(conf)))  # Vision's origin is bottom-left
    return out


def tokens(doc: Doc, source: str) -> list[Word]:
    if source == "gold":
        return doc.gold_tokens
    path = CACHE / source / f"{doc.id}.json"
    if path.exists():
        return [Word(**t) for t in json.loads(path.read_text())]
    toks = vision_tokens(doc.image)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(t) for t in toks]))
    return toks


def cached_tokens(doc_id: str, source: str) -> list[Word]:
    return [Word(**t) for t in json.loads((CACHE / source / f"{doc_id}.json").read_text())]


def rows(toks: list[Word]) -> list[list[Word]]:
    if not toks:
        return []
    line_h = statistics.median(t.h for t in toks) or 0.02
    out: list[list[Word]] = []
    for t in sorted(toks, key=lambda t: t.y + t.h / 2):
        centre = t.y + t.h / 2
        if out and abs(centre - statistics.mean(r.y + r.h / 2 for r in out[-1])) < line_h / 2:
            out[-1].append(t)
        else:
            out.append([t])
    return [sorted(r, key=lambda t: t.x) for r in out]


def text(toks: list[Word]) -> str:
    return "\n".join("  ".join(t.text for t in r) for r in rows(toks))
