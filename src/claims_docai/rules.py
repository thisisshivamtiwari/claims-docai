"""Baseline: keyword rules over OCR rows, the kind of template logic claims teams start with."""

from __future__ import annotations

import re

from .data import Token
from .normalize import AMOUNT
from .ocr import rows

SUBTOTAL = re.compile(r"sub\s*-?\s*t[o0]t", re.I)
TOTAL = re.compile(r"grand\s*t[o0]t|\bt[o0]\s*t\s*a\s*l\b|\bjumlah\b|amount\s*due", re.I)
NOT_TOTAL = re.compile(r"qty|item|disc|tax|change|kemb|cash|tunai", re.I)
TAX = re.compile(r"\btax\b|\bpb\s*1\b|\bppn\b|pajak|\bvat\b", re.I)
MONEY = re.compile(r"\d{1,3}(?:[.,]\d{3})+|\d{4,}")


def _amounts(line: str) -> list[str]:
    return [m.group(0) for m in MONEY.finditer(line)]


def extract(toks: list[Token]) -> dict:
    lines = ["  ".join(t.text for t in r) for r in rows(toks)]
    out: dict = {"total": None, "subtotal": None, "tax": None, "items": []}
    first_summary = len(lines)

    def value_at(i):
        found = _amounts(lines[i])
        if not found and i + 1 < len(lines):
            found = _amounts(lines[i + 1])
        return found[-1] if found else None

    for i, line in enumerate(lines):
        if SUBTOTAL.search(line):
            out["subtotal"] = out["subtotal"] or value_at(i)
            first_summary = min(first_summary, i)
        elif TAX.search(line):
            out["tax"] = out["tax"] or value_at(i)
            first_summary = min(first_summary, i)
        elif TOTAL.search(line) and not NOT_TOTAL.search(line):
            out["total"] = out["total"] or value_at(i)
            first_summary = min(first_summary, i)

    for line in lines[:first_summary]:
        found = _amounts(line)
        head = AMOUNT.sub("", line).strip(" x@-:")
        if found and re.search(r"[A-Za-z]{2,}", head):
            out["items"].append({"name": re.sub(r"\s+", " ", head), "price": found[-1]})
    return out
