"""Amounts on CORD receipts are rupiah written as 60.000, 60,000, 60000 or Rp. 60.000; compare them as integers."""

from __future__ import annotations

import re

AMOUNT = re.compile(r"\d[\d.,]*\d|\d")


def amount(value) -> int | None:
    """Digits of the amount as an int; a trailing 2-digit decimal part (",00") is dropped. None if no digits."""
    if value is None:
        return None
    s = str(value).strip()
    s = re.sub(r"[.,]\d{2}$", "", s) if re.search(r"\d[.,]\d{3}[.,]\d{2}$", s) else s
    digits = re.sub(r"\D", "", s)
    return int(digits) if digits else None


def name(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
