"""Terminal walk-through for the demo video: one receipt through the pipeline, then the test-split numbers.

Replays cached OCR and recorded model outputs, so it runs in seconds and prints exactly what was evaluated.
"""

from __future__ import annotations

import json
import time

from . import llm, rules
from .data import HEADER_FIELDS, truths
from .evaluate import PIPELINE_ARMS, RESULTS, printed_amounts, route
from .normalize import amount
from .ocr import cached_tokens, text

B, G, R, Y, END = "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def fmt(v):
    return "absent" if v is None else f"{v:,}"


def show(doc_id: str, res: dict):
    split = doc_id.split("-")[0]
    truth = truths(split)[doc_id]
    toks = cached_tokens(doc_id, "vision")
    preds = {
        "rules-vision": rules.extract(toks),
        "text-vision": llm.parse(llm.Cassette(f"text-vision-{split}").get(doc_id)),
        "vlm": llm.parse(llm.Cassette(f"vlm-{split}").get(doc_id)),
    }
    print(f"{B}Receipt {doc_id}, Apple Vision OCR (fast){END}")
    for line in text(toks).splitlines()[-12:]:
        print("   " + line)
        time.sleep(0.05)
    time.sleep(0.8)
    print(f"\n{B}Three readers, one vote per field{END}")
    print(f"   {'field':<9}{'rules':>12}{'text 1.5B':>12}{'vision 3B':>12}   truth")
    for f in HEADER_FIELDS:
        vals = [fmt(amount(preds[a][f])) for a in PIPELINE_ARMS]
        print(f"   {f:<9}" + "".join(f"{v:>12}" for v in vals) + f"   {fmt(amount(truth[f]))}")
        time.sleep(0.3)
    p = res["pipeline"]
    routed = route({a: {doc_id: preds[a]} for a in PIPELINE_ARMS}, {doc_id: printed_amounts(toks)}, p["priority"])[doc_id]
    th = p["validation"]["threshold"]
    print(f"\n{B}Routing (auto-approve at confidence >= {th}){END}")
    for f in HEADER_FIELDS:
        c = routed["fields"][f]
        ok = c["value"] == amount(truth[f])
        where = f"{G}AUTO-APPROVE{END}" if c["confidence"] >= th else f"{Y}REVIEW QUEUE{END}"
        mark = f"{G}right{END}" if ok else f"{R}wrong{END}"
        print(f"   {f:<9}{fmt(c['value']):>12}  confidence {c['confidence']:.2f}  {where}  ({mark})")
        time.sleep(0.4)


def main(doc_ids: list[str]):
    res = json.loads((RESULTS / "eval.json").read_text())
    for d in doc_ids:
        show(d, res)
        print()
        time.sleep(0.8)
    t = res["pipeline"]["test"]
    print(f"{B}100 test receipts, 300 field decisions{END}")
    for arm, label in [("rules-vision", "rules on OCR"), ("text-vision", "1.5B text model on OCR"), ("vlm", "3B vision model")]:
        share = res["arms"][arm]["test"]["documents_all_header_fields_correct"]
        print(f"   {label:<24} all 3 fields right on {share:.0%} of receipts")
        time.sleep(0.3)
    print(
        f"   auto-approved {t['auto_approved']} of {t['field_decisions']} fields, "
        f"{t['auto_approved_precision']:.0%} of them right ({t['auto_approved_correct']}/{t['auto_approved']})"
    )
    print(f"   sent to a handler: {t['sent_to_review']} fields")
