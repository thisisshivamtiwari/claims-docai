"""Scoring, the confidence-routed pipeline and the human-review queue.

A field decision is one header field of one receipt: the extracted amount, or "not on the receipt". It is correct if
it equals the ground truth as an integer amount (or both are absent).

Pipeline (inputs a claims team really has: the image and Apple Vision OCR):
- three arms vote per field: rules on OCR rows, the text model on OCR rows, the vision-language model on the image;
- ties go to the arm with the best validation accuracy for that field;
- confidence = 0.5 x share of arms agreeing + 0.25 if the amount is printed in the OCR + 0.25 if the receipt's
  arithmetic holds (subtotal + tax = total, items add up, or tax ~ 10% of subtotal);
- the auto-approve threshold is the lowest one that reaches the target precision on the validation split; test is
  scored once with that threshold. Everything below it goes to a handler. The service runs at 98%: on 300 validation
  decisions no threshold reached 99% (the top bucket was 49/50). Every target in OPERATING_POINTS is reported.
"""

from __future__ import annotations

import json
from collections import Counter

from . import llm, rules
from .data import HEADER_FIELDS, ROOT, Word, truths
from .normalize import amount, name
from .ocr import CACHE, cached_tokens

RESULTS = ROOT / "results"
SPLITS = ("validation", "test")
ARMS = {
    "rules-gold": "rules on gold OCR",
    "rules-vision": "rules on Apple Vision OCR",
    "text-gold": "Qwen2.5-1.5B on gold OCR",
    "text-vision": "Qwen2.5-1.5B on Apple Vision OCR",
    "vlm": "Qwen2.5-VL-3B on the image",
}
PIPELINE_ARMS = ("rules-vision", "text-vision", "vlm")
TARGET_PRECISION = 0.98
OPERATING_POINTS = (0.95, 0.98, 0.99)


def gold_tokens(split: str) -> dict[str, list[Word]]:
    raw = json.loads((CACHE / "gold" / f"{split}.json").read_text())
    return {k: [Word(**t) for t in v] for k, v in raw.items()}


def predictions(split: str) -> dict[str, dict[str, dict]]:
    """arm -> doc id -> {"total", "subtotal", "tax", "items"} (raw strings)."""
    ids = list(truths(split))
    gold = gold_tokens(split)
    out: dict[str, dict[str, dict]] = {}
    out["rules-gold"] = {i: rules.extract(gold[i]) for i in ids}
    out["rules-vision"] = {i: rules.extract(cached_tokens(i, "vision")) for i in ids}
    for arm in ("text-gold", "text-vision", "vlm"):
        tape = llm.Cassette(f"{arm}-{split}")
        if all(tape.get(i) is not None for i in ids):
            out[arm] = {i: llm.parse(tape.get(i)) for i in ids}
    return out


def field_scores(preds: dict[str, dict], truth: dict[str, dict]) -> dict:
    scores = {}
    for f in HEADER_FIELDS:
        c = Counter()
        for i, t in truth.items():
            p, g = amount(preds[i][f]), amount(t[f])
            c["correct"] += p == g
            c["pred"] += p is not None
            c["gold"] += g is not None
            c["tp"] += p is not None and p == g
        prec = c["tp"] / c["pred"] if c["pred"] else 0.0
        rec = c["tp"] / c["gold"] if c["gold"] else 0.0
        scores[f] = {
            "accuracy": round(c["correct"] / len(truth), 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        }
    docs_ok = sum(all(amount(preds[i][f]) == amount(t[f]) for f in HEADER_FIELDS) for i, t in truth.items())
    scores["documents_all_header_fields_correct"] = round(docs_ok / len(truth), 3)
    scores["items"] = item_scores(preds, truth)
    return scores


def _item_match(p: dict, g: dict) -> bool:
    if amount(p.get("price")) != amount(g.get("price")):
        return False
    pn, gn = set(name(p.get("name")).split()), set(name(g.get("name")).split())
    return bool(gn) and len(pn & gn) / len(gn) >= 0.5


def item_scores(preds: dict[str, dict], truth: dict[str, dict]) -> dict:
    tp = n_pred = n_gold = 0
    for i, t in truth.items():
        left = list(preds[i]["items"])
        n_pred += len(left)
        n_gold += len(t["items"])
        for g in t["items"]:
            hit = next((p for p in left if _item_match(p, g)), None)
            if hit is not None:
                left.remove(hit)
                tp += 1
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    return {
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
    }


def _close(a, b, tol=2):
    return a is not None and b is not None and abs(a - b) <= tol


def arithmetic_ok(f: str, vals: dict, items: list[dict]) -> bool:
    total, sub, tax = vals["total"], vals["subtotal"], vals["tax"]
    item_sum = sum(amount(it.get("price")) or 0 for it in items) or None
    if f == "total":
        return (sub is not None and tax is not None and _close(sub + tax, total)) or _close(item_sum, total) or _close(sub, total)
    if f == "subtotal":
        return (
            (tax is not None and _close(sub + tax if sub is not None else None, total))
            or _close(item_sum, sub)
            or _close(sub, total)
        )
    if tax is None:
        return False
    return (sub is not None and (_close(sub + tax, total) or _close(tax, round(sub * 0.1), 2))) or _close(
        tax, round((total or 0) / 11), 2
    )


def printed_amounts(toks: list[Word]) -> set[int]:
    """Every amount the OCR saw, whole tokens and money-shaped substrings."""
    found = {amount(t.text) for t in toks} | {amount(m) for t in toks for m in rules.MONEY.findall(t.text)}
    return found - {None}


def route(
    preds: dict[str, dict], printed: dict[str, set[int]], priority: dict[str, list[str]], arms=PIPELINE_ARMS
) -> dict[str, dict]:
    """Per document: chosen value and confidence for each header field, plus the chosen items."""
    out = {}
    for i in printed:
        chosen, votes = {}, {}
        for f in HEADER_FIELDS:
            cands = [amount(preds[a][i][f]) for a in arms]
            count = Counter(cands)
            best = max(count.values())
            tied = {v for v, n in count.items() if n == best}
            chosen[f] = next(amount(preds[a][i][f]) for a in priority[f] if amount(preds[a][i][f]) in tied)
            votes[f] = best
        items = preds[priority["items"][0]][i]["items"]
        fields = {}
        for f in HEADER_FIELDS:
            v = chosen[f]
            conf = 0.5 * votes[f] / len(arms)
            conf += 0.25 if (v is None or v in printed[i]) else 0.0
            conf += 0.25 if (v is None and votes[f] == len(arms)) or arithmetic_ok(f, chosen, items) else 0.0
            fields[f] = {"value": v, "confidence": round(conf, 4), "votes": votes[f]}
        out[i] = {"fields": fields, "items": items}
    return out


def decisions(routed: dict[str, dict], truth: dict[str, dict]) -> list[tuple[float, bool]]:
    return [
        (r["fields"][f]["confidence"], r["fields"][f]["value"] == amount(truth[i][f]))
        for i, r in routed.items()
        for f in HEADER_FIELDS
    ]


def pick_threshold(decs: list[tuple[float, bool]], target: float = TARGET_PRECISION) -> float | None:
    for th in sorted({c for c, _ in decs}):
        kept = [ok for c, ok in decs if c >= th]
        if kept and sum(kept) / len(kept) >= target:
            return th
    return None


def review_stats(routed: dict[str, dict], truth: dict[str, dict], th: float | None) -> dict:
    decs = decisions(routed, truth)
    auto = [ok for c, ok in decs if th is not None and c >= th]
    docs_auto = sum(all(th is not None and r["fields"][f]["confidence"] >= th for f in HEADER_FIELDS) for r in routed.values())
    return {
        "threshold": th,
        "field_decisions": len(decs),
        "accuracy_all_fields": round(sum(ok for _, ok in decs) / len(decs), 3),
        "auto_approved": len(auto),
        "auto_approved_share": round(len(auto) / len(decs), 3),
        "auto_approved_correct": sum(auto),
        "auto_approved_precision": round(sum(auto) / len(auto), 3) if auto else None,
        "sent_to_review": len(decs) - len(auto),
        "documents": len(routed),
        "documents_fully_auto": docs_auto,
    }


def run_all(write: bool = True) -> bool:
    results = {"arms": {}, "pipeline": {}, "latency_seconds": {}}
    preds = {s: predictions(s) for s in SPLITS}
    tru = {s: truths(s) for s in SPLITS}
    for s in SPLITS:
        for arm, p in preds[s].items():
            results["arms"].setdefault(arm, {})[s] = field_scores(p, tru[s])
    for s in SPLITS:
        for arm in ("text-gold", "text-vision", "vlm"):
            secs = llm.Cassette(f"{arm}-{s}").seconds()
            if secs:
                secs.sort()
                results["latency_seconds"].setdefault(arm, {})[s] = {
                    "median": secs[len(secs) // 2],
                    "p95": secs[int(len(secs) * 0.95) - 1],
                }
    if all(a in preds["validation"] and a in preds["test"] for a in PIPELINE_ARMS):
        val = results["arms"]
        priority = {f: sorted(PIPELINE_ARMS, key=lambda a: -val[a]["validation"][f]["accuracy"]) for f in HEADER_FIELDS}
        priority["items"] = sorted(PIPELINE_ARMS, key=lambda a: -val[a]["validation"]["items"]["f1"])
        routed = {s: route(preds[s], {i: printed_amounts(cached_tokens(i, "vision")) for i in tru[s]}, priority) for s in SPLITS}
        th = pick_threshold(decisions(routed["validation"], tru["validation"]))
        results["pipeline"] = {
            "arms": list(PIPELINE_ARMS),
            "priority": priority,
            "target_precision": TARGET_PRECISION,
            **{s: review_stats(routed[s], tru[s], th) for s in SPLITS},
        }
        results["operating_points"] = {}
        for target in OPERATING_POINTS:
            t = pick_threshold(decisions(routed["validation"], tru["validation"]), target)
            results["operating_points"][str(target)] = {s: review_stats(routed[s], tru[s], t) for s in SPLITS}
        for s in SPLITS:
            chosen = {
                i: {**{f: r["fields"][f]["value"] for f in HEADER_FIELDS}, "items": r["items"]} for i, r in routed[s].items()
            }
            results["arms"].setdefault("pipeline", {})[s] = field_scores(chosen, tru[s])
    path = RESULTS / "eval.json"
    text = json.dumps(results, indent=1, sort_keys=True) + "\n"
    if write:
        RESULTS.mkdir(exist_ok=True)
        path.write_text(text)
        return True
    same = path.exists() and path.read_text() == text
    print("eval matches results/eval.json" if same else "eval differs from results/eval.json")
    return same
