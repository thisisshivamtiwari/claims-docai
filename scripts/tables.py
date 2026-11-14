"""Render the README result tables from results/eval.json (between the RESULTS markers)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "rules-gold": "Keyword rules, gold OCR (upper bound)",
    "text-gold": "Qwen2.5-1.5B text, gold OCR (upper bound)",
    "rules-vision": "Keyword rules, Apple Vision OCR",
    "text-vision": "Qwen2.5-1.5B text, Apple Vision OCR",
    "vlm": "Qwen2.5-VL-3B, receipt image",
    "pipeline": "Pipeline: vote of the 3 real-input arms",
}


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def render(res: dict) -> str:
    out = ["**Field accuracy on the 100 test receipts** (a field is right if the amount matches, or both say it is absent)", ""]
    out += ["| Arm | Total | Subtotal | Tax | All 3 right | Line items F1 | Median s/receipt |", "|---|---|---|---|---|---|---|"]
    for arm, label in ARMS.items():
        if arm not in res["arms"]:
            continue
        t = res["arms"][arm]["test"]
        lat = res["latency_seconds"].get(arm, {}).get("test", {}).get("median")
        out.append(
            f"| {label} | {pct(t['total']['accuracy'])} | {pct(t['subtotal']['accuracy'])} | {pct(t['tax']['accuracy'])} | "
            f"{pct(t['documents_all_header_fields_correct'])} | {pct(t['items']['f1'])} | {'-' if lat is None else lat} |"
        )
    p = res.get("pipeline")
    if p:
        target = pct(p["target_precision"])
        out += ["", f"**Review queue** (threshold chosen on validation for {target} precision, then applied once to test)", ""]
        out += ["| Split | Threshold | Field decisions | Auto-approved | Auto-approved precision | To a handler | Fully auto |"]
        out += ["|---|---|---|---|---|---|---|"]
        for s in ("validation", "test"):
            r = p[s]
            auto = f"{r['auto_approved']} ({pct(r['auto_approved_share'])})"
            prec = f"{pct(r['auto_approved_precision'])} ({r['auto_approved_correct']}/{r['auto_approved']})"
            out.append(
                f"| {s} | {r['threshold']} | {r['field_decisions']} | {auto} | {prec} | {r['sent_to_review']} | "
                f"{r['documents_fully_auto']}/{r['documents']} |"
            )
    return "\n".join(out)


def main():
    res = json.loads((ROOT / "results" / "eval.json").read_text())
    readme = ROOT / "README.md"
    text = readme.read_text()
    block = "<!-- RESULTS:START -->\n" + render(res) + "\n<!-- RESULTS:END -->"
    readme.write_text(re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->", lambda _: block, text, flags=re.S))


if __name__ == "__main__":
    main()
