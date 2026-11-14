"""claims-docai command line.

python -m claims_docai truth                      # write data/truth-<split>.json from the parquet files
python -m claims_docai ocr                        # cache Apple Vision OCR for every document (macOS)
python -m claims_docai record <arm> <split>       # run a model arm, append raw outputs to cassettes/ (macOS, MLX)
python -m claims_docai eval                       # score every arm from cached OCR + cassettes -> results/
python -m claims_docai ci                         # replay eval and fail if it no longer matches results/
python -m claims_docai demo <doc id> ...          # terminal walk-through used for the demo video
"""

from __future__ import annotations

import json
import sys
import time

from . import data, llm, ocr
from .evaluate import run_all

SPLITS = ("validation", "test")
ARMS = ("text-gold", "text-vision", "vlm")


def cmd_truth():
    for split in SPLITS:
        out = {d.id: d.truth for d in data.load(split)}
        (data.ROOT / "data" / f"truth-{split}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
        gold = {d.id: [t.__dict__ for t in d.gold_tokens] for d in data.load(split)}
        path = ocr.CACHE / "gold" / f"{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(gold))


def cmd_ocr():
    for split in SPLITS:
        for d in data.load(split):
            ocr.tokens(d, "vision")


def cmd_record(arm: str, split: str):
    tape = llm.Cassette(f"{arm}-{split}")
    model = llm.VisionModel() if arm == "vlm" else llm.TextModel()
    for d in data.load(split):
        if tape.get(d.id) is not None:
            continue
        start = time.time()
        raw = model(d.image) if arm == "vlm" else model(ocr.text(ocr.tokens(d, arm.split("-")[1])))
        tape.put(d.id, raw, time.time() - start)
        print(d.id, round(time.time() - start, 1), flush=True)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    cmd, *rest = argv
    if cmd == "truth":
        cmd_truth()
    elif cmd == "ocr":
        cmd_ocr()
    elif cmd == "record":
        cmd_record(*rest)
    elif cmd == "eval":
        run_all(write=True)
    elif cmd == "demo":
        from .demo import main as demo

        demo(rest)
    elif cmd == "ci":
        return 0 if run_all(write=False) else 1
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
