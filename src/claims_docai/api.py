"""FastAPI service: upload a receipt, get fields with confidence; low-confidence fields wait in a review queue.

    uv run --extra local uvicorn claims_docai.api:app        # macOS: Apple Vision OCR + both MLX models

The threshold and arm priority come from results/eval.json (chosen on the validation split). The extractor is
injected, so tests and non-Mac machines run the service with a stub.
"""

from __future__ import annotations

import base64
import html
import json
import uuid
from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from .data import HEADER_FIELDS, ROOT
from .evaluate import PIPELINE_ARMS, RESULTS, printed_amounts, route

Extractor = Callable[[bytes], tuple[dict[str, dict], list]]  # image -> (arm -> fields, OCR tokens)


def local_extractor() -> Extractor:
    from . import llm, rules
    from .ocr import text, vision_tokens

    text_model, vision_model = llm.TextModel(), llm.VisionModel()

    def run(image: bytes):
        toks = vision_tokens(image)
        preds = {
            "rules-vision": rules.extract(toks),
            "text-vision": llm.parse(text_model(text(toks))),
            "vlm": llm.parse(vision_model(image)),
        }
        return preds, toks

    return run


def create_app(extractor: Extractor | None = None, settings: dict | None = None, review_log=None) -> FastAPI:
    settings = settings or json.loads((RESULTS / "eval.json").read_text())["pipeline"]
    app = FastAPI(title="claims-docai")
    claims: dict[str, dict] = {}
    log = review_log or ROOT / "data" / "review" / "decisions.jsonl"
    state = {"extractor": extractor}

    @app.post("/claims")
    async def submit(file: Annotated[UploadFile, File()]):
        image = await file.read()
        if not image:
            raise HTTPException(400, "empty file")
        if state["extractor"] is None:
            state["extractor"] = local_extractor()
        preds, toks = state["extractor"](image)
        cid = uuid.uuid4().hex[:8]
        routed = route({a: {cid: preds[a]} for a in PIPELINE_ARMS}, {cid: printed_amounts(toks)}, settings["priority"])[cid]
        th = settings["validation"]["threshold"]
        for f in routed["fields"].values():
            f["needs_review"] = th is None or f["confidence"] < th
        claims[cid] = {"id": cid, "image": image, **routed}
        return {"id": cid, "fields": routed["fields"], "items": routed["items"], "threshold": th}

    @app.get("/claims/{cid}")
    def get(cid: str):
        if cid not in claims:
            raise HTTPException(404, "unknown claim")
        c = claims[cid]
        return {"id": cid, "fields": c["fields"], "items": c["items"]}

    @app.get("/review", response_class=HTMLResponse)
    def queue():
        rows = []
        for c in claims.values():
            todo = [f for f in HEADER_FIELDS if c["fields"][f].get("needs_review")]
            if not todo:
                continue
            img = base64.b64encode(c["image"]).decode()
            inputs = "".join(
                f'<label>{f} <input name="{f}" value="{c["fields"][f]["value"] or ""}"> '
                f"<small>confidence {c['fields'][f]['confidence']:.2f}</small></label><br>"
                for f in todo
            )
            rows.append(
                f'<div class="claim"><img src="data:image;base64,{img}" height="420">'
                f'<form method="post" action="/review/{c["id"]}">{inputs}<button>Approve</button></form></div>'
            )
        body = "".join(rows) or "<p>Nothing waiting for review.</p>"
        return f"<!doctype html><title>Review queue</title><h1>Review queue</h1>{body}"

    @app.post("/review/{cid}")
    async def review(
        cid: str,
        total: Annotated[str | None, Form()] = None,
        subtotal: Annotated[str | None, Form()] = None,
        tax: Annotated[str | None, Form()] = None,
    ):
        if cid not in claims:
            raise HTTPException(404, "unknown claim")
        given = {"total": total, "subtotal": subtotal, "tax": tax}
        c = claims[cid]
        for f in HEADER_FIELDS:
            if c["fields"][f].get("needs_review"):
                c["fields"][f].update(value=html.escape(given[f] or "") or None, needs_review=False, reviewed=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as fh:
            fh.write(json.dumps({"id": cid, "fields": c["fields"]}) + "\n")
        return RedirectResponse("/review", status_code=303)

    return app


def __getattr__(name):  # `uvicorn claims_docai.api:app` builds the local app lazily
    if name == "app":
        return create_app()
    raise AttributeError(name)
