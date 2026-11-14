from fastapi.testclient import TestClient

from claims_docai.api import create_app
from claims_docai.data import Token

SETTINGS = {
    "priority": {f: ["vlm", "text-vision", "rules-vision"] for f in ("total", "subtotal", "tax", "items")},
    "validation": {"threshold": 1.0},
}


def stub(image: bytes):
    good = {"total": "14,300", "subtotal": "13,000", "tax": "1,300", "items": []}
    preds = {"rules-vision": good, "text-vision": {**good, "tax": "130"}, "vlm": good}
    return preds, [Token(t, 0, 0, 0.1, 0.02) for t in ("13,000", "1,300", "14,300")]


def test_submit_review_and_approve(tmp_path):
    client = TestClient(create_app(stub, SETTINGS, tmp_path / "decisions.jsonl"))
    res = client.post("/claims", files={"file": ("r.png", b"fake-image", "image/png")}).json()
    fields = res["fields"]
    assert fields["total"]["value"] == 14300 and not fields["total"]["needs_review"]
    assert fields["tax"]["value"] == 1300 and fields["tax"]["needs_review"]  # one arm disagrees

    page = client.get("/review").text
    assert 'name="tax"' in page and 'name="total"' not in page

    client.post(f"/review/{res['id']}", data={"tax": "1,300"}, follow_redirects=False)
    assert "Nothing waiting" in client.get("/review").text
    assert client.get(f"/claims/{res['id']}").json()["fields"]["tax"]["reviewed"]
    assert (tmp_path / "decisions.jsonl").read_text().count("\n") == 1


def test_empty_upload_rejected(tmp_path):
    client = TestClient(create_app(stub, SETTINGS, tmp_path / "decisions.jsonl"))
    assert client.post("/claims", files={"file": ("r.png", b"", "image/png")}).status_code == 400
