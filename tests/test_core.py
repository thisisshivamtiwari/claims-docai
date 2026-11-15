from claims_docai import llm, rules
from claims_docai.data import Word, parse_truth
from claims_docai.evaluate import arithmetic_ok, item_scores, pick_threshold, route
from claims_docai.normalize import amount
from claims_docai.ocr import rows, text


def tok(t, x, y):
    return Word(t, x, y, 0.1, 0.02)


def test_amount_formats():
    assert amount("60.000") == amount("60,000") == amount("Rp. 60.000") == amount("60000") == 60000
    assert amount("1.591.000,00") == 1591000
    assert amount(None) is None and amount("TOTAL") is None


def test_rows_group_by_line_and_read_left_to_right():
    toks = [tok("45,500", 0.7, 0.501), tok("TOTAL", 0.1, 0.5), tok("CASH", 0.1, 0.55)]
    assert [[t.text for t in r] for r in rows(toks)] == [["TOTAL", "45,500"], ["CASH"]]
    assert text(toks) == "TOTAL  45,500\nCASH"


def test_rules_baseline():
    toks = [
        tok("EGG", 0.1, 0.1), tok("TART", 0.2, 0.1), tok("13,000", 0.7, 0.1),
        tok("SUBTOTAL", 0.1, 0.2), tok("13,000", 0.7, 0.2),
        tok("PB1", 0.1, 0.3), tok("1,300", 0.7, 0.3),
        tok("TOTAL", 0.1, 0.4), tok("14,300", 0.7, 0.4),
        tok("CASH", 0.1, 0.5), tok("20,000", 0.7, 0.5),
    ]  # fmt: skip
    out = rules.extract(toks)
    assert (out["subtotal"], out["tax"], out["total"]) == ("13,000", "1,300", "14,300")
    assert out["items"] == [{"name": "EGG TART", "price": "13,000"}]


def test_parse_model_output():
    raw = 'Sure:\n```json\n{"total": "45,500", "subtotal": null, "tax": "", "items": [{"name": "A", "price": 5}]}\n```'
    assert llm.parse(raw) == {"total": "45,500", "subtotal": None, "tax": None, "items": [{"name": "A", "price": "5"}]}
    assert llm.parse("no json here")["total"] is None


def test_truth_handles_lists_and_missing_fields():
    gt = {
        "gt_parse": {
            "menu": [{"nm": "A", "price": "1,000"}, {"nm": ["B", "b"], "price": "2,000"}],
            "total": {"total_price": ["3,000", "3,000"]},
        }
    }
    assert parse_truth(gt) == {
        "total": "3,000",
        "subtotal": None,
        "tax": None,
        "items": [{"name": "A", "price": "1,000"}, {"name": "B", "price": "2,000"}],
    }


def test_arithmetic_checks():
    vals = {"total": 14300, "subtotal": 13000, "tax": 1300}
    assert arithmetic_ok("total", vals, []) and arithmetic_ok("subtotal", vals, []) and arithmetic_ok("tax", vals, [])
    assert not arithmetic_ok("total", {"total": 50000, "subtotal": None, "tax": None}, [{"price": "45,500"}])


def test_item_matching_needs_price_and_name():
    truth = {"d": {"items": [{"name": "EGG TART", "price": "13,000"}, {"name": "PIZZA", "price": "16,000"}]}}
    preds = {"d": {"items": [{"name": "I EGG TART", "price": "13.000"}, {"name": "COLA", "price": "16,000"}]}}
    assert item_scores(preds, truth) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_route_votes_and_confidence():
    pred = {"total": "45,500", "subtotal": None, "tax": None, "items": [{"name": "X", "price": "45,500"}]}
    wrong = {**pred, "total": "50,000"}
    preds = {"rules-vision": {"d": pred}, "text-vision": {"d": wrong}, "vlm": {"d": pred}}
    priority = {f: ["vlm", "text-vision", "rules-vision"] for f in ("total", "subtotal", "tax", "items")}
    out = route(preds, {"d": {45500, 50000}}, priority)["d"]["fields"]
    assert out["total"] == {"value": 45500, "confidence": round(0.5 * 2 / 3 + 0.5, 4), "votes": 2}
    assert out["tax"]["confidence"] == 1.0  # all three agree it is absent


def test_threshold_is_lowest_reaching_target():
    decs = [(1.0, True)] * 99 + [(0.8, True), (0.8, False), (0.8, False), (0.5, False)]
    assert pick_threshold(decs, 0.99) == 1.0
    assert pick_threshold([(0.5, False)], 0.99) is None


def test_degenerate_vision_output():
    assert llm.degenerate("!!!!!!!!") and llm.degenerate("  ")
    assert not llm.degenerate('```json\n{"total": "8,000"}\n```')
