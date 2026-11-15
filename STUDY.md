# STUDY: claims-docai

Read this before an interview where this project might come up. All numbers are from `results/eval.json` (runs on
14 and 15 Nov 2026 on a MacBook with 8 GB of memory: Apple Vision OCR `fast`, Qwen2.5-1.5B-Instruct 4-bit and
Qwen2.5-VL-3B-Instruct 4-bit through MLX, greedy decoding; CORD v2, 100 validation and 100 test receipts).

## 1. The problem in one paragraph
In travel, home and motor insurance, customers upload receipts and invoices as proof of loss, and a claims handler
re-keys the total, the tax and the line items before paying. It is slow, and a wrong total is paid out. Extraction
models also make mistakes, so "how accurate is the model" is the wrong question. The useful one is: which fields can
skip a human without raising the error rate, and how much handler work is left. This repo measures that per field.

## 2. Design choices (and why)
- **CORD v2 as the stand-in.** Public (CC-BY-4.0), real photographed receipts with labelled totals, subtotals, tax
  and line items, and its own word boxes. No public insurance-claims dataset has labels like that. It is a stand-in
  for the extraction step only, and the README says so.
- **Three readers that fail differently.** Keyword rules on OCR (what claims teams usually start with), a 1.5B text
  model on the same OCR, and a 3B vision-language model on the image. Rules and the text model share the OCR's
  mistakes; the vision model doesn't. Different failure modes are what make agreement informative.
- **Gold OCR as an upper bound.** The dataset's word boxes (text and position only, never its field labels) show
  what the same readers do with perfect OCR. That separates "the OCR is bad" from "the extractor is bad".
- **Score amounts as integers and count "absent" as an answer.** `60.000`, `60,000` and `Rp. 60.000` are the same
  amount. A reader that invents a tax line on a receipt with no tax is wrong, which matters because that is the
  vision model's main failure.
- **Validation chooses everything; test is scored once.** Arm priority per field, and the auto-approve threshold for
  each precision target, are picked on validation. Test numbers are what you get from applying them unchanged.
- **Simple, explainable confidence.** 0.5 × share of readers agreeing, + 0.25 if the amount appears in the OCR,
  + 0.25 if the receipt's arithmetic holds (subtotal + tax = total, items add up, tax ≈ 10%). A handler can be told
  why a field was flagged. Fixed weights, not a fitted model: that is the first thing I would change (section 6).
- **Report several operating points, not one.** 95%, 98% and 99% target precision, with the share of fields
  auto-approved at each. The business chooses the point; the evaluation shows the trade.
- **Record/replay cassettes and a CI gate.** Every model response is stored in `cassettes/`, and OCR in
  `data/ocr/`, so GitHub Actions replays the whole evaluation in seconds with no Mac or GPU and fails if the numbers
  no longer match `results/eval.json`.
- **A review service, not just a notebook.** FastAPI `POST /claims` returns each field with confidence and a
  `needs_review` flag; `GET /review` shows the receipt next to the fields to check; corrections are logged as
  integers to `data/review/decisions.jsonl`, which is future labelled data.

## 3. Results (100 test receipts)

| Reader | Total | Subtotal | Tax | All 3 right | Line items F1 |
|---|---|---|---|---|---|
| Rules, gold OCR (upper bound) | 80% | 81% | 91% | 64% | 68% |
| Rules, Apple Vision OCR | 27% | 51% | 67% | 20% | 32% |
| 1.5B text model, Apple Vision OCR | 36% | 45% | 28% | 16% | 30% |
| 3B vision model, image | 73% | 63% | 41% | 37% | 76% |
| Pipeline (vote) | 74% | 63% | 60% | 32% | 76% |

Review queue on test, thresholds chosen on validation: at the 95% target, 69 of 300 fields auto-approved, 90% right;
at 98%, 35 of 300 auto-approved, 97% right (34/35); 99% was not reachable on validation (the top bucket was 49/50).
Median time per receipt: 10.2 s for the vision model, 2.8 s for the text model, on an 8 GB laptop.

## 4. What I learned
1. **OCR quality dominated.** The same rules fall from 80% to 27% on the total when gold OCR is replaced with
   Apple Vision `fast`. In a real system the first spend is on OCR, before any model.
2. **A small vision model skips the OCR problem** (73% on the total) but invents tax: precision 39%, recall 90%.
3. **Voting barely moves accuracy** (74% vs 73% on the total); its value is the agreement signal for confidence.
4. **High precision costs coverage.** At 97% precision only 12% of fields skip a handler. Honest framing for a claims
   team: this tells you which few fields to skip, not that the handler goes away.
5. **Validation-to-test drift is visible even on 100 receipts**: the 95% threshold gave 90% on test. With slices of
   35–84 fields, one error moves precision by about 3 points.

## 5. Failure modes I saw
- **OCR misreads amounts**: "Totsl Rp 39.6", "50.QOO", "41.COJ". Rules then find no total or the wrong one.
- **Change or cash taken as tax**: on validation-001 the vision model put the change ("Kembali 27.000") in the tax
  field; on validation-000 the text model did the same with "CHANGE 4,500".
- **Invented tax** on receipts with no tax line (the vision model's main error).
- **Degenerate output**: the 4-bit vision model sometimes answers only "!!!!" (an fp16 overflow in this MLX setup).
  Large pages (up to 2304×4096) trigger it; the fix shrinks to 1024 px and retries at 896, 768 and 640. 3 of 200
  pages still failed at every size and are scored as wrong.
- **Label disagreements**: some receipts have a printed total that the dataset does not label as the total; every
  reader is marked wrong there. I kept them rather than cleaning the test set by hand.

## 6. What I would do next
- Fit the confidence (logistic regression on the same features, on validation) and plot precision against coverage.
- Add a stronger OCR (PaddleOCR or docTR) as a fourth reader.
- Feed handler corrections back as labels and re-pick the threshold as they arrive.
- Move to claim-like documents (invoices) and add policy checks, such as a total above the claim limit.

## 7. Interview questions and answers

**1. Why not just report model accuracy?**
Because the claims team's cost is in handler time and wrong payouts. A field-level confidence with a threshold turns
accuracy into "how many fields skip a human at a given error rate". At 98% target, 12% of test fields skipped review
at 97% precision. That number, not 73% accuracy, is what an operations lead can plan with.

**2. How did you avoid overfitting the threshold?**
All choices (arm priority per field, threshold per target) were made on the 100 validation receipts and applied
once to the 100 test receipts. The drift is still visible: the 95% target gave 90% on test. With more data I would
use a larger validation set or cross-validation and report a confidence interval.

**3. Why did the vision model beat the text model so clearly?**
The text model only sees the OCR, so OCR mistakes ("Totsl Rp 39.6") become its mistakes. The vision model reads the
pixels. With gold OCR the text model scores 68% on the total, close to the vision model's 73%, so most of the gap is
OCR, not model size.

**4. Rules with perfect OCR beat both models. What does that tell you?**
Receipts are a structured, keyword-heavy format: with clean text, "TOTAL ... 39,600" is easy. The models' value is
robustness to bad input and to layouts the rules don't know. In production I would keep rules as one voter and as a
cheap explanation for handlers.

**5. How is confidence computed, and why not use the model's token probabilities?**
Agreement between three readers, whether the amount is printed in the OCR, and whether the receipt's arithmetic
holds. It is explainable to a handler and works across different models. Token log-probabilities are a reasonable
extra feature, but small 4-bit models are often confidently wrong, and they don't exist for the rules reader.

**6. Why couldn't you reach 99% precision?**
On validation the highest-confidence bucket held 50 fields with 1 error (98%), and that error was a label
disagreement. With 300 decisions you can't certify 99% anyway: one mistake is a third of a point. Reaching it needs
more validation data and a better-separated score.

**7. How does CI test something that needs a Mac and a GPU?**
Record and replay. Model responses are stored per document in `cassettes/` and OCR in `data/ocr/`; CI re-runs the
scoring, routing and threshold selection and fails if `results/eval.json` changes. A prompt change means re-recording,
so nothing reaches main unevaluated.

**8. What happens to a field below the threshold?**
`POST /claims` marks it `needs_review`; it appears in `GET /review` next to the receipt image with its value and
confidence. The handler's correction is stored as an integer amount in `decisions.jsonl`. Those become labels for the
next threshold or for fine-tuning.

**9. What would you change to run this for a real insurer?**
Real claim documents and labels, a better OCR, a fitted confidence model, authentication and audit logging on the
service, PII handling for names and card numbers on receipts, monitoring of the auto-approve rate and of
handler-overturn rate per field, and a policy layer (claim limits, duplicate receipts).

**10. What surprised you?**
How much OCR mattered (80% vs 27% for the same rules), and that the 4-bit vision model can fail silently with
"!!!!" depending on image size. Without checking raw outputs, that would have looked like a weak model rather than a
numerical bug.
