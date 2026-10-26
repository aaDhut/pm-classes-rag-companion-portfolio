# Evaluation results

## Method

`uv run python -m eval.accuracy [case ids or types] --out <name>`

For every test case, the harness:

1. Runs the question (and any follow-up turns) through the same code path as the live app, including the fact-check.
2. Has **Gemini 3.1 Pro** split the answer into individual claims and grade each one as correct, wrong or unsupported against the source material.
3. Checks that the required facts (`must`) are covered and that nothing in `must_not_mention` appears.
4. For cases with hand-picked evidence, measures retrieval **context precision / recall** (are the retrieved chunks the right ones?).
5. Logs tokens, cost and time for each step (answer, fact-check, slide scoring).

The test set has 190 hand-written cases across 14 question types. It stays private because it quotes course material; the format is in [`cases.example.json`](cases.example.json).

## Main run: 25 Oct 2026 (183 cases)

**99.28% of claims correct · 168/183 answers fully clean · 0 errors**

- Claims checked: 3,748 (3,721 correct, 27 wrong or unsupported)
- Required facts covered: 252/288 (87.5%)
- Fact-check fixes applied: 57
- Cost of the run (answers + grading): $22.13

| Type | Cases | Clean | Claims | Accuracy |
|---|---|---|---|---|
| class_summary | 32 | 27 | 1586 | 99.6% |
| fact | 71 | 67 | 1004 | 99.6% |
| filter | 1 | 1 | 16 | 100.0% |
| followup | 12 | 8 | 196 | 92.3% |
| image_slide | 4 | 4 | 50 | 100.0% |
| language | 1 | 1 | 20 | 100.0% |
| not_covered | 4 | 4 | 16 | 100.0% |
| numbers | 20 | 20 | 244 | 100.0% |
| pictures | 5 | 5 | 59 | 100.0% |
| quiz | 5 | 5 | 25 | 100.0% |
| table | 12 | 11 | 154 | 99.4% |
| topic_summary | 6 | 5 | 269 | 99.6% |
| where | 10 | 10 | 109 | 100.0% |

The remaining failures were reviewed one by one and fixed in targeted re-tests. Some "wrong" verdicts were the grader's mistake (for example, trusting jumbled OCR text from a slide), so every verdict was checked against the transcript before changing the app.

## Later full re-run: 26 Oct 2026 (188 cases)

**97.03% of claims correct** (4,049 of 4,173) after the test set grew and stricter cases were added. The biggest drops were "where was it taught" (83%) and image-only slides (88%). Those types guided the next round of fixes. Both runs are reported so the numbers aren't cherry-picked.

## Cost vs accuracy experiments (50-question set)

| Setting | Accuracy | Cost / answer | Fact-check time | Decision |
|---|---|---|---|---|
| Fact-check thinking = medium | 95.9% | $0.0515 | 10.2 s | previous default |
| Fact-check thinking = low | 95.6% | $0.0355 | 4.3 s | **shipped**: −31% cost, same accuracy |
| Everything on Flash-Lite | 93.5% | ~$0.007 | – | rejected: lost accuracy, and the Lite checker often returned nothing |
| Route easy questions to Flash-Lite | 94.8% | est. ~$0.028 | – | not shipped yet |
