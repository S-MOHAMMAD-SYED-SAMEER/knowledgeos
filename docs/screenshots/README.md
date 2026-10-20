# Screenshots

Taken on 2026-10-20 from the real application (`app.main:app`, default prompt `knowledge_answer_v2`, default threshold 1.4451, real local models, a real PostgreSQL with pgvector, the fictional 12-document fixture corpus). The answers were produced by `gemini-3.5-flash-lite` during the recording of the project video (take 2); these images are page loads of the stored answers, so taking them made no model request. All data is fictional.

| File | What it shows |
|---|---|
| `answer-page-1440x900.png` | An answered question: numbered citations, "Citations verified" (every citation points at a passage the model was given and every sentence has one; it does not check that the passage supports the sentence), and each source with its title, version and a short quotation. The page is 1,162 px tall at 1440 wide with the evidence collapsed. |
| `answer-evidence-open-1440x900.png` | The same page after clicking citation [1]: the retrieved passages open on the cited one. The passage is the stored chunk text as ingested, including the document's own "##" heading marks. |
| `answer-current-version-1440x900.png` | A question whose answer cites the newer Production Database Access SOP (three days) and the older Security Policy (seven days). |
| `abstention-1440x900.png` | A question the documents don't cover: "Abstained, insufficient evidence", no citations, no "Citations verified" badge. It was stopped by the abstention threshold before the model was asked. |
| `evaluation-page-1440x900.png` | The Evaluation page, read from the committed results in `docs/results/`: counts with their samples, and what they were scored against. |
| `answer-page-390x844.png` | The answer page on a phone (no horizontal scroll). |
| `busy-page-test-double-1440x900.png` | The "The model is busy" page. **Produced by a test double, not by a real outage.** |

These are not results about answer quality. The figures, and their limits, are in the README and `docs/results/2026-10-20-answers-eval.md`.
