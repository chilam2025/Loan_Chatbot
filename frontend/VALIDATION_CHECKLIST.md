# Frontend validation checklist

The Playwright suite covers the UI with stubbed API responses. Complete the
human usability, assistive-technology, and live-stack checks below before final
submission. Do not use real applicant identifiers or documents.

## Timed usability sessions

Run with representative participants using the same browser, device class, and
language mix expected for the demo. Do not coach participants through the flow.
Start timing when they choose **Start application**. Stop when the decision,
explanation, and next-step message are visible. Record completion time, whether
they completed without help, the first point of confusion, and any accessibility
or language issue. Capture consent separately; do not collect names or other
identifiers.

Use at least 20 completed sessions if reporting p95. Sort completion times in
ascending order and take the value at rank `ceil(0.95 × n)` (one-based nearest
rank). Report the sample size, device/browser, language counts, median, p95, and
the target of five minutes. Include incomplete sessions in a separate count and
describe why they stopped; do not silently omit them. With fewer than 20
completions, label p95 as preliminary and avoid claiming the five-minute target
is validated.

| Session | Language | Device class | Completion time | Completed unaided? | Issue observed |
| --- | --- | --- | --- | --- | --- |
| 1 |  |  |  |  |  |

## Manual accessibility review

- Use only the keyboard: Tab, Shift+Tab, Enter, Space, and arrow keys. Check
  logical focus order, visible focus, language selector, chat history, form
  controls, retry actions, and decision controls.
- With a screen reader (for example NVDA with Firefox or Chrome), confirm page
  title and language, progress, each prompt and help text, validation errors,
  newly added chat messages, decision, explanation, and next step are announced
  once and in a useful order.
- Trigger a required-field error and a backend error. Confirm the error is
  announced and focus remains on or returns to the relevant control.
- Inspect contrast and text scaling at 200%. Check English and Hindi, including
  the active-loan factor label and both decision outcomes.
- At 320px and 200% zoom, confirm there is no horizontal scrolling, clipped
  text, or obscured controls. Repeat at a typical desktop viewport.
- Record browser, screen reader, viewport, issues, and retest result. Automated
  axe checks are useful coverage but do not replace this review.

## Live Docker/browser walkthrough

Ensure private model and reference-table artifacts are installed locally and
that `.env` is configured from `.env.example`. Start from the repository root:

```powershell
docker compose up --build
```

Wait for the frontend at `http://localhost:8080` and confirm the backend
readiness endpoint reports the model and reference tables loaded. Walk through
approved, declined, and manual-review outcomes in English and Hindi. For each
decision, verify the explanation and localized next step are visible and that
probability, threshold, pincode, and raw model values are not shown. Also test
an unavailable model/readiness response and a reload during an in-progress
application.

Capture screenshots of the welcome/consent state, one in-progress application,
each outcome, both languages, and the unavailable state. Record commit SHA,
browser/version, viewport, locale, and outcome with the evidence. Use synthetic
answers and redact application identifiers. Distinguish these live screenshots
from the existing mocked Playwright screenshots.
