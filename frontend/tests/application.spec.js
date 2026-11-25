import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const questions = [
  { question_id: "age", answer_type: "integer", prompt: "How old are you?", minimum: 18, maximum: 60 },
  { question_id: "pincode", answer_type: "text", prompt: "What is your pincode?", pattern: "^[0-9]{6}$" },
  {
    question_id: "vehicle", answer_type: "choice", prompt: "Choose your vehicle",
    choices: [{ value: "RAW-VALUE", label: "Clean vehicle label" }],
  },
  { question_id: "vehicle_price", answer_type: "integer", prompt: "What is the vehicle price?", minimum: 1 },
  { question_id: "loan_amount", answer_type: "integer", prompt: "How much would you like to borrow?", minimum: 1 },
  {
    question_id: "past_loans_active", answer_type: "choice", prompt: "Do you have an active loan?",
    choices: [
      { value: "PAST_LOANS_ACTIVE", label: "Yes" },
      { value: "NO_PAST_LOANS", label: "No" },
    ],
  },
];

function decisionPayload(decision) {
  return {
    application_id: "e2e-app",
    status: decision,
    decision,
    probability_declined: 0.951,
    threshold: 0.11,
    model_version: "private-model-version",
    model_provider: "trained",
    next_action: decision === "APPROVED" ? "review_sanction_documents" : "view_decision",
    explanation: {
      factors_raising_decline_score: [
        { factor: "Existing active loan", strength: "moderate", log_odds_contribution: 0.87 },
        { factor: "Location", strength: "strong", log_odds_contribution: 1.23 },
      ],
      factors_lowering_decline_score: [
        { factor: "Age", strength: "slight", log_odds_contribution: -0.1 },
      ],
      caveat: "Model explanation",
    },
  };
}

async function mockApi(page, {
  decision = "APPROVED", unavailable = false, offline = false, validationError = false,
  resumeFailOnce = false,
} = {}) {
  let answered = 0;
  let applicationStatus = "STARTED";
  let shouldFailResume = resumeFailOnce;
  let decisionCalls = 0;
  const assistantRequests = [];
  const requestedPaths = [];

  await page.route("**/ready", async (route) => {
    if (offline) return route.abort();
    if (unavailable) return route.fulfill({ status: 503, json: { status: "degraded", model_loaded: false, reference_tables_loaded: true } });
    return route.fulfill({ json: { status: "ready", model_loaded: true, reference_tables_loaded: true } });
  });
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    requestedPaths.push(path);
    if (request.method() === "POST" && path === "/api/v1/applications") {
      return route.fulfill({ status: 201, json: { application_id: "e2e-app", status: "STARTED", locale: request.postDataJSON().locale } });
    }
    if (path.endsWith("/next-question") && request.method() === "GET") {
      return route.fulfill({ json: {
        application_id: "e2e-app", status: applicationStatus, answered_questions: answered,
        total_questions: questions.length, next_action: "answer_questions", question: questions[answered] ?? null,
      } });
    }
    if (path.endsWith("/answers") && request.method() === "PATCH") {
      if (validationError && answered === 0) {
        return route.fulfill({ status: 422, json: { detail: "Age does not meet backend rules" } });
      }
      answered += request.postDataJSON().answers.length;
      applicationStatus = answered === questions.length ? "READY_FOR_REVIEW" : "COLLECTING_INFORMATION";
      return route.fulfill({ json: {
        application_id: "e2e-app", status: applicationStatus, answered_questions: answered,
        total_questions: questions.length,
        next_action: answered === questions.length ? "review_and_submit" : "answer_questions",
        question: questions[answered] ?? null,
      } });
    }
    if (path === "/api/v1/applications/e2e-app" && request.method() === "GET") {
      if (shouldFailResume) {
        shouldFailResume = false;
        return route.fulfill({ status: 503, json: { detail: "Temporary backend failure" } });
      }
      return route.fulfill({ json: { application_id: "e2e-app", status: applicationStatus, locale: "en-IN" } });
    }
    if (path.endsWith("/submit") && request.method() === "POST") {
      decisionCalls += 1;
      applicationStatus = decision;
      return route.fulfill({ json: decisionPayload(decision) });
    }
    if (path.endsWith("/decision") && request.method() === "GET") return route.fulfill({ json: decisionPayload(decision) });
    if (path === "/api/v1/assistant/ask" && request.method() === "POST") {
      const body = request.postDataJSON();
      assistantRequests.push(body);
      return route.fulfill({ json: { answer: `Saarthi answer ${assistantRequests.length}`, source: "generative" } });
    }
    return route.fulfill({ status: 404, json: { detail: "Unmocked request: " + request.method() + " " + path } });
  });
  return { decisionCalls: () => decisionCalls, assistantRequests, requestedPaths };
}

async function beginApplication(page, locale = "en") {
  await page.goto("/");
  if (locale === "hi") await page.getByRole("combobox", { name: "Language" }).selectOption("hi");
  await page.getByRole("button", { name: locale === "hi" ? "आवेदन शुरू करें" : "Start application" }).click();
  await expect(page.getByRole("button", { name: locale === "hi" ? "हाँ, आगे बढ़ें" : "Yes, continue" })).toBeVisible();
  await page.getByRole("button", { name: locale === "hi" ? "हाँ, आगे बढ़ें" : "Yes, continue" }).click();
}

async function answerQuestions(page) {
  const values = ["30", "110001", "RAW-VALUE", "90000", "60000", "NO_PAST_LOANS"];
  for (const [index, value] of values.entries()) {
    const question = questions[index];
    await expect(page.locator(".field-label")).toHaveText(question.prompt);
    const input = page.getByLabel(question.prompt);
    if (question.answer_type === "choice") await input.selectOption(value);
    else await input.fill(value);
    await page.getByRole("button", { name: /continue|आगे बढ़ें/i }).click();
  }
}

async function reachReview(page, locale = "en") {
  await expect(page.getByRole("heading", { name: locale === "hi" ? "आपके आवेदन की जानकारी" : "Your application details" })).toBeVisible();
}

test("approved flow asks only six questions and shows only the result and next steps", async ({ page }) => {
  const api = await mockApi(page);
  await beginApplication(page);
  await answerQuestions(page);
  await reachReview(page);
  await page.getByRole("button", { name: "Submit application" }).click();
  const result = page.locator("#response-form");
  await expect(result.getByText(/approved in principle/i)).toBeVisible();
  await expect(result.getByText(/review the sanction documents/i)).toBeVisible();
  for (const hidden of ["0.951", "0.11", "private-model-version", "Location", "Active-loan status", "Model explanation", "Factors"]) {
    await expect(page.locator("body")).not.toContainText(hidden);
  }
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);
  await expect(page.locator("body")).not.toContainText(/What is your (10-digit mobile number|PAN)|upload proof of income|6-digit code/i);
  await expect(page.locator("input[type=file], input[type=tel]")).toHaveCount(0);
  expect(api.requestedPaths.some((path) => path.includes("verification"))).toBe(false);
  await page.screenshot({ path: "test-results/decision-approved-en.png", fullPage: true });
});

test("declined flow shows the review next step", async ({ page }) => {
  await mockApi(page, { decision: "DECLINED" });
  await beginApplication(page); await answerQuestions(page); await reachReview(page);
  await page.getByRole("button", { name: "Submit application" }).click();
  const result = page.locator("#response-form");
  await expect(result.getByText(/can’t approve/i)).toBeVisible();
  await expect(result.getByText(/review this result with the lender/i)).toBeVisible();
});

test("Hindi flow localizes the result and next step without explanations", async ({ page }) => {
  await mockApi(page, { decision: "DECLINED" });
  await beginApplication(page, "hi"); await answerQuestions(page); await reachReview(page, "hi");
  await page.getByRole("button", { name: "आवेदन जमा करें" }).click();
  const result = page.locator("#response-form");
  await expect(result.getByText(/स्वीकृत नहीं कर सकते/)).toBeVisible();
  await expect(result.getByText(/ऋणदाता से बात कर सकते हैं/)).toBeVisible();
  await expect(result).not.toContainText("सक्रिय ऋण की स्थिति");
  await page.screenshot({ path: "test-results/decision-declined-hi.png", fullPage: true });
});

test("readiness 503 shows localized unavailable message before consent", async ({ page }) => {
  await mockApi(page, { unavailable: true });
  await page.goto("/");
  await page.getByRole("button", { name: "Start application" }).click();
  await expect(page.getByText("The decision service is temporarily unavailable. Please try again later.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Yes, continue" })).toHaveCount(0);
});

test("backend offline is announced in the selected language", async ({ page }) => {
  await mockApi(page, { offline: true });
  await page.goto("/");
  await page.getByRole("combobox", { name: "Language" }).selectOption("hi");
  await page.getByRole("button", { name: "आवेदन शुरू करें" }).click();
  await expect(page.getByRole("alert")).toContainText("आवेदन सेवा");
  await expect(page.getByRole("button", { name: "हाँ, आगे बढ़ें" })).toHaveCount(0);
});

test("backend validation detail is localized instead of exposing English API text", async ({ page }) => {
  await mockApi(page, { validationError: true });
  await beginApplication(page, "hi");
  await page.locator("#answer").fill("30");
  await page.getByRole("button", { name: "आगे बढ़ें" }).click();
  await expect(page.getByRole("alert")).toContainText("कृपया अपना जवाब जाँचकर फिर से कोशिश करें।");
  await expect(page.getByRole("alert")).not.toContainText("Age does not meet backend rules");
});

test("required-answer errors are announced and return focus to the invalid control", async ({ page }) => {
  await mockApi(page);
  await beginApplication(page);
  await page.locator("#answer").fill("");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("alert")).toContainText("Please check your answer");
  await expect(page.locator("#answer")).toBeFocused();
});

test("application resumes at its unanswered question after reload", async ({ page }) => {
  await mockApi(page);
  await beginApplication(page);
  await page.locator("#answer").fill("30");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect.poll(() => page.evaluate(() => JSON.parse(sessionStorage.getItem("loan-chat-application"))))
    .toEqual({ applicationId: "e2e-app" });
  await page.reload();
  await expect(page.getByLabel("What is your pincode?")).toBeVisible();
  await expect(page.locator("#answer")).toHaveAttribute("inputmode", "numeric");
});

test("completed application resumes at review after reload", async ({ page }) => {
  await mockApi(page);
  await beginApplication(page);
  await answerQuestions(page);
  await reachReview(page);
  await page.reload();
  await reachReview(page);
  await expect(page.getByRole("button", { name: "Submit application" })).toBeVisible();
});

test("failed session recovery can be retried successfully", async ({ page }) => {
  await mockApi(page, { resumeFailOnce: true });
  await page.addInitScript(() => sessionStorage.setItem("loan-chat-application", JSON.stringify({ applicationId: "e2e-app" })));
  await page.goto("/");
  await expect(page.getByText("The decision service is temporarily unavailable. Please try again later.")).toBeVisible();
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByLabel("How old are you?")).toBeVisible();
});

test("home is keyboard reachable and meets automated accessibility rules", async ({ page }) => {
  await mockApi(page);
  await page.goto("/");
  await page.keyboard.press("Tab");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("combobox", { name: "Language" })).toBeFocused();
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});

test("Saarthi answers free-form questions with history but no application data", async ({ page }) => {
  const api = await mockApi(page);
  await beginApplication(page);
  await page.getByRole("button", { name: "Ask Saarthi", exact: true }).click();
  const panel = page.getByRole("dialog", { name: "Saarthi" });
  await expect(panel).toBeVisible();
  await panel.getByRole("button", { name: "Do I need any documents?" }).click();
  await expect(panel.getByText("Saarthi answer 1")).toBeVisible();
  await panel.getByLabel("Your question").fill("And how is EMI calculated?");
  await panel.getByRole("button", { name: "Send" }).click();
  await expect(panel.getByText("Saarthi answer 2")).toBeVisible();

  expect(api.assistantRequests[0]).toEqual({ question: "Do I need any documents?", locale: "en-IN", history: [] });
  expect(api.assistantRequests[1].history).toEqual([
    { role: "user", content: "Do I need any documents?" },
    { role: "assistant", content: "Saarthi answer 1" },
  ]);
  expect(JSON.stringify(api.assistantRequests)).not.toContain("e2e-app");
  const accessibility = await new AxeBuilder({ page }).analyze();
  expect(accessibility.violations).toEqual([]);

  await page.keyboard.press("Escape");
  await expect(panel).toBeHidden();
  await expect(page.getByRole("button", { name: "Ask Saarthi", exact: true })).toBeFocused();
});

test("Saarthi speaks Hindi when Hindi is selected", async ({ page }) => {
  const api = await mockApi(page);
  await page.goto("/");
  await page.getByRole("combobox", { name: "Language" }).selectOption("hi");
  await page.getByRole("button", { name: "सारथी से पूछें", exact: true }).click();
  const panel = page.getByRole("dialog", { name: "सारथी" });
  await expect(panel.getByText(/नमस्ते, मैं सारथी हूँ/)).toBeVisible();
  await panel.getByRole("button", { name: "कौन आवेदन कर सकता है?" }).click();
  await expect(panel.getByText("Saarthi answer 1")).toBeVisible();
  expect(api.assistantRequests[0].locale).toBe("hi-IN");
});

test("mobile viewport has no horizontal overflow and application form passes axe", async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 740 });
  await mockApi(page);
  await beginApplication(page);
  const dimensions = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
  }));
  expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport);
  const results = await new AxeBuilder({ page }).analyze();
  expect(results.violations).toEqual([]);
});
