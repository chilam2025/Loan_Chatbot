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
  decision = "APPROVED", unavailable = false, offline = false, manual = false, validationError = false,
} = {}) {
  let answered = 0;
  let applicationStatus = "STARTED";
  let verification = { mobile_verified: false, pan_verified: false, document_received: false };
  let decisionCalls = 0;
  let submittedMobile;

  await page.route("**/ready", async (route) => {
    if (offline) return route.abort();
    if (unavailable) return route.fulfill({ status: 503, json: { status: "degraded", model_loaded: false, reference_tables_loaded: true } });
    return route.fulfill({ json: { status: "ready", model_loaded: true, reference_tables_loaded: true } });
  });
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
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
      applicationStatus = answered === questions.length ? "AWAITING_VERIFICATION" : "COLLECTING_INFORMATION";
      return route.fulfill({ json: {
        application_id: "e2e-app", status: applicationStatus, answered_questions: answered,
        total_questions: questions.length,
        next_action: answered === questions.length ? "verify_identity" : "answer_questions",
        question: questions[answered] ?? null,
      } });
    }
    if (path.endsWith("/verification/otp/request") && request.method() === "POST") {
      submittedMobile = request.postDataJSON().mobile_number;
      return route.fulfill({ json: { application_id: "e2e-app", challenge_id: "e2e-challenge", mock_code: "123456" } });
    }
    if (path.endsWith("/verification/otp/confirm") && request.method() === "POST") {
      if (manual) {
        applicationStatus = "MANUAL_REVIEW";
        return route.fulfill({ json: { status: "MANUAL_REVIEW", mobile_verified: false, pan_verified: false, document_received: false } });
      }
      verification.mobile_verified = true;
      return route.fulfill({ json: { status: "VERIFIED", ...verification } });
    }
    if (path.endsWith("/verification/pan") && request.method() === "POST") {
      verification.pan_verified = true;
      return route.fulfill({ json: { status: "VERIFIED", ...verification } });
    }
    if (path.endsWith("/verification/documents") && request.method() === "POST") {
      verification.document_received = true;
      return route.fulfill({ json: { status: "VERIFIED", ...verification } });
    }
    if (path.endsWith("/verification") && request.method() === "GET") {
      return route.fulfill({ json: { application_id: "e2e-app", status: "PENDING", ...verification } });
    }
    if (path === "/api/v1/applications/e2e-app" && request.method() === "GET") {
      return route.fulfill({ json: { application_id: "e2e-app", status: applicationStatus, locale: "en-IN" } });
    }
    if (path.endsWith("/submit") && request.method() === "POST") {
      decisionCalls += 1;
      applicationStatus = decision;
      return route.fulfill({ json: decisionPayload(decision) });
    }
    if (path.endsWith("/decision") && request.method() === "GET") return route.fulfill({ json: decisionPayload(decision) });
    return route.fulfill({ status: 404, json: { detail: "Unmocked request: " + request.method() + " " + path } });
  });
  return { submittedMobile: () => submittedMobile, decisionCalls: () => decisionCalls };
}

async function beginApplication(page, locale = "en") {
  await page.goto("/");
  if (locale === "hi") await page.getByRole("combobox", { name: "Language" }).selectOption("hi");
  await page.getByRole("button", { name: locale === "hi" ? "आवेदन शुरू करें" : "Start application" }).click();
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

async function completeVerification(page, locale = "en") {
  const labels = locale === "hi"
    ? {
      mobile: "आपका 10 अंकों का मोबाइल नंबर क्या है?",
      otp: "हमारे भेजे गए 6 अंकों का कोड दर्ज करें।",
      pan: "आपका PAN क्या है?",
      document: "कृपया आय का प्रमाण, जैसे वेतन पर्ची, अपलोड करें।",
      review: "आपके आवेदन की जानकारी",
    }
    : {
      mobile: "What is your 10-digit mobile number?",
      otp: "Enter the 6-digit code we sent.",
      pan: "What is your PAN?",
      document: "Please upload proof of income, such as a salary slip.",
      review: "Your application details",
    };
  const continueButton = page.getByRole("button", { name: /continue|आगे बढ़ें/i });

  await expect(page.getByLabel(labels.mobile)).toBeVisible();
  await page.getByLabel(labels.mobile).fill("+91 98765-43210");
  await continueButton.click();
  await expect(page.getByLabel(labels.otp)).toBeVisible();
  await page.getByLabel(labels.otp).fill("123456");
  await continueButton.click();
  await expect(page.getByLabel(labels.pan)).toBeVisible();
  await page.getByLabel(labels.pan).fill("ABCDE1234F");
  await continueButton.click();
  await expect(page.getByLabel(labels.document)).toBeVisible();
  await page.getByLabel(labels.document).setInputFiles({
    name: "salary-slip.pdf", mimeType: "application/pdf", buffer: Buffer.from("%PDF-1.4 test metadata only"),
  });
  await continueButton.click();
  await expect(page.getByRole("heading", { name: labels.review })).toBeVisible();
}

test("approved flow masks mobile and shows safe explanation and next steps", async ({ page }) => {
  const api = await mockApi(page);
  await beginApplication(page);
  await answerQuestions(page);
  await completeVerification(page);
  await page.getByRole("button", { name: "Submit application" }).click();
  const result = page.locator("#response-form");
  await expect(result.getByText(/approved in principle/i)).toBeVisible();
  await expect(result.getByText("Active-loan status")).toBeVisible();
  await expect(result.getByText(/review the sanction documents/i)).toBeVisible();
  await expect(result).not.toContainText("0.951");
  await expect(result).not.toContainText("private-model-version");
  await expect(result).not.toContainText("Location");
  await expect(page.locator("#messages")).not.toContainText("9876543210");
  expect(api.submittedMobile()).toBe("9876543210");
  await page.screenshot({ path: "test-results/decision-approved-en.png", fullPage: true });
});

test("declined flow shows the review next step", async ({ page }) => {
  await mockApi(page, { decision: "DECLINED" });
  await beginApplication(page); await answerQuestions(page); await completeVerification(page);
  await page.getByRole("button", { name: "Submit application" }).click();
  const result = page.locator("#response-form");
  await expect(result.getByText(/can’t approve/i)).toBeVisible();
  await expect(result.getByText(/review this result with the lender/i)).toBeVisible();
});

test("OTP policy routes to manual review without calling model decision", async ({ page }) => {
  const api = await mockApi(page, { manual: true });
  await beginApplication(page); await answerQuestions(page);
  await page.locator("#answer").fill("+91 98765-43210");
  await page.getByRole("button", { name: "Continue" }).click();
  await page.locator("#answer").fill("123456");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.locator("#response-form").getByText(/additional review/i)).toBeVisible();
  expect(api.decisionCalls()).toBe(0);
});

test("Hindi flow localizes active-loan explanation and next step", async ({ page }) => {
  await mockApi(page, { decision: "DECLINED" });
  await beginApplication(page, "hi"); await answerQuestions(page); await completeVerification(page, "hi");
  await page.getByRole("button", { name: "आवेदन जमा करें" }).click();
  const result = page.locator("#response-form");
  await expect(result.getByText("सक्रिय ऋण की स्थिति")).toBeVisible();
  await expect(result.getByText(/ऋणदाता से बात कर सकते हैं/)).toBeVisible();
  await page.screenshot({ path: "test-results/decision-declined-hi.png", fullPage: true });
});

test("readiness 503 shows localized unavailable message before application creation", async ({ page }) => {
  await mockApi(page, { unavailable: true });
  await page.goto("/");
  await page.getByRole("button", { name: "Start application" }).click();
  await page.getByRole("button", { name: "Yes, continue" }).click();
  await expect(page.getByRole("alert")).toContainText("decision service is temporarily unavailable");
});

test("backend offline is announced in the selected language", async ({ page }) => {
  await mockApi(page, { offline: true });
  await page.goto("/");
  await page.getByRole("combobox", { name: "Language" }).selectOption("hi");
  await page.getByRole("button", { name: "आवेदन शुरू करें" }).click();
  await page.getByRole("button", { name: "हाँ, आगे बढ़ें" }).click();
  await expect(page.getByRole("alert")).toContainText("आवेदन सेवा");
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

test("formatted +91 mobile fits the input and is normalized before submission", async ({ page }) => {
  const api = await mockApi(page);
  await beginApplication(page); await answerQuestions(page);
  await expect(page.getByLabel("What is your 10-digit mobile number?")).toBeVisible();
  const phone = page.locator("#answer");
  await expect(phone).toHaveAttribute("maxlength", "16");
  await phone.fill("+91 98765-43210");
  await page.getByRole("button", { name: "Continue" }).click();
  expect(api.submittedMobile()).toBe("9876543210");
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
