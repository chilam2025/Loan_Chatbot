import en from "./locales/en.js";
import hi from "./locales/hi.js";
import {
  answerQuestion,
  confirmOtp,
  getApplication,
  getDecision,
  getNextQuestion,
  getReadiness,
  getVerification,
  recordIncomeDocument,
  requestOtp,
  startApplication,
  submitApplication,
  verifyPan,
} from "./services/applicationApi.js";

// Steps: welcome → consent → question (repeated, driven by the backend) → mobile → otp
// → pan → document → review → decided. The backend decides every outcome.
const CURRENCY_QUESTIONS = new Set(["net_salary", "vehicle_price", "loan_amount"]);
const SAFE_EXPLANATION_FACTORS = new Map([
  ["Loan size relative to vehicle price", "factorLoanVehicle"],
  ["Loan amount", "factorLoanAmount"],
  ["Age", "factorAge"],
  ["Existing active loan", "factorActiveLoan"],
  ["Vehicle", "factorVehicle"],
]);
const EXPLANATION_STRENGTHS = new Map([
  ["slight", "factorStrengthSlight"],
  ["moderate", "factorStrengthModerate"],
  ["strong", "factorStrengthStrong"],
]);
const dictionaries = { en, hi };
const state = {
  language: readLanguage(),
  step: "welcome",
  applicationId: null,
  question: null,
  progress: { answered: 0, total: 0 },
  answers: [],
  challenge: null,
  decision: null,
  busy: false,
};
const SESSION_KEY = "loan-chat-application";
const $ = (selector) => document.querySelector(selector);
const messages = $("#messages");
const form = $("#response-form");

function readLanguage() {
  try { return localStorage.getItem("loan-chat-language") || "en"; } catch { return "en"; }
}
function readSession() {
  try {
    const stored = sessionStorage.getItem(SESSION_KEY);
    const value = stored ? JSON.parse(stored) : null;
    return typeof value?.applicationId === "string" ? value : null;
  } catch { return null; }
}
function persistSession() {
  try {
    if (state.applicationId) sessionStorage.setItem(SESSION_KEY, JSON.stringify({ applicationId: state.applicationId }));
    else sessionStorage.removeItem(SESSION_KEY);
  } catch { /* session storage unavailable */ }
}
function t(key) { return dictionaries[state.language][key] ?? dictionaries.en[key] ?? key; }
function currency(value) {
  return new Intl.NumberFormat(dictionaries[state.language].locale, {
    style: "currency", currency: "INR", maximumFractionDigits: 0,
  }).format(Number(value));
}

function progressPercent() {
  const { answered, total } = state.progress;
  const questionShare = total ? Math.round((answered / total) * 70) : 0;
  return ({
    welcome: 0, consent: 5, consentDeclined: 0, question: 5 + questionShare,
    mobile: 78, otp: 82, pan: 86, document: 90, review: 95, decided: 100,
  })[state.step] ?? 0;
}

function updateChrome() {
  document.documentElement.lang = state.language === "hi" ? "hi" : "en";
  $("#eyebrow").textContent = t("eyebrow"); $("#page-title").textContent = t("title"); $("#intro-copy").textContent = t("intro");
  $("#privacy-note").textContent = t("privacy"); $("#prototype-note").textContent = t("prototype");
  const percent = progressPercent();
  $("#progress-count").textContent = `${percent}%`;
  $(".progress-track").setAttribute("aria-valuenow", String(percent));
  $(".progress-value").style.width = `${percent}%`;
  $("#progress-label").textContent = t(state.step === "welcome" ? "start" : "progress");
  $("#language").value = state.language;
  $("#language").disabled = Boolean(state.applicationId);
  $("#language").title = state.applicationId ? t("languageLocked") : "";
}

function bot(text) {
  const item = document.createElement("article"); item.className = "message bot-message";
  const avatar = document.createElement("span"); avatar.className = "avatar"; avatar.setAttribute("aria-hidden", "true"); avatar.textContent = "A";
  const bubble = document.createElement("p"); bubble.className = "bubble"; bubble.textContent = text;
  item.append(avatar, bubble); messages.append(item); item.scrollIntoView({ block: "nearest" });
}
function applicant(text) {
  const item = document.createElement("p"); item.className = "message applicant-message bubble"; item.textContent = text;
  messages.append(item); item.scrollIntoView({ block: "nearest" });
}
function maskMobile(number) { return `••••••${number.slice(-4)}`; }
function button(label, action, className = "primary-button") {
  const control = document.createElement("button"); control.type = "button"; control.className = className;
  control.textContent = label; control.disabled = state.busy; control.addEventListener("click", action); return control;
}
function submitButton(label) {
  const control = document.createElement("button"); control.type = "submit"; control.className = "primary-button";
  control.textContent = state.busy ? t("working") : label; control.disabled = state.busy; return control;
}
function field(labelText, helpText, input, prefix) {
  const label = document.createElement("label"); label.className = "field-label"; label.htmlFor = "answer"; label.textContent = labelText;
  const help = document.createElement("span"); help.className = "field-help"; help.id = "answer-help"; help.textContent = helpText || "";
  const error = document.createElement("span"); error.id = "answer-error"; error.className = "field-error"; error.setAttribute("role", "alert");
  input.id = "answer"; input.required = true; input.setAttribute("aria-describedby", "answer-help answer-error");
  form.append(label);
  if (prefix) { const span = document.createElement("span"); span.className = "currency-prefix"; span.textContent = prefix; form.append(span); input.classList.add("currency-input"); }
  form.append(input, help, error);
}
function showError(message) {
  const error = $("#answer-error");
  if (error) { error.textContent = message; return; }
  const alert = document.createElement("p");
  alert.className = "message bot-message";
  alert.setAttribute("role", "alert");
  alert.textContent = message;
  messages.append(alert);
  alert.scrollIntoView({ block: "nearest" });
}
function errorMessage(error) {
  if (error?.status === 0) return t("backendOffline");
  if (error?.status === 503) return t("serviceUnavailable");
  if (error?.status === 409) return t("applicationConflict");
  if (error?.status === 404) return t("applicationNotFound");
  if (error?.status === 422) return t("validationSummary");
  if (error?.status >= 500) return t("serviceUnavailable");
  if (error?.status == null && typeof error?.detail === "string") return error.detail;
  return t("validationSummary");
}

function renderQuestionInput(question) {
  let input;
  if (question.answer_type === "choice") {
    input = document.createElement("select");
    const placeholder = document.createElement("option"); placeholder.value = ""; placeholder.textContent = t("choose"); input.append(placeholder);
    for (const choice of question.choices) { const option = document.createElement("option"); option.value = choice.value; option.textContent = choice.label; input.append(option); }
  } else {
    input = document.createElement("input"); input.autocomplete = "off";
    if (question.answer_type === "integer") {
      input.type = "number"; input.step = "1"; input.inputMode = "numeric";
      if (question.minimum !== null) input.min = String(question.minimum);
      if (question.maximum !== null) input.max = String(question.maximum);
    } else {
      input.type = "text";
      if (question.pattern) input.pattern = question.pattern.replace(/^\^|\$$/g, "");
      if (question.question_id === "pincode") { input.inputMode = "numeric"; input.maxLength = 6; }
    }
  }
  if (state.draftAnswer !== undefined) input.value = state.draftAnswer;
  const prefix = CURRENCY_QUESTIONS.has(question.question_id) ? t("currencyPrefix") : null;
  field(question.prompt, t(`${question.question_id}Help`) === `${question.question_id}Help` ? "" : t(`${question.question_id}Help`), input, prefix);
}

function renderForm() {
  form.replaceChildren(); form.hidden = false;
  switch (state.step) {
    case "welcome":
      form.append(button(t("startButton"), () => { state.step = "consent"; bot(t("consent")); refresh(); }));
      return;
    case "consent":
      form.append(
        button(t("no"), () => { state.step = "consentDeclined"; bot(t("consentDeclined")); refresh(); }, "secondary-button"),
        button(t("yes"), begin),
      );
      return;
    case "consentDeclined":
      form.append(button(t("startAgain"), restart));
      return;
    case "resuming":
      form.append(document.createTextNode(t("resuming")));
      return;
    case "resumeError":
      form.append(button(t("retryResume"), restoreExistingApplication), button(t("startAgain"), restart, "secondary-button"));
      return;
    case "decided":
      renderDecision();
      return;
    case "question":
      renderQuestionInput(state.question); break;
    case "mobile": {
      const input = document.createElement("input"); input.type = "tel"; input.inputMode = "tel"; input.maxLength = 16; input.autocomplete = "tel";
      field(t("mobile"), t("mobileHelp"), input); break;
    }
    case "otp": {
      const input = document.createElement("input"); input.type = "text"; input.inputMode = "numeric"; input.maxLength = 6; input.autocomplete = "one-time-code";
      field(t("otp"), `${t("otpHelp")} ${state.challenge.mock_code}`, input); break;
    }
    case "pan": {
      const input = document.createElement("input"); input.type = "text"; input.maxLength = 10; input.autocomplete = "off"; input.style.textTransform = "uppercase";
      field(t("pan"), t("panHelp"), input); break;
    }
    case "document": {
      const input = document.createElement("input"); input.type = "file"; input.accept = "application/pdf,image/jpeg,image/png";
      field(t("document"), t("documentHelp"), input); break;
    }
    case "review":
      renderReview(); return;
    default:
      return;
  }
  form.append(submitButton(t("continue")));
  $("#answer")?.focus();
}

function renderReview() {
  const heading = document.createElement("h2"); heading.className = "review-heading"; heading.textContent = t("reviewTitle");
  const notice = document.createElement("p"); notice.className = "review-notice"; notice.textContent = t("reviewLocked");
  const list = document.createElement("dl"); list.className = "review-list";
  if (!state.answers.length) {
    const savedNotice = document.createElement("p"); savedNotice.className = "review-notice"; savedNotice.textContent = t("savedAnswersNotice");
    form.append(heading, notice, savedNotice, list, button(state.busy ? t("working") : t("submit"), submit));
    return;
  }
  for (const { prompt, display } of state.answers) {
    const term = document.createElement("dt"); term.textContent = prompt;
    const detail = document.createElement("dd"); detail.textContent = display; list.append(term, detail);
  }
  form.append(heading, notice, list, button(state.busy ? t("working") : t("submit"), submit));
}

function renderDecision() {
  const result = state.decision;
  const heading = document.createElement("h2");
  heading.className = "review-heading";
  heading.textContent = t("resultTitle");
  const outcome = document.createElement("p");
  outcome.className = "review-notice decision-result";
  outcome.textContent = t({
    APPROVED: "approved",
    DECLINED: "declinedDecision",
    MANUAL_REVIEW: "manualReview",
  }[result?.decision] ?? "decisionUnknown");
  const caveat = document.createElement("p");
  caveat.className = "review-notice";
  caveat.textContent = t("prototypeDecisionNote");
  form.append(heading, outcome, caveat);

  const explanation = result?.explanation;
  const groups = [
    [explanation?.factors_raising_decline_score ?? [], "factorsRaising"],
    [explanation?.factors_lowering_decline_score ?? [], "factorsLowering"],
  ].map(([items, label]) => [
    items.filter((item) => SAFE_EXPLANATION_FACTORS.has(item.factor) && EXPLANATION_STRENGTHS.has(item.strength)),
    label,
  ]).filter(([items]) => items.length);

  if (groups.length) {
    const factorsHeading = document.createElement("h3");
    factorsHeading.className = "review-heading explanation-heading";
    factorsHeading.textContent = t("explanationTitle");
    form.append(factorsHeading);
    for (const [items, label] of groups) {
      const group = document.createElement("section");
      group.className = "factor-group";
      const groupHeading = document.createElement("h4");
      groupHeading.textContent = t(label);
      const list = document.createElement("ul");
      for (const item of items) {
        const row = document.createElement("li");
        row.textContent = `${t(SAFE_EXPLANATION_FACTORS.get(item.factor))}: ${t(EXPLANATION_STRENGTHS.get(item.strength))}`;
        list.append(row);
      }
      group.append(groupHeading, list);
      form.append(group);
    }
  }
  const nextActionKey = {
    review_sanction_documents: "nextApproved",
    view_decision: "nextDeclined",
  }[result?.next_action];
  if (nextActionKey) {
    const nextStep = document.createElement("p");
    nextStep.className = "review-notice next-step";
    nextStep.textContent = t(nextActionKey);
    form.append(nextStep);
  }
  form.append(button(t("startAgain"), restart, "secondary-button"));
}

function refresh() { updateChrome(); renderForm(); }

async function withBusy(action) {
  if (state.busy) return;
  state.busy = true; renderForm();
  try { await action(); }
  catch (error) { state.busy = false; renderForm(); showError(errorMessage(error)); return; }
  state.busy = false; refresh();
}

function begin() {
  return withBusy(async () => {
    const readiness = await getReadiness();
    if (readiness.status !== "ready" || !readiness.model_loaded || !readiness.reference_tables_loaded) {
      const error = new Error("Readiness check failed"); error.status = 503; throw error;
    }
    applicant(t("yes"));
    const created = await startApplication(dictionaries[state.language].locale);
    state.applicationId = created.application_id;
    persistSession();
    showQuestionnaire(await getNextQuestion(state.applicationId));
  });
}

async function restoreExistingApplication() {
  if (state.busy) return;
  state.busy = true;
  renderForm();
  try {
    const saved = readSession();
    if (!saved) { state.busy = false; restart(); return; }
    const application = await getApplication(saved.applicationId);
    state.applicationId = application.application_id;
    state.language = application.locale === "hi-IN" ? "hi" : "en";
    messages.replaceChildren();
    if (application.status === "APPROVED" || application.status === "DECLINED") {
      state.decision = await getDecision(state.applicationId);
      state.step = "decided";
      bot(t("resumedApplication"));
    } else if (application.status === "MANUAL_REVIEW") {
      state.decision = { decision: "MANUAL_REVIEW" };
      state.step = "decided";
      bot(t("manualReview"));
    } else if (application.status === "AWAITING_VERIFICATION" || application.status === "READY_FOR_REVIEW") {
      const verification = await getVerification(state.applicationId);
      if (!verification.mobile_verified) state.step = "mobile";
      else if (!verification.pan_verified) state.step = "pan";
      else if (!verification.document_received) state.step = "document";
      else state.step = "review";
      bot(t("resumedApplication"));
      bot(t(state.step === "mobile" ? "mobile" : state.step === "pan" ? "pan" : state.step === "document" ? "document" : "review"));
    } else {
      bot(t("resumedApplication"));
      showQuestionnaire(await getNextQuestion(state.applicationId));
    }
    state.busy = false;
    refresh();
  } catch (error) {
    state.busy = false;
    state.step = "resumeError";
    refresh();
    bot(errorMessage(error));
  }
}

function showQuestionnaire(progress) {
  state.progress = { answered: progress.answered_questions, total: progress.total_questions };
  if (progress.question) {
    state.step = "question"; state.question = progress.question; bot(progress.question.prompt);
  } else {
    state.step = "mobile"; state.question = null; bot(t("verificationIntro")); bot(t("mobile"));
  }
}

function displayAnswer(question, raw) {
  if (question.answer_type === "choice") return question.choices.find((choice) => choice.value === raw)?.label ?? raw;
  if (CURRENCY_QUESTIONS.has(question.question_id)) return currency(raw);
  return raw;
}

async function handleAnswer(raw) {
  const question = state.question;
  const value = question.answer_type === "integer" ? Number(raw) : raw;
  if (question.answer_type === "integer" && !Number.isInteger(value)) throw new Error(t("wholeNumber"));
  const progress = await answerQuestion(state.applicationId, question.question_id, value);
  state.draftAnswer = undefined;
  const display = String(displayAnswer(question, raw));
  applicant(display);
  state.answers.push({ prompt: question.prompt, display });
  showQuestionnaire(progress);
}

async function handleVerification(input) {
  if (state.step === "mobile") {
    // Accept common ways of writing an Indian number: spaces, dashes, +91, or a leading 0.
    const mobile = input.value.replace(/[\s-]/g, "").replace(/^(\+?91|0)(?=\d{10}$)/, "");
    if (!/^[6-9]\d{9}$/.test(mobile)) throw new Error(t("mobileError"));
    state.challenge = await requestOtp(state.applicationId, mobile);
    applicant(maskMobile(mobile)); state.step = "otp"; bot(t("otp"));
  } else if (state.step === "otp") {
    const result = await confirmOtp(state.applicationId, state.challenge.challenge_id, input.value.trim());
    applicant("••••••");
    if (result.status === "MANUAL_REVIEW") {
      state.decision = { decision: "MANUAL_REVIEW" };
      state.step = "decided";
      bot(t("manualReview"));
      return;
    }
    state.step = "pan"; bot(t("pan"));
  } else if (state.step === "pan") {
    const pan = input.value.trim().toUpperCase();
    if (!/^[A-Z]{5}\d{4}[A-Z]$/.test(pan)) throw new Error(t("panError"));
    await verifyPan(state.applicationId, pan);
    applicant(t("panMasked")); state.step = "document"; bot(t("document"));
  } else if (state.step === "document") {
    const file = input.files?.[0];
    if (!file) throw new Error(t("documentRequired"));
    await recordIncomeDocument(state.applicationId, file);
    applicant(file.name); state.step = "review"; bot(t("review"));
  }
}

function submit() {
  return withBusy(async () => {
    const decision = await submitApplication(state.applicationId);
    state.decision = decision; state.step = "decided";
    bot(t("resultTitle"));
  });
}

function restart() {
  messages.replaceChildren();
  Object.assign(state, { step: "welcome", applicationId: null, question: null, progress: { answered: 0, total: 0 }, answers: [], challenge: null, decision: null, draftAnswer: undefined });
  persistSession();
  bot(t("welcome")); refresh();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("#answer");
  if (!input) return;
  const raw = input.type === "file" ? "file" : input.value.trim();
  if (!raw || !input.checkValidity()) { showError(t("validationSummary")); input.focus(); return; }
  if (state.step === "question") state.draftAnswer = raw;
  withBusy(async () => {
    try {
      if (state.step === "question") await handleAnswer(raw);
      else await handleVerification(input);
    } catch (error) {
      if (error instanceof Error && !("status" in error)) { const wrapped = new Error(error.message); wrapped.detail = error.message; throw wrapped; }
      throw error;
    }
  });
});

$("#language").addEventListener("change", (event) => {
  state.language = event.target.value;
  try { localStorage.setItem("loan-chat-language", state.language); } catch { /* storage unavailable */ }
  refresh();
});

updateChrome();
const existingSession = readSession();
if (existingSession) {
  state.step = "resuming";
  bot(t("resuming"));
  renderForm();
  void restoreExistingApplication();
} else {
  bot(t("welcome")); renderForm();
}
