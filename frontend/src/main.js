import en from "./locales/en.js";
import hi from "./locales/hi.js";
import {
  answerQuestion,
  askAssistant,
  getApplication,
  getDecision,
  getNextQuestion,
  getReadiness,
  startApplication,
  submitApplication,
} from "./services/applicationApi.js";

// Steps: welcome → consent → question (repeated, driven by the backend) → review → decided.
// No personal or identity details (phone, PAN, documents) are ever collected.
// The backend decides every outcome.
const CURRENCY_QUESTIONS = new Set(["net_salary", "vehicle_price", "loan_amount"]);
// Index of the current stage in the stepper; "decided" marks every stage complete.
const STAGE_BY_STEP = { question: 0, review: 1, decided: 3 };
const SVG_NS = "http://www.w3.org/2000/svg";
// Saarthi's mark: a steering wheel (a "saarthi" is a charioteer — the one who guides).
const AVATAR_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="2.2"/><path d="M12 14.2v6.2M9.9 11.3 3.7 9.6M14.1 11.3l6.2-1.7"/></svg>';
const DECISION_ICONS = {
  APPROVED: "M5 12.5l4.5 4.5L19 7.5",
  DECLINED: "M7 7l10 10M17 7 7 17",
  MANUAL_REVIEW: "M12 7v5l3 2M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z",
};
const dictionaries = { en, hi };
const state = {
  language: readLanguage(),
  step: "welcome",
  applicationId: null,
  question: null,
  progress: { answered: 0, total: 0 },
  answers: [],
  decision: null,
  busy: false,
};
const assistant = { history: [], busy: false, started: false };
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
function locale() { return dictionaries[state.language].locale; }
function currency(value) {
  return new Intl.NumberFormat(locale(), { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(Number(value));
}
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function avatar() {
  const node = el("span", "agent-avatar"); node.setAttribute("aria-hidden", "true"); node.innerHTML = AVATAR_SVG; return node;
}

function progressPercent() {
  const { answered, total } = state.progress;
  const questionShare = total ? Math.round((answered / total) * 80) : 0;
  return ({
    welcome: 0, consent: 5, consentDeclined: 0, question: 5 + questionShare, review: 90, decided: 100,
  })[state.step] ?? 0;
}

function renderStepper() {
  const current = STAGE_BY_STEP[state.step] ?? -1;
  const stepper = $("#stepper");
  stepper.setAttribute("aria-label", t("stagesLabel"));
  stepper.replaceChildren(...t("stages").map((label, index) => {
    const item = el("li");
    if (index < current) item.className = "is-done";
    if (index === current) { item.className = "is-current"; item.setAttribute("aria-current", "step"); }
    const dot = el("span", "step-dot", index < current ? "✓" : String(index + 1)); dot.setAttribute("aria-hidden", "true");
    item.append(dot, el("span", "step-label", label));
    return item;
  }));
}

function updateChrome() {
  document.documentElement.lang = state.language === "hi" ? "hi" : "en";
  const text = {
    "eyebrow": "eyebrow", "page-title": "title", "brand-subtitle": "brandSubtitle",
    "highlight-time": "highlightTime", "highlight-docs": "highlightDocs", "highlight-instant": "highlightInstant",
    "chat-name": "assistantLabel", "chat-status": "chatStatus", "privacy-note": "privacy", "footer-ask": "footerAsk",
    "launcher-label": "launcher", "assistant-title": "assistantLabel", "assistant-tagline": "assistantTagline",
    "assistant-input-label": "assistantInputLabel", "assistant-note": "assistantNote",
  };
  for (const [id, key] of Object.entries(text)) $("#" + id).textContent = t(key);
  $("#assistant-input").placeholder = t("assistantPlaceholder");
  $("#assistant-send").setAttribute("aria-label", t("assistantSend"));
  $("#assistant-close").setAttribute("aria-label", t("assistantClose"));
  const percent = progressPercent();
  $("#progress-count").textContent = `${percent}%`;
  $(".progress-track").setAttribute("aria-valuenow", String(percent));
  $(".progress-value").style.width = `${percent}%`;
  $("#progress-label").textContent = t(state.step === "welcome" ? "start" : "progress");
  renderStepper();
  $("#language").value = state.language;
  $("#language").disabled = Boolean(state.applicationId);
  $("#language").title = state.applicationId ? t("languageLocked") : "";
}

function botMessage(container, text) {
  const item = el("article", "message bot-message");
  const body = el("div", "message-body");
  body.append(el("span", "message-label", t("assistantLabel")), el("p", "bubble message-copy", text));
  item.append(avatar(), body);
  container.append(item);
  container.scrollTop = container.scrollHeight;
  return item;
}
function applicantMessage(container, text) {
  const item = el("article", "message applicant-message");
  item.append(el("span", "message-label", t("applicantLabel")), el("p", "answer-text", text));
  container.append(item);
  container.scrollTop = container.scrollHeight;
}
function typingIndicator(container) {
  const item = el("div", "message bot-message typing-row"); item.setAttribute("aria-hidden", "true");
  const dots = el("span", "typing"); dots.append(el("span"), el("span"), el("span"));
  item.append(avatar(), dots); container.append(item); container.scrollTop = container.scrollHeight;
  return item;
}
function bot(text) { botMessage(messages, text); }
function applicant(text) { applicantMessage(messages, text); }

function button(label, action, className = "primary-button") {
  const control = el("button", className, label); control.type = "button";
  control.disabled = state.busy; control.addEventListener("click", action); return control;
}
function submitButton(label) {
  const control = el("button", "primary-button", state.busy ? t("working") : label); control.type = "submit";
  control.disabled = state.busy; return control;
}
function field(labelText, helpText, input, prefix) {
  const label = el("label", "field-label", labelText); label.htmlFor = "answer";
  const help = el("span", "field-help", helpText || ""); help.id = "answer-help";
  const error = el("span", "field-error"); error.id = "answer-error"; error.setAttribute("role", "alert");
  input.id = "answer"; input.required = true; input.setAttribute("aria-describedby", "answer-help answer-error");
  form.append(label);
  if (prefix) { form.append(el("span", "currency-prefix", prefix)); input.classList.add("currency-input"); }
  form.append(input, help, error);
}
function showError(message) {
  const error = $("#answer-error");
  if (error) { error.textContent = message; return; }
  const alert = el("p", "message alert-message", message);
  alert.setAttribute("role", "alert");
  messages.append(alert);
  messages.scrollTop = messages.scrollHeight;
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
    const placeholder = el("option", "", t("choose")); placeholder.value = ""; input.append(placeholder);
    for (const choice of question.choices) { const option = el("option", "", choice.label); option.value = choice.value; input.append(option); }
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
  const helpKey = `${question.question_id}Help`;
  field(question.prompt, t(helpKey) === helpKey ? "" : t(helpKey), input, prefix);
}

function renderForm() {
  form.replaceChildren(); form.hidden = false;
  switch (state.step) {
    case "welcome":
      form.append(button(t("startButton"), showConsent));
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
    case "review":
      renderReview(); return;
    default:
      return;
  }
  form.append(submitButton(t("continue")));
  $("#answer")?.focus();
}

function renderReview() {
  const heading = el("h2", "review-heading", t("reviewTitle"));
  const notice = el("p", "review-notice", t("reviewLocked"));
  const list = el("dl", "review-list");
  const submitControl = button(state.busy ? t("working") : t("submit"), submit);
  if (!state.answers.length) {
    form.append(heading, notice, el("p", "review-notice", t("savedAnswersNotice")), submitControl);
    return;
  }
  for (const { prompt, display } of state.answers) list.append(el("dt", "", prompt), el("dd", "", display));
  form.append(heading, notice, list, submitControl);
}

function renderDecision() {
  const decision = state.decision?.decision;
  const variant = { APPROVED: "approved", DECLINED: "declined", MANUAL_REVIEW: "review" }[decision];
  const card = el("div", `decision-card${variant ? ` decision-${variant}` : ""}`);
  if (DECISION_ICONS[decision]) {
    const icon = el("span", "decision-icon"); icon.setAttribute("aria-hidden", "true");
    const svg = document.createElementNS(SVG_NS, "svg"); svg.setAttribute("viewBox", "0 0 24 24");
    const path = document.createElementNS(SVG_NS, "path"); path.setAttribute("d", DECISION_ICONS[decision]);
    svg.append(path); icon.append(svg); card.append(icon);
  }
  card.append(
    el("h2", "review-heading", t("resultTitle")),
    el("p", "decision-result", t({ APPROVED: "approved", DECLINED: "declinedDecision", MANUAL_REVIEW: "manualReview" }[decision] ?? "decisionUnknown")),
  );
  const nextActionKey = decision === "MANUAL_REVIEW" ? "nextReview" : {
    review_sanction_documents: "nextApproved",
    view_decision: "nextDeclined",
  }[state.decision?.next_action];
  if (nextActionKey) card.append(el("p", "review-notice next-step", t(nextActionKey)));
  const actions = el("div", "decision-actions");
  actions.append(button(t("startAgain"), restart, "secondary-button"));
  card.append(actions);
  form.append(card);
}

function refresh() { updateChrome(); renderForm(); }

async function withBusy(action) {
  if (state.busy) return;
  state.busy = true; renderForm();
  const typing = typingIndicator(messages);
  try { await action(); }
  catch (error) { typing.remove(); state.busy = false; renderForm(); showError(errorMessage(error)); return; }
  typing.remove();
  state.busy = false; refresh();
}

async function ensureReady() {
  const readiness = await getReadiness();
  if (readiness.status !== "ready" || !readiness.model_loaded || !readiness.reference_tables_loaded) {
    const error = new Error("Readiness check failed"); error.status = 503; throw error;
  }
}

function begin() {
  return withBusy(async () => {
    await ensureReady();
    applicant(t("yes"));
    const created = await startApplication(locale());
    state.applicationId = created.application_id;
    persistSession();
    showQuestionnaire(await getNextQuestion(state.applicationId));
  });
}

function showConsent() {
  return withBusy(async () => {
    await ensureReady();
    state.step = "consent";
    bot(t("consent"));
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
    const resumedLanguage = application.locale === "hi-IN" ? "hi" : "en";
    if (resumedLanguage !== state.language) { state.language = resumedLanguage; resetAssistant(); }
    messages.replaceChildren();
    if (application.status === "APPROVED" || application.status === "DECLINED") {
      state.decision = await getDecision(state.applicationId);
      state.step = "decided";
      bot(t("resumedApplication"));
    } else if (application.status === "READY_FOR_REVIEW") {
      state.step = "review";
      bot(t("resumedApplication"));
      bot(t("review"));
    } else if (application.status !== "STARTED" && application.status !== "COLLECTING_INFORMATION") {
      // An application left over from the old identity-check flow can't be submitted any more.
      state.busy = false;
      restart();
      return;
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
    state.step = "review"; state.question = null; bot(t("review"));
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

function submit() {
  return withBusy(async () => {
    state.decision = await submitApplication(state.applicationId);
    state.step = "decided";
    bot(t("resultIntro"));
  });
}

function restart() {
  messages.replaceChildren();
  Object.assign(state, { step: "welcome", applicationId: null, question: null, progress: { answered: 0, total: 0 }, answers: [], decision: null, draftAnswer: undefined });
  persistSession();
  bot(t("welcome")); refresh();
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("#answer");
  if (!input) return;
  const raw = input.value.trim();
  if (!raw || !input.checkValidity()) { showError(t("validationSummary")); input.focus(); return; }
  state.draftAnswer = raw;
  withBusy(async () => {
    try {
      await handleAnswer(raw);
    } catch (error) {
      if (error instanceof Error && !("status" in error)) { const wrapped = new Error(error.message); wrapped.detail = error.message; throw wrapped; }
      throw error;
    }
  });
});

// --- Saarthi: free-form customer questions -------------------------------------
// Separate from the application: it never sees answers, identifiers or decisions.

const assistantPanel = $("#assistant-panel");
const assistantMessages = $("#assistant-messages");
const assistantInput = $("#assistant-input");
const launcher = $("#assistant-launcher");

function resetAssistant() {
  assistant.history = [];
  assistant.started = false;
  assistantMessages.replaceChildren();
  botMessage(assistantMessages, t("assistantGreeting"));
  renderSuggestions();
}

function renderSuggestions() {
  const container = $("#assistant-suggestions");
  if (assistant.started) { container.replaceChildren(); return; }
  container.replaceChildren(...t("suggestions").map((text) => {
    const chip = el("button", "chip", text); chip.type = "button";
    chip.addEventListener("click", () => askSaarthi(text));
    return chip;
  }));
}

function setAssistantOpen(open) {
  assistantPanel.hidden = !open;
  launcher.setAttribute("aria-expanded", String(open));
  if (open) assistantInput.focus();
  else launcher.focus();
}

async function askSaarthi(question) {
  const text = question.trim();
  if (!text || assistant.busy) return;
  assistant.busy = true;
  assistant.started = true;
  renderSuggestions(); // removes the chips, so keep keyboard focus in the panel
  assistantInput.focus();
  $("#assistant-send").disabled = true;
  applicantMessage(assistantMessages, text);
  const typing = typingIndicator(assistantMessages);
  try {
    const reply = await askAssistant(text, locale(), assistant.history);
    typing.remove();
    botMessage(assistantMessages, reply.answer);
    assistant.history.push({ role: "user", content: text }, { role: "assistant", content: reply.answer.slice(0, 1500) });
    assistant.history = assistant.history.slice(-12);
  } catch {
    typing.remove();
    botMessage(assistantMessages, t("assistantError"));
  } finally {
    assistant.busy = false;
    $("#assistant-send").disabled = false;
  }
}

launcher.addEventListener("click", () => setAssistantOpen(assistantPanel.hidden));
$("#footer-ask").addEventListener("click", () => setAssistantOpen(true));
$("#assistant-close").addEventListener("click", () => setAssistantOpen(false));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !assistantPanel.hidden) setAssistantOpen(false);
});
$("#assistant-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = assistantInput.value;
  assistantInput.value = "";
  assistantInput.focus();
  void askSaarthi(question);
});

$("#language").addEventListener("change", (event) => {
  state.language = event.target.value;
  try { localStorage.setItem("loan-chat-language", state.language); } catch { /* storage unavailable */ }
  resetAssistant();
  if (!state.applicationId) restart();
  else refresh();
});

for (const node of document.querySelectorAll(".agent-avatar")) node.innerHTML = AVATAR_SVG;
updateChrome();
resetAssistant();
const existingSession = readSession();
if (existingSession) {
  state.step = "resuming";
  bot(t("resuming"));
  renderForm();
  void restoreExistingApplication();
} else {
  bot(t("welcome")); renderForm();
}
