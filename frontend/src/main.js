import en from "./locales/en.js";
import hi from "./locales/hi.js";
import { answerQuestion, getReview, startApplication, submitApplication } from "./services/applicationApi.js";
import "./styles.css";

const dictionaries = { en, hi };
const state = { language: localStorage.getItem("loan-chat-language") || "en", step: "welcome", answers: {}, busy: false };
const $ = (selector) => document.querySelector(selector);
const messages = $("#messages");
const form = $("#response-form");
function t(key) { return dictionaries[state.language][key] ?? dictionaries.en[key] ?? key; }
function currency(value) { return new Intl.NumberFormat(dictionaries[state.language].locale, { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(Number(value)); }
function updateChrome() {
  document.documentElement.lang = state.language === "hi" ? "hi" : "en";
  $("#eyebrow").textContent = t("eyebrow"); $("#page-title").textContent = t("title"); $("#intro-copy").textContent = t("intro");
  $("#privacy-note").textContent = t("privacy"); $("#prototype-note").textContent = t("prototype");
  const percent = ({ welcome: 0, consent: 8, age: 25, employment: 45, income: 62, loan: 78, review: 92, submitted: 100, declined: 0 })[state.step] ?? 0;
  $("#progress-count").textContent = `${percent}%`; $(".progress-track").setAttribute("aria-valuenow", String(percent)); $(".progress-value").style.width = `${percent}%`;
  $("#progress-label").textContent = t(state.step === "welcome" ? "start" : "progress"); $("#language").value = state.language;
}
function bot(text) {
  const item = document.createElement("article"); item.className = "message bot-message";
  const avatar = document.createElement("span"); avatar.className = "avatar"; avatar.setAttribute("aria-hidden", "true"); avatar.textContent = "A";
  const bubble = document.createElement("p"); bubble.className = "bubble"; bubble.textContent = text; item.append(avatar, bubble); messages.append(item); item.scrollIntoView({ block: "nearest" });
}
function applicant(text) { const item = document.createElement("p"); item.className = "message applicant-message bubble"; item.textContent = text; messages.append(item); item.scrollIntoView({ block: "nearest" }); }
function button(label, action, className = "primary-button") { const control = document.createElement("button"); control.type = "button"; control.className = className; control.textContent = label; control.addEventListener("click", action); return control; }
function renderForm() {
  form.replaceChildren(); form.hidden = false;
  if (["welcome", "consent", "declined", "submitted"].includes(state.step)) {
    if (state.step === "consent") form.append(button(t("no"), () => { state.step = "declined"; bot(t("consentDeclined")); updateChrome(); renderForm(); }, "secondary-button"));
    form.append(button(state.step === "welcome" ? t("startButton") : state.step === "consent" ? t("yes") : t("startAgain"), handlePrimary));
    return;
  }
  if (state.step === "review") { renderReview(); return; }
  const label = document.createElement("label"); label.className = "field-label"; label.htmlFor = "answer"; label.textContent = t(state.step);
  const help = document.createElement("span"); help.className = "field-help"; help.id = "answer-help"; help.textContent = t(`${state.step}Help`) || "";
  let input; let prefixElement = null;
  if (state.step === "employment") {
    input = document.createElement("select"); input.id = "answer"; input.required = true; input.setAttribute("aria-describedby", "answer-help answer-error");
    for (const [value, key] of [["", "choose"], ["salaried", "salaried"], ["self_employed", "selfEmployed"], ["other", "otherWork"]]) { const option = document.createElement("option"); option.value = value; option.textContent = t(key); input.append(option); }
    if (state.answers.employment) input.value = state.answers.employment;
  } else {
    input = document.createElement("input"); input.id = "answer"; input.required = true; input.autocomplete = "off"; input.setAttribute("aria-describedby", "answer-help answer-error");
    if (state.step === "age") { input.type = "number"; input.min = "1"; input.step = "1"; input.inputMode = "numeric"; }
    else { input.type = "number"; input.min = "1"; input.step = "1"; input.inputMode = "decimal"; input.classList.add("currency-input"); input.setAttribute("aria-label", `${t(state.step)} (${t("currencyPrefix")})`); }
    input.value = state.answers[state.step] ?? "";
    if (state.step !== "age") { prefixElement = document.createElement("span"); prefixElement.className = "currency-prefix"; prefixElement.textContent = t("currencyPrefix"); }
  }
  const error = document.createElement("span"); error.id = "answer-error"; error.className = "field-error"; error.setAttribute("role", "alert");
  form.append(label); if (prefixElement) form.append(prefixElement); form.append(input, help, error, button(t("back"), goBack, "text-button"));
  const next = document.createElement("button"); next.type = "submit"; next.className = "primary-button"; next.textContent = t("continue"); form.append(next); input.focus();
}
async function renderReview() {
  state.answers = (await getReview()).answers;
  const heading = document.createElement("h2"); heading.className = "review-heading"; heading.textContent = t("reviewTitle");
  const notice = document.createElement("p"); notice.className = "review-notice"; notice.textContent = t("review");
  const list = document.createElement("dl"); list.className = "review-list";
  for (const [key, label] of [["age", "ageLabel"], ["employment", "workLabel"], ["income", "incomeLabel"], ["loan", "loanLabel"]]) {
    const term = document.createElement("dt"); term.textContent = t(label); const detail = document.createElement("dd");
    detail.textContent = key === "income" || key === "loan" ? currency(state.answers[key]) : key === "employment" ? t(({ salaried: "salaried", self_employed: "selfEmployed", other: "otherWork" })[state.answers[key]]) : state.answers[key]; list.append(term, detail);
  }
  form.append(heading, notice, list, button(t("edit"), goBack, "secondary-button"));
  const submit = button(state.busy ? t("working") : t("submit"), handlePrimary); submit.disabled = state.busy; form.append(submit);
}
function goBack() { const order = ["age", "employment", "income", "loan", "review"]; const index = order.indexOf(state.step); state.step = order[Math.max(0, index - 1)]; bot(t(state.step)); updateChrome(); renderForm(); }
async function handlePrimary() {
  if (state.busy) return;
  if (state.step === "welcome") { await startApplication(); state.step = "consent"; bot(t("consent")); }
  else if (state.step === "consent") { applicant(t("yes")); state.step = "age"; bot(t("age")); }
  else if (state.step === "review") {
    state.busy = true; renderForm();
    try { const result = await submitApplication(); state.step = "submitted"; bot(t("submitted")); if (result.decision === "MANUAL_REVIEW") bot(t("manualReview")); }
    catch { bot(t("validationSummary")); }
    finally { state.busy = false; }
  } else if (state.step === "declined" || state.step === "submitted") {
    messages.replaceChildren(); state.answers = {}; state.step = "welcome"; await startApplication(); bot(t("welcome"));
  }
  updateChrome(); renderForm();
}
form.addEventListener("submit", async (event) => {
  event.preventDefault(); const input = $("#answer"); const error = $("#answer-error"); const value = input.value.trim();
  if (!value || (state.step === "age" && (!Number.isInteger(Number(value)) || Number(value) < 1)) || (state.step !== "age" && state.step !== "employment" && !(Number(value) > 0))) {
    error.textContent = state.step === "age" ? t("ageError") : t("amountError"); input.focus(); return;
  }
  state.answers[state.step] = state.step === "employment" ? value : Number(value);
  applicant(state.step === "income" || state.step === "loan" ? currency(value) : state.step === "employment" ? t(({ salaried: "salaried", self_employed: "selfEmployed", other: "otherWork" })[value]) : value);
  try { const result = await answerQuestion(state.step, state.answers[state.step]); state.step = result.nextQuestion; bot(state.step === "review" ? t("review") : t(state.step)); }
  catch { error.textContent = t("validationSummary"); return; }
  updateChrome(); renderForm();
});
$("#language").addEventListener("change", (event) => { state.language = event.target.value; localStorage.setItem("loan-chat-language", state.language); updateChrome(); renderForm(); });
updateChrome(); bot(t("welcome")); renderForm();
