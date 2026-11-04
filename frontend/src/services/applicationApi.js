// Replace these functions with the agreed backend contract during integration.
// The mock owns question order and outcomes; presentation components do not.
const questions = ["age", "employment", "income", "loan"];
let mockId = null;
let cursor = 0;
const answers = {};

export async function startApplication() {
  mockId = `demo-${Date.now()}`;
  cursor = 0;
  Object.keys(answers).forEach((key) => delete answers[key]);
  return { id: mockId, status: "STARTED", question: "consent" };
}

export async function answerQuestion(field, value) {
  if (!mockId) throw new Error("Start an application before answering questions.");
  answers[field] = value;
  if (questions[cursor] === field) cursor += 1;
  return { accepted: true, nextQuestion: questions[cursor] ?? "review" };
}

export async function getReview() { return { applicationId: mockId, answers: { ...answers } }; }

export async function submitApplication() {
  if (!mockId) throw new Error("There is no active application.");
  return { status: "SUBMITTED", decision: "MANUAL_REVIEW", prototype: true };
}
