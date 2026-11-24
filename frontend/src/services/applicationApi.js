// Client for the ABC Credit backend API (see docs/contracts/api_contract.md).
// The backend owns question order, validation, verification, and the credit decision;
// this module only transports requests and never derives an outcome itself.

const API_BASE_URL = window.ABC_API_BASE_URL
  ?? document.querySelector('meta[name="api-base-url"]')?.content
  ?? "http://localhost:8000/api/v1";
const API_ROOT_URL = new URL("/", new URL(API_BASE_URL, window.location.href)).origin;

export class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : `Request failed with status ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function request(method, path, body, baseUrl = API_BASE_URL) {
  let response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      method,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, null);
  }
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, payload?.detail ?? null);
  return payload;
}

const application = (id) => `/applications/${encodeURIComponent(id)}`;

export function startApplication(locale) {
  return request("POST", "/applications", { locale, consent_to_data_processing: true });
}

export function getReadiness() {
  return request("GET", "/ready", undefined, API_ROOT_URL);
}

export function getApplication(applicationId) {
  return request("GET", application(applicationId));
}

export function getVerification(applicationId) {
  return request("GET", `${application(applicationId)}/verification`);
}

export function getDecision(applicationId) {
  return request("GET", `${application(applicationId)}/decision`);
}

export function getNextQuestion(applicationId) {
  return request("GET", `${application(applicationId)}/next-question`);
}

export function answerQuestion(applicationId, questionId, value) {
  return request("PATCH", `${application(applicationId)}/answers`, {
    answers: [{ question_id: questionId, value }],
  });
}

export function requestOtp(applicationId, mobileNumber) {
  return request("POST", `${application(applicationId)}/verification/otp/request`, {
    mobile_number: mobileNumber,
  });
}

export function confirmOtp(applicationId, challengeId, code) {
  return request("POST", `${application(applicationId)}/verification/otp/confirm`, {
    challenge_id: challengeId,
    code,
  });
}

export function verifyPan(applicationId, pan) {
  return request("POST", `${application(applicationId)}/verification/pan`, { pan });
}

// Only file metadata is sent; document contents never leave the browser in this prototype.
export function recordIncomeDocument(applicationId, file, documentType = "SALARY_SLIP") {
  return request("POST", `${application(applicationId)}/verification/documents`, {
    document_type: documentType,
    file_name: file.name,
    content_type: file.type,
    size_bytes: file.size,
  });
}

export function submitApplication(applicationId) {
  return request("POST", `${application(applicationId)}/submit`);
}
