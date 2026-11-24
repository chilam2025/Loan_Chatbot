# Production assistant and identity integrations

## Safety boundary

The CatBoost pipeline remains the only component that can return the recorded credit
decision. Qwen answers process questions only. It must not receive PAN, OTP, full phone,
bank-account or Aadhaar values and must never change a decision or explanation factor.

## Qwen through Ollama

1. Install Ollama on the inference host.
2. Pull the configured model: `ollama pull qwen3:4b`.
3. Set `ABC_ASSISTANT_PROVIDER=ollama` and, if needed, change
   `ABC_OLLAMA_BASE_URL` and `ABC_OLLAMA_MODEL`.
4. Confirm `POST /api/v1/assistant/messages` succeeds before exposing the assistant UI.

The model weights have no per-request fee, but compute, monitoring and hosting are not
free. For stronger server hardware, evaluate a larger Qwen3 Instruct checkpoint against
a fixed English/Hindi safety and loan-FAQ test set before changing the model tag.

## MSG91 OTP

1. Register the legal entity on an Indian operator DLT portal.
2. Register the sender/header and the exact service OTP template.
3. Map the approved DLT template in MSG91 and fund the account.
4. Set `ABC_OTP_PROVIDER=msg91`, a random `ABC_VERIFICATION_HASH_KEY` of at least
   32 characters, `ABC_MSG91_AUTH_KEY`, and `ABC_MSG91_TEMPLATE_ID`.
5. Test delivery, retry, expiry, incorrect-code and carrier-failure behavior using
   non-customer test numbers before production.

The API never returns a provider-generated OTP. Changing the phone between request and
confirmation is rejected before MSG91 is called. Do not log request bodies or provider
URLs because they contain the transient phone or OTP.

## Protean Online PAN Verification

Protean requires an eligible-entity registration, user ID, registered signing
certificate, legal name and date of birth. Complete Protean UAT before production.
Mount the private key as a secret (never in the image or repository), then set
`ABC_PAN_PROVIDER=protean`, `ABC_PROTEAN_USER_ID`,
`ABC_PROTEAN_PRIVATE_KEY_PATH`, and any key password. Use the UAT endpoint during
certification and switch to production only after Protean approval.

The adapter treats only an existing PAN with matching name/date of birth and an
operative status as verified. PAN, name and date of birth are used transiently and are
not stored in the application tables or audit details.

## Before serving real applicants

- Add an API gateway with TLS, request-size limits, IP/per-number OTP throttling,
  resend cooldowns, bot protection and security monitoring.
- Put secrets in a managed secret store and rotate them.
- Use PostgreSQL in India-hosted infrastructure, encrypted backups and documented
  retention/deletion controls.
- Complete penetration, privacy, accessibility, model-risk, fairness, lender-policy
  and regulatory reviews.
- Replace document-metadata simulation with encrypted object storage, malware
  scanning, content verification and explicit retention controls.
