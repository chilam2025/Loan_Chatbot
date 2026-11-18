# Loan application conversation flow

**Status:** Draft for review by Persons 1 and 2. The current repository has no
model contract or API contract, so field selection, request shapes, and server
validation messages remain subject to those reviews.

## Conversation principles

- Ask one clear question at a time and show progress through the application.
- Explain why sensitive information is needed before requesting it.
- Let applicants go back and edit answers before submission.
- Treat answers as input only. The backend validates values and returns the
  decision; the conversation must not infer or change credit eligibility.
- Explain that OTP, PAN, DigiLocker, and document checks in this academic
  prototype are simulations, not connections to government services.
- Use the server's application ID and state. Do not create a decision locally.

## Journey

| Stage | Chat experience | Data / action | Branching and safeguards |
|---|---|---|---|
| Welcome | Introduce the vehicle-loan application and estimated time. | Start application through the API. | Explain that a final result comes from the backend and that some applications may need manual review. |
| Consent | Explain the information collected, prototype verification, and use for an application decision. Ask the applicant to agree before continuing. | Record consent through the API. | If consent is declined, end politely without collecting application details. |
| Eligibility | Ask age and residence details required by the agreed model and product rules. | Age, resident type, six-digit Indian pincode. | Validate format deterministically. Eligibility rules and age limits must come from the backend contract. |
| Employment and income | Ask employment type, qualification where required, and monthly net income. | Employment type, qualification, net salary. | Show localized INR input. Request clarification for missing or invalid values; do not estimate income. |
| Vehicle and loan | Ask for vehicle make, model, variant, product, vehicle price or applicable tier, and requested loan amount. | Product code, make, model description, model variant, final tier, loan amount, vehicle price. | Backend calculates LTV. Do not ask for LTV as an applicant answer. Skip vehicle fields that the API/model contract derives. |
| Additional details | Ask for gender and active past-loan information only if required and approved by the model and product contract. | Gender and active past loans, if approved. | Explain optionality only if the backend confirms the field is optional. Never invent categories. |
| Verification | Explain each requested check, then present the prototype OTP, PAN, document, address, income, and DigiLocker-consent steps as applicable. | Submit verification actions and display returned statuses. | Clearly label simulations. Never request or display OTP values in logs; use backend statuses (`PENDING`, `VERIFIED`, `FAILED`, `REQUIRES_REVIEW`). |
| Application review | Summarize personal, employment, vehicle, loan, and verification information. | Read current answers and server-calculated LTV. | Provide edit controls that return to the relevant question. Require explicit confirmation before submission. |
| Submission | Confirm the applicant is ready to submit. | Submit once through the API. | Disable repeat submission while pending and handle duplicate-submit responses safely. |
| Decision | Show the returned approved, declined, or manual-review outcome. | Render backend decision and approved amount/next action if supplied. | Do not infer reasons for declines. Show prototype sanction documents only when supplied and label them prototypes. |
| Next steps | Explain the next action returned for the outcome, plus support/contact details from approved configuration. | No local decision logic. | Manual review explains that more verification is needed; decline wording stays neutral and offers the available review/contact route. |

## Question wording and input contract (draft)

English and Hindi text below is a first wording pass. Each question's required
status, category options, conditions, and validation message must be reconciled
with the API contract before implementation. Inputs are collected only when the
backend says they are needed.

| Field | English | Hindi | Input / validation and help | Required / condition |
|---|---|---|---|---|
| Consent | May we use the information you provide to process this application? | क्या हम आपके आवेदन को संसाधित करने के लिए आपके द्वारा दी गई जानकारी का उपयोग कर सकते हैं? | Choice: Yes / No. Help: Verification steps in this prototype are simulations. | Required to continue. |
| Age | How old are you? | आपकी उम्र कितनी है? | Whole-number age. Error: Enter your age as a whole number. Help: Enter your age in years. | Required if confirmed by backend/model contract. |
| Pincode | What is your six-digit Indian PIN code? | आपका छह अंकों का भारतीय पिन कोड क्या है? | Six digits. Error: Enter a valid six-digit PIN code. Help: We use this to check the address area. | Required if confirmed by backend/model contract. |
| Residence | Which best describes your residence? | आपके निवास का सबसे सही वर्णन कौन-सा है? | Searchable choice from backend-provided options. Error: Choose one of the listed options. | Required if confirmed by backend/model contract. |
| Employment | What type of work do you do? | आप किस प्रकार का काम करते हैं? | Searchable choice from backend-provided options. Error: Choose one of the listed work types. | Required if confirmed by backend/model contract. |
| Qualification | What is your highest qualification? | आपकी उच्चतम शैक्षणिक योग्यता क्या है? | Choice from backend-provided options. Error: Choose one of the listed options. | Ask only if required by the contract. |
| Net salary | What is your monthly take-home income? | आपकी मासिक हाथ में मिलने वाली आय कितनी है? | INR amount. Error: Enter a valid monthly amount in rupees. Help: Enter the amount you receive after deductions. | Required if confirmed by backend/model contract. |
| Product | What type of vehicle loan are you applying for? | आप किस प्रकार के वाहन ऋण के लिए आवेदन कर रहे हैं? | Choice from backend-provided products. Error: Choose one of the listed loan types. | Required if contract requires applicant selection. |
| Vehicle make | What is the vehicle make? | वाहन का निर्माता कौन है? | Searchable selection from backend-provided makes. Error: Choose a listed make. | Required if applicable to selected product. |
| Vehicle model | Which vehicle model are you interested in? | आप किस वाहन मॉडल में रुचि रखते हैं? | Searchable selection filtered by make. Error: Choose a listed model. | Required if applicable to selected product. |
| Variant | Which variant do you want? | आप कौन-सा वेरिएंट चाहते हैं? | Searchable selection filtered by model. Error: Choose a listed variant. | Required if applicable to selected product. |
| Vehicle price | What is the vehicle price in rupees? | वाहन की कीमत रुपये में कितनी है? | INR amount. Error: Enter a valid vehicle price in rupees. | Ask if required to calculate LTV. |
| Loan amount | How much would you like to borrow? | आप कितनी ऋण राशि लेना चाहते हैं? | INR amount. Error: Enter a valid loan amount in rupees. Help: The requested amount must be within the limits shown by the lender. | Required. Backend supplies limits. |
| Gender | Which gender option, if any, would you like to provide? | आप कौन-सा लिंग विकल्प देना चाहेंगे, यदि कोई हो? | Choice from backend-approved options, including a skip option if permitted. Error: Choose an available option. | Ask only if approved and required by the contract. |
| Active past loans | Do you currently have any active loans? | क्या आपके पास अभी कोई सक्रिय ऋण है? | Yes / No / Unsure if supported by backend. Error: Choose one option. | Ask only if approved and required by the contract. |
| OTP | Enter the code sent to your phone. | आपके फ़ोन पर भेजा गया कोड दर्ज करें। | One-time code field. Error: That code could not be verified. Request a new code or try again. | Only during mock OTP verification. |
| PAN | Enter your PAN for the prototype verification step. | प्रोटोटाइप सत्यापन के लिए अपना PAN दर्ज करें। | PAN input; format and verification outcome come from backend. Error: Check the PAN details and try again. | Only if verification is requested; explain purpose first. |

## Validation and display rules

- Store monetary values as INR amounts in the API's agreed numeric format.
- Format display values with Indian grouping, for example `₹25,000`, `₹85,000`,
  and `₹1,50,000`. Keep parsing and formatting in shared localization utilities.
- Pincode input accepts exactly six digits; the backend remains authoritative.
- Display server validation messages next to the relevant question and announce
  errors to assistive technology.
- Do not derive LTV in a visual component. Display only a server-calculated value.
- Preserve the selected language while navigating, reviewing, and correcting answers.
- Do not expose application identifiers in general logs or error messages.

## Open contract decisions

1. Person 1: confirm the minimal model feature set, required fields, approved
   categories, and which questions can be skipped or derived.
2. Person 2: define endpoints, answer payloads, server-side validation errors,
   question sequencing, verification request/status formats, and consent recording.
3. Team: approve the Hindi wording, consent language, help/contact details, and
   which verification steps appear in the academic prototype.
