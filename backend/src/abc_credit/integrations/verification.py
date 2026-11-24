"""External identity-verification providers with explicit test doubles."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from secrets import randbelow
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4


class ProviderConfigurationError(RuntimeError):
    """Raised when a production provider is selected without its credentials."""


class ProviderUnavailableError(RuntimeError):
    """Raised when an external verification provider cannot complete a request."""


@dataclass(frozen=True)
class OtpDispatch:
    """Result of asking a provider to deliver an OTP."""

    delivery: str
    mock_code: str | None = None


class OtpProvider(Protocol):
    name: str

    def send(self, mobile_number: str) -> OtpDispatch: ...

    def verify(self, mobile_number: str, code: str) -> bool: ...


class MockOtpProvider:
    """Local test provider; never use this mode for a real applicant."""

    name = "mock"

    def send(self, mobile_number: str) -> OtpDispatch:
        del mobile_number
        return OtpDispatch(delivery="mock", mock_code=f"{randbelow(1_000_000):06d}")

    def verify(self, mobile_number: str, code: str) -> bool:
        del mobile_number, code
        # Mock codes are verified against their database digest, not by this provider.
        return False


class Msg91OtpProvider:
    """MSG91 OTP V5 adapter for DLT-registered Indian transactional SMS."""

    name = "msg91"
    base_url = "https://control.msg91.com/api/v5/otp"

    def __init__(self, auth_key: str, template_id: str, timeout_seconds: float = 8.0) -> None:
        if not auth_key or not template_id:
            raise ProviderConfigurationError(
                "MSG91 requires ABC_MSG91_AUTH_KEY and ABC_MSG91_TEMPLATE_ID"
            )
        self._auth_key = auth_key
        self._template_id = template_id
        self._timeout_seconds = timeout_seconds

    def send(self, mobile_number: str) -> OtpDispatch:
        query = urlencode(
            {
                "template_id": self._template_id,
                "mobile": f"91{mobile_number}",
                "otp_length": 6,
            }
        )
        payload = self._request(f"{self.base_url}?{query}", method="POST")
        if payload.get("type") != "success":
            raise ProviderUnavailableError("MSG91 did not accept the OTP delivery request")
        return OtpDispatch(delivery="sms")

    def verify(self, mobile_number: str, code: str) -> bool:
        query = urlencode({"otp": code, "mobile": f"91{mobile_number}"})
        payload = self._request(f"{self.base_url}/verify?{query}", method="GET")
        return payload.get("type") == "success" or payload.get("message") == "OTP verified success"

    def _request(self, url: str, *, method: str) -> dict[str, object]:
        request = Request(url, method=method, headers={"authkey": self._auth_key})
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ProviderUnavailableError("MSG91 is unavailable") from error
        if not isinstance(value, dict):
            raise ProviderUnavailableError("MSG91 returned an invalid response")
        return value


@dataclass(frozen=True)
class PanVerificationResult:
    verified: bool
    provider_status: str


class PanProvider(Protocol):
    name: str

    def verify(
        self,
        *,
        pan: str,
        full_name: str,
        date_of_birth: date,
    ) -> PanVerificationResult: ...


class MockPanProvider:
    """Format-only provider retained for tests and local development."""

    name = "mock"

    def verify(
        self,
        *,
        pan: str,
        full_name: str,
        date_of_birth: date,
    ) -> PanVerificationResult:
        del pan, full_name, date_of_birth
        return PanVerificationResult(verified=True, provider_status="MOCK_VALID")


class ProteanPanProvider:
    """Protean Online PAN Verification v4 adapter.

    Production use requires an eligible-entity account and the signing certificate
    registered with Protean. The request signature is created locally; PAN data is
    never logged or persisted by this application.
    """

    name = "protean"

    def __init__(
        self,
        *,
        user_id: str,
        private_key_path: Path,
        endpoint: str,
        private_key_password: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not user_id or private_key_path == Path("."):
            raise ProviderConfigurationError(
                "Protean requires ABC_PROTEAN_USER_ID and ABC_PROTEAN_PRIVATE_KEY_PATH"
            )
        self._user_id = user_id
        self._private_key_path = private_key_path
        self._private_key_password = private_key_password
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds

    def verify(
        self,
        *,
        pan: str,
        full_name: str,
        date_of_birth: date,
    ) -> PanVerificationResult:
        input_data = [
            {
                "pan": pan,
                "name": full_name,
                "fathername": "",
                "dob": date_of_birth.strftime("%d/%m/%Y"),
            }
        ]
        signature = self._sign(input_data)
        now = datetime.now(UTC)
        transaction_id = f"{self._user_id}:{now.strftime('%Y%m%d%H%M%S')}{uuid4().hex[:8]}"
        body = json.dumps({"inputData": input_data, "signature": signature}).encode("utf-8")
        request = Request(
            self._endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User_ID": self._user_id,
                "Records_count": "1",
                "Request_time": now.strftime("%Y-%m-%dT%H:%M:%S"),
                "Transaction_ID": transaction_id,
                "Version": "4",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise ProviderUnavailableError("Protean PAN verification is unavailable") from error
        if not isinstance(payload, dict) or str(payload.get("response_Code")) != "1":
            raise ProviderUnavailableError("Protean rejected the PAN verification request")
        records = payload.get("outputData")
        if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
            raise ProviderUnavailableError("Protean returned an invalid response")
        record = records[0]
        pan_status = str(record.get("pan_status", ""))
        matched = record.get("name") == "Y" and record.get("dob") == "Y"
        operative = record.get("seeding_status") != "R"
        return PanVerificationResult(
            verified=pan_status == "E" and matched and operative,
            provider_status=pan_status or "UNKNOWN",
        )

    def _sign(self, input_data: list[dict[str, str]]) -> str:
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
        except ImportError as error:  # pragma: no cover - dependency is declared
            raise ProviderConfigurationError("cryptography is required for Protean") from error
        try:
            key_bytes = self._private_key_path.read_bytes()
            password = (
                self._private_key_password.encode("utf-8")
                if self._private_key_password
                else None
            )
            key = serialization.load_pem_private_key(key_bytes, password=password)
            canonical = json.dumps(
                input_data, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            signed = key.sign(canonical, padding.PKCS1v15(), hashes.SHA256())
        except (OSError, ValueError, TypeError) as error:
            raise ProviderConfigurationError("Protean signing key could not be loaded") from error
        return base64.b64encode(signed).decode("ascii")


def configured_otp_provider() -> OtpProvider:
    provider = os.getenv("ABC_OTP_PROVIDER", "mock").strip().lower()
    if provider == "mock":
        return MockOtpProvider()
    if provider == "msg91":
        return Msg91OtpProvider(
            os.getenv("ABC_MSG91_AUTH_KEY", ""),
            os.getenv("ABC_MSG91_TEMPLATE_ID", ""),
        )
    raise ProviderConfigurationError(f"Unsupported OTP provider: {provider}")


def configured_pan_provider() -> PanProvider:
    provider = os.getenv("ABC_PAN_PROVIDER", "mock").strip().lower()
    if provider == "mock":
        return MockPanProvider()
    if provider == "protean":
        return ProteanPanProvider(
            user_id=os.getenv("ABC_PROTEAN_USER_ID", ""),
            private_key_path=Path(os.getenv("ABC_PROTEAN_PRIVATE_KEY_PATH", "")),
            private_key_password=os.getenv("ABC_PROTEAN_PRIVATE_KEY_PASSWORD"),
            endpoint=os.getenv(
                "ABC_PROTEAN_ENDPOINT",
                "https://opvapi.egov-nsdl.com/TIN/PanInquiryAPIBackEnd",
            ),
        )
    raise ProviderConfigurationError(f"Unsupported PAN provider: {provider}")
