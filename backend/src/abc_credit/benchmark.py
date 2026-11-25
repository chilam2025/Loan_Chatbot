"""Measure end-to-end decision latency (the brief's Y) against a running API.

Each run creates an application, answers the six questions in one request,
then times ``POST /submit``: the interval between the applicant's final input
and the Approve/Decline response. Answers are synthetic and vary per run.

    uv run --project backend benchmark-latency --runs 50

Start the API first (see README). Results are printed and, with ``--output``,
written as JSON for the report.
"""

import argparse
import json
import random
import statistics
import time
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8000/api/v1"
PINCODES = ["110001", "400001", "560002", "600001", "700001", "500001", "302001", "226001"]


def _call(method: str, url: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def run(base_url: str, runs: int, seed: int) -> dict:
    rng = random.Random(seed)
    created = _call("POST", f"{base_url}/applications", {"locale": "en-IN", "consent_to_data_processing": True})
    first = _call("GET", f"{base_url}/applications/{created['application_id']}/next-question")
    # Vehicle choices come with the vehicle question, which follows age and pincode.
    while first["question"]["question_id"] != "vehicle":
        question_id = first["question"]["question_id"]
        value = 30 if question_id == "age" else "560002"
        first = _call(
            "PATCH",
            f"{base_url}/applications/{created['application_id']}/answers",
            {"answers": [{"question_id": question_id, "value": value}]},
        )
    vehicles = [choice["value"] for choice in first["question"]["choices"]]

    submit_ms: list[float] = []
    outcomes = {"APPROVED": 0, "DECLINED": 0}
    for _ in range(runs):
        application = _call(
            "POST", f"{base_url}/applications", {"locale": "en-IN", "consent_to_data_processing": True}
        )["application_id"]
        price = rng.randrange(50_000, 300_000, 1_000)
        answers = [
            {"question_id": "age", "value": rng.randint(18, 60)},
            {"question_id": "pincode", "value": rng.choice(PINCODES)},
            {"question_id": "vehicle", "value": rng.choice(vehicles)},
            {"question_id": "vehicle_price", "value": price},
            {"question_id": "loan_amount", "value": int(price * rng.uniform(0.5, 0.95))},
            {"question_id": "past_loans_active", "value": rng.choice(["PAST_LOANS_ACTIVE", "NO_PAST_LOANS"])},
        ]
        _call("PATCH", f"{base_url}/applications/{application}/answers", {"answers": answers})
        started = time.perf_counter()
        decision = _call("POST", f"{base_url}/applications/{application}/submit")
        submit_ms.append((time.perf_counter() - started) * 1000)
        outcomes[decision["decision"]] += 1

    return {
        "runs": runs,
        "measure": "POST /submit round trip: final input to Approve/Decline response",
        "median_ms": round(statistics.median(submit_ms), 1),
        "p95_ms": round(_percentile(submit_ms, 0.95), 1),
        "max_ms": round(max(submit_ms), 1),
        "outcomes": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.base_url, args.runs, args.seed)
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
