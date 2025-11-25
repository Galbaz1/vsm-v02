#!/usr/bin/env python
"""
parse_with_landingai.py

Usage:
    python parse_with_landingai.py data/uk_firmware.pdf output_landingai.json

- Uses LandingAI ADE Parse Jobs to parse large PDFs (up to ~1000 pages).
- Saves the full JSON response (chunks, markdown, metadata) which preserves tables/images/headings and grounding info.
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv


API_BASE = "https://api.va.landing.ai/v1"
POLL_INTERVAL_SECONDS = 10
MAX_WAIT_SECONDS = 60 * 60  # 1 hour, adjust if you like


def create_parse_job(pdf_path: str, api_key: str) -> str:
    """Create a Parse Job and return job_id (LandingAI ADE Parse Jobs)."""
    url = f"{API_BASE}/ade/parse/jobs"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # Model: dpt-2-latest (as of Nov 2025, resolves to dpt-2-20251103 snapshot)
    # See "Parse Large Files (Parse Jobs)" – automatically uses latest DPT-2 version.
    with open(pdf_path, "rb") as f:
        files = {"document": f}
        data = {"model": "dpt-2-latest"}
        resp = requests.post(url, files=files, data=data, headers=headers)
    resp.raise_for_status()
    body = resp.json()
    job_id = body.get("job_id")
    if not job_id:
        raise RuntimeError(f"No job_id in response: {body}")
    return job_id


def wait_for_job_and_get_result(job_id: str, api_key: str) -> dict:
    """
    Poll Get Async Job Status until the job is completed.
    Return the full data dictionary (markdown, chunks, metadata, etc.).
    """
    url = f"{API_BASE}/ade/parse/jobs/{job_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    waited = 0
    while True:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        print(f"[LandingAI] Job {job_id} status: {status}")

        if status == "completed":
            # Check for direct data or output_url
            data_block = data.get("data")
            if data_block:
                return data_block
            
            output_url = data.get("output_url")
            if output_url:
                print(f"[LandingAI] Job completed. Downloading results from: {output_url}")
                result_resp = requests.get(output_url)
                result_resp.raise_for_status()
                return result_resp.json()
                
            raise RuntimeError("Job completed but no data or output_url found in response.")

        elif status in ("failed", "error", "canceled"):
            raise RuntimeError(f"Parse job ended with status {status}: {data}")
        else:
            time.sleep(POLL_INTERVAL_SECONDS)
            waited += POLL_INTERVAL_SECONDS
            if waited > MAX_WAIT_SECONDS:
                raise TimeoutError(
                    f"Timed out waiting for job {job_id} after {MAX_WAIT_SECONDS} seconds."
                )


def main():
    if len(sys.argv) < 3:
        print("Usage: python parse_with_landingai.py data/uk_firmware.pdf output_landingai.json")
        sys.exit(1)

    # Load environment variables from .env file
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    pdf_path = sys.argv[1]
    out_json_path = sys.argv[2]

    api_key = os.environ.get("LANDINGAI_API_KEY")
    if not api_key:
        print("Error: Please set LANDINGAI_API_KEY in .env file or environment variable.")
        sys.exit(1)

    if not os.path.isfile(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"[LandingAI] Creating parse job for {pdf_path}...")
    job_id = create_parse_job(pdf_path, api_key)
    print(f"[LandingAI] Created job_id: {job_id}")

    print("[LandingAI] Waiting for parsing to complete...")
    result_data = wait_for_job_and_get_result(job_id, api_key)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"[LandingAI] JSON results saved to: {out_json_path}")


if __name__ == "__main__":
    main()
