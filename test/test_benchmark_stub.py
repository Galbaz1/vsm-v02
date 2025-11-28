import json
import os
import sys
from pathlib import Path
import tempfile
import types

# Configure DSPy cache to a writable repo-local dir before importing api/dspy
_cache_dir = Path(__file__).resolve().parents[1] / ".dspy-cache-test"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["DSPY_CACHE_DIR"] = str(_cache_dir)
os.environ["DSPY_CACHE_PATH"] = str(_cache_dir / "cache.sqlite")
os.environ.setdefault("DSPY_DISABLE_CACHE", "1")

# Provide a lightweight dspy stub to avoid disk-backed cache initialization during tests
if "dspy" not in sys.modules:
    dspy_stub = types.SimpleNamespace()

    class _Field:
        def __init__(self, *args, **kwargs):
            pass

    class _Signature:
        pass

    class _Module:
        pass

    class _ChainOfThought:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

    class _Prediction:
        pass

    dspy_stub.Signature = _Signature
    dspy_stub.InputField = _Field
    dspy_stub.OutputField = _Field
    dspy_stub.Module = _Module
    dspy_stub.ChainOfThought = _ChainOfThought
    dspy_stub.Prediction = _Prediction
    sys.modules["dspy"] = dspy_stub

# Ensure project root is on sys.path for test execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from api.prompts.signatures import SIGNATURE_MAP, TechnicalJudgeSignature
from api.services.benchmark import load_benchmark_dataset, run_benchmark_stub
from api.main import app


def _write_sample_dataset(tmp_path: Path) -> Path:
    data = [
        {
            "id": 1,
            "query": "What is the supply voltage?",
            "expected_answer": "24V",
            "evidence": {
                "document": "techman.pdf",
                "locations": [{"page": 10, "manual": "Technical Manual"}],
            },
        },
        {
            "id": 2,
            "question": "How to reset the alarm?",
            "answer": "Press reset button.",
            "reference": {"answer": "Press reset button."},
        },
    ]
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_technical_judge_signature_registered():
    assert "technical_judge" in SIGNATURE_MAP
    assert SIGNATURE_MAP["technical_judge"] is TechnicalJudgeSignature


def test_load_benchmark_dataset(tmp_path):
    dataset_path = _write_sample_dataset(tmp_path)
    items = load_benchmark_dataset(str(dataset_path))
    assert len(items) == 2
    assert items[0].id == "1"
    assert items[0].query.startswith("What is")
    assert items[0].expected_sources is not None
    assert items[1].expected_answer.startswith("Press reset")


def test_run_benchmark_stub_returns_structure(tmp_path):
    dataset_path = _write_sample_dataset(tmp_path)
    import asyncio

    result = asyncio.run(
        run_benchmark_stub(dataset_path=str(dataset_path), mode="local", enable_tracing=True)
    )
    assert result.total == 2
    assert result.summary["status"] == "not_implemented"
    assert result.summary["enable_tracing"] is True
    assert len(result.records) == 2
    assert result.records[0]["status"] == "pending"


def test_benchmark_endpoint_stub(tmp_path):
    dataset_path = _write_sample_dataset(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/benchmark/evaluate",
        json={"dataset_path": str(dataset_path), "enable_tracing": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["summary"]["status"] == "not_implemented"
    assert body["result"]["total"] == 2
    assert body["result"]["summary"]["enable_tracing"] is True
