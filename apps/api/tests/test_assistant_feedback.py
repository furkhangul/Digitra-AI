import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.services import assistant

client = TestClient(app)


@pytest.fixture
def queue_path(tmp_path, monkeypatch):
    path = tmp_path / "review-queue.jsonl"
    monkeypatch.setattr(get_settings(), "feedback_queue_path", str(path))
    return path


def test_feedback_correction_is_queued_without_media(queue_path):
    response = client.post(
        "/feedback",
        json={"predicted_label": "A", "correct_label": "B", "model_version": "6.0.0", "source": "assistant"},
    )
    assert response.json() == {"received": True, "queued_for_review": True}
    record = json.loads(queue_path.read_text(encoding="utf-8").strip())
    assert record["predicted_label"] == "A"
    assert record["correct_label"] == "B"
    assert record["status"] == "pending_review"
    assert set(record) == {"received_at", "predicted_label", "correct_label", "model_version", "source", "status"}


def test_feedback_without_correction_is_not_queued(queue_path):
    response = client.post("/feedback", json={"predicted_label": "A", "correct_label": "A", "model_version": "6.0.0"})
    assert response.json()["queued_for_review"] is False
    assert not queue_path.exists()


def test_assistant_reports_missing_credentials(monkeypatch):
    monkeypatch.setattr(assistant, "_has_credentials", lambda: False)
    assert client.get("/assistant/status").json()["llm"] is False
    response = client.post("/assistant", json={"message": "Digitra nedir?"})
    assert response.status_code == 503


def test_assistant_rejects_oversized_message():
    response = client.post("/assistant", json={"message": "x" * 1001})
    assert response.status_code == 422


def test_cors_allows_launcher_origin():
    response = client.options(
        "/assistant",
        headers={
            "Origin": "http://127.0.0.1:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"
