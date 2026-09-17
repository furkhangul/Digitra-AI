import base64
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def jpeg_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]["ready"] is True
    assert body["model"]["version"] == "5.0.0"
    assert len(body["model"]["tid_classes"]) == 29


def test_predict_no_hand_is_not_fabricated():
    response = client.post("/predict", json={"mode": "balanced", "landmarks": []})
    assert response.status_code == 200
    body = response.json()
    assert body["hand_detected"] is False
    assert body["reject_reason"] == "NO_HAND"
    assert body["prediction"] is None


def test_only_measured_model_metrics_are_exposed():
    response = client.get("/model-metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert len(metrics) == 3
    assert metrics[0]["internal_accuracy"] == 0.8909512761020881
    assert "kişi bağımsız değildir" in metrics[0]["evaluation_scope"]


def test_live_session_rejects_when_no_hand_is_visible():
    with client.websocket_connect("/ws/recognize") as websocket:
        websocket.send_json({"hands": []})
        body = websocket.receive_json()
        assert body["model_ready"] is True
        assert body["hands_detected"] == 0
        assert body["prediction"] is None
        assert body["reject_reason"] == "NO_HAND"


def test_models_are_registered_with_truthful_scopes():
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert {item["id"] for item in models} == {
        "digitra-tid-robust",
        "digitra-landmark-personal",
        "digitra-tid-specials-temporal",
    }
    legacy = next(item for item in models if item["id"] == "digitra-landmark-personal")
    assert "TİD modeli olarak sunulmaz" in legacy["description"]


def test_tid_image_is_routed_to_robust_v5():
    image = Image.new("RGB", (64, 64), (114, 114, 114))

    response = client.post(
        "/predict",
        json={"mode": "accurate", "image": jpeg_data_url(image)},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "digitra-tid-robust"
    assert body["version"] == "5.0.0"
    assert body["hand_detected"] is True
    assert len(body["top_k"]) == 3


def test_invalid_image_is_rejected():
    response = client.post(
        "/predict",
        json={"mode": "accurate", "image": "data:image/jpeg;base64,not-base64"},
    )
    assert response.status_code == 400


def test_live_session_uses_tid_image_when_crop_is_present():
    image = Image.new("RGB", (64, 64), (114, 114, 114))
    landmarks = [[0.4 + index * 0.001, 0.4, 0.0] for index in range(21)]
    with client.websocket_connect("/ws/recognize") as websocket:
        websocket.send_json(
            {
                "hands": [
                    {
                        "landmarks": landmarks,
                        "handedness": "Right",
                        "handedness_score": 0.99,
                    }
                ],
                "image": jpeg_data_url(image),
            }
        )
        body = websocket.receive_json()
        assert body["model"] == "digitra-tid-robust"
        assert body["version"] == "5.0.0"
        assert body["scope"] == "tid_alphabet_image"
        assert body["mode"] == "tid_image"


def test_live_session_accepts_mediapipe_json_landmark_objects():
    image = Image.new("RGB", (64, 64), (114, 114, 114))
    landmarks = [
        {"x": 0.4 + index * 0.001, "y": 0.4, "z": 0.0}
        for index in range(21)
    ]
    with client.websocket_connect("/ws/recognize") as websocket:
        websocket.send_json(
            {
                "hands": [
                    {
                        "landmarks": landmarks,
                        "world_landmarks": landmarks,
                        "handedness": "Right",
                        "handedness_score": 0.99,
                    }
                ],
                "image": jpeg_data_url(image),
            }
        )
        body = websocket.receive_json()
        assert "error" not in body
        assert body["hands_detected"] == 1
        assert body["mode"] == "tid_image"
