"""API endpoint tests using FastAPI TestClient."""

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    # Create simple 400x500 white rectangle on black background
    canvas = np.zeros((500, 400, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (50, 50), (350, 450), (255, 255, 255), -1)
    _, encoded = cv2.imencode(".jpg", canvas)
    return encoded.tobytes()


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_serve_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Smart Document Scanner" in res.text


def test_detect_endpoint(client, sample_image_bytes):
    res = client.post(
        "/api/detect",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["corners"]) == 4
    assert data["image_width"] == 400
    assert data["image_height"] == 500


def test_process_endpoint(client, sample_image_bytes):
    res = client.post(
        "/api/process",
        files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"filter_mode": "bw", "output_format": "jpg"}
    )
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert len(res.content) > 100
