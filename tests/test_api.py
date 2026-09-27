"""HTTP API tests."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

STEADY = {
    "length": 0.5, "nodes": 5, "thermal_conductivity": 1.0, "heat_generation": 0,
    "left_boundary": {"type": "temperature", "value": 373},
    "right_boundary": {"type": "heat_flux", "value": 1000},
}


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_examples_contains_reference_problems():
    ids = [e["id"] for e in client.get("/api/examples").json()]
    assert "mst-steady" in ids and "mst-transient" in ids


def test_solve_steady_response_shape():
    r = client.post("/api/solve/steady", json=STEADY)
    assert r.status_code == 200
    body = r.json()
    for key in ("success", "dx", "nodes", "matrix", "rhs", "solution", "gaussian", "validation", "explanation"):
        assert key in body
    assert body["success"] and body["dx"] == 0.125
    assert body["gaussian"]["steps"][0]["operations"][0]["description"] == "R2 → R2 + 1·R1"


def test_solve_transient_examples_all_run():
    for ex in client.get("/api/examples").json():
        url = f"/api/solve/{ex['problem_type']}"
        assert client.post(url, json=ex["request"]).status_code == 200, ex["id"]


def test_engineering_error_messages_no_traceback():
    bad = dict(STEADY, right_boundary={"type": "adiabatic"}, left_boundary={"type": "heat_flux", "value": 5})
    r = client.post("/api/solve/steady", json=bad)
    assert r.status_code == 422
    body = r.json()
    assert body["success"] is False and "Neumann-Neumann" in body["error"]
    assert "Traceback" not in r.text


def test_schema_error_is_friendly():
    r = client.post("/api/solve/steady", json={"length": "abc"})
    assert r.status_code == 422 and r.json()["kind"] == "schema"


def test_validate_endpoint():
    r = client.post("/api/validate", json={"problem_type": "steady", "steady": dict(STEADY, length=-1)})
    body = r.json()
    assert body["valid"] is False and "Length must be greater than zero." in body["errors"]


def test_pdf_report():
    r = client.post("/api/report/pdf", json={"problem_type": "steady", "steady": STEADY})
    assert r.status_code == 200 and r.content[:4] == b"%PDF"
