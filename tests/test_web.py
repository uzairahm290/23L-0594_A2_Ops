import io

from conftest import png_bytes


def up(client, data, name="x.png", path="/predict"):
    return client.post(path, data={"image": (io.BytesIO(data), name)}, content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json["status"] == "ok" and r.json["debug"] is False


def test_upload_page(client):
    assert b"Analyse exhibit" in client.get("/").data


def test_valid_image_logged_and_listed(client):
    r = up(client, png_bytes())
    assert r.status_code == 200 and b"confidence in this label" in r.data
    h = client.get("/history")
    assert b"x.png" in h.data and b"test-model" in h.data


def test_json_api(client):
    r = up(client, png_bytes(), path="/api/predict")
    assert r.status_code == 200
    assert {"label", "confidence", "model_name", "created_at"} <= set(r.json)


def test_wrong_extension_rejected(client):
    r = up(client, b"hello", "notes.txt")
    assert r.status_code == 415 and b"Unsupported file type" in r.data


def test_too_large_rejected(client):
    r = up(client, b"\x89PNG" + b"0" * (2 * 1024 * 1024), "big.png")
    assert r.status_code == 413


def test_fake_image_rejected(client):
    r = up(client, b"this is not really a jpeg" * 10, "photo.jpg")
    assert r.status_code == 422 and b"Not a readable image" in r.data


def test_filename_sanitised(client):
    up(client, png_bytes(), "../../etc/passwd.png")
    assert b"etc_passwd.png" in client.get("/history").data


def test_nothing_written_to_disk(client, tmp_path):
    before = set(tmp_path.rglob("*"))
    up(client, png_bytes())
    new = {p for p in set(tmp_path.rglob("*")) - before if not p.name.startswith("t.db")}
    assert not new
