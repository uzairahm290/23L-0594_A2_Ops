from flask import Blueprint, current_app, jsonify, render_template, request

from .inference import RemoteModelError
from .validation import UploadError, validate

bp = Blueprint("main", __name__)


def _deps():
    return current_app.extensions["predictor"], current_app.extensions["store"]


@bp.get("/")
def upload_page():
    return render_template("index.html", max_mb=current_app.config["MAX_UPLOAD_MB"],
                           exts=sorted(current_app.config["ALLOWED_EXTENSIONS"]))


def _run(file_storage):
    cfg = current_app.config
    data, name, meta = validate(file_storage, cfg["ALLOWED_EXTENSIONS"], cfg["MAX_UPLOAD_MB"])
    predictor, store = _deps()
    pred = predictor.predict(data)
    rec = store.add(filename=name, image_sha256=meta["sha256"], width=meta["width"], height=meta["height"],
                    label=pred.label, p_ai=pred.p_ai, confidence=pred.confidence, threshold=pred.threshold,
                    model_name=pred.model_name, model_version=pred.model_version, backend=pred.backend,
                    latency_ms=pred.latency_ms)
    current_app.logger.info("prediction id=%s label=%s p_ai=%.4f backend=%s %dms",
                            rec.id, pred.label, pred.p_ai, pred.backend, pred.latency_ms)
    del data                                    # bytes go out of scope; nothing touched disk
    return pred, rec, meta


@bp.post("/predict")
def predict_page():
    try:
        pred, rec, meta = _run(request.files.get("image"))
    except UploadError as e:
        return render_template("error.html", status=e.status, title=e.title, detail=e.detail), e.status
    except RemoteModelError as e:
        return render_template("error.html", status=502, title="Model service unavailable",
                               detail=str(e)), 502
    return render_template("result.html", pred=pred, rec=rec, meta=meta)


@bp.post("/api/predict")
def predict_api():
    try:
        pred, rec, meta = _run(request.files.get("image"))
    except UploadError as e:
        return jsonify(error=e.title, detail=e.detail), e.status
    except RemoteModelError as e:
        return jsonify(error="model service unavailable", detail=str(e)), 502
    return jsonify(id=rec.id, label=pred.label, p_ai=round(pred.p_ai, 4), confidence=round(pred.confidence, 4),
                   threshold=pred.threshold, model_name=pred.model_name, model_version=pred.model_version,
                   backend=pred.backend, latency_ms=pred.latency_ms, created_at=rec.created_at.isoformat())


@bp.get("/history")
def history():
    page = max(request.args.get("page", 1, type=int), 1)
    _, store = _deps()
    rows, total = store.page(page)
    return render_template("history.html", rows=rows, total=total, page=page,
                           pages=max((total + 19) // 20, 1), stats=store.stats())


@bp.get("/health")
def health():
    predictor, store = _deps()
    model_ok, db_ok = predictor.ready(), store.ping()
    body = {"status": "ok" if model_ok and db_ok else "degraded", "model_loaded": model_ok,
            "database": db_ok, "backend": predictor.backend,
            "debug": current_app.debug, "max_upload_mb": current_app.config["MAX_UPLOAD_MB"]}
    return jsonify(body), 200 if body["status"] == "ok" else 503
