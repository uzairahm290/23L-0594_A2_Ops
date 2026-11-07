import logging
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

# make the shared preprocessing package importable whether we run from the repo or the container
for p in (Path(__file__).resolve().parents[2] / "src", Path("/app/src")):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from .db import Store  # noqa: E402
from .inference import get_predictor  # noqa: E402
from .settings import Settings  # noqa: E402

API_PATHS = ("/api/", "/health")


def create_app(settings=Settings, predictor=None, store=None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    app.config.from_object(settings)
    app.config["MAX_CONTENT_LENGTH"] = int(settings.MAX_UPLOAD_MB * 1024 * 1024) + 64 * 1024

    app.logger.info("backend=%s model=%s db=%s", settings.BACKEND,
                    settings.MODEL_PATH if settings.BACKEND == "local" else settings.MODEL_API_URL,
                    settings.DATABASE_URL.split("@")[-1])          # never log the password
    app.extensions["predictor"] = predictor or get_predictor(settings)   # loaded once, here
    app.extensions["store"] = store or Store(settings.DATABASE_URL)

    from .routes import bp
    app.register_blueprint(bp)

    def wants_json():
        return request.path.startswith(API_PATHS) or request.accept_mimetypes.best == "application/json"

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        msg = f"Limit is {settings.MAX_UPLOAD_MB:g} MB."
        if wants_json():
            return jsonify(error="file too large", detail=msg), 413
        return render_template("error.html", status=413, title="File too large", detail=msg), 413

    @app.errorhandler(HTTPException)
    def http_error(e):
        if wants_json():
            return jsonify(error=e.name, detail=e.description), e.code
        return render_template("error.html", status=e.code, title=e.name, detail=e.description), e.code

    @app.errorhandler(Exception)
    def crash(e):
        app.logger.exception("unhandled error")
        if wants_json():
            return jsonify(error="internal error"), 500
        return render_template("error.html", status=500, title="Something broke on our side",
                               detail="The error has been logged. Try again in a moment."), 500

    @app.context_processor
    def inject():
        return {"backend": settings.BACKEND, "model_name": settings.MODEL_NAME}

    return app
