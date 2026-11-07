"""Every runtime setting comes from the environment (12-factor). Nothing is hard-coded,
so the same image runs on a laptop, in Compose and in CI with different values."""
import os


class Settings:
    # inference
    BACKEND = os.getenv("INFERENCE_BACKEND", "local")            # local | remote
    MODEL_PATH = os.getenv("MODEL_PATH", "/models/champion/model.keras")
    MODEL_NAME = os.getenv("MODEL_NAME", "23L-0594_A2_Ops_visionguard")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")
    THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
    MODEL_API_URL = os.getenv("MODEL_API_URL", "")               # e.g. https://23l-0594-a2-ops.vercel.app
    MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
    MODEL_API_TIMEOUT = float(os.getenv("MODEL_API_TIMEOUT", "30"))

    # storage
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///visionguard.db")

    # uploads (Vercel rejects request bodies over 4.5 MB, so stay under it)
    MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "4"))
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}

    SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
