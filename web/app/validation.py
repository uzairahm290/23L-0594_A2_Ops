"""Task 4.2 - reject anything that isn't a decodable image of an allowed type and size.
Three layers: extension allow-list, size limit (Flask MAX_CONTENT_LENGTH + explicit check),
and a real decode with Pillow (catches renamed .txt files and truncated images)."""
import hashlib
import io

from PIL import Image
from werkzeug.utils import secure_filename


class UploadError(Exception):
    def __init__(self, status: int, title: str, detail: str):
        super().__init__(detail)
        self.status, self.title, self.detail = status, title, detail


def validate(file_storage, allowed: set[str], max_mb: float) -> tuple[bytes, str, dict]:
    if file_storage is None or file_storage.filename == "":
        raise UploadError(400, "No file chosen", "Pick an image before pressing Analyse.")

    name = secure_filename(file_storage.filename) or "upload"   # strips ../, slashes, odd chars
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in allowed:
        raise UploadError(415, "Unsupported file type",
                          f"'.{ext or '?'}' isn't accepted. Use one of: {', '.join(sorted(allowed))}.")

    data = file_storage.read()                 # kept in memory only, never written to disk
    if len(data) > max_mb * 1024 * 1024:
        raise UploadError(413, "File too large", f"Limit is {max_mb:g} MB; this one is {len(data)/1e6:.1f} MB.")
    if not data:
        raise UploadError(400, "Empty file", "The file has 0 bytes.")

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()                        # structural check
        with Image.open(io.BytesIO(data)) as im:
            im.seek(0)
            im.load()                          # full pixel decode (verify() doesn't do this)
            meta = {"width": im.width, "height": im.height, "format": im.format}
    except Exception:
        raise UploadError(422, "Not a readable image",
                          "The file has an image extension but its contents couldn't be decoded. "
                          "It may be corrupt, truncated, or a different file renamed.")
    meta["sha256"] = hashlib.sha256(data).hexdigest()
    return data, name, meta
