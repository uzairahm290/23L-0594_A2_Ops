"""Task 1.2 - walk the dataset and build a per-image audit table.

Output:
  reports/audit.csv            one row per file
  reports/audit_summary.txt    counts, min/median/max, corrupt + duplicate numbers
"""
import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visionguard.config import DATA_DIR, REPORT_DIR, ROOT, label_from_folder  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: Path, label: str) -> dict:
    try:
        rel = path.resolve().relative_to(ROOT)   # portable paths in the CSVs
    except ValueError:
        rel = path.resolve()
    row = {"path": str(rel), "class": label, "file_size": path.stat().st_size,
           "width": None, "height": None, "mode": None, "format": None,
           "sha256": sha256(path), "readable": True, "error": ""}
    try:
        with Image.open(path) as im:
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            im.load()                      # full decode, catches truncated files
    except Exception as exc:
        row.update(readable=False, error=type(exc).__name__ + ": " + str(exc)[:80])
    return row


def main(root: Path):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # label = nearest ancestor folder that maps to a class
        label = next((label_from_folder(p.name) for p in path.parents
                      if label_from_folder(p.name)), None)
        if label is None:
            continue
        rows.append(inspect(path, label))

    df = pd.DataFrame(rows)
    df["duplicate"] = df.duplicated("sha256", keep="first")
    # same bytes under both labels = label noise, worth calling out separately
    cross = df.groupby("sha256")["class"].nunique()
    df["cross_class_dup"] = df["sha256"].map(cross) > 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "audit.csv", index=False)

    ok = df[df.readable]
    lines = ["== Images per class ==", df["class"].value_counts().to_string(), "",
             "== Format per class ==", pd.crosstab(df["format"].fillna("UNREADABLE"), df["class"]).to_string(), "",
             "== Colour mode per class ==", pd.crosstab(ok["mode"], ok["class"]).to_string(), "",
             "== width / height / file_size (readable files) =="]
    stats = ok.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"])
    lines += [stats.to_string(), "",
              f"corrupt / unreadable files : {int((~df.readable).sum())}",
              f"exact duplicates (extra copies) : {int(df.duplicate.sum())}",
              f"duplicates appearing in BOTH classes : {int(df[df.cross_class_dup].sha256.nunique())}",
              f"total files : {len(df)}"]
    if (~df.readable).any():
        lines += ["", "== Unreadable ==", df[~df.readable][["path", "error"]].to_string(index=False)]
    text = "\n".join(lines)
    (REPORT_DIR / "audit_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATA_DIR)
    main(ap.parse_args().root)
