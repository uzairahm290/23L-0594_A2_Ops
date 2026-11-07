"""Stratified 70/15/15 split of clean images into data/splits/{train,val,test}.csv.

Drops unreadable files, keeps one copy of each exact duplicate, and drops any hash that
appears under BOTH labels (we can't know which label is right).
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visionguard.config import CLASSES, REPORT_DIR, SEED, SPLIT_DIR  # noqa: E402

df = pd.read_csv(REPORT_DIR / "audit.csv")
before = len(df)
df = df[df.readable & ~df.duplicate & ~df.cross_class_dup].copy()
df["label"] = df["class"].map(CLASSES.index)

train, rest = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
val, test = train_test_split(rest, test_size=0.50, stratify=rest.label, random_state=SEED)

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[["path", "label"]].to_csv(SPLIT_DIR / f"{name}.csv", index=False)
    print(f"{name:<5} {len(part):>5}  real={int((part.label == 0).sum())}  ai={int((part.label == 1).sum())}")
print(f"dropped {before - len(df)} of {before} files (corrupt / duplicate / conflicting label)")
