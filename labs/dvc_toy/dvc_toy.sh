#!/usr/bin/env bash
# Task 1.4 toy part. Run it OUTSIDE the main repo:  bash labs/dvc_toy/dvc_toy.sh ~/dvc-toy
# Each "STEP" line is a screenshot point. set -x echoes every command so the terminal shot shows it.
set -euo pipefail
TOY=${1:-$HOME/dvc-toy}
REMOTE=${2:-$HOME/dvc-toy-remote}      # local folder acting as remote storage
rm -rf "$TOY" "$REMOTE"; mkdir -p "$TOY" "$REMOTE"; cd "$TOY"
set -x

: "STEP 1 - git + dvc init"
git init -q -b main
dvc init -q
git commit -qm "init dvc"

: "STEP 2 - small CSV, track with DVC (v1)"
printf "id,price\n1,100\n2,200\n" > prices.csv
dvc add prices.csv
cat prices.csv.dvc                     # the pointer: md5 hash + size + path, nothing else
git add prices.csv.dvc .gitignore
git commit -qm "data v1: two rows"

: "STEP 3 - change data (v2)"
printf "3,300\n" >> prices.csv
dvc add prices.csv
git add prices.csv.dvc
git commit -qm "data v2: three rows"
git log --oneline

: "STEP 4 - go back to v1"
git checkout HEAD~1 -- prices.csv.dvc
dvc checkout
cat prices.csv                         # two rows again
git checkout main -- prices.csv.dvc && dvc checkout   # back to v2

: "STEP 5 - remote: push, delete local, pull back"
dvc remote add -d storage "$REMOTE"
git add .dvc/config && git commit -qm "configure remote"
dvc push
rm -rf prices.csv .dvc/cache
ls                                     # prices.csv is gone
dvc pull
cat prices.csv                         # restored from remote
