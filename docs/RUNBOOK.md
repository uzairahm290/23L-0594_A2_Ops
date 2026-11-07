# VisionGuard runbook (23L-0594)

Run these in order on your Mac. Every 📸 is a screenshot the report needs, tagged with its sub-task ID.
Before each terminal screenshot run `whoami; date` so the shot shows your username and the time (the
assignment asks for that). Keep the browser address bar visible in browser shots.

Placeholders: `<gh>` = your GitHub user (uzairahm290), `<dh>` = your DagsHub user.

---

## 0. One-time setup

```bash
# Python 3.11. TensorFlow 2.16.2 is the last release with Intel-Mac wheels and it needs Python <= 3.12
brew install python@3.11 node
git clone https://github.com/<gh>/23L-0594_A2_Ops.git && cd 23L-0594_A2_Ops
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt -r web/requirements.txt pytest ruff
cp .env.example .env          # fill it in as you go
```

Kaggle token: kaggle.com → Settings → API → Create New Token → move `kaggle.json` to `~/.kaggle/` and `chmod 600 ~/.kaggle/kaggle.json`.

---

## Task 1: Data

**1.1 Environment**
```bash
python scripts/env_check.py
```
📸 **1.1**: the version list and the "GPUs visible: none" line.

**Download**
```bash
bash scripts/download_data.sh
```
If the folder names come out different from `AiArtData/` and `RealArt/`, check `label_from_folder()` in `src/visionguard/config.py`. It matches anything starting with "ai" or containing "real".

**1.2 Audit**
```bash
python scripts/audit.py
```
📸 **1.2**: the whole summary (counts, format/mode tables, min/median/max, corrupt and duplicate counts). The full per-image table is `reports/audit.csv`. Open it in Numbers/Excel and screenshot the first 15 rows too.

**1.3 EDA**
```bash
python scripts/eda.py
python scripts/make_splits.py
```
Figures go to `reports/figures/eda_samples.png` and `eda_histograms.png` (they go straight into the report, no screenshot needed). 📸 the `make_splits` output for the report's split table.

**1.4 DVC, toy part**
```bash
bash labs/dvc_toy/dvc_toy.sh ~/dvc-toy
```
📸 **1.4a**: `cat prices.csv.dvc` (the pointer file).
📸 **1.4b**: `git log --oneline` showing both data versions.
📸 **1.4c**: after `dvc checkout`, `prices.csv` back at two rows.
📸 **1.4d**: `ls` with the file missing, then `dvc pull` restoring it.

The brief says to commit both versions to DagsHub. To do that, create an empty DagsHub repo `23L-0594_A2_Ops-dvc-toy`, then run:
```bash
cd ~/dvc-toy && git remote add origin https://dagshub.com/<dh>/23L-0594_A2_Ops-dvc-toy.git && git push -u origin main
```
📸 its DagsHub page showing the commits.

**1.4 DVC, apply to the dataset**
```bash
cd ~/23L-0594_A2_Ops
dvc init
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<dh>/23L-0594_A2_Ops.s3
dvc remote modify origin --local access_key_id <dagshub-token>
dvc remote modify origin --local secret_access_key <dagshub-token>
dvc add data/raw
git add data/raw.dvc data/.gitignore .dvc/config .dvcignore data/splits
git commit -m "Track dataset with DVC (DagsHub storage)"
dvc push
git push
```
(`--local` writes the token to `.dvc/config.local`, which is git-ignored.)
📸 **1.4e**: `cat data/raw.dvc` and the `dvc push` output.
📸 **1.4f**: the DagsHub repo page with the data folder showing the DVC badge.

> DagsHub repo: on dagshub.com choose **Create → New Repository → Connect a repository → GitHub**, pick `23L-0594_A2_Ops`. That one repo then gives you Git mirroring, DVC storage and the MLflow server.

---

## Task 2: Models

```bash
python scripts/augment_preview.py light      # 2.1 figure
python scripts/augment_preview.py strong
```

**Toy Keras Tuner (2.7)**
```bash
python labs/tuner_toy.py
```
📸 **2.7a**: the results summary and the BEST line.

Training (2.2–2.5) happens through the experiment runs in Task 3, so you train once and it is logged.

---

## Task 3: MLflow + DagsHub

**3.1 Local toy**
```bash
python labs/mlflow_toy.py --lr 1e-3
python labs/mlflow_toy.py --lr 1e-2
mlflow ui --port 5000          # open http://127.0.0.1:5000
```
📸 **3.1a/b**: each run's page (params, metrics, artifacts).
📸 **3.1c**: select both runs → Compare → the val_accuracy chart.

**3.2 DagsHub**
DagsHub repo → **Remote → Experiments** shows the MLflow tracking URI. Avatar → **Settings → Tokens** gives the access token. Put them in `.env`:
```
MLFLOW_TRACKING_URI=https://dagshub.com/<dh>/23L-0594_A2_Ops.mlflow
MLFLOW_TRACKING_USERNAME=<dh>
MLFLOW_TRACKING_PASSWORD=<token>
```
```bash
git check-ignore -v .env       # proves .env is ignored
python labs/mlflow_toy.py --lr 1e-3 --dagshub
```
📸 **3.2a**: the DagsHub repo page. 📸 **3.2b**: the toy run on DagsHub's Experiments tab. 📸 **3.2c**: the `git check-ignore` output.

**3.3 / 3.5 Experiment runs**
This is the long one. Each MobileNetV2 run takes roughly 10–20 min on an Intel Mac CPU.
```bash
bash scripts/run_experiments.sh 2>&1 | tee reports/experiments.log
```
Runs R1–R8 all go to DagsHub. R2 uses the manual callback and R8 is identical but uses autolog, which is the 3.3 comparison.
📸 **2.2**: `cat models/R1_cnn_baseline/model_summary.txt`.
📸 **2.3**: `cat models/R2_mnv2_ref/trainable_layers.json`, plus R2's `figures/curves.png` (shaded part = fine-tune stage).
📸 **2.4**: the "best epoch" line in the log (also on every curves.png as a dashed line).
📸 **3.3a**: R2's run page on DagsHub (params, metric charts, tags git_commit / dvc_data_version / seed).
📸 **3.3b**: R8's run page (autolog's params and charts).
📸 **3.4**: R2's Artifacts tab expanded (figures/, splits/, requirements-freeze.txt, keras_model/, model/).
📸 **3.5**: the `compare_runs.py` table at the end of the log, plus DagsHub's compare view with R1–R7 selected.

**2.5 / 2.6 / bonus** (for the chosen run, probably R2)
```bash
python scripts/error_analysis.py models/R2_mnv2_ref
python scripts/gradcam.py models/R2_mnv2_ref
```
Then fill the `hypothesis` column in `models/R2_mnv2_ref/error_hypotheses.csv` by looking at each image.

**2.7 Tuner on VisionGuard (≥ 8 trials)**
```bash
python scripts/tune.py --trials 8 --epochs 6
```
📸 **2.7b**: the trial table and the three "spread" lines.

**3.6 Registry**
Get the run IDs of your best run and one other from the comparison table:
```bash
python scripts/register_model.py <other_run_id> <best_run_id> --champion <best_run_id>
python scripts/predict_by_alias.py data/raw/<some ai image>
```
📸 **3.6a**: DagsHub → Models showing two versions with `champion` on one. 📸 **3.6b**: the terminal output of both commands.
`predict_by_alias.py` also saves `models/champion/model.keras`, which is the file Docker mounts.

---

## Task 4–6: Flask, database, Docker

**4.1 Inference check**
```bash
# use one AI and one real image from your TEST split (data/splits/test.csv)
python scripts/inference_check.py <champion threshold> <ai_image_path> <real_image_path>
```
📸 **4.1**: two predictions, one AI and one real.

**Tests and lint**
```bash
pytest -q && ruff check src web api tests --select E,F,W --ignore E501,E402
```

**6.1 Docker warm-up**
```bash
bash labs/docker_warmup/warmup.sh
```
📸 **6.1**: one shot per STEP.

**6.2 / 6.3 Compose stack (local backend first)**
Fill `.env` (POSTGRES_*, SECRET_KEY, `INFERENCE_BACKEND=local`, `THRESHOLD`, `MODEL_VERSION`).
```bash
docker compose up -d --build
docker compose ps                          # 📸 6.3a: both services (healthy)
docker compose exec web whoami             # 📸 6.2: prints vg, not root
docker compose exec web id
nc -zv localhost 5432                      # 📸 6.3b: connection refused = DB port closed
curl -s localhost:8000/health | python -m json.tool     # 📸 4.4
```
Then the browser:
- 📸 **4.3**: http://localhost:8000 with an image chosen (preview), then the result page.
- 📸 **4.2a/b/c**: upload a `.txt` (415), a >4 MB file (413), and a text file renamed `.jpg` (422).
- 📸 **5**: the History page after several uploads.
- 📸 **5b**: the database directly:
  ```bash
  docker compose exec db psql -U visionguard -d visionguard -c "select id, created_at, label, round(confidence::numeric,3) conf, model_name, model_version, backend from prediction_log order by id desc limit 10;"
  ```

**6.4 Image sizes**
```bash
docker build -f web/Dockerfile.naive -t vg-naive .
docker images | grep -E "vg-naive|23l-0594-a2-ops-web"     # 📸 6.4a: before vs after
docker history 23l-0594-a2-ops-web:latest --format "{{.Size}}\t{{.CreatedBy}}" | head -12   # 📸 6.4b: largest layers
docker build -f web/Dockerfile --build-arg TF_PACKAGE= -t vg-remote .   # the remote-backend image, smallest
docker images | grep vg-remote
```

**6.5 Operations and breaking it**
```bash
docker compose stop && docker compose start && docker compose restart web
docker compose logs --tail 20 web
docker compose exec web sh -c 'ls /models/champion; python -V'
docker compose exec db sh -c 'pg_isready; ls /var/lib/postgresql/data | head'
```
📸 **6.5a**: each command.
Break it:
```bash
sed -i '' 's/^POSTGRES_PASSWORD=.*/&WRONG/' .env      # web now uses a wrong password
docker compose up -d web
docker compose logs --tail 30 web                       # 📸 6.5b: password authentication failed for user
# fix: restore the password line, then
docker compose up -d web && docker compose logs --tail 5 web
```
Why it fails: Postgres only reads POSTGRES_PASSWORD when the volume is first initialised. Changing `.env` changes the URL web uses, but not the password stored in the DB, so authentication fails.

**6.6 GitHub Actions**
- Toy: create a new public repo `ci-toy`, copy in `labs/ci_toy/` (including `.github/`) and push. 📸 green run with 2 matrix jobs. Break `calc.py` (`a - b`) and push. 📸 red run. Revert and push. 📸 green.
- Project: pushing this repo runs `.github/workflows/ci.yml`. 📸 the green run (test + docker jobs) and the README badge.

---

## Task 7: Vercel model API

```bash
python scripts/export_tflite.py models/champion/model.keras     # writes api/model/visionguard.tflite + model_info.json
git add api/model && git commit -m "Champion model exported to TFLite for the API" && git push
npm i -g vercel
cd api
vercel login
vercel link                   # new project named 23l-0594-a2-ops (Vercel lowercases names)
openssl rand -hex 24          # copy this, it's your API key
vercel env add VG_API_KEY production      # paste the key
vercel --prod
cd ..
```
📸 **7a**: `vercel --prod` output with the production URL.
📸 **7b**: the Vercel dashboard → project → Deployments (Ready).
📸 **7c**: Settings → Environment Variables (key name only, value hidden).
📸 **7d**: `https://<project>.vercel.app/health` in the browser.

Alternative without the CLI: Vercel dashboard → Add New → Project → import the GitHub repo → **Root Directory: `api`** → add env var `VG_API_KEY` → Deploy.

**Postman**: Import `docs/VisionGuard.postman_collection.json`, set the `api` and `key` variables, and pick an image file in the form-data field.
📸 **7e**: requests 1, 2 and 3 (200 health, 200 predict, 401 without key).

**Other network**: switch your Mac to your phone's hotspot (not your home Wi-Fi) and re-run request 2. 📸 that shows the hotspot in the menu bar. Or open `/health` on your phone over mobile data.

---

## Task 8: Flask → Vercel

In `.env` set:
```
INFERENCE_BACKEND=remote
MODEL_API_URL=https://<project>.vercel.app
MODEL_API_KEY=<the same key>
TF_PACKAGE=
```
```bash
docker compose up -d --build        # rebuilds without TensorFlow; the model now lives on Vercel
curl -s localhost:8000/health | python -m json.tool     # backend: remote
```
Upload 5–6 different images (mix of AI and real) through the browser.
📸 **8a–8f**: each result page (footer shows `backend: remote`, Served by `remote:vercel`).
📸 **8g**: History page listing them.
📸 **8h**: `docker compose logs web | grep prediction` showing each request.
📸 **8i**: Vercel dashboard → Logs showing the POST /predict calls.
📸 **8j**: Postman request 4 (Flask JSON endpoint).

---

## Finish

```bash
git log -1 --format=%H          # the graded commit hash for the report's front page
```
Fill the report placeholders and export it as `23L-0594_VisionGuard.pdf`.
