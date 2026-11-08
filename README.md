# VisionGuard · 23L-0594_A2_Ops

![ci](https://github.com/uzairahm290/23L-0594_A2_Ops/actions/workflows/ci.yml/badge.svg)

Tells whether an uploaded image is AI-generated or a real photo. A Keras model (MobileNetV2 fine-tuned on the
Kaggle *AI Generated Images vs Real Images* set) is tracked in MLflow on DagsHub and served as a TFLite API on Vercel.
A Flask app with a Postgres request log runs locally in Docker Compose.

```
browser ─► Flask (Compose, :8000) ─► Vercel model API (TFLite, X-API-Key)
                    │
                    └─► Postgres (Compose, internal only, named volume)
```

| Link | |
|---|---|
| DagsHub | https://dagshub.com/uzairahm290/23L-0594_A2_Ops |
| Model API | https://<project>.vercel.app/health |

## Layout

| Path | What |
|---|---|
| `src/visionguard/` | shared code: config, preprocessing (used by training *and* serving), tf.data, models, evaluation, MLflow helpers |
| `scripts/` | audit, EDA, splits, train, tune, compare, register, predict-by-alias, export TFLite, error analysis, Grad-CAM |
| `labs/` | tool labs: DVC toy, MLflow toy, Keras Tuner toy, Docker warm-up, CI toy |
| `web/` | Flask app, Dockerfile (+ deliberately naive Dockerfile for the size comparison) |
| `api/` | Vercel model API (deploy with Root Directory = `api`) |
| `docs/RUNBOOK.md` | every command in order, with the screenshots the report needs |

## Quick start

```bash
cp .env.example .env              # fill in
docker compose up -d --build
open http://localhost:8000
```

Full instructions are in [docs/RUNBOOK.md](docs/RUNBOOK.md).
