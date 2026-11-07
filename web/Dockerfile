# syntax=docker/dockerfile:1
# Build from the repo root:  docker build -f web/Dockerfile -t 23l-0594-a2-ops-web .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 1) dependencies first: this layer is rebuilt only when requirements change,
#    not on every code edit, so day-to-day rebuilds take seconds instead of minutes.
#    CPU-only TensorFlow: no CUDA libraries (~2 GB smaller than `tensorflow`), and inference
#    on one image at a time doesn't need a GPU. Set TF_PACKAGE="" for the remote-backend image.
ARG TF_PACKAGE="tensorflow-cpu==2.20.0"
COPY web/requirements.txt /tmp/requirements.txt
RUN pip install ${TF_PACKAGE} -r /tmp/requirements.txt && rm /tmp/requirements.txt

# 2) unprivileged user: a bug in the app can't write to system paths or install anything
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin vg

# 3) source code last (changes most often)
COPY --chown=vg:vg src/visionguard /app/src/visionguard
COPY --chown=vg:vg web /app/web

USER vg
WORKDIR /app/web
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# gunicorn, not `flask run`: real WSGI server, debug impossible, multiple threads.
# One worker on purpose: each worker would load its own copy of the model into RAM.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--access-logfile", "-", "wsgi:app"]
