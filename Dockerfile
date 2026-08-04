# Railpack cannot detect this project on its own: the repo root holds only
# backend/ and docs/, and requirements.txt lives at backend/app/requirements.txt
# rather than the root. An explicit Dockerfile is clearer than teaching a
# buildpack about the layout, and it pins the interpreter.
#
# Python 3.10 matches the version the test suite is verified against. Newer
# interpreters are very likely fine, but "likely" is not "verified" — bump this
# only after running the suite on the new version.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Dependencies first so a code-only change reuses the cached install layer.
COPY backend/app/requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt

# The application lives under backend/, and app.main:app is resolved relative to
# it — this mirrors how the service is run locally.
COPY backend/ /srv/backend/
WORKDIR /srv/backend

# The bundled Georgian font is what keeps carousel text from rendering as blank
# boxes; there is no Georgian font in this image. Fail the build rather than
# discover it in production.
# The import check also constructs the services, which create their SQLite
# stores. Point those at /tmp so nothing is baked into an image layer — at
# runtime they belong on a mounted volume, not in the container filesystem.
RUN test -f app/assets/fonts/NotoSansGeorgian-Bold.ttf \
 && test -f app/assets/fonts/NotoSansGeorgian-Regular.ttf \
 && BILLING_DB_PATH=/tmp/build-check/billing.sqlite3 \
    JOBS_DB_PATH=/tmp/build-check/jobs.sqlite3 \
    CAROUSEL_MEDIA_DIR=/tmp/build-check/media \
    python -c "import app.main; print('routes:', len(app.main.app.routes))" \
 && rm -rf /tmp/build-check \
 && echo "import + font check OK"

# Railway injects PORT. Default to 8000 so the image also runs locally.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
