FROM python:3.12-slim AS app-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --ingroup app --home /app --no-create-home app \
    && mkdir -p /app/.data/chroma /app/.cache/chroma \
    && chown -R app:app /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

# Copy only the compact runtime corpus. Source notebooks and generated image
# archives are not part of this standalone application repository.
COPY --chown=app:app data/restaurants.json data/reviews.json data/recipes.json ./data/

FROM app-base AS test

COPY tests ./tests
RUN python -m pip install --no-cache-dir ".[dev]"

USER app

CMD ["pytest", "-q"]

FROM app-base AS runtime

USER app

EXPOSE 8000 8001 7860

CMD ["python", "-m", "connoisseur.mcp.server"]
