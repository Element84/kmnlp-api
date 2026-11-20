# Image, intended for production use
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock .
COPY src/ src/

# Excldue the default test & dev groups (i.e. only include the src we need).
RUN uv sync --no-default-groups

COPY entrypoint.py entrypoint_kg.py .

CMD ["/app/.venv/bin/python", "entrypoint.py"]
