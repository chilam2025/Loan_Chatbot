FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock ./backend/
COPY backend/README.md ./backend/README.md
COPY backend/src ./backend/src
RUN uv sync --project backend --frozen --no-dev --extra postgres
COPY backend ./backend
ENV PYTHONPATH=/app/backend/src
EXPOSE 8000
