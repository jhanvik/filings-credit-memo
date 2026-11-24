FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv \
    && useradd --create-home appuser

WORKDIR /app

# Dependencies first (cached unless the lock file changes). README.md is needed because pyproject.toml names it.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY app ./app
RUN uv sync --locked --no-dev \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8501

# The viewer only reads saved memos from data/outputs/memos (mount them; see README). It does not call a model.
CMD ["uv", "run", "--no-sync", "streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
