FROM python:3.13-slim-bookworm

# Pin the installer; application dependencies are locked in uv.lock.
RUN pip install --no-cache-dir uv==0.12.5
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=25 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra cloud --no-editable
COPY app.py ./
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY artifacts/door ./artifacts/door
COPY artifacts/shm ./artifacts/shm
COPY artifacts/acv ./artifacts/acv
COPY artifacts/corrugation ./artifacts/corrugation
RUN useradd --create-home --uid 10001 railguard
USER railguard
EXPOSE 8080
CMD ["sh", "-c", "exec python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080}"]
