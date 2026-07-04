FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv python install 3.12

WORKDIR /app

COPY pyproject.toml uv.lock .
COPY src/ src/
COPY checkpoints/checkpoint_1.pth checkpoints/

RUN uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "sparkle.model.serve:app", "--host", "0.0.0.0", "--port", "8000"]
