FROM pytorch/pytorch:2.7.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

RUN uv python install 3.12

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY checkpoints/checkpoint_1.pth checkpoints/
COPY Pre Processing/preprocessing/flow_split.py Pre Processing/preprocessing/
COPY Pre Processing/preprocessing/pcap_to_packet.py Pre Processing/preprocessing/
COPY Pre Processing/preprocessing/test.py Pre Processing/preprocessing/
COPY Pre Processing/preprocessing/preprocess_ustc_by_flow.py Pre Processing/preprocessing/

RUN uv sync --frozen

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "sparkle.model.serve:app", "--host", "0.0.0.0", "--port", "8000"]
