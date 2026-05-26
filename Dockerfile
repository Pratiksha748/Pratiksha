FROM python:3.10-slim

WORKDIR /app

# install system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libglib2.0-0 libsm6 libxext6 libxrender1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/crop_model.h5
ENV CLASS_INDICES_PATH=/app/models/class_indices.json

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8080"]
