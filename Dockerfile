FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-home \
    HF_HUB_CACHE=/data/hf-home/hub \
    HF_WEBDAV_HOST=0.0.0.0 \
    HF_WEBDAV_PORT=7860

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "run.py"]
