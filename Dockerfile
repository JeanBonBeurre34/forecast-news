FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libxml2-dev libxslt-dev libjpeg-dev zlib1g-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

RUN mkdir /app/data
CMD ["python", "app.py"]

