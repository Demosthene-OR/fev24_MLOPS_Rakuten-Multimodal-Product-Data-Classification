FROM python:3.10.14-slim

# Installation de curl
# RUN apt-get update && apt-get install -y curl

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt


CMD ["uvicorn", "predict_API2:app", "--host", "0.0.0.0", "--port", "8000"]

