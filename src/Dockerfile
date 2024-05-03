FROM python:3.10.14-slim

WORKDIR /app

COPY . .

# COPY ./requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt


CMD ["uvicorn", "predict_API2:app", "--host", "0.0.0.0", "--port", "8000"]

