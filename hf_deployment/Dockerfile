FROM python:3.10

ENV PYTHONUNBUFFERED=1
ENV MYSQL_ROOT_PWD=rakuten

# Airflow Envs
ENV AIRFLOW_HOME=/app/airflow
ENV AIRFLOW__WEBSERVER__BASE_URL=http://localhost:7860/airflow

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 iputils-ping \
    nginx supervisor mariadb-server \
    php-fpm php-mysql wget \
    && rm -rf /var/lib/apt/lists/*

# Fix line endings just in case
RUN sed -i 's/\r$//' /etc/supervisor/conf.d/supervisord.conf || true

# Setup Adminer
RUN mkdir -p /var/www/html/adminer \
    && wget "https://github.com/vrana/adminer/releases/download/v4.8.1/adminer-4.8.1.php" -O /var/www/html/adminer/index.php \
    && chown -R www-data:www-data /var/www/html/adminer \
    && mkdir -p /run/php

# Copy requirements first
COPY requirements.txt .
COPY src/requirements.txt src_requirements.txt
# COPY src/fastapi_oauth/requirements.txt oauth_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    mysql-connector-python \
    apache-airflow \
    tensorboard

# Copy application code
COPY . .

# Copy config files
COPY hf_deployment/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY hf_deployment/start.sh /app/start.sh
# COPY hf_deployment/nginx.conf /etc/nginx/nginx.conf # We pass it via cmdline in supervisor, but copying it is safer
COPY hf_deployment/nginx.conf /app/hf_deployment/nginx.conf

# Make start script executable and fix line endings
RUN chmod +x /app/start.sh && sed -i 's/\r$//' /app/start.sh

# Expose ports (HF Spaces only expose 7860 publicly)
EXPOSE 7860

CMD ["/app/start.sh"]
