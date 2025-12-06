#!/bin/bash

# Initialize MySQL
service mariadb start

# Wait for MySQL to start
sleep 5

# Create Database and User
mysql -e "CREATE DATABASE IF NOT EXISTS rakuten_db;"
mysql -e "CREATE USER IF NOT EXISTS 'root'@'localhost' IDENTIFIED BY '${MYSQL_ROOT_PWD}';"
mysql -e "GRANT ALL PRIVILEGES ON *.* TO 'root'@'localhost';"
mysql -e "FLUSH PRIVILEGES;"

# Apply init.sql (User creation)
mysql < docker/init.sql

# Stop MySQL so supervisor can manage it
service mariadb stop

# Initialize Airflow DB
export AIRFLOW_HOME=/app/airflow
airflow db migrate
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start Supervisor
/usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
