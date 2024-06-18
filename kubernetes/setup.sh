#!/bin/bash

export MYSQL_ROOT_PWD="${MYSQL_ROOT_PWD}"

# Lancement du servie airflow-init avec Docker Compose
docker-compose -f kubernetes/docker-compose.yml run --rm airflow-init

# Lancement de Docker Compose en mode détaché
# docker-compose -f kubernetes/docker-compose-airflow.yml up -d
docker-compose build -f kubernetes/docker-compose-other.yml

# Taguer l'image avec le nom de Docker Hub
docker tag docker-api_predict:latest olivierrenouard1103523/docker-api_predict:latest
docker tag docker-api_oauth:latest olivierrenouard1103523/docker-api_oauth:latest
docker tag docker-api_train:latest olivierrenouard1103523/docker-api_train:latest
docker tag docker-api_flows:latest olivierrenouard1103523/docker-api_flows:latest
docker tag docker-streamlit:latest olivierrenouard1103523/docker-streamlit:latest
docker tag docker-tensorboard:latest olivierrenouard1103523/docker-tensorboard:latest

# Push dans Docker Hub
docker push olivierrenouard1103523/docker-api_predict:latest
docker push olivierrenouard1103523/docker-api_oauth:latest
docker push olivierrenouard1103523/docker-api_train:latest
docker push olivierrenouard1103523/docker-api_flows:latest
docker push olivierrenouard1103523/docker-streamlit:latest
docker push olivierrenouard1103523/docker-tensorboard:latest

# Attendre que MySQL soit prêt
# echo "Attente pour que MySQL soit prêt..."
# while ! docker-compose -f kubernetes/docker-compose-other.yml exec -T mysql mysqladmin ping -h"localhost" -u"root" -p"Rakuten" --silent; do
#     sleep 1
# done


# Attente pour que les services soient prêts (si nécessaire)
# echo "Attente pour que les autres services soient prêts..."
# sleep 15

# Lancement de l'initialisation avec curl
# echo -e "\nFin de l'initialisation..."


# Récupération des logs
# docker-compose -f kubernetes/docker-compose-other.yml logs > kubernetes/log_docker_compose.txt

