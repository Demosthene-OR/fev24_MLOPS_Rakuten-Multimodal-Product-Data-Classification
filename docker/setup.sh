#!/bin/bash

# Lancement de Docker Compose en mode détaché
docker-compose up -d

# Attente pour que les services soient prêts
echo "Attente pour que les services soient prêts..."
sleep 30

# Lancement de l'initialisation avec curl
echo "Lancement de l'initialisation..."
curl http://localhost:8000/initialisation

# Récupération des logs
docker-compose logs > log_docker_compose.txt

