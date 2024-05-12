#!/bin/bash

# Lancement de Docker Compose en mode détaché
docker-compose -f docker/docker-compose.yml up -d

# Attendre que MySQL soit prêt
echo "Attente pour que MySQL soit prêt..."
while ! docker-compose -f docker/docker-compose.yml exec -T mysql mysqladmin ping -h"localhost" -u"root" -p"Rakuten" --silent; do
    sleep 1
done


# Attente pour que les services soient prêts (si nécessaire)
echo "Attente pour que les autres services soient prêts..."
sleep 15

# Lancement de l'initialisation avec curl
echo "Lancement de l'initialisation..."
curl http://localhost:8000/initialisation
echo -e "\nFin de l'initialisation..."


# Récupération des logs
docker-compose -f docker/docker-compose.yml logs > docker/log_docker_compose.txt

