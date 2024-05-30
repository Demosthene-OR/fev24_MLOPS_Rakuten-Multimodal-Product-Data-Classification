# Définition de la variable d'environnement MYSQL_ROOT_PWD
$env:MYSQL_ROOT_PWD = "${MYSQL_ROOT_PWD}"

# Lancement du servie airflow-init avec Docker Compose
docker-compose -f docker/docker-compose.yml run --rm airflow-init

# Lancement de Docker Compose en mode détaché
docker-compose -f docker/docker-compose.yml up -d

# Attendre que MySQL soit prêt
Write-Host "Attente pour que MySQL soit prêt..."
while (-not (docker-compose -f docker/docker-compose.yml exec -T mysql mysqladmin ping -h"localhost" -u"root" -p"Rakuten" --silent)) {
    Start-Sleep -Seconds 1
}

# Attente pour que les autres services soient prêts (si nécessaire)
Write-Host "Attente pour que les autres services soient prêts..."
Start-Sleep -Seconds 15

# Lancement de l'initialisation avec Invoke-RestMethod
Write-Host "Fin de l'initialisation..."

# Récupération des logs
docker-compose -f docker/docker-compose.yml logs | Out-File -FilePath "docker/log_docker_compose.txt"

