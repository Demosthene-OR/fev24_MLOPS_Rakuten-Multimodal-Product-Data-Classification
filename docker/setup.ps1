# Récupération des logs
docker-compose up -d

# Attente pour la fin des tests
Start-Sleep -Seconds 10

# Lancement de l'initialisation
Invoke-WebRequest -Uri "http://localhost:8000/initialisation" -UseBasicParsing
