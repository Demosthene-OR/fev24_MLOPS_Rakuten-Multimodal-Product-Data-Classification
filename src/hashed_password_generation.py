from passlib.hash import bcrypt

# Mot de passe à stocker dans la base de données
password = "John"

# Hash du mot de passe avec bcrypt
hashed_password = bcrypt.hash(password)

# Affichage du mot de passe haché
print("Mot de passe haché:", hashed_password)