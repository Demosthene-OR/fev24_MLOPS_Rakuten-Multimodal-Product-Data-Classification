USE rakuten_db;

-- Création de la table User avec les champs FirstName, LastName, Email et Authorization
CREATE TABLE Users (
    FirstName VARCHAR(30),
    LastName VARCHAR(30),
    Email VARCHAR(30),
    Authorization INT
);

-- Insertion d'un utilisateur pour tester
INSERT INTO Users (FirstName, LastName, Email, Authorization) VALUES ('Olivier', 'Renouard', 'olivier.renouard1103@gmail.com', 2);
INSERT INTO Users (FirstName, LastName, Email, Authorization) VALUES ('Fadimatou', 'Abdoulaye', 'afadimatou@gmail.com', 2);
INSERT INTO Users (FirstName, LastName, Email, Authorization) VALUES ('John', 'Doe', 'john.doe@example.com', 1);

