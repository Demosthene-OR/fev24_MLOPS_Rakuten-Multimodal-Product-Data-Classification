USE rakuten_db;

-- Création de la table User avec les champs FirstName, LastName, Email, Authorization, username et password
CREATE TABLE Users (
    FirstName VARCHAR(30),
    LastName VARCHAR(30),
    Email VARCHAR(30),
    Authorization INT,
    username VARCHAR(30),
    password VARCHAR(128) -- 128 caractères pour stocker le hachage bcrypt
);

-- Insertion d'un utilisateur pour tester
INSERT INTO Users (FirstName, LastName, Email, Authorization, username, password) VALUES ('Olivier', 'Renouard', 'olivier.renouard1103@gmail.com', 2, 'Olivier', '$2b$12$FWFmGn05f245f637.i9uYOdVEcDD2sC7ZO0cC0ePOjtY/OB6pOp0q');
INSERT INTO Users (FirstName, LastName, Email, Authorization, username, password) VALUES ('Fadimatou', 'Abdoulaye', 'afadimatou@gmail.com', 2, 'Fadimatou' , '$2b$12$66yNO9XAmZj2vMbYu.icRueNRVJK/cTWvoSE3czHt/P5BlKSFBdyu');
INSERT INTO Users (FirstName, LastName, Email, Authorization, username, password) VALUES ('John', 'Doe', 'john.doe@example.com', 1, 'John', '$2b$12$pfyjoh3RWS9crj8lFwF3ROWeSZ26Eu5m1WknmMJ0RqCYp.AHiu.MO');

