import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, GRU, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.utils import resample
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
from datetime import datetime
from keras import backend as K
from src.tools import f1_m, load_model, save_model
import pickle
import json
import os
import sys
import glob


class TextRnnModel:
    def __init__(self, max_words=30000, max_sequence_length=50, file_path="models"):
        
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.file_path = file_path

        if not glob.glob(self.file_path+"/best_rnn_model/best_rnn_model*.h5"):
            self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
            self.model = None
        else:
            with open(self.file_path+"/tokenizer_config.json", "r", encoding="utf-8") as json_file:
                tokenizer_config = json_file.read()
            self.tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
            self.model = load_model(self.file_path, "best_rnn_model.h5")

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val, n_epochs=5):
        
        # Si le modele RNN n'existe pas, on initialise le tokenizer (création du vocabulaire)
        if not glob.glob(self.file_path+"/best_rnn_model/best_rnn_model*.h5"):
            self.tokenizer.fit_on_texts(X_train["description"])
            tokenizer_config = self.tokenizer.to_json()
            with open(self.file_path+"/tokenizer_config.json", "w", encoding="utf-8") as json_file:
                json_file.write(tokenizer_config)

        train_sequences = self.tokenizer.texts_to_sequences(X_train["description"])
        train_padded_sequences = pad_sequences(
            train_sequences,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )
        
        val_sequences = self.tokenizer.texts_to_sequences(X_val["description"])
        val_padded_sequences = pad_sequences(
            val_sequences,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
        )
        
        # Si, le modèle n'xiste pas, on le créé: 1 couche d'embedding, 1 couche de GRU bi-directionnel, 1 couche de Dense softmax de classification
        if not os.path.exists(self.file_path+"/best_rnn_model.h5"):
            text_input = Input(shape=(self.max_sequence_length,))
            embedding_layer = Embedding(input_dim=self.max_words, output_dim=512)(  #128
                text_input
            )
            # lstm_layer = LSTM(512)(embedding_layer)
            gru_layer = Bidirectional(GRU(512))(embedding_layer)
            x = Dropout(0.1)(gru_layer)
            output = Dense(27, activation="softmax")(x)
            
            self.model = Model(inputs=[text_input], outputs=output)
        # Sinon, on charge le modele existant.
        else:
            self.model = load_model(self.file_path, "best_rnn_model.h5") 

        # Compile le modèle avec la métrique F1
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=[f1_m,"accuracy"] 
        )

        # Définir le nom de l'expérience TensorBoard avec la date et l'heure
        log_name = f"experience_tensorboard_{datetime.now().strftime('%Y%m%d-%H%M%S')}_text"

        rnn_callbacks = [
            ModelCheckpoint(
                filepath=self.file_path+"/best_rnn_model.h5", save_best_only=True
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir=f"logs/{log_name}"),  # Enregistre les journaux pour TensorBoard
        ]

        history = self.model.fit(
            [train_padded_sequences],
            tf.keras.utils.to_categorical(y_train, num_classes=27),
            epochs=n_epochs,
            batch_size=32,
            validation_data=(
                [val_padded_sequences],
                tf.keras.utils.to_categorical(y_val, num_classes=27),
            ),
            callbacks=rnn_callbacks,
        )
        save_model(self.file_path, "best_rnn_model.h5")
        
        # Récupérer les meilleures valeurs de F1 et l'accuracy correspondante
        best_loss_epoch = np.argmin(history.history['val_loss'])
        # Récupérer les meilleures valeurs de F1 et d'accuracy
        best_f1 = history.history['f1_m'][best_loss_epoch]
        best_accuracy = history.history['accuracy'][best_loss_epoch]
        return history, best_loss_epoch, best_f1, best_accuracy


class ImageVGG16Model:
    def __init__(self, file_path="models"):
        
        self.file_path = file_path
        
        if not glob.glob(self.file_path+"/best_vgg16_model/best_vgg16_model*.h5"):
            self.model = None
        else:
            self.model = load_model(self.file_path, "best_vgg16_model.h5")

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val, n_epochs=5):
        
        # Paramètres
        batch_size = 32
        num_classes = 27

        df_train = pd.concat([X_train, y_train.astype(str)], axis=1)
        df_val = pd.concat([X_val, y_val.astype(str)], axis=1)

        # Créer un générateur d'images pour le set d'entraînement
        train_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=df_train,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),  # Adapter à la taille d'entrée de VGG16
            batch_size=batch_size,
            class_mode="categorical",  # Utilisez 'categorical' pour les entiers encodés en one-hot
            shuffle=True,
        )

        # Créer un générateur d'images pour le set de validation
        val_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
        val_generator = val_datagen.flow_from_dataframe(
            dataframe=df_val,
            x_col="image_path",
            y_col="prdtypecode",
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,  # Pas de mélange pour le set de validation
        )

        if not glob.glob(self.file_path+"/best_vgg16_model/best_vgg16_model*.h5"):
            image_input = Input(
                shape=(224, 224, 3)
            )  # Adjust input shape according to your images
        
            vgg16_base = VGG16(
                include_top=False, weights="imagenet", input_tensor=image_input
            )

            x = vgg16_base.output
            x = Flatten()(x)
            x = Dense(512, activation="relu")(x) 
            x = Dense(512, activation="relu")(x) 
            output = Dense(num_classes, activation="softmax")(x)

            self.model = Model(inputs=vgg16_base.input, outputs=output)

            for layer in vgg16_base.layers:
                layer.trainable = False
        else:
            self.model = load_model(self.file_path, "best_vgg16_model.h5") 
        
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=[f1_m,"accuracy"]
        )

        # Définir le nom de l'expérience TensorBoard avec la date et l'heure
        log_name = f"experience_tensorboard_{datetime.now().strftime('%Y%m%d-%H%M%S')}_img"
        
        vgg_callbacks = [
            ModelCheckpoint(
                filepath=self.file_path+"/best_vgg16_model.h5", save_best_only=True
            ),  # Enregistre le meilleur modèle
            EarlyStopping(
                patience=3, restore_best_weights=True
            ),  # Arrête l'entraînement si la performance ne s'améliore pas
            TensorBoard(log_dir=f"logs/{log_name}"),  # Enregistre les journaux pour TensorBoard
        ]

        history = self.model.fit(
            train_generator,
            epochs=n_epochs,
            validation_data=val_generator,
            callbacks=vgg_callbacks,
        )
        save_model(self.file_path, "best_vgg16_model.h5")
        
        # Récupérer les meilleures valeurs de F1 et l'accuracy correspondante
        best_loss_epoch = np.argmin(history.history['val_loss'])
        # Récupérer les meilleures valeurs de F1 et d'accuracy
        best_f1 = history.history['f1_m'][best_loss_epoch]
        best_accuracy = history.history['accuracy'][best_loss_epoch]
        return history, best_loss_epoch, best_f1, best_accuracy
        
class concatenate:
    def __init__(self, tokenizer, rnn, vgg16):
        self.tokenizer = tokenizer
        self.rnn = rnn
        self.vgg16 = vgg16

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(
        self, X_train, y_train, new_samples_per_class=0, max_sequence_length=50, random_state=42
    ):
               
        num_classes = 27
        
        new_X_train = pd.DataFrame(columns=X_train.columns)
        new_y_train = pd.DataFrame(
            columns=[0]
        )  # Créez la structure pour les étiquettes

        # Boucle à travers chaque classe
        for class_label in range(num_classes):
            # Indices des échantillons appartenant à la classe actuelle
            indices = np.where(y_train == class_label)[0]

            # Sous-échantillonnage aléatoire. Si 'new_samples_per_class'>0, sélectionner 'new_samples_per_class' échantillons
            if (new_samples_per_class>0):
                sampled_indices = resample(
                    indices, 
                    n_samples=new_samples_per_class, 
                    replace=False, random_state=random_state
                    )
            else:
                sampled_indices = resample(
                    indices, 
                    replace=False, random_state=random_state
                    )
                
            # Ajout des échantillons sous-échantillonnés et de leurs étiquettes aux DataFrames
            new_X_train = pd.concat([new_X_train, X_train.loc[sampled_indices]])
            new_y_train = pd.concat([new_y_train, y_train.loc[sampled_indices]])

        # Réinitialiser les index des DataFrames
        new_X_train = new_X_train.reset_index(drop=True)
        new_y_train = new_y_train.reset_index(drop=True)
        new_y_train = new_y_train.values.reshape(new_y_train.shape[0]).astype("int")

        # Charger les modèles préalablement sauvegardés
        tokenizer = self.tokenizer
        rnn_model = self.rnn
        vgg16_model = self.vgg16

        train_sequences = tokenizer.texts_to_sequences(new_X_train["description"])
        train_padded_sequences = pad_sequences(
            train_sequences, maxlen=max_sequence_length, padding="post", truncating="post"
        )

        # Paramètres pour le prétraitement des images
        target_size = (
            224,
            224,
            3,
        )  # Taille cible pour le modèle VGG16, ajustez selon vos besoins

        images_train = new_X_train["image_path"].apply(
            lambda x: self.preprocess_image(x, target_size)
        )

        images_train = tf.convert_to_tensor(images_train.tolist(), dtype=tf.float32)

        rnn_proba = rnn_model.predict([train_padded_sequences])

        vgg16_proba = vgg16_model.predict([images_train])

        return rnn_proba, vgg16_proba, new_y_train

    def optimize(self, rnn_proba, vgg16_proba, y_train):
        # Recherche des poids optimaux en utilisant la validation croisée
        best_weights = None
        best_accuracy = 0.0
        best_weighted_f1 = 0.0

        for rnn_weight in np.linspace(0, 1, 101):  # Essayer différents poids pour RNN
            vgg16_weight = 1.0 - rnn_weight  # Le poids total doit être égal à 1

            combined_predictions = (rnn_weight * rnn_proba) + (
                vgg16_weight * vgg16_proba
            )
            final_predictions = np.argmax(combined_predictions, axis=1)
            accuracy = accuracy_score(y_train, final_predictions)
            weighted_f1 = f1_score(y_train, final_predictions, average='weighted')
            if weighted_f1 > best_weighted_f1:
                best_weighted_f1 = weighted_f1
                best_accuracy = accuracy
                best_weights = (rnn_weight, vgg16_weight)
        
        print('============================')
        print("Train dataset size :", len(y_train))   
        print("best_weighted_f1 =", best_weighted_f1)
        print("best_accuracy =", best_accuracy)
        print("best_weights =", best_weights)
        # print('============================')
        
        
        return best_weights, best_weighted_f1, best_accuracy, len(y_train)
