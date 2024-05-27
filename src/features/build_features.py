import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import math
import json
import os
import sys


class DataImporter:
    def __init__(self, x_train_path="data/preprocessed/X_train_update.csv", y_train_path="data/preprocessed/Y_train_CVw08PX.csv", model_path="models"):
        self.x_train_path = x_train_path
        self.y_train_path = y_train_path
        self.model_path = model_path

    def load_data(self):
        data = pd.read_csv(self.x_train_path)
        # Remplacer les NaN par des chaînes vides
        data["designation"] = data["designation"].fillna('')
        data["description"] = data["description"].fillna('')
        data["description"] = data["designation"] + " " + data["description"]
        data = data.drop(["Unnamed: 0", "designation"], axis=1)

        target = pd.read_csv(self.y_train_path)
        target = target.drop(["Unnamed: 0"], axis=1)
        if not os.path.exists(f"{self.model_path}/mapper.json"):
            modalite_mapping = {
                modalite: i for i, modalite in enumerate(target["prdtypecode"].unique())
            }
            # with open(f"{self.model_path}mapper.pkl", "wb") as fichier:
            #     pickle.dump(modalite_mapping, fichier)
            with open(f"{self.model_path}/mapper.json", "w") as fichier_json:
                json_mapper = {str(v): str(k) for k, v in modalite_mapping.items()}
                json.dump(json_mapper, fichier_json)
        else:
            with open(f"{self.model_path}/mapper.json", "r") as json_file:
                modalite_mapping = json.load(json_file)
                modalite_mapping = {int(v): int(k) for k, v in modalite_mapping.items()}
            
        target["prdtypecode"] = target["prdtypecode"].replace(modalite_mapping)

        df = pd.concat([data, target], axis=1)

        return df

    def split_train_test(self, df, samples_per_class=0, random_state=42, with_test=False):   
        # Dans la suite, si samples_per_class==0, on entraine le modele sur la totalité de df
        # Sinon on l'entraine sur un jeu de donnée equilibrée (X_train = nb de classes * samples_per_class)
        
        print("len(df) = ",len(df))
        grouped_data = df.groupby("prdtypecode")
        
        # Le calcul suivant est nécessaire si on entraine le modele sur la totalité de df (i.e. samples_per_class==0,)
        class_size = grouped_data.size().tolist()
        train_size = [int(n*0.8) for n in class_size]
        # Le nombre de ligne de X_test est plafonné à 50 * nombre de classes
        test_reduc =  1.0 if (len(df)//10) < (27*50) else 1350 / len(df)
        test_size = [(math.ceil(test_reduc *n) if with_test else 0) for n in class_size]
        val_size = [(class_size[i]-train_size[i]-test_size[i]) for i in range(len(class_size))]


        X_train_samples = []
        X_test_samples = []
        i=0
        
        for _, group in grouped_data:
            
            if (samples_per_class > 0):
                samples = group.sample(n=samples_per_class, random_state=random_state)
            else:
                samples = group.sample(n=train_size[i], random_state=random_state)
                i +=1
            X_train_samples.append(samples)
            
            remaining_samples = group.drop(samples.index)
            X_test_samples.append(remaining_samples) 

        X_train = pd.concat(X_train_samples)
        X_test = pd.concat(X_test_samples)
        
        X_train = X_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
        X_test = X_test.sample(frac=1, random_state=random_state).reset_index(drop=True)

        y_train = X_train["prdtypecode"]
        X_train = X_train.drop(["prdtypecode"], axis=1)

        if (samples_per_class > 0):
            val_samples_per_class = max(int(samples_per_class/12),3)
        else:
            val_samples_per_class = 0

        grouped_data_test = X_test.groupby("prdtypecode") 
        X_test = X_test.drop(X_test.index)
        y_test=[]
        X_val_samples = []
        X_test_samples = []
        
        i=0

        for _, group in grouped_data_test:
            if (val_samples_per_class > 0):
                samples = group.sample(n=val_samples_per_class, random_state=random_state)
            else:
                samples = group.sample(n=val_size[i], random_state=random_state)
                i +=1
            X_val_samples.append(samples)
            remaining_samples = group.drop(samples.index)
            if with_test and (val_samples_per_class > 0):
                X_test_samples.append(remaining_samples[:50])
            elif with_test:
                X_test_samples.append(remaining_samples)
        
        X_val = pd.concat(X_val_samples)
        X_val = X_val.sample(frac=1, random_state=random_state).reset_index(drop=True)
        y_val = X_val["prdtypecode"]
        X_val = X_val.drop(["prdtypecode"], axis=1)
        
        if with_test:
            X_test = pd.concat(X_test_samples)
            X_test = X_test.sample(frac=1, random_state=random_state).reset_index(drop=True)
            y_test = X_test["prdtypecode"]
            X_test= X_test.drop(["prdtypecode"], axis=1)
        
        print('============================')
        print("Dataset size : ", len(X_train)+len(X_val)+len(X_test))
        print("Train size   : ", len(X_train))
        print("Val size     : ", len(X_val))
        print("Test size    : ", len(X_test))
        print('============================')
        # sys.exit(0)
        return X_train, X_val, X_test, y_train, y_val, y_test


class ImagePreprocessor:
    def __init__(self, filepath="data/preprocessed/image_train"):
        self.filepath = filepath

    def preprocess_images_in_df(self, df):
        df["image_path"] = (
            f"{self.filepath}/image_"
            + df["imageid"].astype(str)
            + "_product_"
            + df["productid"].astype(str)
            + ".jpg"
        )


class TextPreprocessor:
    def __init__(self):
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(
            stopwords.words("french")
        )  # Vous pouvez choisir une autre langue si nécessaire

    def preprocess_text(self, text):

        if isinstance(text, float) and math.isnan(text):
            return ""
        # Supprimer les balises HTML
        text = BeautifulSoup(text, "html.parser").get_text()

        # Supprimer les caractères non alphabétiques
        text = re.sub(r"[^a-zA-Z]", " ", text)

        # Tokenization
        words = word_tokenize(text.lower())

        # Suppression des stopwords et lemmatisation
        filtered_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]

        return " ".join(filtered_words[:50])

    def preprocess_text_in_df(self, df, columns):
        for column in columns:
            df[column] = df[column].apply(self.preprocess_text)
