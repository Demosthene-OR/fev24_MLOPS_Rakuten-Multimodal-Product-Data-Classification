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

random_state = 42

class DataImporter:
    def __init__(self, filepath="data/preprocessed"):
        self.filepath = filepath

    def load_data(self):
        data = pd.read_csv(f"{self.filepath}/X_train_update.csv")
        data["description"] = data["designation"] + " " + str(data["description"])
        data = data.drop(["Unnamed: 0", "designation"], axis=1)

        target = pd.read_csv(f"{self.filepath}/Y_train_CVw08PX.csv")
        target = target.drop(["Unnamed: 0"], axis=1)
        if not os.path.exists("models/mapper.json"):
            modalite_mapping = {
                modalite: i for i, modalite in enumerate(target["prdtypecode"].unique())
            }
            with open("models/mapper.pkl", "wb") as fichier:
                pickle.dump(modalite_mapping, fichier)
            with open("models/mapper.json", "w") as fichier_json:
                json_mapper = {str(v): str(k) for k, v in modalite_mapping.items()}
                json.dump(json_mapper, fichier_json)
        else:
            with open("models/mapper.json", "r") as json_file:
                modalite_mapping = json.load(json_file)
                modalite_mapping = {int(v): int(k) for k, v in modalite_mapping.items()}
            
        target["prdtypecode"] = target["prdtypecode"].replace(modalite_mapping)

        df = pd.concat([data, target], axis=1)

        return df

    def split_train_test(self, df, samples_per_class=0):   # Dans la suite, si samples_per_class==0, on utilise 70% de chaque classe de df pour X_train_samples
        global random_state

        grouped_data = df.groupby("prdtypecode")
        class_size = grouped_data.size().tolist()
        train_size = [int(n*0.8) for n in class_size]
        test_size = [class_size[i]-train_size[i] for i in range(len(class_size))]
        
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

        y_test = X_test["prdtypecode"]
        X_test = X_test.drop(["prdtypecode"], axis=1)

        val_samples_per_class = int(samples_per_class/12)

        grouped_data_test = pd.concat([X_test, y_test], axis=1).groupby("prdtypecode")

        X_val_samples = []
        y_val_samples = []
        i=0

        for _, group in grouped_data_test:
            if (val_samples_per_class > 0):
                samples = group.sample(n=val_samples_per_class, random_state=random_state)
            else:
                samples = group.sample(n=test_size[i], random_state=random_state)
                i +=1
            X_val_samples.append(samples[["description", "productid", "imageid"]])
            y_val_samples.append(samples["prdtypecode"])

        X_val = pd.concat(X_val_samples)
        y_val = pd.concat(y_val_samples)

        X_val = X_val.sample(frac=1, random_state=random_state).reset_index(drop=True)
        y_val = y_val.sample(frac=1, random_state=random_state).reset_index(drop=True)

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
