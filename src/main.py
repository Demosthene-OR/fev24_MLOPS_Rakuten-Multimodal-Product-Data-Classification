from features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from models.train_model import TextRnnModel, ImageVGG16Model, concatenate
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import tensorflow as tf
import sys
import json
import time
import argparse
import numpy as np
from tools import f1_m, load_model
import sys
import random


def main():
    parser = argparse.ArgumentParser(description= "Input data")
    
    parser.add_argument("--x_train_path", default = "data/preprocessed/X_train_update.csv", type=str,help="File path for the X_train input CSV file.")
    parser.add_argument("--y_train_path", default = "data/preprocessed/Y_train_CVw08PX.csv", type=str,help="File path for the Y_train input CSV file.")
    parser.add_argument("--images_path", default = "data/preprocessed/image_train", type=str,  help="Base path for the images.")
    parser.add_argument("--model_path", default = "models", type=str,  help="Base path for the models")
    parser.add_argument("--n_epochs", default = 1, type=int,  help="Num epochs")
    parser.add_argument("--samples_per_class", default = 50, type=int,  help="Num samples per class") # Caution: If samples_per_class==0 , the Full Dataset will be used
    parser.add_argument("--with_test", default = 0, type=int,  help="Compute performance on Test dataset")
    parser.add_argument("--random_state", default = 42, type=int,  help="random_state")
    args = parser.parse_args()
    with_test = False if args.with_test==0 else True
    samples_per_class = args.samples_per_class
    n_epochs = args.n_epochs
    random_state = args.random_state if args.random_state >= 0 else random.randint(0, 100)

    t_debut = time.time()
    data_importer = DataImporter(args.x_train_path,args.y_train_path, args.model_path )
    df = data_importer.load_data()
      
    X_train, X_val, X_test, y_train, y_val, y_test = \
        data_importer.split_train_test(df, samples_per_class=samples_per_class, random_state=random_state, with_test=with_test) 

    # Preprocess text and images
    text_preprocessor = TextPreprocessor()
    image_preprocessor = ImagePreprocessor(args.images_path)
    text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
    text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
    image_preprocessor.preprocess_images_in_df(X_train)
    image_preprocessor.preprocess_images_in_df(X_val)
    if with_test:
        text_preprocessor.preprocess_text_in_df(X_test, columns=["description"])
        image_preprocessor.preprocess_images_in_df(X_test)

    # sys.exit(0)
    # Train Rnn model
    print("Training RNN Model")
    text_rnn_model = TextRnnModel(file_path=args.model_path)
    text_rnn_model.preprocess_and_fit(X_train, y_train, X_val, y_val, n_epochs=n_epochs)
    print("Finished training RNN")


    print("Training VGG")
    # Train VGG16 model
    image_vgg16_model = ImageVGG16Model(file_path=args.model_path)
    image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val, n_epochs=n_epochs)
    print("Finished training VGG")

    with open(args.model_path+"/tokenizer_config.json", "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

    rnn = load_model(args.model_path,"best_rnn_model.h5")
    vgg16 = load_model(args.model_path,"best_vgg16_model.h5")
           
    print("Training the concatenate model")
    model_concatenate = concatenate(tokenizer, rnn, vgg16)
    if (samples_per_class > 0):
        new_samples_per_class = max(int(samples_per_class/12),3) # 50
    else:
        new_samples_per_class = 0

    rnn_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train, new_samples_per_class=new_samples_per_class, random_state=random_state)  
    best_weights = model_concatenate.optimize(rnn_proba, vgg16_proba, new_y_train)
    print("Finished training concatenate model")

    with open(args.model_path+"/best_weights.pkl", "wb") as file:
        pickle.dump(best_weights, file)
    with open(args.model_path+"/best_weights.json", "w") as file:
        json.dump(best_weights, file)

    num_classes = 27

    proba_rnn = keras.layers.Input(shape=(num_classes,))
    proba_vgg16 = keras.layers.Input(shape=(num_classes,))

    weighted_proba = keras.layers.Lambda(
        lambda x: best_weights[0] * x[0] + best_weights[1] * x[1]
    )([proba_rnn, proba_vgg16])

    concatenate_model = keras.models.Model(
        inputs=[proba_rnn, proba_vgg16], outputs=weighted_proba
    )
    
    t_fin = time.time()
    print("Durée de l'entrainement = {:.2f}".format(t_fin - t_debut))
    print('============================')

    # Enregistre le modèle au format h5
    concatenate_model.save(args.model_path+"/concatenate.h5")


    
    # Calcul de la perforance sur le dataset test
    if with_test:
        t_debut = time.time()
        rnn_proba_test, vgg16_proba_test, new_y_test = model_concatenate.predict(X_test, y_test, new_samples_per_class=0, random_state=random_state) 
        combined_predictions = (best_weights[0] * rnn_proba_test) + (best_weights[1] * vgg16_proba_test)
        final_predictions = np.argmax(combined_predictions, axis=1)
        test_accuracy = accuracy_score(new_y_test, final_predictions)
        test_f1 = f1_score(new_y_test , final_predictions, average='weighted')
        t_fin = time.time()
        print('============================')
        print("Taille du dataset Test :", len(y_test))
        print("Test: f1 score =", test_f1)
        print("Test accuracy score =", test_accuracy)
        print("Durée du test = {:.2f}".format(t_fin - t_debut))
        print('============================')
    
if __name__ == "__main__":
    main()