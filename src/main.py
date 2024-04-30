from features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from models.train_model import TextRnnModel, ImageVGG16Model, concatenate
from tensorflow import keras
from sklearn.metrics import f1_score
import pickle
import tensorflow as tf
import sys
import json
from tools import f1_m, load_model

data_importer = DataImporter()
df = data_importer.load_data()
samples_per_class = 600
X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df, samples_per_class=samples_per_class) 

# Preprocess text and images
text_preprocessor = TextPreprocessor()
image_preprocessor = ImagePreprocessor()
text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
image_preprocessor.preprocess_images_in_df(X_train)
image_preprocessor.preprocess_images_in_df(X_val)

# Train Rnn model
print("Training RNN Model")
text_rnn_model = TextRnnModel()
text_rnn_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
print("Finished training RNN")


print("Training VGG")
# Train VGG16 model
image_vgg16_model = ImageVGG16Model()
image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
print("Finished training VGG")

with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
    tokenizer_config = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

rnn = load_model("best_rnn_model.h5")
vgg16 = load_model("best_vgg16_model.h5")
           
print("Training the concatenate model")
model_concatenate = concatenate(tokenizer, rnn, vgg16)
new_samples_per_class = int(samples_per_class/12) # 50
rnn_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train, new_samples_per_class=new_samples_per_class)  
best_weights = model_concatenate.optimize(rnn_proba, vgg16_proba, new_y_train)
print("Finished training concatenate model")

with open("models/best_weights.pkl", "wb") as file:
    pickle.dump(best_weights, file)
with open("models/best_weights.json", "w") as file:
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

# Enregistrer le modèle au format h5
concatenate_model.save("models/concatenate.h5")
