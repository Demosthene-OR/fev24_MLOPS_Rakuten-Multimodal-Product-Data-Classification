import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from filesplit.merge import Merge
from filesplit.split import Split
import os

# Definition de la metrique weighted-F1 score
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

file_path = "models"

@tf.function
def f1_m(y_true, y_pred):
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)
    precision_result = precision.result()
    recall_result = recall.result()
    
    # Calculer le F1-score
    f1 = 2 * ((precision_result * recall_result) / (precision_result + recall_result + K.epsilon()))
    
    # Calculer le weighted F1-score
    weights = tf.reduce_sum(y_true, axis=0) / tf.reduce_sum(y_true)
    weighted_f1 = tf.reduce_sum(weights * f1)
    
    return weighted_f1

# Voici 2 fonctions qui permettent de passer sous la barre des 100 Mo,
# et permettent ainsi d'enregistrer le model sur Github sans LFS .....

def load_model(file_path, file_name):
    
    merge = Merge(file_path+"/"+file_name[:-3],  file_path, file_name).merge(cleanup=False)
    with keras.utils.custom_object_scope({"f1_m": f1_m}):
        return keras.models.load_model(file_path+"/"+file_name)

def save_model(file_path, file_name):
    
    # if not os.path.exists(file_path):
    #     os.makedirs(file_path)
    Split(file_path+"/"+file_name,file_path+"/"+file_name[:-3]).bysize(9500000)
    return
    