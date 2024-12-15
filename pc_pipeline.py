#Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from keras import preprocessing
from prepare_data import normalize, load_data_from_directory
import pc_structure as pc
import numpy as np
import configparser

# load configuration
config = configparser.ConfigParser()
config.read("config.ini")
dir_train = config.get("PATHS","train_dir")
dir_test = config.get("PATHS","test_dir")

#load dataset

data_train = preprocessing.image_dataset_from_directory(
    dir_train,
    labels="inferred",
    seed=123,
    image_size=(256, 256),
    batch_size=32
)
data_test = preprocessing.image_dataset_from_directory(
    dir_test,
    labels="inferred",
    seed=123,
    image_size=(256, 256),
    batch_size=32
)

#preprocessing data

data_train = data_train.map(lambda x,y: (normalize(x), y))
data_test = data_test.map(lambda x,y: (normalize(x), y))
total_batches = tf.data.experimental.cardinality(data_train).numpy()
data_train = data_train.take(total_batches)
total_batches = tf.data.experimental.cardinality(data_test).numpy()
data_test = data_test.take(total_batches)

#creating feature Extractor

model = pc.resnet50_feature_extractor()

#creating memory_bank

memory_bank = pc.extract_aggregate(model,data_train)

#extracting features from test data and calculate distance from data in memory bank

anomaly_scores_per_image = []
image_batch, label_batch = next(iter(data_test))  # Take first batch
label_batch = np.array(label_batch) 
for i in range(1,6):
    image = image_batch[i]  # Take first image from batch
    anomalies = pc.calculate_anomalies(model, image, memory_bank) # calculate scores

    anomaly_scores_per_image.append(anomalies)

# saving extracted anomaly scores
output_dir = config.get("PC_PARAMETERS", "output_dir")
np.save(output_dir, anomaly_scores_per_image)    

print("Training completed successfully.")

# Visualize anomalies on first image in the batch
for i in range(anomaly_scores_per_image.shape[0]):      
    pc.visualize_anomalies(image_batch[i], anomaly_scores_per_image[i],label_batch[i])  # Visualisiere die Anomalien auf das erste Bild im Batch