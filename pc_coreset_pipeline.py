#Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from keras import preprocessing
from prepare_data import normalize, load_data_from_directory, crop_images_to_224
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
data_train = data_train.map(crop_images_to_224)
data_test = data_test.map(lambda x,y: (normalize(x), y))
data_test = data_test.map(crop_images_to_224)
total_batches = tf.data.experimental.cardinality(data_train).numpy()
data_train = data_train.take(total_batches-4)
total_batches = tf.data.experimental.cardinality(data_test).numpy()
data_test = data_test.take(total_batches-4)

#creating feature Extractor

model = pc.resnet50_feature_extractor()

#creating memory_bank

memory_bank = pc.extract_aggregate(model,data_train)

#Coreset Subsampling

flatten_memory_bank = memory_bank.reshape(-1, memory_bank.shape[-1])
#flatten_memory_bank = np.load("trained_models/coreset_capsule_0.01.npy")
print(flatten_memory_bank.shape)

coreset = pc.coreset_subsampling(flatten_memory_bank)

print(coreset.shape)

output_dir = config.get("PC_PARAMETERS", "output_dir")
np.save(output_dir,coreset)

#extracting features from test data and calculate distance from data in memory bank

anomaly_scores_per_image = []
image_batch, label_batch = next(iter(data_test))  # Take first batch
label_batch = np.array(label_batch) 
for i in range(1):
    image = image_batch[i]  # Take first image from batch
    anomalies = pc.calculate_anomalies(model, image, flatten_memory_bank) # calculate scores

    anomaly_scores_per_image.append(anomalies)
anomaly_scores = np.array(anomaly_scores_per_image)

# saving extracted anomaly scores
   
print("Training completed successfully.")

print(anomaly_scores)
# Visualize anomalies on first image in the batch
for i in range(anomaly_scores.shape[0]):      
    pc.visualize_anomalies(image_batch[i], anomaly_scores[i],label_batch[i])  # Visualisiere die Anomalien auf das erste Bild im Batch