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
import matplotlib.pyplot as plt

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
data_train = data_train.take(total_batches)
total_batches = tf.data.experimental.cardinality(data_test).numpy()
data_test = data_test.take(total_batches)


model = pc.resnet50_feature_extractor()
features = model.predict(data_train)
image_feature =features[0]


plt.figure(figsize=(10, 10))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(image_feature[:, :, i], cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.show()

#creating memory_bank

memory_bank = pc.extract_aggregate(model,data_train)
print(memory_bank.shape)
#Coreset Subsampling

flatten_memory_bank = memory_bank.reshape(-1, memory_bank.shape[-1])
memory_bank_dir = config.get("PC_PARAMETERS", "memory_bank")
np.save(memory_bank_dir,memory_bank)
print(flatten_memory_bank.shape)
coreset = pc.coreset_subsampling(flatten_memory_bank)
#coreset = np.load("./trained_models/capsule_anomaly_scores.npy")

output_dir = config.get("PC_PARAMETERS", "coreset")
np.save(output_dir,coreset)
