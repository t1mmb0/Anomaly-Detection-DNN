import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import configparser
config = configparser.ConfigParser()
from prepare_data import load_data_from_directory, normalize, crop_images_to_224
import pc_structure as pc
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

#Visualization of Anomaly Detection for PC
config.read("config.ini")
training_mode=True

capsule_label_dict = {
    (1, 0, 0, 0, 0, 0): "crack",
    (0, 1, 0, 0, 0, 0): "faulty_imprint",
    (0, 0, 1, 0, 0, 0): "good",
    (0, 0, 0, 1, 0, 0): "poke",
    (0, 0, 0, 0, 1, 0): "scratch",
    (0, 0, 0, 0, 0, 1): "squeeze"
}

#Load dataset and prepare
data_dir = config.get("PATHS", "test_dir")
data = load_data_from_directory(data_dir, seed=123, image_size=(256,256))
data = data.map(crop_images_to_224)

memory_bank = np.load(config.get("PC_PARAMETERS", "coreset"))
print(memory_bank.shape)
model = pc.resnet50_feature_extractor()



data = data.map(lambda x,y: (normalize(x),y))

all_images = []
all_labels = []

for image_batch, label_batch in data:
    all_images.append(image_batch)
    all_labels.append(label_batch)

all_images = tf.concat(all_images, axis=0)
all_labels = tf.concat(all_labels, axis=0)


#extracting features from test data and calculate distance from data in memory bank

anomaly_scores_per_image = []

for i in range(5):
    image = all_images[i]  # Take first image from batch
    anomalies = pc.calculate_anomalies(model, image, memory_bank, projection_dim=50) # calculate scores

    anomaly_scores_per_image.append(anomalies)
anomaly_scores = np.array(anomaly_scores_per_image)

   
print("Training completed successfully.")

# Visualize anomalies on first image in the batch
for i in range(anomaly_scores.shape[0]):      
    pc.visualize_anomalies(all_images[i], anomaly_scores[i],all_labels[i])  # Visualisiere die Anomalien auf das erste Bild im Batch