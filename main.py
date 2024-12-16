import numpy as np
import tensorflow as tf
import configparser
config = configparser.ConfigParser()
from prepare_data import load_data_from_directory, normalize, crop_images_to_224
import pc_structure as pc

#Visualization of Anomaly Detection for PC
config.read("config.ini")
model_dir = config.get("PC_PARAMETERS", "model_dir")



#Load dataset and prepare
data_dir = config.get("PATHS", "test_dir")
data = load_data_from_directory(data_dir, seed=123, image_size=(256,256))
data = data.map(crop_images_to_224)
memory_bank = np.load(config.get("PC_PARAMETERS", "model_dir"))

print(memory_bank.shape)
model = pc.resnet50_feature_extractor()

anomaly_scores_per_image = []

data = data.map(lambda x,y: (normalize(x),y))

all_images = []
all_labels = []

for image_batch, label_batch in data:
    all_images.append(image_batch)
    all_labels.append(label_batch)

all_images = tf.concat(all_images, axis=0)
all_labels = tf.concat(all_labels, axis=0)
print(all_images.shape, all_labels.shape)

for i in range(len(all_images)):
    
    image = all_images[i]  # Take first image from batch
    anomalies = pc.calculate_anomalies(model, image, memory_bank) # calculate scores

    anomaly_scores_per_image.append(anomalies)
    print(f"Image {i} done!")

#Visualization of Anomaly Detection for PC
anomaly_scores_per_image = np.array(anomaly_scores_per_image)

np.save(config.get("PC_PARAMETERS", "output_dir"), anomaly_scores_per_image)

for i in range(anomaly_scores_per_image.shape[0]):      
    pc.visualize_anomalies(all_images[i], anomaly_scores_per_image[i],all_labels[i])  # Visualisiere die Anomalien auf das erste Bild im Batch