import numpy as np
import configparser
config = configparser.ConfigParser()
from prepare_data import load_data_from_directory, normalize, crop_images_to_224
import pc_structure as pc

#Visualization of Anomaly Detection for PC
config.read("config.ini")
model_dir = config.get("PC_PARAMETERS", "model_dir")



#Load dataset and prepare
data_dir = config.get("PATHS", "test_dir")
data = load_data_from_directory(data_dir, seed=123)


data = data.map(lambda x,y: (normalize(x),y))
image_batch, label_batch = next(iter(data))
label_batch = np.array(label_batch) 
#Load the model
anomaly_scores_per_image = np.load(model_dir)

#Visualization of Anomaly Detection for PC
for i in range(anomaly_scores_per_image.shape[0]):      
    pc.visualize_anomalies(image_batch[i], anomaly_scores_per_image[i],label_batch[i])  # Visualisiere die Anomalien auf das erste Bild im Batch