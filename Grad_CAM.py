import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import applications, models, preprocessing

import matplotlib.pyplot as plt
import cv2


visualize = False



# 1. Vortrainiertes Modell laden
model = applications.resnet50.ResNet50(weights="imagenet")
model.trainable = False
target_layer_name = "conv5_block3_out"

# 2. Gute Eingabebild vorbereiten
from prepare_data import load_data_from_directory, normalize
image_path = "./Images/capsule_images_train/"
data =load_data_from_directory(image_path, image_size=(224,224))
data = data.map(lambda x,y: (normalize(x),y))

# 3. Validierungsbilder vorbereiten
from prepare_data import load_data_from_directory, normalize
image_path_test = "./Images/capsule_images_test/"
data_test =load_data_from_directory(image_path_test, image_size=(224,224))
data_test = data.map(lambda x,y: (normalize(x),y))


#Verwendung einer vortrainierten Resnet Architektur zur Feature Extraction:

from ResNet_model import resnet50_feature_extractor
resnet = resnet50_feature_extractor(output_layer="conv5_block3_out")

normal_features = resnet.predict(data)
print(normal_features.shape)
validation_features = resnet.predict(data_test)


normal_features_flat = normal_features.reshape(normal_features.shape[0], -1)  # (219, 10048)
validation_features_flat = validation_features.reshape(validation_features.shape[0], -1)  # (219, 10048)

if visualize:
    # Visualisierung eines einzelnen Featurekanals
    plt.figure(figsize=(20,20))
    for i in range(64):
        plt.subplot(8, 8, i+1)
        feature_map = normal_features[0, :, :, i]
        plt.imshow(feature_map, cmap='jet')
        plt.colorbar()
    plt.show()



#4. NearestNeighbors

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error

# NearestNeighbors mit den normalen Features trainieren
knn = NearestNeighbors(n_neighbors=5)  # k = 5 Nachbarn
knn.fit(normal_features_flat)

# Berechne die Distanz zu den k-nächsten Nachbarn für Anomalie-Daten
distances, indices = knn.kneighbors(validation_features_flat) #
print(distances.shape, indices.shape)
distances_normal, indices_normal = knn.kneighbors(normal_features_flat)
# Berechne den mittleren Abstand zu den k-nächsten Nachbarn
mean_distances = np.mean(distances, axis=1)
print(mean_distances)

# Definiere einen Schwellenwert (z. B. das 95. Perzentil der Distanzen)
mean_distances_normal = np.mean(distances_normal, axis=1)
print(mean_distances_normal)
threshold = np.percentile(mean_distances_normal, 95)
print(threshold)
# Identifiziere Anomalien: Wenn der Abstand größer als der Schwellenwert ist, wird das Bild als anormal betrachtet
anomalies = mean_distances > threshold

# Ausgabe der Ergebnisse
print(f"Anomalien erkannt: {np.sum(anomalies)} von {len(anomalies)} Testbildern.")

