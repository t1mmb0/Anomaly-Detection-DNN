import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from keras import layers, applications, preprocessing
import matplotlib.pyplot as plt
from prepare_data import normalize
import numpy as np
from scipy.ndimage import zoom

"""
    PC_Structure
"""

def resnet50_feature_extractor(weights="imagenet",input_shape=(224,224,3), output_layer="conv2_block3_out"):

    """

    Erstellt einen ResNet50-Feature-Extraktor.

    Parameter:
    weights = (str) ; Gewichte für den Feature-Extraktor. Default ist "imagenet".
    input_shape = (array) ; Die Dimensionen des Input-Bildes. Default ist (224,224,3).
    ouput_layer = (str) ; Der Name des Output-Layers des ResNet50-Modells. Default ist "conv4_block6_out".

    Rückgabewert:
    Gibt das Feature-Extraktor-Modell zurück.

    """
    resnet50 = applications.ResNet50(
        weights=weights,
        include_top=False, 
        input_shape=input_shape
    )
    resnet50.trainable = False

    feature_extractor = keras.models.Model(
        inputs=resnet50.input, outputs=resnet50.get_layer(output_layer).output
    )
    return feature_extractor

def aggregate_patches(feature_map, patch_size=3):
    """
    Aggregiert lokale 3x3 Patches innerhalb der Feature-Map.
    
    feature_map: numpy array oder Tensor der Form (Height, Width, Channels)
    patch_size: Größe des Patch-Fensters (standardmäßig 3x3)
    """
    height, width, channels = feature_map.shape
    padding = patch_size // 2  # Padding für die Rand-Patches

    # Padding hinzufügen
    padded_map = np.pad(feature_map, 
                        pad_width=((padding, padding), (padding, padding), (0, 0)), 
                        mode='constant')

    patches = []
    for i in range(height):
        for j in range(width):
            patch = padded_map[i:i + patch_size, j:j + patch_size, :]
            aggregated_patch = np.mean(patch, axis=(0, 1))  # Durchschnitt der 9 Pixel (3x3 Patch)
            patches.append(aggregated_patch)
    
    return np.array(patches)  # Shape: (Height * Width, Channels)

from sklearn.neighbors import NearestNeighbors

def calculate_anomalies(model, image, memory_bank): 
    anomalies = []

    # KNN-Ähnlichkeitsberechnung
    knn = NearestNeighbors(n_neighbors=1, metric='euclidean')



    # Feature Map für das Testbild extrahieren
    feature_map = model(tf.expand_dims(image, axis=0))  # Feature Map für Block 2
    feature_map = feature_map.numpy().squeeze()  # Entferne die Batch-Dimension

    # Aggregiere Patches für das Testbild
    patches = aggregate_patches(feature_map)

    # Berechne Ähnlichkeit (Abstand) der Patches des Testbildes zu den Patches in der Memorybank
    distances = []

    # Berechne KNN für jedes Patch des ersten Testbildes
    for patch in patches:
        # Berechne die Distanz von diesem Patch zu allen Patches der Memorybank
        knn.fit(memory_bank.reshape(-1, 256))  # Memorybank auf 2D umformen
        dist, _ = knn.kneighbors(patch.reshape(1, -1))  # Berechne Abstand zu den nächstgelegenen Patches
        distances.append(dist[0][0])  # Speichere den Abstand zum nächstgelegenen Nachbarn



    return np.array(distances)

def visualize_anomalies(image, anomalies, label):
    """
    Visualisiert die Anomalien auf den Patches eines Testbildes und stellt sicher, dass das Bild sichtbar bleibt.
    
    image: Das Testbild, das auf Anomalien überprüft wird.
    anomalies: Der Anomalie-Score für jedes Patch des Testbildes.
    """

    # Beispiel für Schwellenwert
    threshold = 0.12  # Der Schwellenwert für Anomalien

    # Setze Anomalien unter dem Schwellenwert auf 0
    anomalies_thresholded = np.where(anomalies >= threshold, anomalies, 0)
    # Normalisiere die Anomalie-Werte
    anomalies_normalized = (anomalies_thresholded - np.min(anomalies_thresholded)) / (np.max(anomalies_thresholded) - np.min(anomalies_thresholded))
    
    # Reshape der Anomalie-Werte zu einer 2D-Heatmap
    heatmap = anomalies_normalized.reshape(56, 56)  # Angenommen 56x56 Patches
    heatmap_rescaled = zoom(heatmap, (224/56, 224/56), order=1)
    # Visualisiere das zugrundeliegende Bild

    plt.imshow(image)  # Originalbild anzeigen
    plt.imshow(heatmap_rescaled, cmap='hot', interpolation='nearest', alpha=0.2)  # Heatmap mit Transparenz überlagern
    plt.colorbar(label="Anomalie-Score")
    plt.title(f"Anomalie-Lokalisierung Heatmap/ Label: {label}")
    plt.show()

def extract_aggregate(model, dataset):
    aggregated_features = []

    for image_batch,_ in dataset:
        for image in image_batch:
            feature_map = model(tf.expand_dims(image, axis=0))  # Feature Map für Block 2
            feature_map = feature_map.numpy().squeeze()
            patches = aggregate_patches(feature_map)
            aggregated_features.append(patches)


    return np.array(aggregated_features)


if __name__ == "__main__":
    pass