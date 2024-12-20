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

def calculate_anomalies(model, image, memory_bank, k_neighbors=5): 

    # KNN-Ähnlichkeitsberechnung
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')



    # Feature Map für das Testbild extrahieren
    feature_map = model(tf.expand_dims(image, axis=0))  # Feature Map für Block 2
    
    feature_map = feature_map.numpy().squeeze()  # Entferne die Batch-Dimension

    # Aggregiere Patches für das Testbild
    patches = aggregate_patches(feature_map)
    # Berechne Ähnlichkeit (Abstand) der Patches des Testbildes zu den Patches in der Memorybank
    
    knn.fit(memory_bank)
    # Berechne KNN für jedes Patch des ersten Testbildes

    # Berechne die Distanz von diesem Patch zu allen Patches der Memorybank       
    dist,_ = knn.kneighbors(patches)  # Berechne Abstand zu den nächstgelegenen Patches
    
    # Abstand zum nächsten Nachbarn (1. Spalte von dist)
    s_star = dist[:, 0]  # Shape: (3136,)

    # Exponentielle Abstände berechnen
    distance_to_one = np.exp(-s_star)  # Shape: (3136,)
    distance_to_others = np.sum(np.exp(-dist), axis=1)  # Summe über alle k-Nachbarn, Shape: (3136,)

    # Skalierungsfaktor berechnen
    factor = 1 - (distance_to_one / distance_to_others)  # Shape: (3136,)

    # Skalierten Score berechnen
    scaled_scores = factor * s_star  # Shape: (3136,)

    return np.array(scaled_scores)

def visualize_anomalies(image, anomalies, label):
    """
    Visualisiert die Anomalien auf den Patches eines Testbildes und stellt sicher, dass das Bild sichtbar bleibt.
    
    image: Das Testbild, das auf Anomalien überprüft wird.
    anomalies: Der Anomalie-Score für jedes Patch des Testbildes.
    """

    # Beispiel für Schwellenwert
    threshold = 0.32  # Der Schwellenwert für Anomalien

    # Setze Anomalien unter dem Schwellenwert auf 0
    anomalies_thresholded = np.where(anomalies >= threshold, anomalies, 0)
    # Normalisiere die Anomalie-Werte
    anomalies_normalized = (anomalies_thresholded - np.min(anomalies_thresholded)) / (np.max(anomalies_thresholded) - np.min(anomalies_thresholded))
    
    # Reshape der Anomalie-Werte zu einer 2D-Heatmap
    heatmap = anomalies_normalized.reshape(56, 56)  # Angenommen 56x56 Patches
    heatmap_rescaled = zoom(heatmap, (224/56, 224/56), order=1)
    # Visualisiere das zugrundeliegende Bild

    plt.imshow(image)  # Originalbild anzeigen
    plt.imshow(heatmap_rescaled, cmap='hot', interpolation='nearest', alpha=0.5)  # Heatmap mit Transparenz überlagern
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

from sklearn.random_projection import GaussianRandomProjection
# Funktion zur zufälligen linearen Projektion
def compute_projection_dimension(num_vectors, max_dim=256, min_dim=10, scaling_factor=2):
    """
    Berechnet die Projektion-Dimension d* basierend auf der Formel:
    d* = min(max_dim, max(min_dim, int(scaling_factor * log(num_vectors))))
    
    Parameter:
    - num_vectors (int): Anzahl der Vektoren in der Memory Bank.
    - max_dim (int): Maximale Ziel-Dimension (Standard: 256).
    - min_dim (int): Minimale Ziel-Dimension (Standard: 10).
    - scaling_factor (float): Skalierungsfaktor C für die logarithmische Abhängigkeit (Standard: 2).
    
    Rückgabe:
    - int: Berechnete Projektion-Dimension d*.
    """
    # Berechne den logarithmischen Wert
    log_value = np.log(num_vectors)
    
    # Skalierter logarithmischer Wert
    scaled_log = scaling_factor * log_value
    
    # Anwenden der min- und max-Grenzen
    projection_dim = min(max_dim, max(min_dim, int(scaled_log)))
    
    return projection_dim

from sklearn.metrics.pairwise import euclidean_distances

def coreset_subsampling(flatten_memory_bank, coreset_fraction=0.01):
    """
    Optimierte Coreset-Subsampling-Methode mit Zwischenspeicherung der Abstände.
    
    Parameter:
    - flatten_memory_bank: numpy array der Form (num_vectors, feature_dim)
    - coreset_fraction: Anteil der Vektoren, die im Coreset enthalten sein sollen
    - random_seed: Seed für Reproduzierbarkeit
    
    Rückgabe:
    - coreset: numpy array der Form (coreset_size, feature_dim)
    """
    np.random.seed(42)
    num_vectors = flatten_memory_bank.shape[0]
    print(num_vectors)
    coreset_size = int(num_vectors * coreset_fraction)
    
    # Initialisierung: Wähle einen zufälligen Startpunkt
    first_index = np.random.randint(0, num_vectors)
    coreset_indices = [first_index]
    

    
    # Berechne initial die Abstände von allen Punkten zu diesem Startpunkt
    distances = euclidean_distances(flatten_memory_bank, flatten_memory_bank[first_index].reshape(1, -1)).flatten()
    
    for i in range(coreset_size - 1):
        if i % 100 == 0:
            print(f"Iteration {i} of {coreset_size}.")
        
        # Finde den Punkt, der den maximalen Abstand zu den bisherigen Coreset-Punkten hat
        max_dist_idx = np.argmax(distances)
        coreset_indices.append(max_dist_idx)
        
        # Aktualisiere die minimalen Abstände: Nur neue Abstände berücksichtigen
        new_distances = euclidean_distances(flatten_memory_bank, flatten_memory_bank[max_dist_idx].reshape(1, -1)).flatten()
        distances = np.minimum(distances, new_distances)
    
    # Baue das finale Coreset aus den gewählten Indizes
    coreset = flatten_memory_bank[coreset_indices]
    return coreset


if __name__ == "__main__":
    pass