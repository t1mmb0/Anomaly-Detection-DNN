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


#Visualization of Anomaly Detection for PC
config.read("config.ini")
model_dir = config.get("PC_PARAMETERS", "model_dir")
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

memory_bank = np.load(config.get("PC_PARAMETERS", "model_dir"))
model = pc.resnet50_feature_extractor()



data = data.map(lambda x,y: (normalize(x),y))

all_images = []
all_labels = []

for image_batch, label_batch in data:
    all_images.append(image_batch)
    all_labels.append(label_batch)

all_images = tf.concat(all_images, axis=0)
all_labels = tf.concat(all_labels, axis=0)

# Ordnerstruktur für die Bildspeicherung erstellen
output_dir = config.get("PC_PARAMETERS", "output_dir")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialisiere einen Zähler pro Label-Kategorie
label_counters = {}

for i in range(len(all_images)):
    
    image = all_images[i]  # Nimm das aktuelle Bild
    label = all_labels[i].numpy()
    label_name = capsule_label_dict[tuple(label)]

    # Erstelle den Ordner für das Label, falls noch nicht vorhanden
    label_folder = os.path.join(output_dir, label_name)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
        label_counters[label_name] = 0  # Starte den Zähler für neue Label-Kategorie

    # Anomalien berechnen und normalisieren
    anomalies = pc.calculate_anomalies(model, image, memory_bank)
    min_score = np.min(anomalies)
    max_score = np.max(anomalies)
    anomalies = (anomalies - min_score) / (max_score - min_score)
    anomalies = anomalies.reshape(56, 56)

    # Konvertiere Anomalien zu einem Bild
    anomaly_score_image = Image.fromarray((anomalies * 255).astype(np.uint8))

    # Dateinamen mit dreistelliger Formatierung erstellen
    file_index = label_counters[label_name]  # Hole aktuellen Zählerstand
    filename = f"{file_index:03d}.tiff"  # Formatierung zu '000.tiff', '001.tiff', etc.
    image_path = os.path.join(label_folder, filename)

    # Bild speichern
    anomaly_score_image.save(image_path)

    # Zähler für das aktuelle Label erhöhen
    label_counters[label_name] += 1

    print(f"Saved anomaly score for image {i} in {image_path}")

