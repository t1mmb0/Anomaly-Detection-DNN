import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import numpy as np
from prepare_data import normalize, load_saved_dataset, load_data_from_directory, add_gausian_noise, crop_images_to_224
import matplotlib.pyplot as plt
from pc_structure import *
from convAE_structure import *


def image_split(image):   
    patch_w, patch_h = 112, 37
    patches = []
    for i in range(3): #Zeilen
        for j in range(1): #Spalten
            patch = (image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w])
            patches.append(patch)

    patches = np.array(patches)
    return patches
def calculate_ssim(input_images, reconstructed_images):
    ssim = tf.image.ssim(input_images, reconstructed_images, max_val=1.0)
    return ssim

def calculate_mse(input_images, reconstructed_images):

    mse = np.mean((input_images - reconstructed_images) ** 2)
    return mse

def compute_error_map(original, reconstruction):
    return np.abs(original - reconstruction)  # Pixelweiser absoluter Fehler

def visualization(original, reconstruction, error_map, label, ssim, mse, dict):
    plt.figure(figsize=(10,10))  # Dynamische Größe basierend auf Anzahl der Bilder
    # Originalbild
    plt.subplot(3,3,1)  # Erste Spalte: Original
    plt.imshow(original, cmap="gray")
    if dict == None:
        label_title = "good"
    else:
        label_tuple = tuple(label.numpy())
        label_title = dict[label_tuple]

    plt.title(f"Original\nLabel: {label_title}", fontsize=10)
    plt.axis("off")

    # Rekonstruiertes Bild
    plt.subplot(3,3,2)  # Zweite Spalte: Rekonstruktion
    plt.imshow(reconstruction, cmap="gray")
    plt.title(f"Rekonstruktion\nSSIM: {ssim:.4f}, MSE: {mse:.4f}", fontsize=10)
    plt.axis("off")

    # Fehlerkarte als Heatmap
    normalized_error_map = np.linalg.norm(error_map, axis=-1)


    plt.subplot(3,3,3)  # Dritte Spalte: Fehlerkarte
    im = plt.imshow(normalized_error_map, cmap="hot", interpolation="nearest")
    plt.title("Fehlerkarte", fontsize=10)
    plt.axis("off")
    plt.colorbar(im, shrink=0.8)

    plt.tight_layout(pad=5)  # Abstand zwischen den Plots
    plt.show()

# Load config and extract parameters
import configparser
config = configparser.ConfigParser()
#Dataset Preparation
config.read("config.ini")
data_filename = config.get("PATHS", "test_dir")
model_path = config.get("CONV_AE_PARAMETERS", "model_dir")


capsule_label_dict = {
    (1, 0, 0, 0, 0, 0): "crack",
    (0, 1, 0, 0, 0, 0): "faulty_imprint",
    (0, 0, 1, 0, 0, 0): "good",
    (0, 0, 0, 1, 0, 0): "poke",
    (0, 0, 0, 0, 1, 0): "scratch",
    (0, 0, 0, 0, 0, 1): "squeeze"
}

#Load data

data = load_data_from_directory(data_filename,image_size=(256,256))
data = data.map(crop_images_to_224)
dataset = data.map(lambda x,y: (normalize(x),y))

#Load model

custom_objects = {
    'Encoder': Encoder,
    'Decoder': Decoder,
    'Autoencoder': Autoencoder,
    'ResNetEncoder': ResNetEncoder,
    'loss': ssim_loss
}

autoencoder = keras.models.load_model(model_path, custom_objects=custom_objects)
predictions = autoencoder.encoder.predict(dataset)

# Prepare data
images, labels = next(iter(dataset))

#Calculate anomaly scores
for i in range(10):
    image, label = images[i], labels[i]
    noisy_image = add_gausian_noise(image, stddev=0.05)
    noisy_image_batch = tf.expand_dims(noisy_image, axis=0)

    prediction = autoencoder.predict(noisy_image_batch)
  
    original = image
    reconstruction = prediction[0]

    ssim_error = calculate_ssim(original, reconstruction)
    mse_error = calculate_mse(original, reconstruction)
    error_map = compute_error_map(original, reconstruction)


    # Funktion aufrufen
    visualization(image, reconstruction, error_map, label, ssim_error, mse_error, capsule_label_dict)