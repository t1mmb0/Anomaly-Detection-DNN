#Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import preprocessing


def load_data_from_directory(directory, label_type="inferred", batch_size=32, validation_split=None, subset=None, seed=42, label_mode="categorical", image_size=(224,224),shuffle=False):

    """
    lädt das Dataset und gibt es aus.
    """
    data = preprocessing.image_dataset_from_directory(
        directory=directory,
        labels=label_type,
        batch_size=batch_size,
        validation_split=validation_split,
        subset=subset,
        seed=seed,
        label_mode=label_mode,
        image_size=image_size,
        shuffle=shuffle
    )
    return data

def normalize(images):
    
    """

    Diese Funktion normalisiert die Bilddaten.
    Anwendbar mit dataset.map(normalize).

    """

    images = images/255.0
    
    return images


def labels_to_images(images, labels):

    """

    Diese Funktion ersetzt die Labels durch die Bilddaten. Wird für das Training eines Autoencoders verwendet.
    Anwendbar mit dataset.map(labels_to_images).

    """

    return images, images

def discard_labels(images,labels):
    
    """
    Diese Funktion entfernt die Labels aus dem Datensatz.
    Anwendbar mit dataset.map(discard_labels).

    """

    return images, None

def  replaceall_labels(images, labels, new_labels):
    """
    Diese Funktion ersetzt die Labels durch neue Labels.
    Diese Funktion wird mit dataset.map(replace_labels) verwendet.

    Parameter:
    new_labels = int-Wert der dem kompleten Datensatz zugewiesen werden soll.

    """
    images = images
    labels = tf.fill(tf.shape(labels), new_labels)

    return images, labels
    

def train_test_split(dataset, train_size=0.8):

    """
    Diese Funktion teilt den Datensatz in Trainings- und Validierungsdatensätze auf.
    
    Parameter:
    dataset = tf.dataset
    train_size = float ; gibt an welcher Anteil des Datensatzes zum Training verwendet wird.

    Rückgabe:
    train_dataset = tf.dataset
    val_dataset = tf.dataset

    """


    total_batches = tf.data.experimental.cardinality(dataset).numpy()
    train_batches = int(train_size * total_batches)

    train_dataset = dataset.take(train_batches)
    val_dataset = dataset.skip(train_batches)

    return train_dataset, val_dataset


def show_shape_images_labels(dataset):

    """
    Diese Funktion zeigt die Dimensionen der Bilddaten und Labels an.
    
    Parameter:
    dataset = tf.dataset
    
    Rückgabe:
    Diese Funktion gibt keine Werte zurück, sondern zeigt die Dimensionen der Bilddaten und Labels an.

    """

    for image, label in dataset.take(1):
        print(f"Shape of images: {image.shape}")
        if label is not None:
            print(f"Shape of labels: {label.shape}")
        else:
            print("Labels are not available (None).")

def print_example_images_labels(dataset):
    for image, label in dataset:
        print(image[0])
        print(label)

def show_images(dataset, num_images=1):

    """
    Diese Funktion zeigt bis zu 16 Bilder aus dem Datensatz in einem Fenster.
    
    Parameter:
    dataset = tf.dataset 
    num_images = 1-16 Bilder, die angezeigt werden sollen
    
    Rückgabe:
    Diese Funktion gibt keine Werte zurück, sondern zeigt die Bilder an.
    
    """

    count = 0
    plt.figure(figsize=(10, 10))

    for image, label in dataset.take(1):
        for i in range(min(num_images, 16)):
            count += 1
            plt.subplot(4, 4, count)
            plt.imshow(image[i])
            plt.axis("off")
    
    plt.show()


def load_saved_dataset(file_path):
    
    """
    Diese Funktion lädt ein gespeichertes Dataset aus einer Datei.

    Parameter:
    file_path = str ; Pfad der Datei, aus der das Dataset geladen werden soll.

    Rückgabe:
    tf.data.Dataset ; geladenes Dataset.

    """

    return tf.data.Dataset.load(file_path)


def augment_data(images, labels, config):

    """
    Diese Funktion augmented das Dataset durch zufällige Bildveränderungen.
    Anpassung über config.py.

    """

    if config.get("flip_left_right", False):
        images = tf.image.random_flip_left_right(images)
    if config.get("flip_up_down", False):
        images = tf.image.random_flip_up_down(images)
    if config.get("brightness", False):
        max_delta = config.get("brightness_max_delta", 0.2)
        images = tf.image.random_brightness(images, max_delta=max_delta)
    return images, labels

def add_gausian_noise(image, stddev=0.1):

    """
    Fügt einem Bild Gaußsches Rauschen hinzu.
    
    Parameter:
        image: Ein TensorFlow-Tensor, der das Bild repräsentiert (Normalisiert [0, 1]).
        stddev: Die Standardabweichung des Rauschens.

    Rückgabe:
        Ein TensorFlow-Tensor mit dem verrauschten Bild.
    """

    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=stddev)
    noisy_image = tf.clip_by_value(image + noise, 0.0, 1.0)

    return noisy_image

def crop_images_to_224(images, labels=None):
    """
    Schneidet Bilder auf die Größe (224, 224) zu, ohne das Seitenverhältnis zu verändern.
    Nimmt an, dass die Eingabebilder mindestens 224x224 groß sind.
    
    Parameter:
    images: TensorFlow-Tensor ; Die Bilder aus dem Dataset.
    labels: TensorFlow-Tensor ; Labels der Bilder (optional).

    Rückgabe:
    images: TensorFlow-Tensor ; Bilder mit Größe (224, 224).
    labels: TensorFlow-Tensor ; Labels der Bilder (falls vorhanden).
    """
    target_size = [224, 224]
    
    # Zentrierter Crop auf (224, 224)
    images = tf.image.resize_with_crop_or_pad(images, target_size[0], target_size[1])
    
    return images, labels
#Testumgebung

if __name__ == "__main__":

    pass