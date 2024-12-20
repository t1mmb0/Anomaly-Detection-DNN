import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, applications
from pc_structure import resnet50_feature_extractor
import matplotlib.pyplot as plt
import numpy as np


"""Model Structure"""


@keras.utils.register_keras_serializable()
class ResNetEncoder(keras.Model):
    def __init__(self, shape, **kwargs):
        super(ResNetEncoder, self).__init__(**kwargs)        
        self.shape = shape

        self.resnet = resnet50_feature_extractor(input_shape=self.shape, output_layer="conv3_block4_out")
        

    def call(self, inputs):

        x = self.resnet(inputs)

        return x

@keras.utils.register_keras_serializable()
class Encoder(keras.Model):
    def __init__(self, shape,):
        super(Encoder, self).__init__()
        self.shape = shape

        # Convolutional Layers
        self.conv_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', padding="same")
        self.pool_1 = layers.MaxPooling2D((2,2),padding="same")
        self.conv_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation='relu', padding="same")
        self.pool_2 = layers.MaxPooling2D((2,2),padding="same")
        self.conv_3 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu', padding="same")
        self.H = layers.MaxPooling2D((2,2),padding="same")
        

    def call(self, input):
        x = self.conv_1(input)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.conv_3(x)

        x = self.H(x)

        return x

@keras.utils.register_keras_serializable()
class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv_1 = layers.Conv2D(256, (3, 3), strides=1, activation = "relu", padding = "same")
        self.up_1 = layers.Conv2DTranspose(256, (3, 3), strides=(2,2), padding="same")
        self.conv_2 = layers.Conv2D(256, (3, 3), strides=1, activation = "relu", padding = "same")
        self.up_2 = layers.Conv2DTranspose(256, (3, 3), strides=(2,2), padding="same")
        self.conv_3 = layers.Conv2D(128, (3, 3), strides=1, activation = "relu", padding = "same")
        self.up_3 = layers.Conv2DTranspose(128, (3, 3), strides=(2,2), padding="same")
        self.conv_4 = layers.Conv2D(64, (3, 3), strides=1, activation = "relu", padding = "same")
        self.R = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")

    def call(self, inputs):    
        x = self.conv_1(inputs)
        x = self.up_1(x)
        x = self.conv_2(x)
        x = self.up_2(x)
        x = self.conv_3(x)
        x = self.up_3(x)
        x = self.conv_4(x)
        x = self.R(x)
        return x
    
@keras.utils.register_keras_serializable()
class Decoder_Upsampling(tf.keras.Model):
    def __init__(self):
        super(Decoder_Upsampling, self).__init__()

        self.conv_1 = layers.Conv2D(64, (3, 3), strides=1, activation = "relu", padding = "same")
        self.up_1 = layers.UpSampling2D((2,2))
        self.conv_2 = layers.Conv2D(64, (3, 3), strides=1, activation = "relu", padding = "same")
        self.up_2 = layers.UpSampling2D((2,2))
        self.conv_3 = layers.Conv2D(128, (3, 3), strides=1, activation = "relu", padding = "same")
        self.up_3 = layers.UpSampling2D((2,2))
        self.conv_4 = layers.Conv2D(64, (3, 3), strides=1, activation = "relu", padding = "same")
        self.R = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")

    def call(self, inputs):    
        x = self.conv_1(inputs)
        x = self.up_1(x)
        x = self.conv_2(x)
        x = self.up_2(x)
        x = self.conv_3(x)
        x = self.up_3(x)
        x = self.conv_4(x)
        x = self.R(x)
        return x
    
@keras.utils.register_keras_serializable()
class Autoencoder(keras.Model):
    def __init__(self, resnet, upsampling, shape, **kwargs):
        super(Autoencoder, self).__init__( **kwargs)

        self.shape = shape
        self.resnet = resnet
        self.upsampling = upsampling

        if self.resnet:
            self.encoder = ResNetEncoder( shape)
        else:
            self.encoder = Encoder(shape)

        if self.upsampling:
            self.decoder = Decoder_Upsampling()
        else:
            self.decoder = Decoder()

    def call(self, inputs):

      encoded = self.encoder(inputs)
      decoded = self.decoder(encoded)

      return decoded

    def get_config(self):
        config = super(Autoencoder, self).get_config()
        config.update({
            'shape': self.shape,
            'resnet':self.resnet,
            'upsampling': self.upsampling
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.utils.register_keras_serializable()
def ssim_loss(y_true, y_pred):
    """
    SSIM-basierte Loss-Funktion.
    Wertebereich der Bilder muss zwischen [0, 1] sein.
    """
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def calculate_patch_differences(original, reconstructed, patch_size=32):

    h, w, _ = original.shape
    patch_scores = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            original_patch = original[i:i + patch_size, j:j + patch_size]
            reconstructed_patch = reconstructed[i:i + patch_size, j:j + patch_size]
            patch_diff = np.mean(np.abs(original_patch - reconstructed_patch))
            patch_scores = np.append(patch_scores, patch_diff)

    return patch_scores

class VisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, interval=10):
        super(VisualizationCallback, self).__init__()
        self.val_data = val_data  # Die Validierungsdaten für Visualisierungen
        self.interval = interval   # Wie oft (alle n Epochen) visualisieren

    def on_epoch_end(self, epoch, logs=None):
        # Nur alle 10 Epochen visualisieren
        if (epoch + 1) % self.interval == 0:
            # Beispielbilder aus dem Validierungsdatensatz holen (nur die Eingabebilder)
            example_images, _ = next(iter(self.val_data.take(5)))  # Beispiel aus dem Dataset holen
            example_images = example_images[:10]  # Die ersten 10 Bilder verwenden

            # Rekonstruktion der Eingabebilder
            reconstructed_images = self.model.predict(example_images)  # Vorhersage für Rekonstruktionen

            # Visualisierung der Eingabebilder und Rekonstruktionen
            self.visualize_images(example_images, reconstructed_images)

    def visualize_images(self, original, reconstructed):
        plt.figure(figsize=(12, 6))

        for i in range(len(original)):
            # Originalbilder
            plt.subplot(2, len(original), i + 1)
            plt.imshow(original[i])
            plt.title("Original")
            plt.axis('off')

            # Rekonstruktionsbilder
            plt.subplot(2, len(original), len(original) + i + 1)
            plt.imshow(reconstructed[i])
            plt.title("Rekonstruiert")
            plt.axis('off')

        plt.show()




if __name__ == "__main__":

    pass