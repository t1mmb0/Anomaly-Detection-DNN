import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from prepare_data import load_data_from_directory, normalize, add_gausian_noise
import convAE_structure

import configparser
config = configparser.ConfigParser()
import ast

#Dataset Preparation
config.read("config.ini")


shape= config.get("CONV_AE_PARAMETERS", "shape")
shape = ast.literal_eval(shape) 
resnet = config.getboolean("CONV_AE_PARAMETERS", "resnet")
denoising =config.getboolean("CONV_AE_PARAMETERS", "denoising")
training_epochs = config.getint("CONV_AE_PARAMETERS", "training_epochs")
dir = config.get("PATHS", "train_dir")


#Datensatz Loader:
data = load_data_from_directory(dir,image_size=(shape[0],shape[1]))
data = data.map(lambda x,y: (normalize(x),y))

#Datensatz preprocessing
total_batches = tf.data.experimental.cardinality(data).numpy()
train_batches = int(0.8 * total_batches)
train_dataset = data.take(train_batches)
val_dataset = data.skip(train_batches)

train_dataset = train_dataset.map(lambda x,y: (x, x))
val_dataset = val_dataset.map(lambda x,y: (x,x))

if denoising:
    train_dataset = train_dataset.map(lambda x,y: (add_gausian_noise(x, 0.05), y))
    val_dataset = val_dataset.map(lambda x,y: (add_gausian_noise(x, 0.05), y))


train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

#Erstellung einer Modellinstanz und Compiling


autoencoder = convAE_structure.Autoencoder(resnet, shape)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss=convAE_structure.ssim_loss)



#Callback Definitionen, um Overfitting zu unterbinden und für die Visualizierung während des Trainings.

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Überwache den Validation Loss
    patience=10,          # Warte 3 Epochen auf eine Verbesserung
    restore_best_weights=True  # Stelle die besten Modellgewichte wieder her
)

visualization_callback = convAE_structure.VisualizationCallback(val_dataset, interval = 20)

#Training des Modells / Optimierung der Parameter

history = autoencoder.fit(train_dataset, epochs=training_epochs, validation_data=val_dataset, callbacks=[early_stopping])

#Visualisierung mit Beispielbildern

example_images = next(iter(train_dataset.take(5)))
example_images = example_images[0]
reconstructed_images = autoencoder.predict(example_images)


plt.figure(figsize=(15, 5))

# Originalbilder anzeigen
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(example_images[i].numpy())
    plt.title("Original")
    plt.axis('off')

# Rekonstruierte Bilder anzeigen
for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed_images[i])
    plt.title("Rekonstruiert")
    plt.axis('off')

plt.show()
output_dir = config.get("CONV_AE_PARAMETERS", "output_dir" )
keras.models.save_model(autoencoder, output_dir)