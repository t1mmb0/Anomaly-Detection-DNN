import numpy as np
import prepare_data
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models

# Load the dataset
dir = "./images/transistor_images_train"
data_train = prepare_data.load_data_from_directory(dir)
data_train = data_train.map(lambda x,y: (prepare_data.normalize(x),y))


images = []

# Take one batch of images
for image, label in data_train.take(1):
    images.extend(image)

images = np.array(images)

# Create a simple CNN model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3), padding="same")
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check the shape of the images
print("Input image shape:", images.shape)

# Predict the activations (feature maps)
activations = model.predict(images)

print("Activations shape:", activations.shape)

# Get the number of filters
num_filters = activations.shape[-1]

# Take the activations of the first image in the batch
single_activation = activations[0]

# Create a grid of subplots
fig, axes = plt.subplots(4, 4, figsize=(15, 8))
axes = axes.flatten()

# Plot all feature maps
for i in range(num_filters):
    ax = axes[i]
    ax.imshow(single_activation[:, :, i], cmap='viridis')  # Visualisiere Filter i
    ax.axis('off')

# Adjust layout
plt.tight_layout()
plt.show()