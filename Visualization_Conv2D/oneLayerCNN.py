import numpy as np
from prepare_data import load_data_from_directory
import matplotlib.pyplot as plt

from ResNet_model import resnet50_feature_extractor


data = load_data_from_directory("./Images/capsule_images_train")

images, labels = next(iter(data.take(1)))

image = images[0]


feature_extractor = resnet50_feature_extractor(output_layer="conv1_conv")

image_batch = np.expand_dims(image, axis=0)

feature_maps = feature_extractor.predict(image_batch)
feature_map = feature_maps[0]

fig, axes = plt.subplots(1, 4, figsize=(12, 12))
axes = axes.ravel() 

for i in range(4):
    ax = axes[i]
    ax.imshow(feature_map[:, :, i], cmap='gray')
    ax.axis('off')
    ax.set_title(f'Feature Map {i + 1}')

plt.show()