import numpy as np
from prepare_data import load_saved_dataset
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

data = load_saved_dataset("capsule_train_normalized.keras")

images, labels = next(iter(data.take(1)))


image = images[0]



grayscale_image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

kernel = np.ones((4, 4)) / 16

convolved_image = convolve2d(grayscale_image, kernel, mode='valid')

stride = 4
downsampled_image = convolved_image[::stride, ::stride]
print(downsampled_image.shape)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Originales Bild")
plt.imshow(grayscale_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Verkleinertes Bild")
plt.imshow(downsampled_image, cmap='gray')
plt.axis('off')

plt.show()


kernel2 = np.ones((3, 3)) / 9

valid = convolve2d(downsampled_image, kernel2, mode="valid")
print(valid.shape)
plt.subplot(1, 3, 1)
plt.title("Valid Padding")
plt.imshow(valid, cmap='gray')
plt.axis('off')

same = convolve2d(downsampled_image, kernel2, mode="same")
print(same.shape)
plt.subplot(1, 3, 2)
plt.title("Same Padding")
plt.imshow(same, cmap='gray')
plt.axis('off')

full = convolve2d(downsampled_image, kernel2, mode="full")
print(full.shape)
plt.subplot(1, 3, 3)
plt.title("Full Padding")
plt.imshow(full, cmap='gray')
plt.axis('off')
plt.show()