import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import prepare_data
from skimage import transform


dir = "./images/capsule_images_train"
data_train = prepare_data.load_data_from_directory(dir)
data = data_train.map(lambda x,y: (prepare_data.normalize(x),y))

images, labels = next(iter(data.take(1)))


image = images[0]



grayscale_image = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

plt.subplot(2, 2, 1)
plt.title("grayscale")
plt.axis("off")
plt.imshow(grayscale_image, cmap="gray")


vertical_filter = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])
horizontal_filter = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])

blurred_filter = np.array([
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,8,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1]
])
blurred_filter = blurred_filter / 32.0
vertical_edges = convolve2d(grayscale_image, vertical_filter, mode="same", boundary="symm")
horizontal_edges = convolve2d(grayscale_image, horizontal_filter,mode="same", boundary="symm")
blurred_image = convolve2d(grayscale_image, blurred_filter, mode="same", boundary="symm")


plt.subplot(2, 2, 2)
plt.imshow(vertical_edges, cmap='gray')
plt.title("vertical_convolution")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(horizontal_edges, cmap='gray')
plt.title("horizontal_convolution")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.title("blurred")
plt.axis("off")
plt.imshow(blurred_image, cmap='gray')



plt.subplots_adjust(wspace=0, hspace=0)
plt.show()