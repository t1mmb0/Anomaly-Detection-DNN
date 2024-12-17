from convAE_structure import *
from prepare_data import load_data_from_directory, crop_images_to_224, normalize

data = load_data_from_directory("./Images/capsule_images_test",image_size=(256,256))
data = data.map(crop_images_to_224)
dataset = data.map(lambda x,y: (normalize(x),y))

custom_objects = {
    'Encoder': Encoder,
    'Decoder': Decoder,
    'Autoencoder': Autoencoder,
    'ResNetEncoder': ResNetEncoder,
    'loss': ssim_loss
}

autoencoder = keras.models.load_model("./trained_models/ae_gausian_0.1_capsule.keras", custom_objects=custom_objects)


predictions = autoencoder.predict(dataset)
autoencoder.summary()
print(predictions.shape)
images = predictions[0]
plt.figure(figsize=(10,5))
for i in range(16):
    plt.subplot(2,8,i+1)
    plt.imshow(images[:,:,i], cmap='gray')
    plt.axis('off')
plt.show()