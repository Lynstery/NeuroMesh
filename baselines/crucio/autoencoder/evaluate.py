import os

import matplotlib.pyplot as plt
import torch

from crucio.autoencoder.dataset import IMAGE_DIR
from crucio.autoencoder.network2d import get_networks, print_autoencoder_info
from crucio.autoencoder.util import (CUDA_ENABLED, IMAGE_EXT, YUV_ENABLED,
                                     convert_image_to_tensor,
                                     convert_tensor_to_image,
                                     load_compressed_data, 
                                     save_compressed_data)
from crucio.autoencoder.gpu import print_device_info

print_device_info()
print_autoencoder_info()

# Load trained encoder and decoder
encoder, _, decoder, _ = get_networks('eval', True)

# Encoder compresses test image
name = IMAGE_DIR+'/000000013291'
img_path = name+IMAGE_EXT
img_size = os.path.getsize(img_path)/1024
print(f"Size of original image {img_path} is {img_size:.4f} KB")
img_tensor = convert_image_to_tensor(
    img_path, YUV_ENABLED, CUDA_ENABLED).unsqueeze(0)
with torch.no_grad():
    compressed_data = encoder(img_tensor)

# Save compressed data
data_path = save_compressed_data(img_path, compressed_data)
data_size = os.path.getsize(data_path)/1024
print(f"Size of compressed data {data_path} is {data_size:.4f} KB")

# Load compressed data
# data_path = name+'.pkl'
compressed_data = load_compressed_data(data_path)

# Extract compressed data to image
with torch.no_grad():
    decoded_tensor = decoder(compressed_data)
reconstructed_img = convert_tensor_to_image(decoded_tensor[0], False)

# Save image
reconstructed_path = name+'_rec'+IMAGE_EXT
reconstructed_img.save(reconstructed_path)
reconstructed_size = os.path.getsize(reconstructed_path)/1024
print(
    f"Size of reconstructed image {reconstructed_path} is {reconstructed_size:.4f} KB")

# Load image
input_img = plt.imread(img_path)
reconstructed_img = plt.imread(reconstructed_path)

# Show image
plt.subplot(1, 2, 1)
plt.imshow(input_img)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_img)
plt.title('Reconstructed Image')
plt.show()
