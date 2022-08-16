# !wget https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz
# Use the Tiny-NeRF dataset (a subset of the original Blender dataset)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

data = np.load("tiny_nerf_data.npz")
images = data["images"]
poses = data["poses"]
focal = float(data["focal"])

_, image_ht, image_wid, _ = images.shape

# use first 100 images&poses as training data
train_images, train_poses = images[:100], poses[:100]

# use a single image&pose pair for validation
val_images, val_pose = images[101], poses[101]

# visualize the training data
fig = plt.figure(figsize=(16, 16))
grid = ImageGrid(fig, 111, nrows_ncols=(4,4), axes_pad=0.1)

random_images = images[np.random.choice(np.arange(images.shape[0]), 16)]
for ax, image in zip(grid, random_images):
    ax.imshow(image)
plt.title("Sample images from Tiny-NeRF dataset")
plt.show()