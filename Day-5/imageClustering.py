import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# Load the image
image_path = 'Day-3\image_2.jpg'  
image = io.imread(image_path)

# Normalize the image data to the range [0, 1] if needed
image = image / 255.0 if image.max() > 1 else image

# Reshape the image to a 2D array (pixels as rows, RGB values as columns)
pixels = image.reshape(-1, 3)

# Number of clusters (colors)
k = 3

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, random_state=200)
cluster_labels = kmeans.fit_predict(pixels)

# Get the clustered colors (centroids)
clustered_colors = kmeans.cluster_centers_

# Reshape the clustered labels to the original image shape
segmented_image = clustered_colors[cluster_labels].reshape(image.shape)

# Visualize the original and clustered images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title('Clustered Image')
ax[1].axis('off')

plt.tight_layout()
plt.show()
