import os
import cv2
import h5py
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

file_path = './processed_data/normalized_rawdata.h5'
augm_path = './processed_data/augmented_traindata.h5'

## Reading data from file
#with h5py.File(file_path, 'r') as file:
#    x_train = file['train_data'][:]
#    y_train = file['train_label'][:]

with h5py.File(augm_path, 'r') as file:
    x_train = file['augmented_train_data'][:]
    y_train = file['augmented_train_label'][:]

#### STep1: Z-score based outlier detection

# Calculate mean intensity for each image
means = np.array([np.mean(img) for img in x_train])

# Calculate Z-scores
z_scores = np.abs(stats.zscore(means))

# Filter out outliers
threshold = 3  # Common choice for outlier cutoff
good_indices = z_scores < threshold

# Apply the filter to your data
x_train_Zfiltered = x_train[good_indices]
y_train_Zfiltered = y_train[good_indices]

print(f"Original data count: {len(x_train)}")
print(f"Filtered data count: {len(x_train_Zfiltered)}")

plt.figure(figsize=(12, 6))
sns.histplot(means, bins=30, kde=True, label='Original Train Data')
sns.histplot(means[good_indices], bins=30, kde=True, color='red', label='Filtered Train Data')
plt.title('Comparison of Image Mean Intensities Before and After Z-score Outlier Removal')
plt.xlabel('Mean Intensity')
plt.legend()
plt.show()

####Step:2 IQR outlier detection

# Calculate mean intensity for each image
means1 = np.array([np.mean(img) for img in x_train_Zfiltered])

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = np.percentile(means1, 25)
Q3 = np.percentile(means1, 75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 2.5 * IQR
upper_bound = Q3 + 2.5 * IQR

# Filter out outliers
good_indices = (means1 > lower_bound) & (means1 < upper_bound)

# Apply the filter to your data
x_train_IQRfiltered = x_train_Zfiltered[good_indices]
y_train_IQRfiltered = y_train_Zfiltered[good_indices]

print(f"Original data count: {len(x_train_Zfiltered)}")
print(f"Filtered data count: {len(x_train_IQRfiltered)}")

# Save data after outliers removal
out_path = './processed_data/traindata_after_outlier.h5'

with h5py.File(out_path, 'w') as file:
    file.create_dataset('train_data', data=x_train_IQRfiltered)
    file.create_dataset('train_label', data=y_train_IQRfiltered)

plt.figure(figsize=(12, 6))
sns.histplot(means1, bins=30, kde=True, label='Original Train Data')
sns.histplot(means1[good_indices], bins=30, kde=True, color='red', label='Filtered Train Data')
plt.title('Comparison of Image Mean Intensities Before and After IQR Outlier Removal')
plt.xlabel('Mean Intensity')
plt.legend()
plt.show()

def plot_aggregated_image_data(data):
    # Calculate the mean or median pixel value for each image
    image_means = np.mean(data.reshape(data.shape[0], -1), axis=1)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=image_means)
    plt.title('Distribution of Mean Pixel Intensities Across All Images')
    plt.xlabel('Mean Pixel Intensity')
    plt.show()

# Visualizing the distribution of mean pixel intensities
plot_aggregated_image_data(x_train)
plot_aggregated_image_data(x_train_IQRfiltered)


def plot_category_distribution(data, labels):
    # Calculate the mean pixel values for each image
    image_means = np.mean(data.reshape(data.shape[0], -1), axis=1)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=image_means, y=labels.astype(str))
    plt.title('Mean Pixel Intensity Distribution by Category')
    plt.xlabel('Mean Pixel Intensity')
    plt.ylabel('Category')
    plt.show()

# Assuming `y_train` are your labels corresponding to x_train
plot_category_distribution(x_train, y_train)
plot_category_distribution(x_train_IQRfiltered, y_train_IQRfiltered)
