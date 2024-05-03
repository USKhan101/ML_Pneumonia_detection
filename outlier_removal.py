import os
import cv2
import h5py
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

file_path = './processed_data/normalized_rawdata.h5'

## Reading data from file
with h5py.File(file_path, 'r') as file:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]

    x_val = file['val_data'][:]
    y_val = file['val_label'][:]

    x_test = file['test_data'][:]
    y_test = file['test_label'][:]

def count_plot(label):
    l = ["Normal" if i == 0 else "Pneumonia" for i in label]
    sns.set_style('darkgrid')
    ax = sns.countplot(x=l)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

#### Step1: Z-score based outlier detection

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

# Plot bar plot for data count
labels = ['Before Z-score', 'After Z-score']
counts = [len(x_train), len(x_train_Zfiltered)]

plt.figure(figsize=(4, 4))
bars = plt.bar(labels, counts, color=['orange', 'green'])

plt.title('Train data count before and after Z-score outlier removal')
plt.ylabel('Train data count')
plt.xticks(labels, ['Before Z-score', 'After Z-score'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.savefig('./plots/data_count_zcore.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Plot image mean intensity after Z-score outlier removal
plt.figure(figsize=(12, 6))
sns.histplot(means, bins=30, kde=True, label='Original Train Data')
sns.histplot(means[good_indices], bins=30, kde=True, color='red', label='Filtered Train Data')
plt.title('Comparison of Image Mean Intensities Before and After Z-score Outlier Removal')
plt.xlabel('Mean Intensity')
plt.legend()
plt.savefig('./plots/mean_zremoval.pdf', dpi=300, bbox_inches='tight')
plt.show()

####Step:2 IQR outlier detection

# Calculate mean intensity for each image
means1 = np.array([np.mean(img) for img in x_train_Zfiltered])

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = np.percentile(means1, 25)
Q3 = np.percentile(means1, 75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
good_indices = (means1 > lower_bound) & (means1 < upper_bound)

# Apply the filter to your data
x_train_IQRfiltered = x_train_Zfiltered[good_indices]
y_train_IQRfiltered = y_train_Zfiltered[good_indices]

print(f"Original data count: {len(x_train_Zfiltered)}")
print(f"Filtered data count: {len(x_train_IQRfiltered)}")

# Plot bar plot for data count
labels = ['Before IQR', 'After IQR']
counts = [len(x_train_Zfiltered), len(x_train_IQRfiltered)]

plt.figure(figsize=(4, 4))
bars = plt.bar(labels, counts, color=['orange', 'green'])

plt.title('Train data count before and after IQR outlier removal')
plt.ylabel('Train data count')
plt.xticks(labels, ['Before IQR', 'After IQR'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.savefig('./plots/data_count_iqr.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Plot image mean intensity after IQR outlier removal
plt.figure(figsize=(12, 6))
sns.histplot(means1, bins=30, kde=True, label='Original Train Data')
sns.histplot(means1[good_indices], bins=30, kde=True, color='red', label='Filtered Train Data')
plt.title('Comparison of Image Mean Intensities Before and After IQR Outlier Removal')
plt.xlabel('Mean Intensity')
plt.legend()
plt.savefig('./plots/mean_IQRremoval.pdf', dpi=300, bbox_inches='tight')
plt.show()

print (x_train_IQRfiltered.dtype)

# Save data after outliers removal
out_path = './processed_data/outlier_removed_traindata.h5'

with h5py.File(out_path, 'w') as file:
    file.create_dataset('train_data', data=x_train_IQRfiltered)
    file.create_dataset('train_label', data=y_train_IQRfiltered)

def plot_aggregated_image_data(data, name):
    # Calculate the mean pixel value for each image
    image_means = np.mean(data.reshape(data.shape[0], -1), axis=1)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=image_means)
    plt.title('Distribution of Mean Pixel Intensities Across All Images')
    plt.xlabel('Mean Pixel Intensity')
    plt.savefig(f'./plots/aggr_image_{name}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Visualizing the distribution of mean pixel intensities
plot_aggregated_image_data(x_train, 'train_data')
plot_aggregated_image_data(x_train_IQRfiltered, 'filtered_data')


def plot_category_distribution(data, labels, name):
    # Calculate the mean pixel values for each image
    image_means = np.mean(data.reshape(data.shape[0], -1), axis=1)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=image_means, y=labels.astype(str))
    plt.title('Mean Pixel Intensity Distribution by Category')
    plt.xlabel('Mean Pixel Intensity')
    plt.ylabel('Category')
    plt.savefig(f'./plots/category_image_{name}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Visualizing the distribution of mean pixel intensities for each category
plot_category_distribution(x_train, y_train, 'train_data')
plot_category_distribution(x_train_IQRfiltered, y_train_IQRfiltered, 'filtered_data')

# Count plot for training data after outlier removal
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
count_plot(y_train_IQRfiltered)
plt.title('Filtered Train Data')

#pie plot to show the ratio of train, val and test dataset
plt.subplot(1, 2, 2)
plt.pie([len(y_train_IQRfiltered), len(y_val), len(y_test)], labels=['train', 'validation', 'test'], autopct='%1.1f%%', colors=['orange', 'red', 'lightblue'], explode=(0.05, 0, 0))
plt.title('Data Ratio')
plt.tight_layout()

plt.savefig('./plots/outlier_data_count.pdf', dpi=300, bbox_inches='tight')
plt.show()

