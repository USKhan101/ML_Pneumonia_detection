import cv2
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

file_path = './processed_data/augmented_traindata.h5'
test_path = './processed_data/normalized_rawdata.h5'

## Reading data from file
with h5py.File(file_path, 'r') as file:
    x_train = file['augmented_train_data'][:]
    y_train = file['augmented_train_label'][:]

with h5py.File(test_path, 'r') as file1:
    x_test = file1['test_data'][:]
    y_test = file1['test_label'][:]

x_train = x_train.astype(np.uint8)
x_test = x_test.astype(np.uint8)
################  Train Data Feature Extraction  #######################

def feature_extraction(image):


# Feature1: Variance of each image
variances = np.var(x_train, axis=(1, 2)) 
variances = variances.reshape ((-1, 1))

# Feature2: Mean of each image
means = np.mean(x_train, axis=(1, 2))
means = means.reshape ((-1, 1))

# Feature3: Standard Deviation of each image
stds = np.std(x_train, axis=(1, 2))
stds = stds.reshape ((-1, 1))

# Feature4: Skewness of each image 
skewness = np.array([stats.skew(image.ravel()) for image in x_train])
skewness = skewness.reshape ((-1, 1))

# Feature5: Kurtosis of each image 
kurtosis = np.array([stats.kurtosis(image.ravel()) for image in x_train])
kurtosis = kurtosis.reshape ((-1, 1))

# Feature6: Entropy of each image 
def calculate_entropy(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    histogram_normalized = histogram / histogram.sum()
    histogram_normalized = histogram_normalized[histogram_normalized > 0] # Avoid zero probabilities
    return stats.entropy(histogram_normalized, base=2)

# Calculating entropy for each image in the dataset
entropies = np.array([calculate_entropy(image) for image in x_train])
entropies = entropies.reshape ((-1, 1))

# Feature7: Canny Edge detection
canny_edge = np.array([cv2.Canny(image, 140, 200) for image in x_train])
canny_edge = canny_edge.reshape (canny_edge.shape[0], -1)

# Feature8: Sobel X
sobel_x = np.array([cv2.Sobel(image,cv2.CV_8UC1,1,0,ksize=5) for image in x_train])
sobel_x = sobel_x.reshape(sobel_x.shape[0], -1)

# Feature9: Sobel Y
sobel_y = np.array([cv2.Sobel(image,cv2.CV_8UC1,0,1,ksize=5) for image in x_train])
sobel_y = sobel_y.reshape(sobel_y.shape[0], -1)

# Feature10: Binary threshold
bin_thresh = np.array([cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)[1] for image in x_train])
bin_thresh = bin_thresh.reshape(bin_thresh.shape[0], -1)

## Combined Features
train_features = np.hstack((variances, means, stds, skewness, kurtosis, entropies, canny_edge, sobel_x, sobel_y, bin_thresh))

print(train_features.shape)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(train_features)

#################  Test Data Feature Extraction  #######################

# Feature1: Variance of each image
test_variances = np.var(x_test, axis=(1, 2)) 
test_variances = test_variances.reshape ((-1, 1))

# Feature2: Mean of each image
test_means = np.mean(x_test, axis=(1, 2))
test_means = test_means.reshape ((-1, 1))

# Feature3: Standard Deviation of each image
test_stds = np.std(x_test, axis=(1, 2))
test_stds = test_stds.reshape ((-1, 1))

# Feature4: Skewness of each image 
test_skewness = np.array([stats.skew(image.ravel()) for image in x_test])
test_skewness = test_skewness.reshape ((-1, 1))

# Feature5: Kurtosis of each image 
test_kurtosis = np.array([stats.kurtosis(image.ravel()) for image in x_test])
test_kurtosis = test_kurtosis.reshape ((-1, 1))

# Feature6: Entropy of each image 
test_entropies = np.array([calculate_entropy(image) for image in x_test])
test_entropies = test_entropies.reshape ((-1, 1))

# Feature7: Canny Edge detection
test_canny_edge = np.array([cv2.Canny(image, 140, 200) for image in x_test])
test_canny_edge = test_canny_edge.reshape (test_canny_edge.shape[0], -1)

# Feature8: Sobel X
test_sobel_x = np.array([cv2.Sobel(image,cv2.CV_8UC1,1,0,ksize=5) for image in x_test])
test_sobel_x = test_sobel_x.reshape(test_sobel_x.shape[0], -1)

# Feature9: Sobel Y
test_sobel_y = np.array([cv2.Sobel(image,cv2.CV_8UC1,0,1,ksize=5) for image in x_test])
test_sobel_y = test_sobel_y.reshape(test_sobel_y.shape[0], -1)

# Feature10: Binary threshold
test_bin_thresh = np.array([cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)[1] for image in x_test])
test_bin_thresh = test_bin_thresh.reshape(test_bin_thresh.shape[0], -1)

## Combined Features
test_features = np.hstack((test_variances, test_means, test_stds, test_skewness, test_kurtosis, test_entropies, test_canny_edge, test_sobel_x, test_sobel_y, test_bin_thresh))

print(test_features.shape)

scaled_test_features = scaler.transform(test_features)

out_path = './processed_data/feature_data.h5'

## Reading data from file
with h5py.File(out_path, 'w') as file:
    file.create_dataset('train_feature', data=scaled_features)
    file.create_dataset('train_label', data=y_train)

    file.create_dataset('test_feature', data=scaled_test_features)
    file.create_dataset('test_label', data=y_test)

## Plot the histogram of pixel values for the first image
#plt.hist(x_train[0].ravel(), bins=50, color='gray', alpha=0.75)
#plt.title('Pixel Value Distribution')
#plt.xlabel('Pixel Intensity')
#plt.ylabel('Frequency')
#plt.show()
#
## Reshape to feed into random forest
#x_train_augmented = x_train_augmented.reshape(x_train_augmented.shape[0], -1)
#x_test = x_test.reshape(x_test.shape[0], -1)
#
#print (x_train_augmented.shape)
#print (x_test.shape)
