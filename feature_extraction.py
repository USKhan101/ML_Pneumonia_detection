import cv2
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#file_path = './processed_data/augmented_traindata.h5'
data_path = './processed_data/normalized_rawdata.h5'

## Reading data from file
#with h5py.File(file_path, 'r') as file:
#    x_train = file['augmented_train_data'][:]
#    y_train = file['augmented_train_label'][:]

with h5py.File(data_path, 'r') as file1:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]

    x_test = file1['test_data'][:]
    y_test = file1['test_label'][:]

x_train = x_train.astype(np.uint8)
x_test = x_test.astype(np.uint8)

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

################  Train Data Feature Extraction  #######################

def calculate_entropy(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    histogram_normalized = histogram / histogram.sum()
    histogram_normalized = histogram_normalized[histogram_normalized > 0] # Avoid zero probabilities
    return stats.entropy(histogram_normalized, base=2)

def feature_extract(image):
    # Feature1: Variance of each image
    var = np.var(image)

    # Feature2: Mean of each image
    mean = np.mean(image)

    # Feature3: Standard Deviation of each image
    std = np.std(image)

    # Feature4: Skewness of each image 
    skewness = stats.skew(image.ravel())

    # Feature5: Kurtosis of each image 
    kurtosis = stats.kurtosis(image.ravel())

    # Feature6: Entropy of each image 
    entropies = calculate_entropy(image)

    # Feature7: Canny Edge detection
    canny_edge = cv2.Canny(image, 140, 200) 
    canny_edge_count = np.sum(canny_edge > 0)

    # Feature8: Otsu binary threshold
    bin_thresh = np.array([cv2.threshold(image, 120, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] for image in x_train])
    bin_thresh = bin_thresh.reshape(bin_thresh.shape[0], -1)

    ## Combined Features
    train_features = np.hstack((variances, means, stds, skewness, kurtosis, entropies, canny_edge, sobel_x, sobel_y, bin_thresh))
    
    print(train_features.shape)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(train_features)

## Train, val and test data


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
