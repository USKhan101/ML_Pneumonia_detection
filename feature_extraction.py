import cv2
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data_path = './processed_data/normalized_rawdata.h5'

with h5py.File(data_path, 'r') as file:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]

    x_val = file['val_data'][:]
    y_val = file['val_label'][:]

    x_test = file['test_data'][:]
    y_test = file['test_label'][:]

print (x_train.dtype)
print (y_train.dtype)
print (x_val.dtype)
print (y_val.dtype)
print (x_test.dtype)
print (y_test.dtype)

#x_train_flat = x_train.reshape(x_train.shape[0], -1)
#x_test_flat = x_test.reshape(x_test.shape[0], -1)

################  Train Data Feature Extraction  #######################

def calculate_entropy(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    histogram_normalized = histogram / histogram.sum()
    histogram_normalized = histogram_normalized[histogram_normalized > 0] # Avoid zero probabilities
    return stats.entropy(histogram_normalized, base=2)

def feature_extract(image):
    image = np.uint8(image*255)
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
    bin_thresh = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    bin_thresh = np.sum(bin_thresh > 0)

    ## Combined Features
    features = np.array([var, mean, std, skewness, kurtosis, entropies, canny_edge, bin_thresh], dtype=object).astype(np.float32)

    return features
    
# Feature data for x_train, x_val, x_test

def data_preprocess (data):
    processed_data = []

    for img in data:
        # Extract features
        features = feature_extract(img)
        
        # Flatten the image data
        flat_image_data = img.flatten()
        
        # Concatenate features with image data
        full_features = np.concatenate((flat_image_data, features))
        
        # Append to the new dataset
        processed_data.append(full_features)

    return processed_data

## Train, val and test data
train_feature = np.array (data_preprocess(x_train)).astype(np.float32)
val_feature = np.array (data_preprocess(x_val)).astype(np.float32)  
test_feature = np.array (data_preprocess(x_test)).astype(np.float32)

# Save the feature data
out_path = './processed_data/feature_data.h5'

with h5py.File(out_path, 'w') as file:
    file.create_dataset('train_feature', data=train_feature)
    file.create_dataset('train_label', data=y_train)

    file.create_dataset('val_feature', data=val_feature)
    file.create_dataset('val_label', data=y_val)

    file.create_dataset('test_feature', data=test_feature)
    file.create_dataset('test_label', data=y_test)

