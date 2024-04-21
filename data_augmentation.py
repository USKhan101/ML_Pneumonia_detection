import cv2
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, mean_squared_error
from tensorflow.keras.preprocessing.image import ImageDataGenerator

file_path = './processed_data/normalized_rawdata.h5'

## Reading data from file
with h5py.File(file_path, 'r') as file:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]

# Configure the ImageDataGenerator to augment images
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augment only "NORMAL" labeled images because of it's lower % in train data
# Separate the NORMAL images
x_normal = x_train[y_train == 0]  # 0 is the class_num for 'NORMAL'
y_normal = y_train[y_train == 0]  # 0 is the class_num for 'NORMAL'

# Reshape images to add a channel dimension (needed for ImageDataGenerator)
x_normal = x_normal[..., np.newaxis]  # Shape (N, 256, 256, 1) if grayscale

# Number of augmented images to generate per original image
num_augments_per_image = 2

augmented_images = []
augmented_labels = []

for x in x_normal:
    x_expanded = np.expand_dims(x, 0)  # Shape (1, 256, 256, 1)
    temp_gen = data_gen.flow(x_expanded, batch_size=1)
    for _ in range(num_augments_per_image):
        augmented_image = next(temp_gen)
        augmented_images.append(augmented_image[0])
        augmented_labels.append(0)  # Label for 'NORMAL'

# Convert augmented data to numpy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Separate the NORMAL images
x_pneumonia = np.array(x_train[y_train == 1])[..., np.newaxis]  # 1 is the class_num for 'PNEUMONIA'
y_pneumonia = np.array(y_train[y_train == 1])  # 1 is the class_num for 'PNEUMONIA'

# Combine all training data
x_train_augmented = np.concatenate((x_pneumonia, x_normal, augmented_images))
y_train_augmented = np.concatenate((y_pneumonia, y_normal, augmented_labels))

# Randomize the augmented training data
indices = np.arange(x_train_augmented.shape[0])
np.random.shuffle(indices)

x_train_augmented = x_train_augmented[indices]
y_train_augmented = y_train_augmented[indices]

#x_train_augmented = np.squeeze(x_train_augmented, axis=-1)

augm_datapath = './processed_data/augmented_traindata.h5'

with h5py.File(augm_datapath, 'w') as file:
    file.create_dataset('augmented_train_data',  data=x_train_augmented)
    file.create_dataset('augmented_train_label', data=y_train_augmented)

print (x_train_augmented.shape)
print (x_train_augmented[0].shape)
print (y_train_augmented.shape)
print (y_train_augmented[0].shape)

#################  Train Data Feature Extraction  #######################
#
## Feature1: Variance of each image
#variances = np.var(x_train, axis=(1, 2)) 
#variances = variances.reshape ((-1, 1))
#
## Feature2: Mean of each image
#means = np.mean(x_train, axis=(1, 2))
#means = means.reshape ((-1, 1))
#
## Feature3: Standard Deviation of each image
#stds = np.std(x_train, axis=(1, 2))
#stds = stds.reshape ((-1, 1))
#
## Feature4: Skewness of each image 
#skewness = np.array([stats.skew(image.ravel()) for image in x_train])
#skewness = skewness.reshape ((-1, 1))
#
## Feature5: Kurtosis of each image 
#kurtosis = np.array([stats.kurtosis(image.ravel()) for image in x_train])
#kurtosis = kurtosis.reshape ((-1, 1))
#
## Feature6: Entropy of each image 
#def calculate_entropy(image):
#    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
#    histogram_normalized = histogram / histogram.sum()
#    histogram_normalized = histogram_normalized[histogram_normalized > 0] # Avoid zero probabilities
#    return stats.entropy(histogram_normalized, base=2)
#
## Calculating entropy for each image in the dataset
#entropies = np.array([calculate_entropy(image) for image in x_train])
#entropies = entropies.reshape ((-1, 1))
#
## Feature7: Canny Edge detection
#canny_edge = np.array([cv2.Canny(image, 40, 200) for image in x_train])
#canny_edge = canny_edge.reshape (canny_edge.shape[0], -1)
#
## Feature8: Sobel X
#sobel_x = np.array([cv2.Sobel(image,cv2.CV_8UC1,1,0,ksize=5) for image in x_train])
#sobel_x = sobel_x.reshape(sobel_x.shape[0], -1)
#
## Feature9: Sobel Y
#sobel_y = np.array([cv2.Sobel(image,cv2.CV_8UC1,0,1,ksize=5) for image in x_train])
#sobel_y = sobel_y.reshape(sobel_y.shape[0], -1)
#
## Feature10: Binary threshold
#bin_thresh = np.array([cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)[1] for image in x_train])
#bin_thresh = bin_thresh.reshape(bin_thresh.shape[0], -1)
#
### Combined Features
#train_features = np.hstack((variances, means, stds, skewness, kurtosis, entropies, canny_edge, sobel_x, sobel_y, bin_thresh))
#
#print(train_features.shape)
#
#################  Test Data Feature Extraction  #######################
#
## Feature1: Variance of each image
#test_variances = np.var(x_test, axis=(1, 2)) 
#test_variances = test_variances.reshape ((-1, 1))
#
## Feature2: Mean of each image
#test_means = np.mean(x_test, axis=(1, 2))
#test_means = test_means.reshape ((-1, 1))
#
## Feature3: Standard Deviation of each image
#test_stds = np.std(x_test, axis=(1, 2))
#test_stds = test_stds.reshape ((-1, 1))
#
## Feature4: Skewness of each image 
#test_skewness = np.array([stats.skew(image.ravel()) for image in x_test])
#test_skewness = test_skewness.reshape ((-1, 1))
#
## Feature5: Kurtosis of each image 
#test_kurtosis = np.array([stats.kurtosis(image.ravel()) for image in x_test])
#test_kurtosis = test_kurtosis.reshape ((-1, 1))
#
## Feature6: Entropy of each image 
#test_entropies = np.array([calculate_entropy(image) for image in x_test])
#test_entropies = test_entropies.reshape ((-1, 1))
#
## Feature7: Canny Edge detection
#test_canny_edge = np.array([cv2.Canny(image, 40, 200) for image in x_test])
#test_canny_edge = test_canny_edge.reshape (test_canny_edge.shape[0], -1)
#
## Feature8: Sobel X
#test_sobel_x = np.array([cv2.Sobel(image,cv2.CV_8UC1,1,0,ksize=5) for image in x_test])
#test_sobel_x = test_sobel_x.reshape(test_sobel_x.shape[0], -1)
#
## Feature9: Sobel Y
#test_sobel_y = np.array([cv2.Sobel(image,cv2.CV_8UC1,0,1,ksize=5) for image in x_test])
#test_sobel_y = test_sobel_y.reshape(test_sobel_y.shape[0], -1)
#
## Feature10: Binary threshold
#test_bin_thresh = np.array([cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)[1] for image in x_test])
#test_bin_thresh = test_bin_thresh.reshape(test_bin_thresh.shape[0], -1)
#
### Combined Features
#test_features = np.hstack((test_variances, test_means, test_stds, test_skewness, test_kurtosis, test_entropies, test_canny_edge, test_sobel_x, test_sobel_y, test_bin_thresh))
#
#print(test_features.shape)

## Plot the histogram of pixel values for the first image
#plt.hist(x_train[0].ravel(), bins=50, color='gray', alpha=0.75)
#plt.title('Pixel Value Distribution')
#plt.xlabel('Pixel Intensity')
#plt.ylabel('Frequency')
#plt.show()

# Reshape to feed into random forest
x_train_augmented = x_train_augmented.reshape(x_train_augmented.shape[0], -1)
x_val = x_val.reshape(x_val.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Random forest classifier

# Fit the model
rf_classifier = RandomForestClassifier(n_estimators=75, max_depth=15,
        min_samples_split= 2, min_samples_leaf= 1, max_features= 15,
        random_state=43)
rf_classifier.fit(x_train_augmented, y_train_augmented)

# Predicting the test results
y_pred = rf_classifier.predict(x_test)

# Generating the classification report
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.5f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])

cm_display.plot()
plt.show()

# Predict probabilities for the test set
y_scores = rf_classifier.predict_proba(x_test)[:, 1]  # score for the positive class

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

