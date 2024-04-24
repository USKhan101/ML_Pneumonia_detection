import cv2
import h5py
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

file_path = './processed_data/outlier_removed_traindata.h5'

## Reading raw train data from file
with h5py.File(file_path, 'r') as file:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]

# Configure the ImageDataGenerator to augment images
data_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    #horizontal_flip=True,
    brightness_range=[0.5,1.2])

# Function to augment images based on the label
def augment_images(x_data, label, num_augments=4):
    augmented_images = []
    augmented_labels = []

    for x , y in zip(x_data, label):
        # Reshape to add channel dimension
        x = x[..., np.newaxis]
        x_expanded = np.expand_dims(x, 0)
        temp_gen = data_gen.flow(x_expanded, batch_size=1)
        for _ in range(num_augments):
            augmented_image = next(temp_gen)
            augmented_images.append(augmented_image[0])
            augmented_labels.append(y)
    return np.array(augmented_images), np.array(augmented_labels)

# Augment data
augmented_images, augmented_labels = augment_images(x_train, y_train)

# Combine the original and augmented data
x_train_augmented = np.concatenate((x_train[..., np.newaxis], augmented_images)) 
y_train_augmented = np.concatenate((y_train, augmented_labels))

# Randomize the augmented training data
indices = np.random.permutation(x_train_augmented.shape[0])
x_train_augmented = x_train_augmented[indices]
y_train_augmented = y_train_augmented[indices]

# Squeeze out the channel dimension
x_train_augmented = np.squeeze(x_train_augmented, axis=-1)

# Plot some of original and augmented images
plt.figure(figsize=(10, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(x_train_augmented[i], cmap='gray')
    plt.title(f'Train_data {i + 1}: {"NORMAL" if y_train_augmented[i] == 0 else "PNEUMONIA"}')
plt.tight_layout()
plt.show()

# Count bar plot for dataset
def count_plot(label):
    l = ["Normal" if i == 0 else "Pneumonia" for i in label]
    plt.figure()
    sns.set_style('darkgrid')
    sns.countplot(x=l)
    plt.show()

count_plot(y_train_augmented)

# Save the augmented training data in another .h5 file
augm_datapath = './processed_data/augmented_traindata.h5'

with h5py.File(augm_datapath, 'w') as file:
    file.create_dataset('train_data',  data=x_train_augmented)
    file.create_dataset('train_label', data=y_train_augmented)

print (x_train_augmented.shape)
print (x_train_augmented[0].shape)
print (y_train_augmented.shape)
print (y_train_augmented[0].shape)
