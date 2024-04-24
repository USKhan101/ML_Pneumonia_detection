import cv2
import h5py
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
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

x_train_augmented = np.squeeze(x_train_augmented, axis=-1)

plt.figure(figsize=(10,8))

plt.subplot(2, 3, 1) 
plt.imshow(x_normal[0])
plt.title('Train_data 1: ORIGINAL')

plt.subplot(2, 3, 2) 
plt.imshow(augmented_images[0])
plt.title('Train_data 1: AUGMENTED 1')

plt.subplot(2, 3, 3) 
plt.imshow(augmented_images[1])
plt.title('Train_data 1: AUGMENTED 2')

plt.subplot(2, 3, 4) 
plt.imshow(x_normal[1])
plt.title('Train_data 2: ORIGINAL')

plt.subplot(2, 3, 5) 
plt.imshow(augmented_images[2])
plt.title('Train_data 2: AUGMENTED 1')

plt.subplot(2, 3, 6) 
plt.imshow(augmented_images[3])
plt.title('Train_data 2: AUGMENTED 2')

## Count bar plot for dataset
def count_plot (label):
    l= []
    
    for i in label:
        if (i == 0):
            l.append("Normal")
        else:
            l.append("Pneumonia")
    plt.figure()
    sns.set_style('darkgrid')
    sns.countplot(x=l)
    plt.show()

count_plot (y_train_augmented)

augm_datapath = './processed_data/augmented_traindata.h5'

with h5py.File(augm_datapath, 'w') as file:
    file.create_dataset('augmented_train_data',  data=x_train_augmented)
    file.create_dataset('augmented_train_label', data=y_train_augmented)

#print (x_train_augmented.shape)
#print (x_train_augmented[0].shape)
#print (y_train_augmented.shape)
#print (y_train_augmented[0].shape)
