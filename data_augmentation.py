import cv2
import h5py
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

file_path = './processed_data/outlier_removed_traindata.h5'
data_path = './processed_data/normalized_rawdata.h5'

## Reading raw train data from file
with h5py.File(file_path, 'r') as file:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]

with h5py.File(data_path, 'r') as file1:
    x_val = file1['val_data'][:]
    y_val = file1['val_label'][:]

    x_test = file1['test_data'][:]
    y_test = file1['test_label'][:]

print (x_train.dtype)
print (y_train.dtype)

# Configure the ImageDataGenerator to augment images
data_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2
    )

# Function to augment images based on the label
def augment_images(x_data, label, num_augments=3):
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

print (x_train_augmented.dtype)
print (y_train_augmented.dtype)

# Plot some of original and augmented images
plt.figure(figsize=(10, 8))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(x_train_augmented[i], cmap='gray')
    plt.title(f'Augmented_data {i + 1}')
plt.tight_layout()
plt.savefig('augmented_data.png', dpi=300, bbox_inches='tight')
plt.show()

# Count bar plot for dataset
def count_plot(label):
    l = ["Normal" if i == 0 else "Pneumonia" for i in label]
    plt.figure()
    sns.set_style('darkgrid')
    ax = sns.countplot(x=l)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# Count plot for training data after augmentation
plt.figure(figsize=(12, 10))

plt.subplot(1, 2, 1)
count_plot(y_train_augmented)
plt.title('Augmented Data')

#pie plot to show the ratio of train, val and test dataset
plt.subplot(1, 2, 2)
plt.pie([len(y_train_augmented), len(y_val), len(y_test)], labels=['train', 'validation', 'test'], autopct='%1.1f%%', colors=['orange', 'red', 'lightblue'], explode=(0.05, 0, 0))
plt.title('Augmented Data')

plt.savefig('Augmented_data_count.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the augmented training data in another .h5 file
augm_datapath = './processed_data/augmented_traindata.h5'

with h5py.File(augm_datapath, 'w') as file:
    file.create_dataset('train_data',  data=x_train_augmented)
    file.create_dataset('train_label', data=y_train_augmented)

print (x_train_augmented.shape)
print (x_train_augmented[0].shape)
print (y_train_augmented.shape)
print (y_train_augmented[0].shape)
