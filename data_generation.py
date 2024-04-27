import os
import cv2
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#path = './chest_xray/'
path = './chest_xray_new/'
data_file = './processed_data/normalized_rawdata.h5'

# define paths
train_dir = path + 'train'
val_dir = path + 'val'
test_dir = path + 'test'

labels = ['NORMAL', 'PNEUMONIA']
img_size = 256

def array_data (data_dir):
    data = []
    for label in labels:
        path  = os.path.join(data_dir, label)
        class_num = labels.index(label)

        for img in os.listdir(path):
            # Reading images in gray scale
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            data.append ([resized_arr, class_num])

    return np.array(data, dtype=object)

train_data = array_data (train_dir)
val_data = array_data (val_dir)
test_data = array_data (test_dir)

## Count bar plot for dataset
def count_plot (data):
    l= []
    
    for i in data:
        if (i[1] == 0):
            l.append("Normal")
        else:
            l.append("Pneumonia")
    
    sns.set_style('darkgrid')
    ax = sns.countplot(x=l)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')

# Divide the data and labels
x_train = []
y_train = []

x_val = []
y_val = []

x_test = []
y_test = []

for feature, label in train_data:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test_data:
    x_test.append(feature)
    y_test.append(label)

for feature, label in val_data:
    x_val.append(feature)
    y_val.append(label)

# Normalizing Data
x_train = np.array(x_train) / 255.0
x_val = np.array(x_val) / 255.0
x_test = np.array(x_test) / 255.0

x_train = x_train.astype(np.float32)
x_val  = x_test.astype(np.float32)
x_test = x_test.astype(np.float32)

y_train = np.array(y_train).astype(np.uint8)
y_val = np.array(y_val).astype(np.uint8)
y_test = np.array(y_test).astype(np.uint8)

print (x_train.dtype)
print (y_train.dtype)

# Plot count bar and pie chart
plt.figure(figsize=(18, 12))

# Subplot 1: Count plot for training data
plt.subplot(2, 3, 1)
count_plot (train_data)
plt.title('Training Data')

# Subplot 2: Count plot for validation data
plt.subplot(2, 3, 2)
count_plot (val_data)
plt.title('Validation Data')

# Subplot 3: Count plot for testing data
plt.subplot(2, 3, 3)
count_plot (test_data)
plt.title('Testing Data')

# Subplot 4: Pie chart for training data
plt.subplot(2, 3, 4)
train_counts = np.unique(y_train, return_counts=True)
plt.pie(train_counts[1], labels=['Normal' if label == 0 else 'Pneumonia' for label in train_counts[0]], autopct='%1.2f%%')
plt.title('Training Data')

# Subplot 5: Pie chart for validation data
plt.subplot(2, 3, 5)
val_counts = np.unique(y_val, return_counts=True)
plt.pie(val_counts[1], labels=['Normal' if label == 0 else 'Pneumonia' for label in val_counts[0]], autopct='%1.2f%%')
plt.title('Validation Data')

# Subplot 6: Pie chart for testing data
plt.subplot(2, 3, 6)
test_counts = np.unique(y_test, return_counts=True)
plt.pie(test_counts[1], labels=['Normal' if label == 0 else 'Pneumonia' for label in train_counts[0]], autopct='%1.2f%%')
plt.title('Testing Data')

plt.tight_layout()
plt.savefig('new_count_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Show random grayscale x-ray images from train, test, val dataset
plt.figure(figsize=(10,8))

plt.subplot(2, 3, 1) 
plt.imshow(x_train[0])
plt.title('Train_data: NORMAL')

plt.subplot(2, 3, 4) 
plt.imshow(x_train[-1])
plt.title('Train_data: PNEUMONIA')

plt.subplot(2, 3, 2) 
plt.imshow(x_val[0])
plt.title('Val_data: NORMAL')

plt.subplot(2, 3, 5) 
plt.imshow(x_val[-1])
plt.title('Val_data: PNEUMONIA')

plt.subplot(2, 3, 3) 
plt.imshow(x_test[0])
plt.title('Test_data: NORMAL')

plt.subplot(2, 3, 6) 
plt.imshow(x_test[-1])
plt.title('Test_data: PNEUMONIA')

plt.tight_layout()
plt.savefig('new_rand_image.png', dpi=300, bbox_inches='tight')
plt.show()

#pie plot to show the ratio of train, val and test dataset
plt.figure()
plt.pie([len(y_train), len(y_val), len(y_test)], labels=['train', 'validation', 'test'], autopct='%1.1f%%', colors=['orange', 'red', 'lightblue'], explode=(0.05, 0, 0))
plt.savefig('new_pie_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot the histogram of pixel values from each dataset
plt.figure(figsize=(10,8))

plt.subplot(2, 3, 1) 
plt.hist(x_train[0].ravel(), bins=50, color='gray', alpha=0.75)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Train_data: NORMAL')

plt.subplot(2, 3, 4) 
plt.hist(x_train[-1].ravel(), bins=50, color='gray', alpha=0.75)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Train_data: PNEUMONIA')

plt.subplot(2, 3, 2) 
plt.hist(x_val[0].ravel(), bins=50, color='gray', alpha=0.75)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Val_data: NORMAL')

plt.subplot(2, 3, 5) 
plt.hist(x_val[-1].ravel(), bins=50, color='gray', alpha=0.75)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Val_data: PNEUMONIA')

plt.subplot(2, 3, 3) 
plt.hist(x_test[0].ravel(), bins=50, color='gray', alpha=0.75)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Test_data: NORMAL')

plt.subplot(2, 3, 6) 
plt.hist(x_test[-1].ravel(), bins=50, color='gray', alpha=0.75)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Test_data: PNEUMONIA')

plt.tight_layout()
plt.savefig('new_image_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the normalized dataset into h5 file
with h5py.File(data_file, 'w') as file:
    file.create_dataset('train_data',  data=x_train)
    file.create_dataset('train_label', data=y_train) 

    file.create_dataset('val_data',  data=x_val)
    file.create_dataset('val_label', data=y_val) 

    file.create_dataset('test_data',  data=x_test)
    file.create_dataset('test_label', data=y_test) 
