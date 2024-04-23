import os
import cv2
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = './chest_xray/'
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
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            data.append ([resized_arr, class_num])

    #np.random.shuffle(data)

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
    sns.countplot(x=l)
    plt.show()

count_plot (train_data)
count_plot (val_data)
count_plot (test_data)

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

# Normalization
x_train = np.array(x_train) / 255.0
x_val = np.array(x_val) / 255.0
x_test = np.array(x_test) / 255.0

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

plt.subplot(1, 2, 1) 
plt.imshow(x_train[0])
plt.title('Train_data: NORMAL')

plt.subplot(1, 2, 2) 
plt.imshow(x_train[-1])
plt.title('Train_data: PNEUMONIA')

plt.subplot(2, 2, 1) 
plt.imshow(x_val[0])
plt.title('Val_data: NORMAL')

plt.subplot(2, 2, 2) 
plt.imshow(x_val[-1])
plt.title('Val_data: PNEUMONIA')

plt.subplot(3, 2, 1) 
plt.imshow(x_test[0])
plt.title('Test_data: NORMAL')

plt.subplot(3, 2, 2) 
plt.imshow(x_test[-1])
plt.title('Test_data: PNEUMONIA')

plt.pie([len(x_train), len(x_val), len(x_test)], labels=['train', 'validation', 'test'], autopct='%.1f%%', colors=['orange', 'red', 'lightblue'], explode=(0.05, 0, 0))
plt.show()

with h5py.File(data_file, 'w') as file:
    file.create_dataset('train_data',  data=x_train)
    file.create_dataset('train_label', data=y_train) 

    file.create_dataset('val_data',  data=x_val)
    file.create_dataset('val_label', data=y_val) 

    file.create_dataset('test_data',  data=x_test)
    file.create_dataset('test_label', data=y_test) 
