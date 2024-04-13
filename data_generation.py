import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = './chest_xray/'

# define paths
train_dir = path + 'train'
val_dir = path + 'val'
test_dir = path + 'test'

labels = ['NORMAL', 'PNEUMONIA']
img_size = 200

def array_data (data_dir):
    data = []
    for label in labels:
        path  = os.path.join(data_dir, label)
        class_num = labels.index(label)

        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            data.append ([resized_arr, class_num])

    np.random.shuffle(data)

    return np.array(data, dtype=object)

train_data = array_data (train_dir)
val_data = array_data (val_dir)
test_data = array_data (test_dir)

### Ploting 1 image from train, val, test data set
#plt.imshow(train_data[0][0], cmap='gray')
#plt.show()
#
#plt.imshow(val_data[0][0], cmap='gray')
#plt.show()
#
#plt.imshow(test_data[0][0], cmap='gray')
#plt.show()

## Accessing elements
#print("First element, matrix:\n", train_data[0][0])

# Count plot for dataset
#l= []
#
#for i in train_data:
#    if (i[1] == 0):
#        l.append("Normal")
#    else:
#        l.append("Pneumonia")
#
#sns.set_style('darkgrid')
#sns.countplot(x=l)
#plt.show()

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
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

#x_train = x_train.reshape(-1, img_size, img_size, 1)
#x_val = x_val.reshape(-1, img_size, img_size, 1)
#x_test = x_test.reshape(-1, img_size, img_size, 1)

print ("train Data:", x_train.shape)
print ("val Data:", x_val.shape)
print ("test Data:", x_test.shape)
print ("train Data:", y_train.shape)
print ("val Data:", y_val.shape)
print ("test Data:", y_test.shape)
