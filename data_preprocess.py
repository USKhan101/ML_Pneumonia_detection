import cv2
import h5py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

file_path = './processed_data/traindata_after_outlier.h5'
test_path = './processed_data/normalized_rawdata.h5'

## Reading data from file
with h5py.File(file_path, 'r') as file:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]

with h5py.File(test_path, 'r') as file1:
    x_test = file1['test_data'][:]
    y_test = file1['test_label'][:]

print (len(x_train[0].shape))
x_train = x_train.astype(np.uint8)
#x_test = x_test.astype(np.uint8)

################# Pre-processing of image ###################
def preprocess (image):
    size = (128,128)

    #blurred = cv2.GaussianBlur(image, (5, 5), 0)

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #thresh = cv2.erode(thresh, None, iterations=2)
    #thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    c = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(c)
    
    cropped_image = image[y:y+h, x:x+w]
    
    resized_image = cv2.resize(cropped_image, size)

    return resized_image

processed_traindata = np.array ([preprocess (img) for img in x_train])
print (processed_traindata.shape)

#out_path = './processed_data/preprocessed_data.h5'
#
### Saving data
#with h5py.File(out_path, 'w') as file:
#    file.create_dataset('train_feature', data=scaled_features)
#    file.create_dataset('train_label', data=y_train)
#
