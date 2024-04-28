import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt

file_path = './processed_data/normalized_rawdata.h5'
data_path = './processed_data/outlier_removed_traindata.h5'

## Reading data from file
with h5py.File(data_path, 'r') as file:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]

with h5py.File(file_path, 'r') as file:
    x_val = file['val_data'][:]
    y_val = file['val_label'][:]
    
    x_test = file['test_data'][:]
    y_test = file['test_label'][:]

def data_enhance (data_dir):
    data = []
    for img in data_dir:
        image = np.uint8(img*255)
        
        # Median Filtering for noise reduction
        median_filtered = cv2.medianBlur(image, 5)

        # CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply (median_filtered)

        ## Compute the median pixel value
        #median_val = np.median(clahe_image)
        #
        ### thresholds on the median value 
        #sigma = 0.33 
        #threshold1 = int(max(0, (1.0 - sigma) * median_val))
        #threshold2 = int(min(255, (1.0 + sigma) * median_val))
        #
        ## Canny edge detector
        #canny_image = cv2.Canny(clahe_image, threshold1, threshold2)

        # Sharpening images
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened_image = cv2.filter2D(clahe_image, -1, kernel)

        data.append (sharpened_image)

    return np.array(data)

x_train = np.float32(data_enhance (x_train) / 255)
x_val = np.float32(data_enhance (x_val) / 255)
x_test = np.float32(data_enhance (x_test) / 255)

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
plt.savefig('./plots/enhanced_rand_image.pdf', dpi=300, bbox_inches='tight')
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
plt.savefig('./plots/enhanced_image_histogram.pdf', dpi=300, bbox_inches='tight')
plt.show()


# Save enhanced data 
out_path = './processed_data/enhanced_data.h5'

with h5py.File(out_path, 'w') as file:
    file.create_dataset('train_data', data=x_train)
    file.create_dataset('train_label', data=y_train)
    file.create_dataset('val_data', data=x_val)
    file.create_dataset('val_label', data=y_val)
    file.create_dataset('test_data', data=x_test)
    file.create_dataset('test_label', data=y_test)
