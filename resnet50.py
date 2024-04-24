import os
import cv2
import time
import h5py
import random
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Normalize, Lambda, CenterCrop
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda

file_path = './processed_data/normalized_rawdata.h5'
augm_path = './processed_data/augmented_traindata.h5'

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Reading data from file
with h5py.File(file_path, 'r') as file:

    x_val =   np.array(file['val_data'][:])
    y_val =   np.array(file['val_label'][:])

    x_test =  np.array(file['test_data'][:])
    y_test =  np.array(file['test_label'][:])

## Reading data from file
with h5py.File(augm_path, 'r') as file:

    x_train = np.array(file['augmented_train_data'][:])
    y_train = np.array(file['augmented_train_label'][:])

#print (x_train.shape)
#print (x_train[0].shape)

# Transform into tensor and data loaders
transform = Compose([
    Lambda(lambda x: Image.fromarray(x.astype('uint8'), 'L')),
    Resize((256,256)),
    ToTensor()
    #Normalize(mean=[0.485], std=[0.229])  # Normalizing for grayscale
])

# Load data
x_train = torch.stack([transform(img) for img in x_train])
y_train = torch.from_numpy(y_train).long()

train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# validation data
x_val = torch.stack([transform(img) for img in x_val])
y_val = torch.from_numpy(y_val).long()

val_data = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Test data
x_test = torch.stack([transform(img) for img in x_test])
y_test = torch.from_numpy(y_test).long()

test_data = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Loading resnet50 model
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Change the input layer for grayscale image input
input_layer = model.conv1
model.conv1 = nn.Conv2d(1, input_layer.out_channels, 
                            kernel_size=input_layer.kernel_size, 
                            stride=input_layer.stride, 
                            padding=input_layer.padding, 
                            bias=input_layer.bias)
print(model.conv1)

# Binary classification
model.fc = nn.Linear(model.fc.in_features, 2)

# Run model to GPU if available
model.to(device)

# Optimizer and Loss function configuration
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        total_loss = 0
        iteration = 0
        for inputs, labels in train_loader:
            #print(inputs.shape)  
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iteration += 1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            #print (f'Epoch {epoch+1}, Iteration {iteration+1} ,Loss per iter: {total_loss}')

        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}')
        print('Training Accuracy: %d %%' % (100 * correct / total))

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                images, labels = inputs.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                #print(f'Pred: {predicted}, total: {total}, correct: {correct}')

            print('Validation Accuracy: %d %%' % (100 * correct / total))

# Execute the training
train(model, train_loader, val_loader, criterion, optimizer)

print ("Ended in", time.time() - start_time, "seconds.")


############## Extra ##################
initial_bias = np.log([COUNT_PNEUMONIA/COUNT_NORMAL])
initial_bias

weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0 
weight_for_1 = (1 / COUNT_PNEUMONIA)*(TRAIN_IMG_COUNT)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

with strategy.scope():
    model = build_model()

    METRICS = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=METRICS
    )

history = model.fit(
    train_ds,
    steps_per_epoch=TRAIN_IMG_COUNT // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=VAL_IMG_COUNT // BATCH_SIZE,
    class_weight=class_weight,
)
