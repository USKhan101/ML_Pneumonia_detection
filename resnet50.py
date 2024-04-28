import os
import cv2
import time
import h5py
import torch
import random
import numpy as np 
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, NAdam, SGD
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Normalize, Lambda, CenterCrop
from torchvision.models import resnet50, resnet18, resnet101, ResNet50_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

file_path = './processed_data/normalized_rawdata.h5'
augm_path = './processed_data/augmented_traindata.h5'

num_epochs = 50

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

augmentation = False # To fit the model for raw/ augmented data

## Reading data from file
with h5py.File(file_path, 'r') as file:

    x_val =   np.array(file['val_data'][:])
    y_val =   np.array(file['val_label'][:])

    x_test =  np.array(file['test_data'][:])
    y_test =  np.array(file['test_label'][:])

if (augmentation):
    with h5py.File(augm_path, 'r') as file:
        x_train = np.array(file['train_data'][:])
        y_train = np.array(file['train_label'][:])

else:
    with h5py.File(file_path, 'r') as file:
        x_train = np.array(file['train_data'][:])
        y_train = np.array(file['train_label'][:])

print (x_train.shape)
print (y_train.shape)

print (x_val.shape)
print (y_val.shape)

print (x_test.shape)
print (y_test.shape)

##### Steps due to imbalance dataset #### 
## Total train image and count train data for each category
#count_normal = np.sum(y_train == 0)
#count_pneumonia = np.sum(y_train == 1)
#total_train = count_normal + count_pneumonia
#
#print (total_train, count_normal, count_pneumonia)
#
## Calculate initial bias for the readout layer
#initial_bias = np.log(count_pneumonia/ count_normal)
#initial_bias_tensor = torch.tensor([initial_bias]).to(device)
#
## Calculate class weights
#weight_for_0 = (1 / count_normal) * (total_train) / 2.0
#weight_for_1 = (1 / count_pneumonia) * (total_train) / 2.0
#class_weights = torch.tensor([weight_for_0, weight_for_1]).to(device)
#
#print (class_weights)

##########################################

# Transform into tensor and data loaders
transform = Compose([
    Lambda(lambda x: Image.fromarray(x.astype('uint8'), 'L')),
    Resize((256,256)),
    ToTensor()
])

# Load data
x_train = torch.stack([transform(img) for img in x_train])
y_train = torch.from_numpy(y_train).long()

train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# validation data
x_val = torch.stack([transform(img) for img in x_val])
y_val = torch.from_numpy(y_val).long()

val_data = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

# Test data
x_test = torch.stack([transform(img) for img in x_test])
y_test = torch.from_numpy(y_test).long()

test_data = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Loading resnet model
model = resnet50(pretrained=True)

# Change the input layer for grayscale image input
input_layer = model.conv1
model.conv1 = nn.Conv2d(1, input_layer.out_channels, 
                            kernel_size=input_layer.kernel_size, 
                            stride=input_layer.stride, 
                            padding=input_layer.padding, 
                            bias=input_layer.bias)

# Binary classification
num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Linear(32, 2),
    nn.Sigmoid()
)

last_linear_layer = model.fc[-2]

# Set initial bias for last linear output layer
#last_linear_layer.bias.data.fill_(initial_bias.item())

# Run model to GPU if available
model.to(device)

# Optimizer and Loss function configuration
#criterion = nn.BCELoss()
#criterion = CrossEntropyLoss(weight=class_weights)

criterion = CrossEntropyLoss()
optimizer = NAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)

train_loss = []
val_loss = []
test_loss = []

train_accuracy = []
val_accuracy = []
test_accuracy = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct    = 0 
    total      = 0
    iteration  = 0

    for inputs, labels in train_loader:
        #print(inputs.shape)  
        inputs, labels = inputs.to(device), labels.to(device)
        labels_one_hot = torch.eye(2, device=device)[labels].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_one_hot)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        iteration += 1
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # print(f'Pred: {predicted}, labels: {labels}, total: {total}, correct: {correct}')
        # print (f'Epoch {epoch+1}, Iteration {iteration+1} ,Loss per iter: {total_loss}')

    accuracy = 100 * correct / total
    train_loss.append(total_loss/len(train_loader))
    train_accuracy.append(accuracy)

    print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}')
    print('Training Accuracy: %d %%' % (accuracy))

    # Validation
    model.eval()

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels_one_hot = torch.eye(2, device=device)[labels].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels_one_hot)

            total_loss += loss.item() * inputs.size(0)
            predicted = torch.round(outputs).argmax(dim=1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(f'Pred: {predicted}, labels: {labels}, total: {total}, correct: {correct}')

        accuracy = 100 * correct / total
        val_loss.append(total_loss/len(val_loader))
        val_accuracy.append(accuracy)

        print(f'Epoch {epoch+1}, Validation Loss: {total_loss/len(val_loader)}')
        print('Validation Accuracy: %d %%' % (accuracy))


    # Testing
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            images, labels = inputs.to(device), labels.to(device)
            labels_one_hot = torch.eye(2, device=device)[labels].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels_one_hot)
            total_loss += loss.item()
            predicted = torch.round(outputs).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print(f'Pred: {predicted}, labels: {labels}, total: {total}, correct: {correct}')
            #all_labels.extend(labels.to(device).numpy())
            #all_preds.extend(predicted.to(device).numpy())

        accuracy = 100 * correct / total
        test_loss.append(total_loss/len(val_loader))
        test_accuracy.append(accuracy)
        print(f'Epoch {epoch+1}, Testing Loss: {total_loss/len(test_loader)}')
        print('Testing Accuracy: %d %%' % (accuracy))

    ## Score report
    #print(classification_report(all_labels, all_preds, target_names=['class_0', 'class_1', 'class_n']))
    #
    ## Accuracy
    #accuracy = accuracy_score(all_labels, all_preds)
    #print(f'Accuracy: {accuracy:.2f}')

print ("Ended in", time.time() - start_time, "seconds.")

# Train Loss and Accuracy by Epochs

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)

plt.plot(train_accuracy, label ="Train Accuracy")
plt.plot(test_accuracy, label ="Test Accuracy")
plt.plot(val_accuracy, label ="Validation Accuracy")
plt.legend()
plt.xlabel('Total number of epochs')
plt.ylabel('Accuracy (%)')

plt.subplot(1, 2, 2)

plt.plot(train_loss, label ="Train Loss")
plt.plot(test_loss, label ="Test Loss")
plt.plot(val_loss, label ="Validation Loss")
plt.legend()
plt.xlabel('Total number of epochs')
plt.ylabel('Loss')
plt.show()

