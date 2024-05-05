import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.callbacks import History

start_time = time.time()

# Load data
aug_data = './processed_data/augmented_traindata.h5'
with h5py.File(aug_data, 'r') as file:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]
 
enhance_path= './processed_data/enhanced_data.h5'
with h5py.File(enhance_path, 'r') as file:
    x_val = file['val_data'][:]
    y_val = file['val_label'][:]
 
    x_test = file['test_data'][:]
    y_test = file['test_label'][:]

# Concatenate train and val data
x_train = np.concatenate([x_train, x_val])
y_train = np.concatenate([y_train, y_val])

def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create model
cnn_model = create_cnn_model((256, 256, 1))

# Train model
history = cnn_model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Plot training & validation accuracy values
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Predicting the test set results
y_pred = (cnn_model.predict(x_test) > 0.5).astype("int32")

# Calculate the ROC curve and AUC
y_scores = cnn_model.predict(x_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()	

# Print other evaluation metrics
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.5f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cm_display.plot()
plt.show()

print("Ended in", time.time() - start_time, "seconds.")
