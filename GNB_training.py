import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

#out_path = 'outlier_removed_traindata.h5'
file_path = 'normalized_rawdata.h5'
enhance_path = 'enhanced_data.h5'
augm_path = 'augmented_traindata.h5'

start_time = time.time()

# Reading data from file, taken from Ummay's Random Forest Classifier code
# only training on raw, enhanced, and augmented

with h5py.File(augm_path, 'r') as file:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]
 
with h5py.File(enhance_path, 'r') as file:
    x_val = file['val_data'][:]
    y_val = file['val_label'][:]
 
    x_test = file['test_data'][:]
    y_test = file['test_label'][:]

    

    

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

# Concatenate train and val data
x_train = np.concatenate([x_train, x_val])
y_train = np.concatenate([y_train, y_val])

print(x_train.shape)
print(y_train.shape)

# Flattening the data
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Initialize Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Fit the model
gnb.fit(x_train, y_train)

# Predict on the test data
y_pred = gnb.predict(x_test)

# Evaluate the model
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report on Test Data:")
print(classification_report(y_test, y_pred))

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.5f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cm_display.plot()
plt.savefig('gnb_cm.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Predict probabilities for the test set
y_scores = gnb.predict_proba(x_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('gnb_roc.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Ended in", time.time() - start_time, "seconds.")
