import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

file_path = './processed_data/normalized_rawdata.h5'
out_path = './processed_data/outlier_removed_traindata.h5'
enhance_path = './processed_data/enhanced_data.h5'
augm_path = './processed_data/augmented_traindata.h5'

augment = False

start_time = time.time()

### Reading raw data from file
#with h5py.File(file_path, 'r') as file:
#    x_train = file['train_data'][:]
#    y_train = file['train_label'][:]
#
#    x_val = file['val_data'][:]
#    y_val = file['val_label'][:]
#
#    x_test = file['test_data'][:]
#    y_test = file['test_label'][:]

## Reading enhanced or augmented data from file
if (augment):
    with h5py.File(augm_path, 'r') as file:
        x_train = file['train_data'][:]
        y_train = file['train_label'][:]

    with h5py.File(enhance_path, 'r') as file:
        x_val = file['val_data'][:]
        y_val = file['val_label'][:]

        x_test = file['test_data'][:]
        y_test = file['test_label'][:]

else:
    with h5py.File(file_path, 'r') as file:
        x_train = file['train_data'][:]
        y_train = file['train_label'][:]
    
        x_val = file['val_data'][:]
        y_val = file['val_label'][:]
    
        x_test = file['test_data'][:]
        y_test = file['test_label'][:]

print (x_train.shape)
print (y_train.shape)
print (x_val.shape)
print (y_val.shape)
print (x_test.shape)
print (y_test.shape)

# Concatanate train and val data
x_train = np.concatenate([x_train, x_val])
y_train = np.concatenate([y_train, y_val])

print (x_train.shape)
print (y_train.shape)

# Flattening the data
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

## Initialize Random Forest Classifier
rf = RandomForestClassifier()

rf.fit(x_train, y_train)

# Predicting the test results
y_pred = rf.predict(x_test)

# Generating the classification report
print(classification_report(y_test, y_pred))
print(f"Accuracy:", accuracy_score(y_test, y_pred))

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.5f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
cm_display.plot()
plt.savefig('./plots/mcs_cm_rf.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Predict probabilities for the test set
y_scores = rf.predict_proba(x_test)[:, 1]  # score for the positive class

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
#plt.savefig('./plots/aug_roc_rf.pdf', dpi=300, bbox_inches='tight')
plt.savefig('./plots/mcs_roc_rf.pdf', dpi=300, bbox_inches='tight')
plt.show()

print ("Ended in", time.time() - start_time, "seconds.")
