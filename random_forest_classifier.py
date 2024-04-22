import time
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

file_path = './processed_data/normalized_rawdata.h5'
augm_path = './processed_data/augmented_traindata.h5'
outl_path = './processed_data/traindata_after_outlier.h5'
feat_path = './processed_data/feature_data.h5'

start_time = time.time()

## Reading data from file
with h5py.File(feat_path, 'r') as file:
    x_train = file['train_feature'][:]
    y_train = file['train_label'][:]

    x_test = file['test_feature'][:]
    y_test = file['test_label'][:]

## Augmented train data
#with h5py.File(augm_path, 'r') as file:
#    x_train = file['augmented_train_data'][:]
#    y_train = file['augmented_train_label'][:]

# Train data after outliers removal
#with h5py.File(outl_path, 'r') as file:
#    x_train = file['train_data'][:]
#    y_train = file['train_label'][:]

# Flattening the data
# do not need for feature data
#x_train = x_train.reshape(x_train.shape[0], -1)
#x_test = x_test.reshape(x_test.shape[0], -1)

## Random forest classifier
#
## Define the parameter grid
#param_grid = {
#    'n_estimators': [50, 75 , 100, 120],
#    'max_depth': [10, 15, 20, 25],
#    'min_samples_split': [1, 2, 5, 10, 20],
#    'min_samples_leaf': [1, 2, 5, 7, 10],
#    'max_features': ['sqrt', 0.5, 5, 10, 15]
#}
#

## Initialize the classifier
#rf = RandomForestClassifier(random_state=42)
#
## Initialize the GridSearchCV object
#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
#
## Fit the grid search to the data
#grid_search.fit(x_train, y_train)
#
## Print the best parameters and best score
#print("Best parameters:", grid_search.best_params_)
#print("Best score:", grid_search.best_score_)

# Fit the model

# For raw/ augmented data
#rf_classifier = RandomForestClassifier(n_estimators=75, max_depth=15, min_samples_split= 2, min_samples_leaf= 1, max_features= 15, random_state=43)

# for extracted features
rf_classifier = RandomForestClassifier(n_estimators=120, max_depth=45, min_samples_split= 2, min_samples_leaf= 1, max_features= 15, random_state=43)

rf_classifier.fit(x_train, y_train)

# Predicting the test results
y_pred = rf_classifier.predict(x_test)

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
plt.show()

# Predict probabilities for the test set
y_scores = rf_classifier.predict_proba(x_test)[:, 1]  # score for the positive class

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
plt.show()

print ("Ended in", time.time() - start_time, "seconds.")
