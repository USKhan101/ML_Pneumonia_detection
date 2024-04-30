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

start_time = time.time()

## Reading data from file
with h5py.File(enhance_path, 'r') as file:
#    x_train = file['train_data'][:]
#    y_train = file['train_label'][:]

    x_val = file['val_data'][:]
    y_val = file['val_label'][:]

    x_test = file['test_data'][:]
    y_test = file['test_label'][:]

with h5py.File(augm_path, 'r') as file:
    x_train = file['train_data'][:]
    y_train = file['train_label'][:]

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
rf = RandomForestClassifier(random_state=43)

## For Grid Search
#param_grid = {
#    'n_estimators': [200, 250],
#    'max_depth': [35, 45],
#    'min_samples_split': [2, 5],
#    'min_samples_leaf': [1, 5 ],
#    'max_features': ['sqrt', 15]
#}
#
## Initialize the GridSearchCV object
#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2)
#
## Fit the grid search to the data
#grid_search.fit(x_train, y_train)
#
## Print the best parameters and best score
#print("Best parameters:", grid_search.best_params_)
#print("Best cross-validation accuracy:", grid_search.best_score_)
#
## Evaluate on the test set
#y_pred = grid_search.predict(x_test)
#print("Test Accuracy: ", accuracy_score(y_test, y_pred))
#print("Classification Report on Test Data:")
#print(classification_report(y_test, y_pred))

# For random search
param_rand = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [30, 40, 50, 60],  
    'min_samples_split': [2, 5, 10, 15, 20], 
    'min_samples_leaf': [1, 2, 5, 10],  
    'max_features': ['sqrt', 15, 30]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_rand,
                                   n_iter=100, 
                                   cv=5,  # 5-fold cross-validation
                                   verbose=1,  
                                   random_state=43,  
                                   scoring='accuracy') 

# Fit the model
random_search.fit(x_train, y_train)

# Best parameters and score from random search
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation accuracy: ", random_search.best_score_)

# Evaluate on the test set
y_pred = random_search.predict(x_test)
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report on Test Data:")
print(classification_report(y_test, y_pred))

# Fit the model for found parameters
#rf = RandomForestClassifier(n_estimators=120, max_depth=45, min_samples_split= 2, min_samples_leaf= 1, max_features= 'sqrt', random_state=43)

#rf.fit(x_train, y_train)
#
## Predicting the test results
#y_pred = rf.predict(x_test)
#
## Generating the classification report
#print(classification_report(y_test, y_pred))
#print(f"Accuracy:", accuracy_score(y_test, y_pred))

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.5f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
cm_display.plot()
plt.savefig('./plots/aug_cm_rf.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Predict probabilities for the test set
# y_scores = rf.predict_proba(x_test)[:, 1]  # score for the positive class
y_scores = random_search.predict_proba(x_test)[:, 1]

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
plt.savefig('./plots/aug_roc_rf.pdf', dpi=300, bbox_inches='tight')
plt.show()

print ("Ended in", time.time() - start_time, "seconds.")
