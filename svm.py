import time
import h5py
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def main():
    start_time = time.time()

    # Load data
    file_path = './processed_data/normalized_rawdata.h5'
    with h5py.File(file_path, 'r') as file:
        x_train = file['train_data'][:]
        y_train = file['train_label'][:]
        x_val = file['val_data'][:]
        y_val = file['val_label'][:]
        x_test = file['test_data'][:]
        y_test = file['test_label'][:]
        print("Data loaded in {:.2f} seconds. Shapes: Train: {}, Val: {}, Test: {}".format(
            time.time() - start_time, x_train.shape, x_val.shape, x_test.shape))

    # Concatenate train and val data
    x_train = np.concatenate([x_train, x_val])
    y_train = np.concatenate([y_train, y_val])
    print("Data concatenated. New train shape: {}, {}".format(x_train.shape, y_train.shape))

    # Flattening the data
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], x-1)
    print("Data flattened for SVM input.")

    # Initialize and fit the SVM model
    print("Starting SVM training...")
    svm_start_time = time.time()
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(x_train_flat, y_train)
    print("SVM training completed in {:.2f} seconds.".format(time.time() - svm_start_time))

    # Predicting the test set results
    print("Starting predictions...")
    predict_start_time = time.time()
    y_pred = svm_model.predict(x_test_flat)
    y_scores = svm_model.predict_proba(x_test_flat)[:, 1]  # Get probabilities for the positive class
    print("Prediction completed in {:.2f} seconds.".format(time.time() - predict_start_time))

    # Calculate and print classification metrics
    print("Generating classification report...")
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.5f}")

    # Confusion matrix display
    print("Displaying confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    cm_display.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Calculate the ROC curve and AUC
    print("Calculating and plotting ROC curve...")
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

    print("Total elapsed time: {:.2f} seconds.".format(time.time() - start_time))

if __name__ == "__main__":
    main()
