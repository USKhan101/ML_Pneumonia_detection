Machine Learning for Chest X-Ray Based Pneumonia Detection.

Files description:

dataset_fixing.py: The dataset has inconsistent data distribution in validation and test directory. We took all the data, shufled them and then put them into directories after spliting 70%, 10%, 20% for train, val and test data.

data_generation.py: Visualizes data distribution and show some of the image datas. The file outputs normalized raw train, test, val data and labels.

outlier_removal.py: The input of the file will be the raw training data and it outputs training data after outliers removal. We do 2 steps outlierdetection. Step1: Z-score based, Step2: IQR based.

data_enhance.py: The input of the file is train data outlier removal. it will enhance contrast, reduce noise and sharp the image data.

data_augmentation.py: This file does augmentation of the enhanced data after outlier removal. Generates 2 new augmented images for each training image. We will augmnet only the training data for training the model well.

Model files:

Random_forest_classifier.py: Classifies the raw/augmenetd data or Extracted features for pneumonia detection.

resnet50.py: RESNET50 for classifying the raw/augmenetd data or Extracted features for pneumonia detection.


How to run:
conda create -n myenv
conda activate myenv
conda install numpy opencv h5py matplotlib scikit-learn

source data_all.sh
python3 model_file_name

conda deactivate
