# Breast Cancer Classification Using Machine Learning

## a. Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification models to predict whether a breast tumour is **Benign (non-cancerous)** or **Malignant (cancerous)** based on the characteristics of cell nuclei obtained from digitized images of breast mass samples.

Five supervised machine learning classification algorithms have been implemented and evaluated using the same dataset and train-test split:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The performance of these models is compared using six evaluation metrics: Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

The project also includes an interactive web application developed using Streamlit. Users can upload a test dataset, select a trained machine learning model, and view its evaluation results without retraining the models.

**Target variable encoding:**
- `0` – Benign
- `1` – Malignant

## b. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Dataset Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The Breast Cancer Wisconsin (Diagnostic) dataset contains information extracted from digitized images of fine needle aspirates (FNA) of breast masses. It includes numerical characteristics of cell nuclei that can be used to distinguish between benign and malignant tumours.

**Dataset characteristics:**

| Attribute | Description |
|---|---|
| Total observations | 569 |
| Number of input features | 30 |
| Target variable | Binary classification |
| Number of classes | 2 |
| Missing values | 0 |
| Duplicate records | 0 |
| Feature type | Numerical |
| Class 0 | Benign |
| Class 1 | Malignant |

The dataset contains 30 numerical features, including measurements related to radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

**Data preprocessing and splitting:**

- The dataset was checked for missing values and duplicate records.
- The target variable was encoded as 0 (Benign) and 1 (Malignant).
- The dataset was divided into training and testing subsets using an 80:20 split.
- Stratified sampling was used to maintain the class distribution in both subsets.
- `random_state=42` was used to ensure reproducibility.
- Feature scaling using StandardScaler was applied to Logistic Regression, KNN, and SVM.
- Decision Tree and Gaussian Naive Bayes were trained without feature scaling.

| Dataset subset | Number of records |
|---|---:|
| Training dataset (80%) | 455 |
| Testing dataset (20%) | 114 |
| Total | 569 |

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/himanshumishra10000/breast-cancer-ml-deployment

The repository contains the Streamlit application, trained machine learning models, test dataset, requirements file, README documentation, and source code used for model development.

## d. Models Used and Performance Comparison

Five machine learning classification algorithms were trained and evaluated using the same training and testing datasets.

### Evaluation Metrics Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | 0.9756 | **0.9524** | **0.9639** | **0.9433** |

*Note: All metrics are calculated on the same held-out test dataset containing 114 observations. The positive class is Malignant (1).*

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression achieved an accuracy of 96.49% and the highest AUC score of 0.9960 among all five models. It demonstrated excellent class discrimination and a strong balance between precision and recall. Its recall of 0.9286 indicates that it correctly identified a high proportion of malignant cases. |
| Decision Tree | Decision Tree achieved an accuracy of 92.98%, which was the lowest among the five models. Its AUC of 0.9246 was also the lowest. Although it provided reasonable classification performance, it was less effective than the other models on this dataset. |
| KNN | KNN achieved an accuracy of 95.61% and an AUC of 0.9823. It provided strong precision of 0.9744 but a comparatively lower recall of 0.9048. This indicates that it performed well overall but missed some malignant cases. |
| Naive Bayes | Gaussian Naive Bayes achieved a perfect precision score of 1.0000, meaning that all cases predicted as malignant were actually malignant in the test set. However, its recall of 0.8333 was the lowest among the five models, indicating that it missed a comparatively larger number of actual malignant cases. |
| Support Vector Machine | SVM achieved the highest accuracy of 97.37%, recall of 0.9524, F1-score of 0.9639, and MCC of 0.9433. It demonstrated the strongest overall performance and provided an effective balance between precision and recall. Its high recall is particularly important because identifying malignant cases correctly is a key consideration in this classification problem. |

### Overall Winner for the Dataset

**Support Vector Machine (SVM) is the overall winner for this dataset.**

SVM achieved the highest accuracy (97.37%), recall (95.24%), F1-score (96.39%), and Matthews Correlation Coefficient (0.9433) among all five evaluated models.

Although Logistic Regression achieved the highest AUC score of 0.9960, SVM demonstrated better overall classification performance across the other key metrics.

In this application, recall is particularly important because the positive class represents malignant tumours. A higher recall means that a greater proportion of actual malignant cases are correctly identified by the model.

Therefore, based on the combined evaluation metrics, SVM is selected as the best-performing model for this dataset.

## Streamlit Web Application

An interactive web application has been developed using Streamlit to demonstrate the trained classification models.

**Application features:**

- Upload a test dataset in CSV format.
- Select one of the five trained machine learning models.
- Evaluate the selected model on the uploaded test data.
- View Accuracy, AUC, Precision, Recall, F1-score, and MCC.
- View the confusion matrix and classification report.
- Compare the performance of all five models.

The application loads the previously trained and saved models. It does not retrain the models during evaluation.

**Note:** This project is developed for academic and educational purposes only. The predictions are not intended to replace professional medical diagnosis or clinical evaluation.
