# Midterm Exam: End-to-End Machine Learning Pipeline

**Student Identification**
* **Name:** FATTAH AHMAD RASYAD
* **Class:** TK-46-03
* **NIM:** 1103220215

---

## 📌 Repository Purpose
This repository serves as a submission for the Midterm Exam. It demonstrates the implementation of **End-to-End Machine Learning & Deep Learning Pipelines** across three distinct domains:
1.  **Classification:** Fraud Detection (Supervised Learning).
2.  **Regression:** Song Year Prediction (Supervised Learning).
3.  **Clustering:** Customer Segmentation (Unsupervised Learning).

The goal is to show proficiency in data preprocessing, feature engineering, model training, hyperparameter tuning, and model evaluation.

---

## 📂 Project Overview & Results

### 1. Fraud Detection (Binary Classification)
* **Objective:** Predict the probability of a transaction being fraudulent (`isFraud`) using transaction and identity data.
* **Key Challenges:** Handling highly imbalanced data (fraud cases are very rare).
* **Methodology:**
    * **Preprocessing:** Median imputation for missing values, Label Encoding for categorical features.
    * **Handling Imbalance:** Used `class_weight='balanced'` parameter to penalize misclassification of the minority class.
* **Models Used:** Random Forest Classifier.
* **Evaluation Metrics:**
    * **ROC-AUC Score:** Used as the primary metric to measure the model's ability to distinguish between classes.
    * **Confusion Matrix:** To visualize False Positives and False Negatives.

### 2. Song Year Prediction (Regression)
* **Objective:** Predict the release year of a song based on 90 audio timbre features.
* **Methodology:**
    * **Preprocessing:** Standardization (`StandardScaler`) was crucial due to the varying scales of audio features.
    * **Training:** Implemented a pipeline to handle large datasets efficiently.
* **Models Used:**
    * **Linear Regression:** Established as a baseline.
    * **Random Forest Regressor:** Main model to capture non-linear relationships in audio data.
* **Evaluation Metrics:**
    * **RMSE (Root Mean Squared Error):** Measures the average deviation of the predicted year from the actual year.
    * **R2 Score:** Indicates how well the features explain the variance in the target variable.

### 3. Customer Clustering (Unsupervised Learning)
* **Objective:** Segment credit card customers based on their usage behavior (Balance, Purchases, Payments, etc.) to identify distinct user groups.
* **Methodology:**
    * **Cleaning:** Removed non-numeric IDs (`CUST_ID`) and handled missing values.
    * **Dimensionality Reduction:** Used **PCA (Principal Component Analysis)** to visualize high-dimensional data in 2D.
* **Models Used:** K-Means Clustering.
* **Key Steps:**
    * **Elbow Method:** Used to determine the optimal number of clusters (*k*).
    * **Silhouette Score:** Used to evaluate the separation distance between resulting clusters.
* **Insights:** Identified groups such as "Big Spenders", "Frugal Users", and "Installment Payers".

---

## 💻 How to Navigate

The repository is structured as follows:

├── Task1_Fraud_Detection.ipynb   # Classification Task Code
├── Task2_Song_Regression.ipynb   # Regression Task Code
├── Task3_Customer_Clustering.ipynb # Clustering Task Code
├── submission.csv                # Output file for Task 1
├── README.md                     # Project Documentation
### How to Run the Notebooks
1.  **Open in Google Colab:** It is recommended to run these notebooks in Google Colab for access to free GPU/TPU resources.
2.  **Data Setup:**
    * Ensure the datasets (`train_transaction.csv`, `midterm-regresi-dataset.csv`, `clusteringmidterm.csv`) are located in your Google Drive or uploaded directly to the session.
    * Adjust the `BASE_PATH` variable in the "Load Data" cell to match your file location.
3.  **Execution:** Run the cells sequentially (Run All). The notebooks are self-contained and include:
    * Library Installation/Import
    * Data Loading
    * EDA & Preprocessing
    * Model Training
    * Evaluation & Visualization

---

**Note:**
* All models include a `verbose` parameter to monitor training progress.
* Deep Learning (Neural Networks) or Machine Learning approaches were selected based on the specific requirements of each task and dataset size.
