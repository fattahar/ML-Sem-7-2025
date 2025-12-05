# Midterm Exam: End-to-End Machine Learning Pipeline

**Student Identification**
* **Name:** FATTAH AHMAD RASYAD
* **Class:** TK-46-03
* **NIM:** 1103220215

---

## 📌 Repository Purpose
This repository serves as a submission for the Midterm Exam. It demonstrates the implementation of **End-to-End Machine Learning Pipelines** across three distinct domains:
1.  **Classification:** Fraud Detection (Supervised Learning).
2.  **Regression:** Song Year Prediction (Supervised Learning).
3.  **Clustering:** Customer Segmentation (Unsupervised Learning).

The project emphasizes robust data preprocessing, handling real-world data challenges (imbalance, high dimensionality, missing values), and justifying model parameter choices.

---

## 📂 Project Details & Technical Rationale

### 1. Fraud Detection (Binary Classification)
**Objective:** Predict the probability of a transaction being fraudulent (`isFraud`) using transaction and identity data.

#### 🛠️ Technical Implementation
* **Data Cleaning:**
    * **Imputation strategy:** Used **Median** for numerical columns. *Rationale:* Financial data often contains outliers (e.g., extremely high transaction amounts). Median is more robust to outliers compared to Mean.
    * **Categorical Encoding:** Used **Label Encoding**. *Rationale:* Tree-based models (like Random Forest) can handle ordinal integers well. We avoided One-Hot Encoding to prevent the "Curse of Dimensionality" as the dataset already has high cardinality.
* **Handling Class Imbalance:**
    * The dataset is highly imbalanced (~3% fraud).
    * **Solution:** Applied `class_weight='balanced'` in the model parameters. *Rationale:* This automatically adjusts weights inversely proportional to class frequencies, heavily penalizing the model if it misclassifies the minority class (Fraud), thus preventing the model from just predicting "Normal" for everything.

#### 🤖 Model & Parameters
* **Algorithm:** Random Forest Classifier.
* **Key Parameters:**
    * `n_estimators=50`: Reduced from default 100 to balance training speed on Google Colab with model performance.
    * `n_jobs=-1`: Utilized all available CPU cores for parallel processing.
* **Evaluation Metric:** **ROC-AUC Score**. Accuracy was disregarded because a model predicting "No Fraud" 100% of the time would still have 97% accuracy but 0% utility.

---

### 2. Song Year Prediction (Regression)
**Objective:** Predict the release year of a song based on 90 audio timbre features.

#### 🛠️ Technical Implementation
* **Feature Engineering:**
    * **Standardization (`StandardScaler`):** Applied to all features. *Rationale:* Audio timbre features come in varying scales. Without scaling, features with larger magnitudes would dominate the loss function, making convergence slower and less accurate.
* **Pipeline Strategy:**
    * **Baseline Model:** Implemented **Linear Regression** first to establish a baseline performance and check for linear relationships.
    * **Main Model:** Implemented **Random Forest Regressor** to capture complex, non-linear patterns in audio data.

#### 🤖 Model & Parameters
* **Algorithm:** Random Forest Regressor.
* **Key Parameters:**
    * `n_estimators=50`: Chosen to prevent timeout on the large dataset (~500k rows) while maintaining ensemble diversity.
    * `verbose=2`: Enabled to monitor tree building progress in real-time and ensure the training process was not hanging.
* **Evaluation Metric:** **RMSE (Root Mean Squared Error)**. We aimed to minimize the average deviation in years.

---

### 3. Customer Clustering (Unsupervised Learning)
**Objective:** Segment credit card customers based on usage behavior to identify distinct user groups.

#### 🛠️ Technical Implementation
* **Data Preprocessing:**
    * **Dropping ID:** Removed `CUST_ID`. *Rationale:* Unique identifiers carry no pattern information and would confuse distance-based algorithms.
    * **Scaling:** Applied `StandardScaler`. *Rationale:* Essential for K-Means. `BALANCE` (thousands) and `PRC_FULL_PAYMENT` (0-1) must be on the same scale, otherwise K-Means would only cluster based on Balance.
* **Dimensionality Reduction:**
    * **PCA (Principal Component Analysis):** Reduced 17 features to 2 principal components. *Rationale:* To enable 2D visualization of the clusters, allowing us to visually verify if the groups are well-separated.

#### 🤖 Model & Parameters
* **Algorithm:** K-Means Clustering.
* **Finding *k* (Elbow Method):**
    * Tested *k* range from 1 to 10.
    * **Result:** The "Elbow" point (where inertia decrease slows down) was observed at **k=4**.
* **Interpretation:**
    * Clusters were analyzed by calculating the mean of original features (Balance, Purchases, etc.) for each group to assign business labels (e.g., "Big Spenders", "Frugal Users").

---

## 💻 How to Navigate

The repository is structured as follows:

```text
├── midterm_transaction_data.ipynb    # Classification Task Code
├── midterm_regresi.ipynbb            # Regression Task Code
├── midterm_clustering.ipynb          # Clustering Task Code
├── README.md                         # Project Documentation
```
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
