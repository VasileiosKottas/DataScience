# README: Facial Expression Recognition and Clustering

## Overview
This project focuses on classifying and clustering facial expressions using the Cohn-Kanade dataset. The key objectives are:
- **Classification**: Predicting facial expressions using a supervised neural network.
- **Clustering**: Grouping expressions with KMeans and evaluating clustering performance.

The project involves feature selection, dimensionality reduction, and evaluation of classification and clustering models.

---

## Workflow
### 1. Data Preparation
- The dataset is loaded from `cohn-kanade-rev_new.xls`.
- Features are selected using `SelectKBest` to retain the most relevant attributes.
- Rows with missing values (`NaN`) are removed to ensure clean data.
- Data is split into training and testing sets (80:20 ratio).
- Features are standardized to have zero mean and unit variance.

### 2. Classification
- A neural network is implemented using PyTorch with the following structure:
  - **Input Layer**: Matches the number of selected features.
  - **Hidden Layer**: Fully connected with ReLU activation (64 nodes).
  - **Output Layer**: Matches the number of expression categories.
- The model is trained for 50 epochs using the Adam optimizer and cross-entropy loss.
- Evaluation metrics include classification accuracy and a confusion matrix.

### 3. Clustering
- **KMeans** is applied to the dataset (both original features and PCA-reduced dimensions).
- The quality of clustering is evaluated using clustering accuracy (against true labels) and the silhouette score.

### 4. Visualization
- Results are visualized through:
  - Confusion matrix for classification performance.
  - Scatter plots for clustering results and true labels.
  - PCA-based visualizations for dimensionality reduction.

---

## File Descriptions
1. **`facial_expression_recognition.py`**: Main script containing the complete implementation.
2. **`cohn-kanade-rev_new.xls`**: Input dataset.
3. **Saved Plots**:
   - `confusion_matrix.png`: Visualization of classification performance.
   - `kmeans_clustering_original.png`: Scatter plot of KMeans clustering on original features.
   - `true_labels.png`: Scatter plot of true labels in PCA-reduced space.

---

## Installation
### Prerequisites
- Python 3.8 or above
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - tqdm
  - torch

### Installation Steps
1. Create an env and activate it (Windows):
   ```bash
   python -m venv .venv
   .venv/Scripts/activate
   cd Face expression recognition
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset file (`cohn-kanade-rev_new.xls`) in the project directory, if not already in.

4. Run the script:
   ```bash
   python facial_expression_recognition.py
   ```

---

## Results
1. **Classification Accuracy**: Achieved [92.86]% accuracy in predicting facial expressions.
2. **Clustering Accuracy**: Achieved [17.14]% accuracy using KMeans.
3. **Silhouette Score**: Evaluated cluster quality with a score of [0.32].
4. **Visualizations**:
   - Confusion matrix and scatter plots provide insights into model performance and clustering patterns.

---

