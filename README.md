# Decision Tree Classification System

A machine learning application built with **Python and Streamlit** that demonstrates decision tree classification using ID3 and C4.5-style algorithms. The application allows users to upload their own CSV datasets, preprocess the data, train classification models, visualize decision trees, generate predictions, and compare model performance through evaluation metrics.

## Features

- **CSV Dataset Upload:** Upload custom CSV files for classification.
- **Dataset Preview:** View dataset records, column names, dimensions, and basic information.
- **Target Column Selection:** Choose the target attribute for classification.
- **Data Preprocessing:**
  - Handle missing values.
  - Convert categorical features into a suitable format.
  - Discretize numerical features using quantile-based binning.
- **ID3 Algorithm:** Build decision trees using Information Gain to select the best splitting attribute.
- **C4.5-Style Algorithm:** Build decision trees using Gain Ratio for attribute selection.
- **Tree Visualization:** Display trained decision trees using Graphviz.
- **Model Predictions:** Generate predictions for test dataset records.
- **Performance Evaluation:** Compare models using:
  - Accuracy
  - Precision
  - Recall
  - Weighted F1-score
  - Confusion Matrix
- **Prediction Export:** Download prediction results as a CSV file.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Data preprocessing and evaluation metrics |
| Graphviz | Decision tree visualization |

## Application Workflow

1. Upload a classification dataset in CSV format.
2. Preview the dataset and inspect its attributes.
3. Select the target column.
4. Preprocess the dataset by handling missing values and converting features.
5. Split the dataset into training and testing sets.
6. Train the ID3 and C4.5-style decision tree models.
7. Visualize the generated decision trees.
8. Evaluate model predictions using classification metrics.
9. Compare the performance of both algorithms.
10. Download the prediction results.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser. If it does not open automatically, visit:

```text
http://localhost:8501
```

## GitHub Repository Setup

If you are creating a new GitHub repository, use:

```bash
git init
git add .
git commit -m "Initial commit: Decision Tree Classification System"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

## Algorithm Implementation

### 1. ID3 (Iterative Dichotomiser 3)

ID3 is a decision tree algorithm that selects the best attribute for splitting the dataset using **Information Gain**.

It calculates the reduction in entropy after splitting the dataset based on an attribute. The attribute with the highest Information Gain is selected for the next split.

### 2. C4.5-Style Decision Tree

The C4.5-style implementation selects splitting attributes using **Gain Ratio**.

Gain Ratio adjusts Information Gain by considering the intrinsic information of a split. This helps reduce the bias toward attributes with many distinct values.

### Important Implementation Note

This project is an educational implementation of decision tree classification. It uses Information Gain for ID3 and Gain Ratio for the C4.5-style algorithm.

Numerical features are discretized into a maximum of three quantile-based bins before model training. The implementation does not cover every feature of the original C4.5 algorithm, including its complete continuous-attribute threshold selection, missing-value handling strategy, and pruning procedure.

## Project Objective

The main objective of this project is to understand and demonstrate how decision tree classification algorithms work, how different splitting criteria affect tree construction, and how their classification performance can be evaluated and compared using a practical dataset.

## Future Improvements

- Support continuous numerical splitting without prior discretization.
- Implement confidence-based pruning.
- Improve missing-value handling.
- Support additional classification algorithms.
- Add more interactive decision tree visualizations.
- Provide detailed model comparison reports.

---

**Developed as an educational Machine Learning project to explore decision tree classification, data preprocessing, model visualization, and performance evaluation.**

## Live Demo

Experience the application online:

**[Launch Decision Tree Classification System](https://decision-tree-classification-system-ecduomwhwfvw6qpebqgwxk.streamlit.app/)**

Upload a CSV dataset, train decision tree models, visualize their structure, and compare classification performance directly in your browser.
