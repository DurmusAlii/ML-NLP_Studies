# Experiments with KNN, SVM, and Decision Trees

## Project Overview
This project explores the performance of three different machine learning classifiers—K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Decision Trees (DT)—on a dataset. The goal is to compare their classification accuracy and computational efficiency using cross-validation.

## Repository Structure
```
/Experiments_with_KNN_SVM_DT/
│-- Experiments_with_KNN_SVM_DT.ipynb  # Jupyter Notebook with code and analysis
│-- Hw1.pdf                             # PDF version of the Jupyter Notebook
│-- README.md                            # Project documentation (this file)
```

## Dataset
The dataset used in this project is the **Wisconsin Breast Cancer Diagnostic Dataset**. The features have been standardized using `StandardScaler` to improve the model performance.

## Implemented Methods
### 1. K-Nearest Neighbors (KNN)
- Custom implementation of KNN using Euclidean distance.
- Evaluated using 6-fold cross-validation.
- Confusion matrices and accuracy results analyzed for each fold.

### 2. Support Vector Machines (SVM)
- Utilized `sklearn`'s `SVC` for classification.
- Compared different kernel functions (linear, polynomial, and RBF).
- Evaluated accuracy and runtime performance.

### 3. Decision Trees (DT)
- Used `sklearn`'s `DecisionTreeClassifier`.
- Tuned hyperparameters such as tree depth.
- Compared performance with KNN and SVM.

## Results & Observations
- KNN achieved high accuracy but was computationally intensive for larger datasets.
- SVM provided strong generalization, especially with an RBF kernel.
- Decision Trees performed well but were prone to overfitting.

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/DurmusAlii/ML-NLP_Studies/Experiments_with_KNN_SVM_DT.git
   ```

## Dependencies
Ensure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `jupyter`

## Contact
For any questions, feel free to reach out via email at [d.alisucu@hotmail.com](mailto:d.alisucu@hotmail.com).
