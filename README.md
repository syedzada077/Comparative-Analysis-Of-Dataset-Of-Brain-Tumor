# Brain Tumor Classification with Machine Learning

## Overview

This project demonstrates a comparative analysis of various machine learning algorithms for classifying brain tumors using MRI images. The dataset used for this project is sourced from Kaggle, and the models include K-Nearest Neighbors (KNN), Decision Trees, Support Vector Machines (SVM), Logistic Regression, Artificial Neural Networks (ANN), and others.

## Dataset

The dataset used in this project is the [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) dataset from Kaggle. It contains MRI images categorized into four classes: No Tumor, Glioma Tumor, Meningioma Tumor, and Pituitary Tumor.

### Dataset Structure

- **Training Set**: MRI images used to train the models.
- **Testing Set**: MRI images used to evaluate the models.

## Project Structure

- `brain_tumor_classification.ipynb`: Jupyter Notebook containing the entire workflow including data loading, preprocessing, model training, and evaluation.
- `brain_tumor_classification.py`: Python script with the same functionality as the notebook for running the project in a non-interactive environment.

## Models Implemented

The following machine learning models were implemented and evaluated:

- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Artificial Neural Network (ANN)**

## Total Images
Total Images of this dataset are **3265**
- **Total-Affected**:2765
- **Total-Not-Affected**:500
- **Total-Train**:2,870 (88%)
- **Total-Test**:395 (12%)

## Evaluation Metrics

Each model was evaluated based on the following metrics:

- **Accuracy**: Percentage of correct predictions.
- **Precision**: Precision of the positive class predictions.
- **Recall**: Recall of the positive class predictions.
- **F1 Score**: Harmonic mean of precision and recall.

## Usage

To run this project, follow these steps:

### Prerequisites

- Python 3.x
- Jupyter Notebook (if running `.ipynb`)
- Required libraries: `scikit-learn`, `numpy`, `matplotlib`, `pandas`, `Pillow`


## Results

The table below summarizes the performance of different models:

| Algorithm             | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
|-----------------------|--------------|---------------|------------|--------------|
| Decision Tree         | 69.03        | 74.40         | 69.03      | 64.01        |
| K-Nearest Neighbors   | 77.918        | 80.29         | 77.91      | 73.97        |
| Logistic Regression   | 73.35        | 77.59         | 73.35      | 69.22        |
| Support Vector Machine| 66.24        | 71.66         | 66.24      | 62.39        |
| Artificial Neural Network | 76.14     | 82.16         | 76.14      | 72.14        |

## Contributing

Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.



## Acknowledgements

- The dataset is sourced from Kaggle: [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- Thanks to all contributors and the open-source community for making this project possible.

