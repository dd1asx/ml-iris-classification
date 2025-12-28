# Iris Flower Classification with K-Nearest Neighbors (KNN)

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

A basic machine learning project demonstrating classification of Iris flower species using the **K-Nearest Neighbors (KNN)** algorithm.

This is a classic beginner project in machine learning, implementing data loading, preprocessing, model training, and evaluation on the famous Iris dataset.

## Project Overview

- **Dataset**: Iris dataset (150 samples, 3 classes: Iris-setosa, Iris-versicolor, Iris-virginica)
- **Features**: Sepal length, sepal width, petal length, petal width (in cm)
- **Algorithm**: K-Nearest Neighbors (from scikit-learn)
- **Results**: Achieved **100% accuracy** on the test set (common with this well-separated dataset and optimal hyperparameters)

The KNN model classifies a flower by finding the "k" closest training examples (using Euclidean distance) and assigning the most common class among them.

## Dataset

The dataset is sourced from the UCI Machine Learning Repository:  
[https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)

**Citation**:
> Fisher, R. A. (1936). Iris [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/53/iris

The dataset contains 50 samples from each of three Iris species, with four numeric features and no missing values.

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
