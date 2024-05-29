
# Rain prediction project Repository
This repository contains a data science project aimed at predicting whether it will rain tomorrow using 10 years of daily weather measurements across various weather stations in Australia. The project is structured into four main stages, each detailed in its respective Jupyter notebook.
The dataset is taken from https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package 

## Project Overview
The primary objective of this project is to predict whether it will rain tomorrow based on historical weather data. The project is divided into the following stages:

### Data Analysis
### Feature Engineering
### Models' Design and Testing
### Models' Comparison

## Data Analysis
The first stage involves exploring and analyzing the weather data. The notebook combines the data exploratory analysis and data treatment. The objectives are to understand the data distribution, identify missing values, value imputation as much as possible within reason and gain insights into the weather patterns. This stage is documented in the notebook: **[`Data analysis`](https://github.com/antoniocreal/Rain-prediction-project/blob/main/Data%20Analysis.ipynb)**.

- **Key tasks**: 
   - Data cleaning
   - Statistical analysis
   - Value imputation using Linear regressions, KNN imputer, conditional grouped means and interpolation
   - New features creation
   - Data visualization of weather patterns
     
## Feature Engineering
In this stage, the target variable classes are balanced by undersamling the class that has the majority of values and the most important features are chosen using feature selection algorithms. This process is detailed in the notebook: **[`Feature Engineering`](https://github.com/antoniocreal/Rain-prediction-project/blob/main/Feature%20Engineering.ipynb)**.

**Key tasks**:
   - Balancing target variable classes
   - Selecting the most important features

## Model Design and Testing
Here, various models are designed and tested to predict whether it will rain tomorrow. The models include ensemble methods, Random Forest and XGBoost, as well as neural networks. This stage is divided into two parts:

 - Models trained on the dataset from the data analysis stage
 - Models trained on the dataset from the feature engineering stage

**Key tasks**:
  - Model training
  - Hyperparameter tuning
  - Performance evaluation

The notebooks for this stage are:

[`Ensemble methods`](https://github.com/antoniocreal/Rain-prediction-project/blob/main/Tree%20methods.ipynb)
[`Neural networks full data`](https://github.com/antoniocreal/Rain-prediction-project/blob/main/Neural%20networks%20full%20data.ipynb)
[`Neural networks feature engineering data`](https://github.com/antoniocreal/Rain-prediction-project/blob/main/Neural%20networks%20feature%20engineering%20data.ipynb)

## Models Comparison
In the final stage, the performance of the models developed in the previous stage is compared. The comparison focuses on metrics such as accuracy, precision, recall, and F1-score. Special attention is given to the differences in performance between the models trained on raw data and those trained on engineered features and some conclusions are drawn.

**Key tasks**:
  - Model evaluation
  - Performance comparison

The notebook for this stage is: 
[`Models comparison`](https://github.com/antoniocreal/Rain-prediction-project/blob/main/Models%20comparison.ipynb) Models comparison.ipynb.

The neural networks, the random forest model and the xgboost models are all made availabe.
