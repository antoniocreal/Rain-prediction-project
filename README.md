
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
The first stage involves exploring and analyzing the weather data. The notebook combines the data exploratory analysis and data treatment. The objectives are to understand the data distribution, identify missing values, value imputation as much as possible within reason and gain insights into the weather patterns. This stage is documented in the notebook: **Data_Analysis.ipynb**.

- **Key tasks**: 
   - Data cleaning
   - Statistical analysis
   - Value imputation using Linear regressions, KNN imputer, conditional grouped means and interpolation
   - New features creation
   - Data visualization of weather patterns
     
## Feature Engineering
In this stage, the target variable classes are balanced by undersamling the class that has the majority of values and the most important features are chosen using feature selection algorithms. This process is detailed in the notebook: **Feature_Engineering.ipynb**.

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

Tree methods.ipynb
Neural networks full data.ipynb
Neural networks feature engineering data.ipynb

## Models Comparison
In the final stage, the performance of the models developed in the previous stage is compared. The comparison focuses on metrics such as accuracy, precision, recall, and F1-score. Special attention is given to the differences in performance between the models trained on raw data and those trained on engineered features and some conclusions are drawn.

**Key tasks**:
  - Model evaluation
  - Performance comparison

The notebook for this stage is: Models comparison.ipynb.





The theoretical notes are also available, made mostly through LLM's and some personal notes on [`Notes`](https://github.com/antoniocreal/Evolutionary_algorithms/blob/main/Notes.odt)

## Folder Structure
- **Genetic Algoritm**: This folder contains implementations of Genetic Algorithms.

  - **Sum of vectors fitness function**:
    - [`Fitness_function_vectors_sum.py`](https://github.com/antoniocreal/Evolutionary_algorithms/blob/main/Genetic%20Algorithm/Fitness_function_vectors_sum.py): Implementation of Genetic Algorithm for optimizing the sum of vectors fitness function in a sequential manner.
  
  - **Match string case fitness function**:
      - [`Match_String_Case.py`](https://github.com/antoniocreal/Evolutionary_algorithms/blob/main/Genetic%20Algorithm/Match_String_Case.py): Sequential implementation of Genetic Algorithm for string matching.
      -  [`Match_String_Case_OOP_format.py`](https://github.com/antoniocreal/Evolutionary_algorithms/blob/main/Genetic%20Algorithm/Match_String_Case_OOP_format.py); [`Match_String_Case_OOP_format_2.py`](https://github.com/antoniocreal/Evolutionary_algorithms/blob/main/Genetic%20Algorithm/Match_String_Case_OOP_format_2.py): 2 different Object-oriented implementation of Genetic Algorithm for string matching.
        
  - **Complex optimization function**:
      - [`Complex_Optimization_equation.py`](https://github.com/antoniocreal/Evolutionary_algorithms/blob/main/Genetic%20Algorithm/Complex_Optimization_equation.py): Sequential implementation of Genetic Algorithm for a complex optimization function

- **Ant Colony Optimization**: Here, you'll find implementations of Ant Colony Optimization.

  - **Travelling salesman**:
    - [`TravellingSalesman_ACO.py`](https://github.com/antoniocreal/Evolutionary_algorithms/blob/main/Genetic%20Algorithm/TravellingSalesman_ACO.py): Sequential implementation of Ant Colony Optimization for solving the Travelling Salesman Problem.
    - [`TravellingSalesman_ACO_OOP_format.py`](https://github.com/antoniocreal/Evolutionary_algorithms/blob/main/Genetic%20Algorithm/TravellingSalesman_ACO_OOP_format.py): Object-oriented implementation of Ant Colony Optimization for solving the Travelling Salesman Problem.

- **Particle Swarm Optimizagion**: This folder contains an implementation of Particle Swarm Optimization.
  - **Optimization function**:
    - [`PSO_Simple_fitness_function.py`](https://github.com/antoniocreal/Evolutionary_algorithms/blob/main/Genetic%20Algorithm/PSO_Simple_fitness_function.py) : Implementation of Particle Swarm Optimization in a sequential manner.
