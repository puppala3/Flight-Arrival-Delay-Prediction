# Flight_Arrival_Delay_Prediction
This project aims to predict whether a flight will experience an arrival delay of more than 15 minutes using machine learning techniques: Logistic Regression and Linear Support Vector Machine (SVM) using Apache Spark's MLlib.It was implemented and run on Google Cloud Platform.
# Project Overview
The project consists of three main components:
- Exploratory Data Analysis (EDA)
- Linear Support Vector Machine (SVM) model
- Logistic Regression model
# Features
- Data preprocessing and feature engineering
- Implementation of Linear SVM and Logistic Regression models
- Hyperparameter tuning using CrossValidator
- Model evaluation using various metrics (AUC, Accuracy, Precision, Recall, F1 Score)
# Dataset
The dataset, named "connected_flights_2018", was obtained from Kaggle. It contains 61 columns and is approximately 1.8 GB in size, making it suitable for big data analysis.
# Models
Both models use a pipeline that includes:
- String indexing and one-hot encoding for categorical variables
- Feature assembly and standardization
- Cross-validation for hyperparameter tuning
# Evaluation
Models are evaluated using
- Area Under ROC Curve (AUC)
- Accuracy, Precision, Recall, and F1 Score
- Confusion Matrix
# Results
Both models performed well, with Linear SVM slightly outperforming Logistic Regression:
- Logistic Regression achieved an AUC of 91.59% and a Recall of 99.78%.
- Linear SVM achieved an AUC of 92.07% and a Recall of 99.99%.
# Conclusion
The Linear SVM model showed slightly better performance in terms of AUC and Recall, making it marginally preferable for this flight delay prediction task. However, both models demonstrated strong predictive capabilities.

# Technologies Used
- Apache Spark
- PySpark MLlib
- Python
- Jupyter Notebook (for exploratory data analysis)
