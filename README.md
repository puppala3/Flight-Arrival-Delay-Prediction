# Flight_Arrival_Delay_Prediction
Project Overview
This project aims to predict whether a flight will experience an arrival delay of more than 15 minutes using machine learning techniques. We utilized a large dataset of flight information and implemented two models: Logistic Regression and Linear Support Vector Machine (SVM) using Apache Spark's MLlib.
Dataset
The dataset, named "connected_flights_2018", was obtained from Kaggle. It contains 61 columns and is approximately 1.8 GB in size, making it suitable for big data analysis.
Methodology
Data Preprocessing:
Selected relevant features including airline, departure time, arrival time, origin state, destination state, and others.
Converted the target variable (ArrDel15) to binary (1 for delay > 15 minutes, 0 otherwise).
Removed rows with null values.
Performed feature engineering: indexing and one-hot encoding for categorical variables.
Exploratory Data Analysis:
Analyzed the distribution of delays across months and days.
Identified class imbalance in the target variable.
Model Implementation:
Implemented Logistic Regression and Linear SVM models.
Used pipeline for streamlined data processing and model training.
Performed hyperparameter tuning using CrossValidator and ParamGridBuilder.
Evaluation:
Split data into 80% training and 20% testing sets.
Evaluated models using metrics such as AUC, Accuracy, Precision, Recall, and F1 Score.
Created confusion matrices to visualize model performance.
Results
Both models performed well, with Linear SVM slightly outperforming Logistic Regression:
Metric	Logistic Regression	Linear SVM
Accuracy	90.98%	83.01%
AUC	91.59%	92.07%
Precision	90.08%	82.58%
Recall	99.78%	99.99%
F1 Score	89.91%	77.23%
Conclusion
The Linear SVM model showed slightly better performance in terms of AUC and Recall, making it marginally preferable for this flight delay prediction task. However, both models demonstrated strong predictive capabilities.
Technologies Used
Apache Spark
PySpark MLlib
Python
Jupyter Notebook (for exploratory data analysis)
