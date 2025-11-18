Customer Conversion Analysis using Clickstream Data
📌 Project Overview

This project analyzes clickstream data from an e-commerce platform to understand customer behavior, conversion patterns, and factors influencing purchases.
We apply supervised (classification & regression) and unsupervised (clustering) learning models to derive insights.
🛠️ Technologies Used

Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

Machine Learning Models:

Regression: Linear, Random forest, Gradient Boosting

Classification: Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Network

Clustering: KMeans, DBSCAN, Agglomerative

Streamlit → Interactive app for model results
Workflow

Data Preprocessing

Handle nulls, outliers, and data types

Encode categorical variables

Feature scaling & engineering

Exploratory Data Analysis (EDA)

Customer journey patterns

Conversion behavior by product category & price

Correlation insights

Model Development

Regression for predicting price

Classification for predicting conversion

Clustering for customer segmentation

Model Evaluation

Accuracy, Precision, Recall, F1, ROC-AUC (Classification)

RMSE, R² (Regression)

Silhouette Score, Davies-Bouldin Index (Clustering)

Results

Random Forest gave best performance for regression (price prediction).

Gradient Boosting performed best for classification (conversion prediction).

KMeans Clustering achieved good separation with Silhouette Score ~0.68.

Deployment

Streamlit app to visualize model outputs & insights
