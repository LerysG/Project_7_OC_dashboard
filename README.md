# Project 7 - Create a Dashboard - OpenClassrooms
by Lérys Granado, PhD. Sept. 2021

## General purposes.
This is an app for education purposes and answering to the closed Kaggle competition:
"Home Credit Default Risk" (https://www.kaggle.com/c/home-credit-default-risk)

This project was composed of several objectives:
- Develop a scoring machine learning model to predict the solvency of clients of a bank-like company (i.e. probability of credit payment failure).
It is therefore a **binary classification issue**.
- Build an interactive dashboard allowing interpretations of these probabilities and improve the company's knowledge on its clients.

Dashboard specifications: 
- Scoring visualization  for a given client.
- Filter and visualize client's features.
- Compare client's features to other clients.

## Implementation and methods.
### Pretests: exploratoraty data analysis and machine learning.
EDA were performed on Jupyter Notebook herein associated.  Implementation was custom-made but inspired from various Kaggle kernels. Only the main dataset 'application_train.csv' was considered. Data were partially answered (25% missing value)
One major observation is that the two classes are severely imbalanced (class=0/class=1 : 10/1). Common gradient boosting such as the one of scikit-learn, as well as XGBoost were found to slow.

### Machine Learning im

