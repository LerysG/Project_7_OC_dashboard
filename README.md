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

## Implementation and Methods.
### Materials
Project was carried out locally, using JupyterNotebooks, Python 3.8 and necessary packages (especially: NumPy, Pandas, MatPlotLib, Seaborn, Scikit-Learn, LightGBM).

### Exploratoraty data analysis (EDA)
EDA was performed on Jupyter Notebook herein associated.  Implementation was custom-made but inspired from various Kaggle kernels. Only the main dataset 'application_train.csv' was considered. Data were partially answered (25% missing value). One major observation is that the two classes are severely imbalanced (class=0/class=1 : 10/1). 

### Preprocessing.
First of all, train data were split in a train set (80%) and a validation set (20%). Data were preprocessed using a simple imputer (median) and standard scaler for numeric features and one-hot-encoder for categorical features. No sampling or data creation were necessary with the selected machine learning model (see next section). A dimension reduction approach was tested but unsuccessful. 

### Model selection.
Neither Gradient Boosting Classifier (scikit-learn) nor XGBoost Classifier were enough performant to train model under reasonnable computation times. Only LightGBM Cassifier (Microsoft) provided satisfying performances (CPU time = few seconds-one minute). Note that LightGBM classifier allows for weighing of data. This was my strategy to handle imbalanced classes (sampling was tested but did not yield any improvements). Randomized search with cross-validation was perform to estimate best model hyperparameters (on train set, using 3 stratified folds). Only poor improvements of scores were observed throughout the search. AUC-ROC score plateaus off near the 69-77% range. 

### Prediction analysis.
ROC and precision-recall curves, confusion matrix and threshold optimization were subsequently performed on a validation set (splitted from application_train data).


<div>
<img src="https://github.com/LerysG/Project_7_OC_dashboard/blob/main/output.png?raw=true" width="500"/>
</div>


![alt text](https://github.com/LerysG/Project_7_OC_dashboard/blob/main/output.png?raw=true)




## 

