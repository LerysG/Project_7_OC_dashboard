# Project 7 - Create a Dashboard - OpenClassrooms
by Lérys Granado, PhD. Sept. 2021

## General purposes.
This is an app for education purposes and answering to the closed Kaggle competition:
"Home Credit Default Risk" (https://www.kaggle.com/c/home-credit-default-risk)

This project was composed of two main objectives:
- Develop a scoring machine learning model to predict the solvency of clients of a bank-like company (i.e. probability of credit payment failure).
It is therefore a **binary classification issue**.
- Build an interactive dashboard allowing interpretations of these probabilities and improve the company's knowledge on its clients.

Dashboard specifications: 
- Scoring visualization  for a given client.
- Filter and visualize client's features.
- Compare client's features to other clients.

## Methods & Discussions.
### Materials.
Project was carried out locally, using JupyterNotebooks, Python 3.8 and necessary packages (especially: NumPy, Pandas, MatPlotLib, Seaborn, Scikit-Learn, LightGBM).

### Exploratoraty data analysis.
EDA was performed on Jupyter Notebook herein associated.  Implementation was custom-made but inspired from various Kaggle kernels. Only the main dataset 'application_train.csv' was considered. Data were partially answered (25% missing value). One major observation is that the two classes are severely imbalanced (class=0/class=1 : 10/1). 

### Preprocessing.
First of all, train data were split in a train set (80%) and a validation set (20%). Data were preprocessed using a simple imputer (median) and standard scaler for numeric features and one-hot-encoder for categorical features. No sampling or data creation were necessary with the selected machine learning model (see next section). A dimension reduction approach was tested but unsuccessful. 

### Model selection.
Neither Gradient Boosting Classifier (scikit-learn) nor XGBoost Classifier were enough performant to train model under reasonnable computation times. Only LightGBM Cassifier (Microsoft) provided satisfying performances (CPU time = few seconds-one minute). Note that LightGBM classifier allows for weighing of data. This was my strategy to handle imbalanced classes (sampling was tested but did not yield any improvements). Randomized search with cross-validation was perform to estimate best model hyperparameters (on train set, using 3 stratified folds). Only poor improvements of scores were observed throughout the search. AUC-ROC score plateaus off near the 69-77% range. 

### Prediction analysis.
ROC and precision-recall curves, confusion matrix and threshold optimization were subsequently performed on a validation set (splitted from application_train data).
Here's some observations:

- **ROC curve**: Score is only fair, improvements can be achieved (up to +5% AUC) using rather complex feature engineering (as shown by some in Kaggle). Yet, this is not within the framework of this project. 

- **G-means vs. threshold**: G-means is a scoring adapted to imbalanced data (comparing true and false positive rates). Finding maximum of the curve gives the output probability threshold above which sample is considerd positive. We note that this curve is not skewed, showing that the weighing approach to handle imbalanced classes was efficient during training. Probability threshold is near 50%.

- **Precision-recall curve and conf. matrix**: We see that recall is near 66%, false negative (FN) is actually worst than false positive (FP), but considering the number of false positive there is a balance to find according to business teams, i.e. net loss (FN) vs. shortfall (FP). 

<p align="center">
  <img width="300" height="300" src="https://github.com/LerysG/Project_7_OC_dashboard/blob/main/output.png?raw=true">
</p>

## Dashboard.
### Implementation.
The interactive dashboard was implemented using **Streamlit** framework and coded in Python. The application *app.py* comes with the following items:

-  Input data that are in the *input* folder. Note that for deployement purposes trainset was randomly sampled to reduce size (125 MB -> 25 MB). 

- A personnal package named *my_functions* containing a module *functions* for function definitions. Note that for app optimization it is of essence to cache the functions. 

- file: *optimized_model.sav*, containing the trained model. The package *pickle* was used to export the model after cross-validation and training. This file is not necessary even for the 1st utilization (if so, you must train model prior to predict).  One is free to better tune hyperparameters and overwrite the given model file.

### Outlines.
The app is composed of 4 indepenant sections:
- **🏠 Home**: presenting the project and packages.
- **👁️ Data, at glance**: showing data overview.
- **🔎 Further explore data**: plotting data distribution along target.
- **💪 Model training**: cross-validation with tunable hyperparameters with a possibility to export model.
- **📈 Test model prediction**: probability predictions for a given client taken from test set (client ID choosed by user) and showing comparison of client with some clients of train set.




