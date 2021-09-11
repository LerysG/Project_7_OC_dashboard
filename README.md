# Project 7 - Create a Dashboard - OpenClassrooms
by Lérys Granado, PhD. Sept. 2021

## Run the dashboard app.
Go to: https://share.streamlit.io/lerysg/project_7_oc_dashboard/main/app.py

## General purposes.
This repo contains a dashboard app for education purposes and answering to the closed Kaggle competition:
"Home Credit Default Risk" (https://www.kaggle.com/c/home-credit-default-risk)

This project was composed of two main objectives:
- Develop a scoring machine learning model to predict the solvency of clients of a bank-like company (i.e. probability of credit payment failure).
It is therefore a **binary classification issue**. Class 0 is solvent client whereas class 1 represents clients with payment difficulties.
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
First of all, raw train data were split in a train set (80%) and a validation set (20%). Data were preprocessed using a simple imputer (median) and standard scaler for numeric features and one-hot-encoder for categorical features. No sampling or data creation were necessary with the selected machine learning model (see next section). A dimension reduction approach was tested but unsuccessful. 

### Model selection.
Neither Gradient Boosting Classifier (scikit-learn) nor XGBoost Classifier were enough performant to train model under reasonnable computation times. Only LightGBM Classifier (Microsoft) provided satisfying performances. Note that LightGBM classifier allows for weighing of data. This was my strategy to handle imbalanced classes (sampling was tested but did not yield any further improvements). Positives were weighed by approx. 10 (as their count represents 10% of the negative count). Randomized search with cross-validation was performed to estimate best model hyperparameters (on train set, using 3 stratified folds). Only poor improvements of scores were observed throughout the search. AUC-ROC score plateaus off near the 69-77% range. 

**Comments on the LightGBM Classifier model**:
- Type: decision tree classifier.
- Principle: LGBM grows trees at the leaf-level (not from all leaves like other GBs)
- Optimization: gradient boosting, histogram-based algorithms.
- Loss function: LogLoss.
- Scoring (here): AUC-ROC.
- Typical fit time: few sec-1 min (for 300k samples, 200 features).
- Interesting for: high performances, class weighing, categorical and sparse data (i.e., one-hot-encoded), accepts NaNs.

<p align="center">
  <img width="350" src="https://lightgbm.readthedocs.io/en/latest/_images/leaf-wise.png?raw=true">
</p>
https://lightgbm.readthedocs.io/en/latest/Features.html

### Prediction analysis.
ROC and precision-recall curves, confusion matrix and threshold optimization were subsequently performed on a validation set (splitted from application_train data).
Here's some observations:

- **ROC curve**: Score is only fair, improvements can be achieved (up to +5% AUC) using rather complex feature engineering (as shown by some in Kaggle). Yet, this is not within the framework of this project. 

- **G-means vs. threshold**: G-means is a scoring adapted to imbalanced data (comparing true and false positive rates). Finding maximum of the curve gives the output probability threshold above which sample is considerd positive. We note that this curve is not skewed, showing that the weighing approach to handle imbalanced classes was efficient during training. Probability threshold is near 50%.

- **Precision-recall curve and conf. matrix**: We see that recall is near 66%, false negative (FN) is actually worst than false positive (FP), but considering the number of false positive there is a balance to find according to business teams, i.e. net loss (FN) vs. shortfall (FP). 

In addition, feature importance are displayed in JupyterNotebook. 

<p align="center">
  <img width="350" height="350" src="https://github.com/LerysG/Project_7_OC_dashboard/blob/main/output.png?raw=true">
</p>

## Dashboard.
### Implementation.
This interactive dashboard was implemented using **Streamlit** framework and coded in Python. The packages needed to run this app are listed in file *requirements.txt*. The main application *app.py* comes with the following items:

- Input data that are in the *input* folder (sampled-train and test raw datasets). Note that for deployement purposes trainset was randomly sampled to reduce size (125 MB -> 25 MB). 

- A personnal package named *my_functions* containing a module *functions* for function definitions. Note that for app optimization it is of essence to cache the functions. 

- The file: *optimized_model.sav*, containing the trained model. The package *pickle* was used to export the model after cross-validation and training. This file is not necessary even for the 1st utilization (if so, you must train model prior to predict).  One is free to better tune hyperparameters and overwrite the given model file.

### Outlines.
The app will change drastically depending on who consults it:
- **👨‍⚕️ Data Scientist** being data-oriented, with:
  - **🏠 Home**: presenting the project and packages.
  - **👁️ Data, at glance**: showing data overview.
  - **🔎 Further explore data**: plotting data distribution along with target.
  - **💪 Model training**: cross-validation with tunable hyperparameters with a possibility to export model.

- **🤵 Bank Clerk** being client-oriented, allowing selection of a client to scout directly from the sidebar, with:
  - **🏠 Home**: presenting the project and packages.
  - **🔎 Client data**: showing client's data overview.
  - **📉 Client prediction**: predicting client's probability of being positive or negative, along with the most important features that lead to this decision.

### Comments on this app.
- It was very easily implemented since coding was only done in Python (no HTML nor JS required).
- It is quite slow since every change make reload the page, so thanks for being patient. The only reported workaround is to cache function, which has been done.




