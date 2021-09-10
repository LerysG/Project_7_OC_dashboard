# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_validate
from lightgbm import LGBMClassifier
import plotly.express as px
import pickle
from my_functions.functions_cached import * # personnal functions pkg and module
#######################################################################################
# To run this code, type in terminal at the file path: 
# streamlit run app3.py
#######################################################################################
# Setting layout
st.set_page_config(page_title="Lerys' Notebook",
                    page_icon="☮",
                    initial_sidebar_state="expanded")
sb = st.sidebar
sb.image('https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png' ,width=300)
sb.image('https://enetelsolutions.co.il/wp-content/uploads/2019/08/brain@3x.webp' ,width=300)

rad = sb.radio('', ['🏠 Home', 
'👁️ Data, at glance', 
'🔎 Further explore data', 
'💪 Model training',
'📈 Test model prediction'])

header = st.container()
dataset = st.container()
eda = st.container()
model_training = st.container()
model_predict = st.container()

#######################################################################################
# Stating graphical parameters
COLOR_BR_r = ['#00CC96', '#EF553B'] #['dodgerblue', 'indianred']
COLOR_BR = ['indianred', 'dodgerblue']

#######################################################################################
# Managing data import/export
PATH = "/app/project_7_oc_dashboard/"
PATH_INPUT = PATH+"input/"
FILENAME_TRAIN = PATH_INPUT+'application_train_sample.csv'
FILENAME_TEST = PATH_INPUT+'application_test.csv'
FILENAME_MODEL = PATH+'optimized_model.sav'

#######################################################################################
# Implementing containers
#######################################################################################

if rad == '🏠 Home':
    with header:
        a,z,e,r,t = st.columns(5)
        a.image('https://icon-icons.com/downloadimage.php?id=168039&root=2699/PNG/512/&file=python_vertical_logo_icon_168039.png', width=60)
        z.image('https://icon-icons.com/downloadimage.php?id=168071&root=2699/PNG/512/&file=numpy_logo_icon_168071.png', width=60)
        e.image('https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Pandas_mark.svg/360px-Pandas_mark.svg.png', width=50)
        r.image('https://github.com/scikit-learn/scikit-learn/raw/main/doc/logos/scikit-learn-logo-notext.png', width=100)
        t.image('https://upload.wikimedia.org/wikipedia/commons/3/37/Plotly-logo-01-square.png', width=170)

        st.title("Welcome to Lerys' Dashboard! \n ----")
        st.markdown("**Lérys Granado, OpenClassrooms Data Scientist, Project 7, Sept. 2021**")
        st.markdown("This project was composed of two main objectives:")
        st.markdown("- **Develop a scoring machine learning model** to predict the solvency of clients of a bank-like company (i.e. probability of credit payment failure). It is therefore a **binary classification issue**. Class 0 is solvent client whereas class 1 represents clients with payment difficulties.")
        st.markdown("- **Build an interactive dashboard** allowing interpretations of these probabilities and improve the company's knowledge on its clients.")

#######################################################################################

if rad == '👁️ Data, at glance':
    with dataset:
        st.header("**The data, at glance.** \n ----")
        st.markdown("In this project, we focus only on the application train dataset.")
        
        st.subheader("Here's the dataframe.")
        df = get_data(FILENAME_TRAIN)
        max_row = st.slider("Select at many row you wanna visualize", value=1000, min_value=1, max_value=len(df)) 
        st.write(df.head(max_row))
        
        st.subheader("Heatmap with missing data.")
        st.markdown('Showing records in dark, missing values in light. Numeric values will be subsequently imputed with median for model training.')

        st.plotly_chart(heatmap(df, max_row))

#######################################################################################

if rad == '🔎 Further explore data':
    df = get_data(FILENAME_TRAIN)
    with eda:
        st.header("**Overview of exploratory data analysis.** \n ----")
        st.subheader("Plotting distributions of target and some features.")      
        
        col1, col2, col3 = st.columns(3)
        col1.plotly_chart(histogram(df, x='TARGET'), use_container_width=True)
        col2.plotly_chart(histogram(df, x='CODE_GENDER'), use_container_width=True)
        col3.plotly_chart(histogram(df, x='EXT_SOURCE_1'), use_container_width=True)
        
        st.subheader("Let's plot some extra numerical features of your choice.")
        
        col1, col2, col3 = st.columns(3)
        num_col = df.select_dtypes(include=np.number).columns.sort_values()
        input1 = col1.selectbox('1st plot', num_col)
        input2 = col2.selectbox('2nd plot', num_col[1:])
        input3 = col3.selectbox('3rd plot', num_col[2:])

        st.subheader("Now, you may pick some categorical features to plot.")
        col4, col5, col6 = st.columns(3)
        cat_col = df.select_dtypes(exclude=np.number).columns.sort_values()
        input4 = col4.selectbox('1st plot', cat_col[1:])
        input5 = col5.selectbox('2nd plot', cat_col[2:])
        input6 = col6.selectbox('3rd plot', cat_col[3:])

        button = st.button('Plot it! ')
        if button:
            col1.plotly_chart(histogram(df, x=input1, legend=False),use_container_width=True)
            col2.plotly_chart(histogram(df, x=input2, legend=False),use_container_width=True)
            col3.plotly_chart(histogram(df, x=input3, legend=False),use_container_width=True)
            col4.plotly_chart(histogram(df, x=input4, legend=False),use_container_width=True)
            col5.plotly_chart(histogram(df, x=input5, legend=False),use_container_width=True)
            col6.plotly_chart(histogram(df, x=input6, legend=False),use_container_width=True)

#######################################################################################

if rad == '💪 Model training': 
    with model_training:
        st.header("**Model training.** \n ----")
        st.markdown("We'll be using LightGBM Classifier (Microsoft),\
            as state-of-the-art gradient boosting classifier.\
                \n You can tune  hyperparameters, fit and observe\
                cross-validation scores (using 3 folds).")

        _, col2, _ = st.columns(3)
        col2.image('https://raw.githubusercontent.com/microsoft/LightGBM/master/docs/logo/LightGBM_logo_black_text_tiny.png')
        
        df_train = get_data(FILENAME_TRAIN)
        df_test = get_data(FILENAME_TEST)
        X_train_sc, _ = preprocess(df_train, df_test)
        y_train = df_train['TARGET']
        
        col1, col2 = st.columns(2)
        col1.subheader("**Tuning best hyperparameters.**")
        n_estimators = col1.slider("Number of trees", value=300, min_value=200, max_value=1000)
        num_leaves = col1.slider("Number of leaves", value=10, min_value=5, max_value=100)
        lr = col1.select_slider("Learning rate", options=[1e-4, 1e-3, 1e-2, 1e-1, 1e0], value=1e-1)
        scale_pos_weight = col1.select_slider("Weight of positives (>10 highly recommanded)",\
            options=[1e-1, 1e0, 1e1, 2e1, 5e1, 1e2], value=1e1)
        reg_alpha = col1.slider("L1 regularization term", value=0, min_value=0, max_value=100)
        reg_lambda = col1.slider("L2 regularization term", value=0,  min_value=0, max_value=100)
        checkbox = col1.checkbox("Export optimized model 🥒🥒🥒")

        if col1.button('Fit using cross-validation!'):
            col2.subheader('**Validation set fit scores.**')
            st.spinner('Fitting...')
            model = LGBMClassifier(max_depth=-1,
                                    random_state=13,
                                    silent=True,
                                    metric='none',
                                    n_jobs=-1,
                                    n_estimators=n_estimators,
                                    num_leaves=num_leaves,
                                    learning_rate=lr,
                                    scale_pos_weight=scale_pos_weight,
                                    reg_alpha=reg_alpha,
                                    reg_lambda=reg_lambda
                                )

            scoring = ['roc_auc','precision','recall','f1']
            x_val = cross_validate(model, X_train_sc, y_train, cv=3, scoring=scoring)
            
            time, unk, auc, precision, recall, f1 = pd.DataFrame(x_val).mean(axis=0)
            d_time, d_unk, d_auc, d_precision, d_recall, d_f1 = pd.DataFrame(x_val).std(axis=0)

            col2.subheader('Mean fit time (s)')
            col2.write(f'{time:.0f} ± {d_time:.0f}')
            col2.subheader('AUC-score')
            col2.write(f'{auc:.0%} ± {d_auc:.0%}')
            col2.subheader('Recall')
            col2.write(f'{recall:.0%} ± {d_recall:.0%}')
            col2.subheader('Precision')
            col2.write(f'{precision:.0%} ± {d_precision:.0%}')
            col2.subheader('f1-score')
            col2.write(f'{f1:.0%} ± {d_f1:.0%}')

            if checkbox:
                model.fit(X_train_sc, y_train)
                filename = FILENAME_MODEL
                pickle.dump(model, open(filename, 'wb'))
                st.header('**Successful model export!**')
                st.balloons()

#######################################################################################

if rad == '📈 Test model prediction': 
    with model_predict:
        st.header("**Testset model prediction.** \n ----")

        df_train = get_data(FILENAME_TRAIN)
        df_test = get_data(FILENAME_TEST)

        np.random.seed(13)
        label_test = df_test['SK_ID_CURR'].sample(50).sort_values()
        col1, col2 = st.columns(2)
        col1.subheader("**Pick a test client to scout.**")

        radio = col1.radio('Select strategy:', ['Random client ID', 'Type client ID'])
        if radio == 'Random client ID':
            input_ = col1.selectbox('Select random client ID', label_test)
        if radio == 'Type client ID':
            input_ = int(col1.text_input('Type client ID', value=100001))


        if col1.button('Predict & plot!'):            
        
            X_train_sc, X_test_sc = preprocess(df_train, df_test)
            y_train = df_train['TARGET']
            
            model = pickle.load(open(FILENAME_MODEL, 'rb'))

            idx = df_test.SK_ID_CURR[df_test.SK_ID_CURR == input_].index
            y_prob = model.predict_proba(X_test_sc[idx, :])
            y_prob = [y_prob.flatten()[0], y_prob.flatten()[1]]
            
            col2.subheader(f"**Successful payment probability for client #{input_}.**")
            fig = px.pie(values=y_prob, names=[0,1], color_discrete_sequence=COLOR_BR_r, 
            width=270, height=270)
            fig.update_layout(margin=dict(l=70, r=0, t=0, b=70))
            fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
            col2.plotly_chart(fig, use_container_width=True)


        st.subheader(f"**Let's compare client (as ----) to other ones from trainset.**")

        col1, col2 = st.columns(2)
        col1.markdown("Pick one numerical feature.")
        num_col = df_train.select_dtypes(include=np.number).columns.sort_values()
        input_num = col1.selectbox('Num plot', num_col)

        col2.markdown("Pick one categorical feature.")
        cat_col = df_train.select_dtypes(exclude=np.number).columns.sort_values()
        input_cat = col2.selectbox('Cat plot', cat_col)

        client = df_test[df_test.SK_ID_CURR == input_]
        vline_num = client[input_num].to_numpy()[0]
        vline_cat = client[input_cat].to_numpy()[0]

        df = df_train.copy()

        col1.plotly_chart(histogram(df, x=input_num, legend=False, plot_vline=True, vline=vline_num),use_container_width=True)
        col2.plotly_chart(histogram(df, x=input_cat, legend=False, plot_vline=True, vline=vline_cat),use_container_width=True)
        
        st.subheader("More information about this client.")
        col1, col2 = st.columns(2)
        info = col1.selectbox('What info?', client.columns.sort_values())     
        info_print = round(client[info].to_numpy()[0],1)

        col1.subheader(info_print)
        col2.write("All client's data.")
        col2.write(client)

#######################################################################################
if __name__ == "__main__":
    print("Script runned directly")
else:
    print("Script called by other")