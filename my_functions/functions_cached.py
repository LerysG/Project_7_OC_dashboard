# -*- coding: utf-8 -*-
import csv
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import plotly.express as px

COLOR_BR_r = ['dodgerblue', 'indianred']
COLOR_BR =['#EF553B', '#00CC96'] # ['#AB63FA', '#00CC96'] ['indianred', 'dodgerblue']
# Plotly colorscales: https://plotly.com/python/builtin-colorscales/

def reduce_memory_usage(df):  
    ''' 
    All types that we want to change for "lighter" ones.
    int8 and float16 are not include because we cannot reduce
    those data types.
    float32 is not include because float16 has too low precision.
     '''
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    for col, col_type in df.dtypes.iteritems():
        if col_type == "object":
            df[col] = df[col].astype("category")
        elif col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
    return df


@st.cache(allow_output_mutation=True)
def get_data(finename):
    df = pd.read_csv(finename,
                    sep=',',
                    low_memory=False,
                    verbose=False,
                    encoding='ISO-8859-1',
                    dtype={'Special': 'object'}
                    )
    df = reduce_memory_usage(df)
    return df


@st.cache(allow_output_mutation=True)
def heatmap(df, max_row):
    fig = px.imshow(df.head(max_row).isna(), color_continuous_scale='OrRd_r')
    fig.update(layout_coloraxis_showscale=True)
    fig.update_layout(coloraxis_colorbar=dict(
    title=" ",
    tickvals=[0,250],
    ticktext=['Recorded','Missing'],
))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig


@st.cache(allow_output_mutation=True)
def histogram(df, x='str', legend=True, plot_vline=False, vline=None):
    if x == "TARGET":
        fig = px.histogram(df,
                        x=x,
                        color="TARGET",
                        width=300,
                        height=200,
                        category_orders={"TARGET": [1, 0]},
                        color_discrete_sequence=COLOR_BR)
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=50))
    else:
        fig = px.histogram(df,
                x=x,
                color="TARGET",
                width=300,
                height=200,
                category_orders={"TARGET": [1, 0]},
                color_discrete_sequence=COLOR_BR,
                barmode="group",
                histnorm='percent')
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    if legend == True:
        fig.update_layout(legend=dict(yanchor="top",xanchor="right"))
    else:
        fig.update_layout(showlegend=False)
    if plot_vline == True:
        fig.add_vline(x=vline, line_width=3, line_dash="dash", line_color="black")
    return fig  


@st.cache(suppress_st_warning=True)
def preprocess(df_train, df_test):
    ''' 
    Here we apply preprocessing to X_train and X_test, 
    from their respectives dfs.
    /!\ num_feat must be fit on train and
    cat_feat needs to be fit on both train+test...
    /!\ there are 2 outputs X_train and X_test!!!
    '''
    st.write('Preprocessing data...')
    my_bar = st.progress(0)

    X_train = df_train.drop(columns=['SK_ID_CURR', 'TARGET'])
    X_test = df_test.drop(columns=['SK_ID_CURR'])
    #X_train, _ = X_train.align(X_test)
    X_test, _ = X_test.align(X_train)
    my_bar.progress(1) 

    for col in X_train.columns:
        if col.endswith("_MEDI") or col.endswith("_MODE"):
            X_train.drop(columns=col, inplace=True)
            X_test.drop(columns=col, inplace=True)
    my_bar.progress(8)

    X = pd.concat([X_train, X_test])
    my_bar.progress(20) 

    num_feat = X_train.select_dtypes(include=np.number).columns
    num_transfo = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    cat_feat = X.select_dtypes(exclude=np.number).columns
    cat_transfo = OneHotEncoder(handle_unknown='ignore')

    prep = ColumnTransformer(
        transformers=[('num', num_transfo, num_feat),
                    ('cat', cat_transfo, cat_feat)])
    my_bar.progress(40)

    prep.fit(X)
    my_bar.progress(60) 

    prep.named_transformers_['num'].fit(X_train[num_feat])
    my_bar.progress(70) 

    prep.named_transformers_['cat'].fit(X[cat_feat])
    my_bar.progress(83) 

    output_train, output_test =  prep.transform(X_train), prep.transform(X_test)
    my_bar.progress(100) 

    return output_train, output_test  


#######################################################################################
__name__ = "functions_cached"