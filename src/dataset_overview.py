import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib as plt

# Title of the main page
def get_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    url = str(dir_path) + "/data.csv"
    return pd.read_csv(url, encoding="ISO-8859-1")

def app():
  st.markdown("### Dataset Overview")
  df = get_data()
  
  st.write("The DataFrame shape is: {}".format(df.shape))

  st.write("Now lets see the datatypes of each column and the null values of it: ")
  columns_info = pd.DataFrame(df.dtypes.astype(str)).T.rename(index={0: 'Column Type'})
  st.dataframe(columns_info)

  st.write("Empty Values on each volumn: ")
  st.dataframe(pd.DataFrame(df.isna().sum(), columns=['NaN']).transpose())

  st.write("Percentage of Empty Values:")
  df_nan_perc = pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0:'Null Values (%)'})
  st.dataframe(df_nan_perc)

  st.markdown("**There are null values in the columns Description and CustomerID, these null values represent a ~0.26% and ~25% respectively.**")


  



