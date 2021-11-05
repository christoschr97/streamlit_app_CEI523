import pandas as pd
import streamlit as st
import numpy as np

def get_data():
    url = "https://raw.githubusercontent.com/christoschr97/CEI523-online-retail-data/master/data.csv"
    return pd.read_csv(url, encoding="ISO-8859-1")
# pd.read_csv('/content/drive/MyDrive/PROJECT_CIS523/data/data.csv',)
df = get_data()

st.title("CEI523 - FINAL ASSIGNMENT")

st.markdown("Welcome to our project. The goal of this project is to user Retail Data to predict sales using customer segmentation")

st.markdown("Here we is a sample of the raw dataset used in our assignment")

st.dataframe(df[:10])

col1, col2 = st.columns(2)
col1.write('This is column1')
col2.write('This is column2')

