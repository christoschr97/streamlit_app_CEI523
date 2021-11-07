import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd


def app():
    st.title("Data Preparation Section")
    st.markdown("""
        ### Data Cleaning Section
        Here we will briefly explain what we did and how we did it in terms of data cleansing

        Now we know that almost 25% of the transactions are not assigned to a particular client and arround 0.27% of the transactions descriptions are not specified, 
        there are several ways to deal with missing values:

        Impute values for the CustomerID and Description, in this case it is impossible, does not make sense.
        Apply clustering analysis and see patterns in those unknown clients and unknown description of products. Once these patterns are detected we can assign a labels for them and use these labels as a generic CustomerID and Description.
        Delete the rows where these missing values are found.
    """)

    st.write("===")
    st.write("By utilizing the code below we can drop the rows where the customer is NaN and thus we cannot utilize them in our project")

    st.code("df.dropna(axis = 0, subset = ['CustomerID'], inplace = True")