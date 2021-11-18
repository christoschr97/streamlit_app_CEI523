import streamlit as st
import os

def app():
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.image("https://media.istockphoto.com/photos/online-shopping-concept-picture-id1135609382?k=20&m=1135609382&s=612x612&w=0&h=eEuuYY_b4rql38pXe4Bue2KVoDl8IZwLbbBlRbrZFeo=")


    with col3:
        st.write("")

    st.subheader('Final Project - CEI523 - Data Science')
    
    ## FALTA O CHECK ON GITHUB
    st.write("""
    #### Problem Domain
    Predict sales using Customer Segmentation\n
    Steps:
    - Getting Familiar with the Data (EDA)
    - Data Preparation: Dataset Overview & Data Cleaning
    - Data Preparation: Feature Engineering & Data Transformation
    - Data Modeling: Training and evaluationg of the first 10 Months & the Last 2 Months separately. We cannot train the model in the first 10 months and test it in the last 2 months because of seasonality.
    - Deployment and Communication 
    The prediction are made regarding to the Product Categories which a customer spends money.\n
    All the operations in the dataset were already done and stored as csv files inside the data directory.\n 
    The data are not cached, but they are modified, stored and reloaded\n
    If you want to check the code, go through the notebook directory in the [github repository](https://github.com/christoschr97/streamlit_app_CEI523).
    """)

    st.write("""
    **Note**\n
    Follow the steps on the left one by one in order to get results
    """)