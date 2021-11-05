import streamlit as st
import pandas as pd

# Custom imports for pages
from multipage import MultiPage
import data_cleaning
import EDA

# Create an instance of the app 
app = MultiPage()

# Title of the main page
def get_data():
    url = "https://raw.githubusercontent.com/christoschr97/CEI523-online-retail-data/master/data.csv"
    return pd.read_csv(url, encoding="ISO-8859-1")
# pd.read_csv('/content/drive/MyDrive/PROJECT_CIS523/data/data.csv',)
df = get_data()

st.title("CEI523 - FINAL ASSIGNMENT")

st.markdown("Welcome to our project. The goal of this project is to user Retail Data to predict sales using customer segmentation")

st.markdown("Here we is a sample of the raw dataset used in our assignment")

# st.dataframe(df[:10])

# Add all your applications (pages) here
app.add_page("Exploratory Data Analysis", EDA.app)
app.add_page("Data Cleaning", data_cleaning.app)

# The main app
app.run()