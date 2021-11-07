import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve, KFold
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate 
from sklearn.neural_network import MLPClassifier
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
import os
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)



# Custom imports for pages
from multipage import MultiPage
import data_cleaning # the data cleaning file
import EDA # the EDA File
import dataset_overview # the dataset overview File


# Create an instance of the app 
app = MultiPage()

# # Title of the main page
# def get_data():
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     url = str(dir_path) + "/data.csv"
#     return pd.read_csv(url, encoding="ISO-8859-1")

# # Load the data utilizing the function get_data()
# df = get_data()

# Write a general title
st.title("CEI523 - FINAL ASSIGNMENT")
st.markdown("## Welcome to our project. The goal of this project is to user Retail Data to predict sales using customer segmentation")

st.markdown("### Here we is a sample of the raw dataset used in our assignment")

# st.dataframe(df[:10])

# Add all your applications (pages) here
app.add_page("Dataset Overview", dataset_overview.app)
app.add_page("Data Preparation", data_cleaning.app)
app.add_page("Exploratory Data Analysis", EDA.app)

# The main app
app.run()