import pandas as pd
import streamlit as st
import numpy as np
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


# def get_data():
#     url = "https://raw.githubusercontent.com/christoschr97/CEI523-online-retail-data/master/data.csv"
#     return pd.read_csv(url, encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceID': str})
# # pd.read_csv('/content/drive/MyDrive/PROJECT_CIS523/data/data.csv',)

def get_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    url = str(dir_path) + "/data.csv"
    return pd.read_csv(url, encoding="ISO-8859-1")
df = get_data()


st.title("CEI523 - FINAL ASSIGNMENT")

st.markdown("### Welcome to our project. The goal of this project is to user Retail Data to predict sales using customer segmentation")

st.markdown("#### Here we is a sample of the raw dataset used in our assignment")

st.dataframe(df[:10])

# Add a selectbox to the sidebar (dummy for now):
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Data Preparation', 'EDA', 'Machine Learning')
)

st.title("Dataset Overview")

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


### DATA PREPARATION SECTION ###
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

st.write(" ")
st.write("By utilizing the code below we can drop the rows where the customer is NaN and thus we cannot utilize them in our project")

st.code("df.dropna(axis = 0, subset = ['CustomerID'], inplace = True")
df.dropna(axis = 0, subset = ['CustomerID'], inplace = True)

st.write("Now we have cleared the missing values of Customer ID lets see again whats happening")

columns_info=pd.DataFrame(df.dtypes.astype(str)).T.rename(index={0:'Column Type'})
columns_info=columns_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'Null Values (NB)'}).astype(str))
columns_info=columns_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.rename(index={0:'Null Values (%)'}).astype(str))
st.write(columns_info)





