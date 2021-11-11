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

def get_data():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    url = str(dir_path) + "/data/data_cleaned.csv"
    return pd.read_csv(url, encoding="ISO-8859-1")

def app():
    st.title("Data Preparation: Feature Engineering")
    df_cleaned = get_data()

    st.markdown("""
        ### Data Preparation section consits of some steps such as:
        * Feature Engineering
    """)

    st.dataframe(df_cleaned.describe())

    st.write("""
    Lets create a dataframe CartPrice which will contain the data for each transaction (group by invoice number)
    """)

    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['FullPrice'].sum()
    cart_price = temp.rename(columns = {'FullPrice':'Cart Price'})

    st.dataframe(cart_price)
    
    