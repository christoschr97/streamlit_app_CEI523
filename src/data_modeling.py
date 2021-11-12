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


def get_x_y_train():
    X = pd.read_csv('./src/data/x_data.csv')
    y = pd.read_csv('./src/data/y_data.csv')
    return X, y


def clusters_histogram(y_cluster, title ):
    x_unique, y_counts = np.unique(y_cluster, return_counts=True)

    x = x_unique
    y = y_counts
    # Use the hovertext kw argument for hover text
    fig = go.Figure(data=[go.Bar(x=x,
                                 y=y,
                                 text='Clusters and Quantity of Clients')])
    # Customize aspect
    fig.update_traces(marker_color='rgb(158,202,225)',
                      marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)

    fig.update_layout(
        title_text='Histogram of Clusters Distribution and Quantity Of Clients in set {}'.format(title),
        title_x=0.5,
        xaxis = dict(
                type='category',
                title='Clusters'),
        yaxis = dict(title='Quantity of Clients')
    )
    fig.show()

def clusters_predictions_histogram(y_cluster, y_predictions, title):
    # Calculate y_cluster unique values and their frequency
    x_unique, y_counts = np.unique(y_cluster, return_counts=True)

    x = x_unique
    y = y_counts
    # Use the hovertext kw argument for hover text
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=x,
                         y=y,
                         name='Y true Clusters',
                         text='Cluster and Quantity of Clients',
                         marker_color='rgb(158,202,225)',
                         marker_line_color='rgb(8,48,107)',
                         marker_line_width=1.5,
                         opacity=0.6))
    
    # Calculate y_predictions unique values and their frequency
    x_unique, y_counts = np.unique(y_predictions, return_counts=True)

    x = x_unique
    y = y_counts
    
    fig.add_trace(go.Bar(x=x,
                         y=y,
                         name='Y Predictions Clusters',
                         marker_color='lightsalmon',
                         marker_line_color='rgb(18,38,207)',
                         marker_line_width=1.5,
                         opacity=0.6))

    fig.update_layout(barmode='group',
                      bargroupgap=0.2,
                      title_text='Histogram of Clusters Distribution and Quantity Of Clients in {}'.format(title),
                      title_x=0.5,
                      xaxis = dict(
                          type='category',
                          title='Clusters'),
                      yaxis = dict(title='Quantity of Clients')
    )
    fig.show()

def crossval_learningcurve(k=None, scores=None, k_fp=None, scores_fp=None, k_lp=None, scores_lp=None):
    fig = go.Figure()
    
    if k:
        fig.add_trace(go.Scatter(x=np.arange(1,k+1),
                                     y=scores,
                                     name='Full Year',
                                     text='Cross Validation Fold and Accuracy Score'))
    if k_fp:
        fig.add_trace(go.Scatter(x=np.arange(1,k_fp+1),
                                 y=scores_fp,
                                 name='First Ten months',
                                 text='Cross Validation Fold and Accuracy Score'))
    if k_lp:
        fig.add_trace(go.Scatter(x=np.arange(1,k_lp+1),
                                 y=scores_lp,
                                 name='Last Two months',
                                 text='Cross Validation Fold and Accuracy Score'))
    
    fig.update_layout(
        title_text='Learning Curve for Cross Validation Folds with {} Folds'.format(k),
        title_x=0.5,
        xaxis = dict(
                type='category',
                title='k'),
        yaxis = dict(title='Accuracy Score')
    )
    
    fig.show()

def comparing_training_loss_and_val_acc(epochs, loss_curve, validation_scores):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs,
                             y=loss_curve,
                             name='Training Loss',
                             text='Epoch and Training Loss'))
    
    fig.add_trace(go.Scatter(x=epochs,
                             y=validation_scores,
                             name='Validation Accuracy',
                             text='Epoch and Validation Accuracy Score'))
    fig.update_layout(
        title_text='Validation Accuracy and Training Loss',
        title_x=0.3,
        xaxis = dict(
            title='Epochs'),
        yaxis = dict(title='Loss and Accuracy Score')
    )
    fig.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def MLP_Classifier():
    mlp_clf = MLPClassifier(hidden_layer_sizes=(30, 30, 30),
                            activation='relu',
                            solver='adam',
                            alpha=0.000099,
                            batch_size=32,
                            learning_rate='adaptive',
                            learning_rate_init=0.001,
                            power_t=0.5,
                            max_iter=246,
                            shuffle=True,
                            random_state=42,
                            tol=0.0001,
                            verbose=False,
                            warm_start=True,
                            momentum=0.9,
                            nesterovs_momentum=True,
                            early_stopping=True,
                            validation_fraction=0.05,
                            beta_1=0.9,
                            beta_2=0.999,
                            epsilon=1e-08,
                            n_iter_no_change=100)
    return mlp_clf

def app():
    st.title("DATA MODELING SECTION")
    X, y = get_x_y_train()

    st.write("Length: {}".format(len(X)))
    st.dataframe(X.head(5))
    st.write("Length: {}".format(len(y)))
    st.dataframe(y.head(5))

    st.markdown("""
    ## We have to split our dataset to X_train, X_test, y_train, y_test 
    """)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.8, random_state=42)

    st.markdown("""
    ## We need to scale our data
    `scaler = StandardScaler()`
    `scaler.fit(X_train)`
    `X_train = scaler.transform(X_train)`
    `X_test = scaler.transform(X_test)`
    """)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    with st.spinner('MLP CLASSIFIER TRAINING: Wait for it...'):
        cv_results_fp = cross_validate(MLP_Classifier(),
                                X_train,
                                y_train,
                                cv=10, 
                                return_train_score=True, 
                                scoring='accuracy')

        for k_fp, s_fp in enumerate(cv_results_fp['test_score']):
            st.write("Fold {} with Test Accuracy Score: {}".format(k_fp, s_fp))

    print("Average Test Accuracy Score: {}".format(np.sum(cv_results_fp['test_score'])/10))

    st.markdown("""
    ##### Now lets train MLP properly
    """)
    st.code("""
    mlp_clf = MLP_Classifier()
    predictions = mlp_clf.predict(X_test)
    metrics.accuracy_score(y_test, predictions)
    mlp_clf.fit(X_train, y_train)
    """)

    mlp_clf = MLP_Classifier()
    predictions = mlp_clf.predict(X_test)
    metrics.accuracy_score(y_test, predictions)
    mlp_clf.fit(X_train, y_train)
    st.write(metrics.classification_report(y_test, predictions))

