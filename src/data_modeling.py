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
from sklearn.metrics import plot_confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
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

import pickle
models = {"svm": svm.SVC(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier(),
          "Gauusian": GaussianNB(),
          "BaggingClassifier": BaggingClassifier(),
          "ExtraTreesClassifier": ExtraTreesClassifier(),
          "DecisionTreeClassifier": DecisionTreeClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of differetn Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    """
    # Set random seed
    np.random.seed(42)
    # Make a dictionary to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores

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

def plot_cfmtrx(clf, X_test, y_test):
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(clf, X_test, y_test, ax=ax)
    st.pyplot(fig)

def RandomForestTraining(X_train, y_train, X_test, y_test):
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    predictions = clf_rf.predict(X_test)
    score = clf_rf.score(X_test, y_test)
    st.markdown("##### Confusion Matrix: Random Forest Classifier")
    plot_cfmtrx(clf_rf, X_test, y_test)
    report = metrics.classification_report(y_test, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.markdown("##### Classification Report: Random Forest Classifier")
    st.dataframe(df)
    return clf_rf, predictions, score

def ExtraTreesTraining(X_train, y_train, X_test, y_test):
    clf_et = ExtraTreesClassifier()
    clf_et.fit(X_train, y_train)
    predictions = clf_et.predict(X_test)
    score = clf_et.score(X_test, y_test)
    st.markdown("##### Confusion Matrix: Extra Trees Classifier")
    plot_cfmtrx(clf_et, X_test, y_test)
    report = metrics.classification_report(y_test, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.markdown("##### Classification Report: Extra Trees Classifier")
    st.dataframe(df)
    return clf_et, predictions, score

def BaggingTraining(X_train, y_train, X_test, y_test):
    clf_bg = ExtraTreesClassifier()
    clf_bg.fit(X_train, y_train)
    predictions = clf_bg.predict(X_test)
    score = clf_bg.score(X_test, y_test)
    st.markdown("##### Confusion Matrix: Bagging Classifier")
    plot_cfmtrx(clf_bg, X_test, y_test)
    report = metrics.classification_report(y_test, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.markdown("##### Classification Report: Bagging Classifier")
    st.dataframe(df)
    return clf_bg, predictions, score

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

    st.code("""
    cv_results_fp = cross_validate(MLP_Classifier(),
                        X_train,
                        y_train,
                        cv=10, 
                        return_train_score=True, 
                        scoring='accuracy')
    """)

    with st.spinner('MLP 10-FOLD VALIDATION CLASSIFIER TRAINING: Wait for it...'):
        cv_results_fp = cross_validate(MLP_Classifier(),
                                X_train,
                                y_train,
                                cv=10, 
                                return_train_score=True, 
                                scoring='accuracy')

        for k_fp, s_fp in enumerate(cv_results_fp['test_score']):
            st.write("Fold {} with Test Accuracy Score: {}".format(k_fp, s_fp))

    st.write("Average Test Accuracy Score: {}".format(np.sum(cv_results_fp['test_score'])/10))

    st.markdown("""
    ##### Now lets train MLP properly
    From the K-fold validation we can see that the algorithm performs quite well with `cv=10`
    """)
    
    st.code("""
    mlp_clf = MLP_Classifier()
    predictions = mlp_clf.predict(X_test)
    metrics.accuracy_score(y_test, predictions)
    mlp_clf.fit(X_train, y_train)
    """)

    with st.spinner('MLP CLASSIFIER TRAINING: Wait for it...'):
        mlp_clf = MLP_Classifier()
        mlp_clf.fit(X_train, y_train)
        predictions = mlp_clf.predict(X_test)
        metrics.accuracy_score(y_test, predictions)
        report = metrics.classification_report(y_test, predictions, output_dict=True)
        df = pd.DataFrame(report).transpose()
        st.dataframe(df)
        # st.write(metrics.classification_report(y_test, predictions))

    plot_cfmtrx(mlp_clf, X_test, y_test)

    st.markdown("""
    ## Find some more classifiers that perform well utilizing the custom made function bellow:
    """)
    st.code("""
    models = {"svm": svm.SVC(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier(),
          "Gauusian": GaussianNB(),
          "BaggingClassifier": BaggingClassifier(),
          "ExtraTreesClassifier": ExtraTreesClassifier(),
          "DecisionTreeClassifier": DecisionTreeClassifier()}

    # Create a function to fit and score models
    def fit_and_score(models, X_train, X_test, y_train, y_test):
        Fits and evaluates given machine learning models.
        models : a dict of differetn Scikit-Learn machine learning models
        X_train : training data (no labels)
        X_test : testing data (no labels)
        y_train : training labels
        y_test : test labels
        # Set random seed
        np.random.seed(42)
        # Make a dictionary to keep model scores
        model_scores = {}
        # Loop through models
        for name, model in models.items():
            # Fit the model to the data
            model.fit(X_train, y_train)
            # Evaluate the model and append its score to model_scores
            model_scores[name] = model.score(X_test, y_test)
        return model_scores
    """)
    with st.spinner('Fast Training of 6 models: Wait for it...'):
        results = fit_and_score(models, X_train, X_test, y_train, y_test)

    st.markdown("Results for a fast training of the models:\n")
    st.markdown(results)

    st.markdown("""
    ## From the results above we can see the three best performing algorithms are the ensemble algorithms:
    * Ranndom Forest
    * Extra Trees Classifier
    * Baggin Classifier
    So, we decided to train also these three algorithms and evaluate them.
    """)

    st.markdown("### Random Forest Classifier")

    with st.spinner('Training Random Forest: Wait for it...'):
        clf_rf, rf_predictions, rf_score = RandomForestTraining(X_train, y_train, X_test, y_test)

    with st.spinner('Training Extra Trees Classifier: Wait for it...'):
        clf_et, et_predictions, et_score = ExtraTreesTraining(X_train, y_train, X_test, y_test)

    with st.spinner('Training Bagging Classifier: Wait for it...'):
        clf_bg, bg_predictions, bg_score = BaggingTraining(X_train, y_train, X_test, y_test)

    st.markdown("""
    # TRAINING WITH THE LAST PERIOD DATA
    ## Now lets train MLP Algorithm with the test set we saved in the previous step and thus check if the classifiers are going to perform the same as the first_period
    """)

    cart_price = pd.read_csv('./src/data/test_set.csv')
    cart_price['InvoiceDate'] = cart_price['InvoiceDate'].astype('datetime64[ns]')
    print(cart_price.dtypes) 

    st.markdown("""
    ##### Test set from Last Period:
    For these we need to use the test_set we prepared in the Feature Engineering section and reconstruct the input X as X_lp (Last Period) and the Y_lp respectively
    """)
    transactions_per_user=cart_price.groupby(by=['CustomerID'])['Cart Price'].agg(['count','min','max','mean','sum'])
    for i in range(5):
        col = 'Cat_{}'.format(i)
        transactions_per_user.loc[:,col] = cart_price.groupby(by=['CustomerID'])[col].sum() / transactions_per_user['sum']*100

    transactions_per_user.reset_index(drop = False, inplace = True)
    cart_price.groupby(by=['CustomerID'])['Cat_0'].sum()

    # Correcting time range
    transactions_per_user['count'] = 5 * transactions_per_user['count']
    transactions_per_user['sum']   = transactions_per_user['count'] * transactions_per_user['mean']

    transactions_per_user.sort_values('CustomerID', ascending = True).head(5)
    st.dataframe(cart_price.head())
    # st.write(cart_price.dtypes)

    last_date = cart_price['InvoiceDate'].max().date()

    first_registration = pd.DataFrame(cart_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
    last_purchase      = pd.DataFrame(cart_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

    test_fp  = first_registration.applymap(lambda x:(last_date - x.date()).days)
    test_lp = last_purchase.applymap(lambda x:(last_date - x.date()).days)

    transactions_per_user.loc[:, 'FirstPurchase'] = test_fp.reset_index(drop=False)['InvoiceDate']
    transactions_per_user.loc[:, 'LastPurchase'] = test_lp.reset_index(drop=False)['InvoiceDate']

    list_cols = ['count','min','max','mean', 'Cat_0','Cat_1','Cat_2','Cat_3','Cat_4', 'LastPurchase', 'FirstPurchase']
    test_matrix = transactions_per_user[list_cols].to_numpy()

    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(test_matrix)
    minmaxscaled_test_matrix = minmax_scaler.transform(test_matrix)

    kmeans = st.session_state.kmeans
    Y_last_period = kmeans.predict(minmaxscaled_test_matrix)
    X_last_period = transactions_per_user[list_cols]

    st.dataframe(Y_last_period)
    st.dataframe(X_last_period)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_last_period, Y_last_period, train_size = 0.8, random_state=42)


    st.markdown("""
    ## MLP Classifier with the Last period data
    """)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    cv_results_lp = cross_validate(mlp_clf,
                            X_train,
                            y_train,
                            cv=10, 
                            return_train_score=True, 
                            scoring='accuracy')

    for k_lp, s_lp in enumerate(cv_results_lp['test_score']):
        print("Fold {} with Test Accuracy Score: {}".format(k_lp, s_lp))

    print("Average Test Accuracy Score: {}".format(np.sum(cv_results_lp['test_score'])/10))
    st.write("Average Test Accuracy Score: {}".format(np.sum(cv_results_lp['test_score'])/10))

    mlp_clf = MLP_Classifier()
    mlp_clf.fit(X_train, y_train)
    predictions = mlp_clf.predict(X_test)

    metrics.accuracy_score(y_test, predictions)
    report = metrics.classification_report(y_test, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    st.dataframe(df)

    plot_cfmtrx(mlp_clf, X_test, y_test)

    st.markdown("### Random Forest Classifier with Last Period Data")
    with st.spinner('Training Random Forest with Last Period Data: Wait for it...'):
        clf_rf, predictions, score = RandomForestTraining(X_train, y_train, X_test, y_test)

    with st.spinner('Training Extra Trees Classifier: Wait for it...'):
        clf_et, et_predictions, et_score = ExtraTreesTraining(X_train, y_train, X_test, y_test)

    with st.spinner('Training Bagging Classifier: Wait for it...'):
        clf_bg, bg_predictions, bg_score = BaggingTraining(X_train, y_train, X_test, y_test)
