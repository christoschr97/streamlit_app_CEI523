#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import streamlit as st

# Exploratory Data analysis packages

import pandas as pd

# Visualizations packages

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import eda_summary as eda


def get_raw_cleaned_df():
    raw = pd.read_csv('./src/data/data.csv', encoding="ISO-8859-1")
    cleaned = pd.read_csv('./src/data/data_cleaned_2.csv')
    return (raw, cleaned)


def app():
    """ Common ML Dataset Explorer """

    st.title('Common ML Dataset Explorer')
    st.subheader('Datasets For ML Explorer with Streamlit')
    st.markdown("""##### This page has 2 sections: 
	1. Dataset Overview: Understanding the dataset
	2. Dataset Explorer (Interactive)""")

    type_of_data = st.radio(
        "Type of Data",
        ('Raw Data', 'Cleaned Data'),
        help='Data source that will be displayed in the charts'
    )

    raw_df, clean_df = get_raw_cleaned_df()

    if type_of_data == 'Raw Data':
        data = raw_df.copy()
    else:
        data = clean_df.copy()

    with st.container():
        st.header('Descriptive Statistics\n')
        st.dataframe(eda.summary_table(data))
        st.dataframe(data.describe())

    st.markdown("""
    ### To Do:
    ##### From the Descriptive statistics we need to do in the next steps:**
    **Data Cleaning and preparation**
    * 1. Remove the rows without Customer ID
    * 2. Remove the duplicate rows
    * 3. Remove the canceled orders / returns (Negative Quantities)
    * 4. Deal with Zero Unit Prices
    * 5. Explore the DataFrame Again and continue with Feature Engineering and Data Transformation
    """)

    # # DATASET EXPLORER ##

    html_temp = \
        """
	<div style="background-color:tomato;"><p style="color:white;font-size:30px;padding:10px">NOTE: Take Care of The columns Selecting due to the large amount of the dataset</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    st.markdown("""
    Select the dataframe below:
    * `data.csv` is the raw dataset
    * `data_cleaned_2.csv` is the cleaned dataset (see following steps)
    """)

    def file_selector(folder_path='./src/data'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select A file', filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()
    st.info('You Selected {}'.format(filename))

    # Read Data

    df = pd.read_csv(filename, encoding='ISO-8859-1',
                     dtype={'CustomerID': str, 'InvoiceNo': str})

    # Show Dataset

    if st.checkbox('Show Dataset'):
        st.markdown('**Showing the first 5 rows of the dataset**')
        st.dataframe(df.head(5))

    # Show Columns

    if st.button('Column Names'):
        st.write(df.columns)

    # Show Shape

    if st.checkbox('Shape of Dataset'):
        data_dim = st.radio('Show Dimension By ', ('Rows', 'Columns'))
        if data_dim == 'Rows':
            st.text('Number of Rows')
            st.write(df.shape[0])
        elif data_dim == 'Columns':
            st.text('Number of Columns')
            st.write(df.shape[1])
        else:
            st.write(df.shape)

    # Select Columns

    if st.checkbox('Select Columns To Show'):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect('Select', all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    # Show Values

    if st.button('Value Counts'):
        st.text('Value Counts By Target/Class')
        st.write(df.iloc[:, -1].value_counts())

    # Show Datatypes

    if st.button('Data Types'):
        st.write(df.dtypes)

    # Show Summary

    if st.checkbox('Summary'):
        st.write(df.describe().T)

    # # Plot and Visualization

    st.subheader('Data Visualization')

    # Correlation
    # Seaborn Plot

    if st.checkbox('Correlation Plot[Seaborn]'):
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot()

    # Pie Chart

    if st.checkbox('Pie Plot'):
        all_columns_names = df.columns.tolist()
        if st.button('Generate Pie Plot'):
            st.success('Generating A Pie Plot')
            st.write(df.iloc[:,
                     -1].value_counts().plot.pie(autopct='%1.1f%%'))
            st.pyplot()

    # Count Plot

    if st.checkbox('Plot of Value Counts'):
        st.text('Value Counts By Target')
        all_columns_names = df.columns.tolist()
        primary_col = st.selectbox('Primary Columm to GroupBy',
                                   all_columns_names)
        selected_columns_names = st.multiselect('Select Columns',
                all_columns_names)
        if st.button('Plot'):
            st.text('Generate Plot')
            if selected_columns_names:
                vc_plot = \
                    df.groupby(primary_col)[selected_columns_names].count()
            else:
                vc_plot = df.iloc[:, -1].value_counts()
            st.write(vc_plot.plot(kind='bar'))
            st.pyplot()

    # Customizable Plot

    st.subheader('Customizable Plot')
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox('Select Type of Plot', [
        'area',
        'bar',
        'line',
        'hist',
        'box',
        'kde',
        ])
    selected_columns_names = st.multiselect('Select Columns To Plot',
            all_columns_names)

    if st.button('Generate Plot'):
        st.success('Generating Customizable Plot of {} for {}'.format(type_of_plot,
                   selected_columns_names))

        # Plot By Streamlit

        if type_of_plot == 'area':
            cust_data = df[selected_columns_names]
            st.area_chart(cust_data)
        elif type_of_plot == 'bar':

            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)
        elif type_of_plot == 'line':

            cust_data = df[selected_columns_names]
            st.line_chart(cust_data)
        elif type_of_plot:

        # Custom Plot

            cust_plot = \
                df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()

    if st.button('Thanks'):
        st.balloons()