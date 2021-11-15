import collections
from numpy.core.defchararray import lower
import streamlit as st
import numpy as np
import pandas as pd


def app():
    st.title("Visualization and Communication Section")
    st.write("Here we will write the results and our conclusions")
    st.write(st.session_state.key)
