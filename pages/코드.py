from importlib.resources import path
import streamlit as st
from contents.contents import *

# load data
st.markdown('## load data')
st.code(data_path_1, language='python')
st.code(data_path_2, language='python')
st.code(concat_data, language='python')