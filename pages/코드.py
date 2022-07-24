import streamlit as st
from contents.contents import *

# load data
st.markdown('## Load Data')
st.code(data_path_1, language='python')
st.code(data_path_2, language='python')
st.code(concat_data_1, language='python')

# concat data
st.markdown('## concat data')
st.code(concat_data_1, language="python")

# EDA & Preprocessing
st.markdown('## EDA & Preprocessing')
st.code(data_check_1, language='python')

# mapping & drop
st.markdown("## mapping & drop")
st.code(mapping_drop, language='python')

# after EDA
st.markdown('## after EDA')
st.code(data_check_2, language='python')

# concat dataset for train
st.markdown("## concat dataset for train")
st.code(concat_data_2, language='python')
st.code(display_data, language='python')

# text preprocessing
st.markdown("## text preprocessing")
st.code(text_preprocessing, language='python')
st.code(apply_data, language='python')

# Train and Test Split
st.markdown("## Train and Test Split")
st.code(xy_split, language='python')
st.code(data_encoding, language='python')
st.code(train_test, language='python')

# tokenization
st.markdown('## tokenization')
st.code(set_param, language='python')
st.code(token, language='python')

# padding & clipping
st.markdown("## padding & clipping")
st.code(padding, language='python')
st.code(to_cate, language='python')

# LSTM

st.markdown("## LSTM")
st.code(lstm, language='python')
