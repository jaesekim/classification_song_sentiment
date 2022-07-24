import streamlit as st
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv1D, Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.python.keras.callbacks import EarlyStopping
from contents.contents import *

st.markdown("# Write lyrics!")
st.markdown("### How does this song feel?")
st.markdown("##### You can predict more appropriately by writing lyrics \
            that are 15 bits or more with fewer other things (such as spaces).")
lyrics = st.text_input("Write down lyrics over 15bits. More ")

model = tf.keras.models.load_model("./data/lstm_model.h5")
st.write(model.summary())