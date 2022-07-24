import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from tqdm import tqdm

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC  #####
from sklearn.metrics import classification_report  ####
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Conv1D, Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from contents.contents import *

st.markdown("# Write lyrics!")
st.markdown("### How does this song feel?")
st.markdown("##### You can predict more appropriately by writing lyrics \
            that are 15 bits or more with fewer other things (such as spaces).")
lyrics = st.text_input("Write down lyrics")


nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
stop_words.extend(STOPWORDS)
stop_words.extend(['twitter', 'twitpic', 'im', 'feel', 'feeling', 'thing', 'things' 'something', 'think', 
                  'thought', 'know', 'go', 'got', 'going', 'come', 'time',' make', 'take', 'day', 'days', 
                  'something', 'someone', 'see', 'one', 'made', 'say', 'quot', 'feel', 'feeling', ])


## 이메일 주소 제거
def remove_email(text):
    return re.sub('([A-Za-z0-9._%+-\]+@[A-Za-z0-9.-]+\.[A-Za-z0-9.-]+)', '', text)


## 세 번 이상 반복되는 문자 제거
def remove_repeated_char(text):
    return re.sub(r'(.)\1\1{1,}', r'\1\1', text)


## 사용자태그 제거
def remove_account_tag(text):
    return re.sub(r'@[\w]+', '', text)


## 해시태그 제거
def remove_hashtag(text):
    return re.sub(r'#[\w]+', '', text)


## url 제거
def remove_url(text):
    return re.sub(r'(http|https|ftp)[^\s]*', '', text)


## 알파벳 한/두 글자인 단어 제거
def remove_less_2_characters(text):
    return re.sub(r'\W*\b\w{1,2}\b', '', text)


## 구두점 제거
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


## 이중 space 제거
def remove_spaces(text):
    text = re.sub(r"\s+", ' ', text)
    return text


## 표제어 추출
def lemmatize(text):
    tokens = text.split()
    tokens_tags = nltk.pos_tag(tokens)
    result = []
    lemmatizer = WordNetLemmatizer()
    for word, tag in tokens_tags:
        if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:  # 동사(Verb)를 원형으로
            result.append(lemmatizer.lemmatize(word, pos='v'))
        else:
            result.append(lemmatizer.lemmatize(word))
    return ' '.join(result)


## 불용어 제거
def remove_stop_words(text):
    word_tokens = word_tokenize(text)
    return ' '.join([word for word in word_tokens if word not in stop_words])


def preprocess_text(text):
    text = remove_email(text)
    text = remove_repeated_char(text)
    text = remove_account_tag(text)
    text = remove_hashtag(text)
    text = remove_url(text)
    text = remove_spaces(text)
    text = remove_less_2_characters(text)
    text = remove_punctuation(text)
    text = text.strip()  # 양 끝 공백 제거
    text = ' '.join(text.split())
    text = text.lower()
    text = lemmatize(text)
    text = remove_stop_words(text)
    return text


tqdm.pandas()

def preprocess_df_col(df, col='input'):
    df[f'{col}_p'] = df[col].progress_apply(lambda x: preprocess_text(x))  # col_preprocessed
    return df[df['input_p'] != '']

max_words = 20000
max_len = 30
tokenizer = Tokenizer(num_words=max_words, oov_token='<oov>')


word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

model = tf.keras.models.load_model("./data/lstm_model.h5")

label_sentiment_map = {0: 'love', 
                      1: 'sadness', 
                      2: 'anger', 
                      3: 'joy', 
                      4: 'anxiety'}

def lyrics_preprocessing(lyrics):
    df_lyrics = pd.DataFrame(lyrics.strip().split('\n'), columns=['input'])
    df_lyrics = preprocess_df_col(df_lyrics, 'input')
    
    lyrics_p = df_lyrics['input_p']
    tokenizer.fit_on_texts(lyrics_p)
    sequences  = tokenizer.texts_to_sequences(lyrics_p)
    return pad_sequences(sequences, maxlen=max_len)

def lyrics_predict(test_lyrics):
    lyrics_pred = pd.DataFrame(model.predict(lyrics_preprocessing(test_lyrics)))
    
    
    lyrics_pred['max_label'] = lyrics_pred.idxmax(axis=1)
#     lyrics_pred['max_prop'] = lyrics_pred.max(axis=1)
    
    label_count = lyrics_pred['max_label'].value_counts()
    labels = label_count.index.map(label_sentiment_map)
    plt.pie(label_count, autopct='%.1f%%', labels=labels);
    print(label_count.idxmax())
    return

    
lyrics_predict(lyrics)





# prob = model.predict()
# for idx in prob.argsort()[0][::-1][:3]:
#         print(idx, "{:.2f}%".format(prob[0][idx]*100))
