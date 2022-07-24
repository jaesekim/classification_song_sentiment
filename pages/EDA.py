import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from contents.contents import *


# load df1 
df1 = pd.read_csv('./data/tweet_emotions.csv')
df1 = df1.drop(columns='tweet_id')
df1.columns = ['sentiment', 'input']
df1 = df1[['input', 'sentiment']]

# loda df2
df2_train = pd.read_csv('./data/train.txt', header=None, sep =';', names=['Input','Sentiment'], encoding='utf-8')
df2_val = pd.read_csv('./data/val.txt', header=None, sep =';', names=['Input','Sentiment'], encoding='utf-8')
df2_test = pd.read_csv('./data/test.txt', header=None, sep =';', names=['Input','Sentiment'], encoding='utf-8')
df2 = pd.concat([df2_train, df2_val, df2_test], axis=0)
df2.columns = ['input', 'sentiment']
df2 = df2.drop_duplicates()

st.markdown("## Before Preprocessing Dataset")
click = st.radio(
    "Select Dataset",
    ('df1_sentiment', 'df2_sentiment')
)

if click == 'df1_sentiment':
    st.markdown('##### df1.value_counts')
    st.code(df_1_value_cnt, language='python')
    st.dataframe(df1.head())
    
else:
    st.markdown('##### df2.value_counts')
    st.code(df_2_value_cnt, language='python')
    st.dataframe(df2.head())

# After Mapping and Drop

st.markdown("## After Mapping and Drop")

sentiment_map1 = {'love': 'love', 
                  'sadness': 'sadness', 
                  'anger': 'anger', 
                  'hate': 'anger', 
                  'happiness': 'joy', 
                  'worry': 'anxiety' }

sentiment_map2 = {'love': 'love', 
                  'sadness': 'sadness', 
                  'anger': 'anger', 
                  'joy': 'joy', 
                  'fear': 'anxiety' }

df1['sentiment'] = df1['sentiment'].map(sentiment_map1)
df2['sentiment'] = df2['sentiment'].map(sentiment_map2)
df1 = df1.dropna(axis=0).reset_index(drop=True)
df2 = df2.dropna(axis=0).reset_index(drop=True)

sentiments = ['love', 'sadness', 'anger', 'joy', 'anxiety']
fig, ax = plt.subplots(1,2, figsize=(35,10))

sns.countplot(y=df1['sentiment'], orient='h', order=sentiments, ax=ax[0])
ax[0].set_title("df1['sentiment']")

sns.countplot(y=df2['sentiment'], orient='h', order=sentiments, ax=ax[1])
ax[1].set_title("df2['sentiment']")

plt.tight_layout()
st.pyplot(fig)

df = pd.concat([df1, df2])
df = df.drop_duplicates()
df['sentiment'] = df['sentiment'].astype('category')

# After Concat
st.markdown("## After Concat")
fig = plt.figure(figsize=(12,3))
sns.countplot(y=df1['sentiment'], orient='h', order=sentiments, ax=ax[0])  
plt.title('Distribution of label values')
st.pyplot(fig)