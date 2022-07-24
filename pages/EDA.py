import streamlit as st

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
    fig = plt.figure(figsize=(12,3))
    sns.countplot(y=df1['sentiment'], orient='h')
    plt.title("df1['sentiment']")
    st.pyplot(fig)
    
else:
    st.markdown('##### df2.value_counts')
    st.code(df_2_value_cnt, language='python')
    st.dataframe(df2.head())
    fig = plt.figure(figsize=(12,3))
    sns.countplot(y=df2['sentiment'], orient='h')
    plt.title("df2['sentiment']")
    st.pyplot(fig)

st.markdown('---')

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
df['input'] = df['input'].map(lambda x : x if len(x) > 15 else np.nan)
df = df.dropna(axis=0).reset_index(drop=True)

st.markdown('---')

# After Concat
st.markdown("## After Concat")
fig = plt.figure(figsize=(12,3))
sns.countplot(y=df1['sentiment'], orient='h', order=sentiments)  
plt.title('Distribution of label values')
st.pyplot(fig)

st.markdown('---')

# Text Preprocessing

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
stop_words.extend(STOPWORDS)
stop_words.extend(['twitter', 'twitpic', 'im', 'feel', 'feeling', 'thing', 'things' 'something', 'think', 
                  'thought', 'know', 'go', 'got', 'going', 'come', 'time',' make', 'take', 'day', 'today',
                  'something', 'someone', 'see', 'one', 'made', 'days', 'say', 'quot', ])


def remove_emoji(text):
    regex_pattern = re.compile("["                               
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U0001F1E6-\U0001F1FF"  # flags                              
                               "]+", flags=re.UNICODE)

    return regex_pattern.sub(r'', text)


def remove_email(text):
    return re.sub('([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]+)', '', text)


def remove_repeated_char(text):
    return re.sub(r'(.)\1\1{1,}', r'\1\1', text)


def remove_account_tag(text):
    return re.sub(r'@[\w]+', '', text)


def remove_hashtag(text):
    return re.sub(r'#[\w]+', '', text)


def remove_url(text):
    return re.sub(r'(http|https|ftp)[^\s]+', '', text)


def remove_spaces(text):
    text = re.sub(r"\d+", ' ', text)
    text = re.sub(r"\n+", ' ', text)
    text = re.sub(r"\t+", ' ', text)
    text = re.sub(r"\r+", ' ', text)
    text = re.sub(r"\s+", ' ', text)
    return text


def remove_stop_words(text):
    word_tokens = word_tokenize(text)
    return " ".join([w for w in word_tokens if not w in stop_words])


def remove_less_2_characters(text):  ## 동작 안됨
    return re.sub(r"\W*\b\w{1,2}\b", '', text)


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


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


def preprocess_text_sample(text):
    text = remove_emoji(text)
    text = remove_email(text)
    text = remove_repeated_char(text)
    text = remove_account_tag(text)
    text = remove_hashtag(text)
    text = remove_url(text)
    text = remove_stop_words(text)
    text = remove_spaces(text)
    text = remove_less_2_characters(text)
    text = remove_punctuation(text)
    text = text.strip()
    text = text.lower()
    text = lemmatize(text)
    return text


tqdm.pandas()

def preprocess_df_col(df, col='input'):
    df[f'{col}_p'] = df[col].progress_apply(lambda x: preprocess_text_sample(x))  # col_preprocessed
    return df[df['input_p'] != '']

df = preprocess_df_col(df, 'input')

# Word Cloud
st.markdown("## Word Cloud")
def print_wordcloud(df, sentiment):

    print(f"WordCloud of most frequent words for the sentiment: {sentiment}".format(sentiment))
    temp_df = df[df['sentiment'] == sentiment]
    print("Number of Rows:", len(temp_df))

    corpus = ''
    for text in temp_df['input']:
        text = str(text)
        corpus += text
        
    wordcloud = WordCloud(max_words=300, width=1600, height=800).generate(corpus)
    plt.figure(figsize = (20,20))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    st.pyplot
print_wordcloud(df, 'sadness')