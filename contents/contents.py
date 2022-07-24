df_1_value_cnt = '''
neutral       8638
worry         8459
happiness     5209
sadness       5165
love          3842
surprise      2187
fun           1776
relief        1526
hate          1323
empty          827
enthusiasm     759
boredom        179
anger          110
Name: sentiment, dtype: int64'''

df_2_value_cnt = '''
joy         6760
sadness     5797
anger       2709
fear        2373
love        1641
surprise     719
'''

df_value_cnt = '''
joy        11877
sadness    10826
anxiety    10654
love        5392
anger       4111
'''




# load data
data_path_1 = '''df1 = pd.read_csv('your_path/data/tweet_emotions.csv')
df1 = df1.drop(columns='tweet_id')
df1.columns = ['sentiment', 'input']
df1 = df1[['input', 'sentiment']]
'''
data_path_2 = '''df2_train = pd.read_csv('your_path/data/train.txt', header=None, sep =';', names=['Input','Sentiment'], encoding='utf-8')
df2_val = pd.read_csv('your_path/data/val.txt', header=None, sep =';', names=['Input','Sentiment'], encoding='utf-8')
df2_test = pd.read_csv('your_path/data/test.txt', header=None, sep =';', names=['Input','Sentiment'], encoding='utf-8')
'''

# concat data

concat_data_1 = '''df2 = pd.concat([df2_train, df2_val, df2_test], axis=0)
df2.columns = ['input', 'sentiment']
df2 = df2.drop_duplicates()
'''
# EDA & Preprocessing

data_check_1 = '''sns.countplot(y=df1['sentiment'], orient='h');
plt.title("df1['sentiment']");
sns.countplot(y=df2['sentiment'], orient='h');
plt.title("df2['sentiment']");'''

# mapping & drop

mapping_drop = '''sentiment_map1 = {'love': 'love', 
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
'''

# after EDA

data_check_2 = '''sentiments = ['love', 'sadness', 'anger', 'joy', 'anxiety']
fig, ax = plt.subplots(1,2, figsize=(12,3))

sns.countplot(y=df1['sentiment'], orient='h', order=sentiments, ax=ax[0])
ax[0].set_title("df1['sentiment']")

sns.countplot(y=df2['sentiment'], orient='h', order=sentiments, ax=ax[1])
ax[1].set_title("df2['sentiment']")

plt.tight_layout()
plt.show()
'''

# concat dataset for train

concat_data_2 = '''df = pd.concat([df1, df2])
df = df.drop_duplicates()
'''
display_data = '''sns.countplot(y=df1['sentiment'], orient='h', order=sentiments)
plt.title('Distribution of label values');
'''

# text preprocessing

text_preprocessing = '''wordnet_lemmatizer = WordNetLemmatizer()
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
'''

apply_data = '''df = preprocess_df_col(df, 'input')
'''

# Train and Test Split

xy_split = '''X = df['input_p']
y = df['sentiment']
'''
data_encoding = '''y_ohe = pd.get_dummies(y)'''
train_test = '''X_train, X_test, y_train, y_test = train_test_split(X, y_ohe, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
'''

# tokenization

set_param = '''
max_words = 20000
max_len = 30
'''
token = '''
tokenizer = Tokenizer(num_words=max_words, oov_token='<oov>')
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
'''

# texts_to_sequences
texts_to_sequences = '''
train_sequences = tokenizer.texts_to_sequences(X_train)
val_sequences = tokenizer.texts_to_sequences(X_val)
test_sequences = tokenizer.texts_to_sequences(X_test)
'''

# padding & clipping

padding = '''
X_train = pad_sequences(train_sequences, maxlen=max_len)
X_val = pad_sequences(val_sequences, maxlen=max_len)
X_test = pad_sequences(test_sequences, maxlen=max_len)
'''
to_cate = '''
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)
'''
# LSTM

lstm = '''
max_words = 20000
embedding_dim = 128
n_class = y_train.shape[1]  # 5

model = Sequential()
model.add(Embedding(input_dim=max_words, 
                    output_dim=embedding_dim, 
                    input_length=max_len))
model.add(Conv1D(64, 5, activation='relu'))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(u64))
model.add(Dropout(0.2))
model.add(Dense(n_class, activation='softmax'))


model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])'''