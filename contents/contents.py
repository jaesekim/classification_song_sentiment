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

concat_data = '''df2 = pd.concat([df2_train, df2_val, df2_test], axis=0)
df2.columns = ['input', 'sentiment']
df2 = df2.drop_duplicates()
'''