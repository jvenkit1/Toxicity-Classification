import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
from nltk.stem import PorterStemmer
import string, re
ps = PorterStemmer()
data = pd.read_csv("train.csv")
for (index,comments) in enumerate(data['comment_text']):
    comments = re.sub('[%s]' % re.escape(string.punctuation), '', comments)
    print (comments)
    l = comments.decode('utf-8').split()
    data['comment_text'][index] = ' '.join([ps.stem(word) for word in l])

print (data['comment_text'])
count_vect = CountVectorizer(min_df=30, stop_words=stop_words.ENGLISH_STOP_WORDS)
X_train_count = count_vect.fit_transform(data['comment_text'])
feature_array = X_train_count.toarray()
print (feature_array)
