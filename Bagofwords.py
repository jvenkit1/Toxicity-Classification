import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_csv("train.csv")
count_vect = CountVectorizer(min_df=30)
y_train_count = data['target'].to_numpy()
X_train_count = count_vect.fit_transform(data['comment_text'])
#X_test_count = count_vect.fit_transform(data_test['comment_text'])
feature_array = X_train_count.toarray()
loss = 0
#Initialize the average square loss
num_instances, num_features = feature_array.shape[0], feature_array.shape[1]
theta = np.ones(num_features)
num_instances = y_train_count.shape[0]
loss = np.sum((np.dot(feature_array, theta) - y_train_count) ** 2) / num_instances
grad = 2.0 / num_instances * np.dot((np.dot(feature_array, theta) - y_train_count), feature_array)

