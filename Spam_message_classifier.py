# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:04:03 2022

@author: hp
"""

# Loading the data set
import pandas as pd

message = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

# Data cleaning and preprocessing

import nltk
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []

for i in range(0, len(message)):
    review = re.sub("[^a-zA-Z]", " ", message["message"][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words("english"))]
    review = " ".join(review)
    corpus.append(review)

# Creating the Bag of words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000)
X = cv.fit_transform(corpus).toarray()    

y = pd.get_dummies(message['label'])
y = y.iloc[:,1].values

# Train test split (spliting the data set)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# training model using Naive bayes clssifier

from sklearn.naive_bayes import MultinomialNB
spam_detect = MultinomialNB()
spam_detect.fit(X_train, y_train)

y_pred = spam_detect.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

acc = accuracy_score(y_test, y_pred)

confusion = confusion_matrix(y_test, y_pred)