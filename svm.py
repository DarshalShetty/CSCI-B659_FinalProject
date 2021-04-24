# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 19:36:03 2021

@author: Midnight Blast
"""

# Imports
# import numpy as np
# from nltk.util import ngrams
import re  # import regex
from sklearn.feature_extraction.text import CountVectorizer  # count vectorizer
from sklearn import svm
from sklearn.metrics import classification_report


def deEmojify(text):
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        u"\U00002702-\U000027B0"
                                        u"\U000024C2-\U0001F251"
                                        "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)


def cleanitup(datafile):
    df_clean = datafile.lower()  # Make Lower Case

    df_clean = deEmojify(df_clean)  # Remove emojis from file#Remove emojis from file
    # Note:  if the ORD value of a word is greater than 255 its an emojii

    df_clean = df_clean.replace("&amp", "and")  # Replace &amp
    df_clean = df_clean.replace("&gt", "and")
    df_clean = df_clean.replace("&", "and")

    df_clean = re.sub('RT\s', '',
                      df_clean)  # Remove the retweet indicator "RT" from file, when rt was lowercase it removed 8 tweets
    df_clean = re.sub('http\S+', '', df_clean)  # Remove all URLs from file
    df_clean = re.sub('@\S+', '', df_clean)  # Remove all @username from file
    df_clean = re.sub('"', '', df_clean)  # Note:  Remove quotes and potentially commas, but keep major punctuation
    # df_clean = re.sub('  +', '', df_clean) #Remove any extra (more than one) whitespace in file, when active this removed 1 tweet
    df_clean = re.sub('[^\w\s@#\-—]', '',
                      df_clean)  # remove all non-Alphanumeric characters from file, excluding spaces, @, hyphen, and em-dash
    return df_clean


# prepare train file to be vectorized

waseem_data_dir = r'Data/waseem-data/'
file = open(r'%swaseemtrain.txt' % waseem_data_dir, encoding="utf8")
fileInTrainX = file.read()

trainXPreSplit = cleanitup(fileInTrainX)

trainXSplit = trainXPreSplit.splitlines()

# prepare test file to be vectorized

file = open(r'%swaseemtest.txt' % waseem_data_dir, encoding="utf8")
fileInTestX = file.read()

testXPreSplit = cleanitup(fileInTestX)

testXSplit = testXPreSplit.splitlines()

# vectorize and fit train file, can change ngram range here
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3))
# tokenize and build vocab
trainX = vectorizer.fit_transform(trainXSplit)

# vectorize test file
# tokenize and build vocab
testX = vectorizer.transform(testXSplit)

# import train key
file = open(r'%swaseemtrainGold.txt' % waseem_data_dir, encoding="utf8")
fileInTrainY = file.read()

trainY = fileInTrainY.splitlines()

# import test key
file = open(r'%swaseemtestGold.txt' % waseem_data_dir, encoding="utf8")
fileInTestY = file.read()

testY = fileInTestY.splitlines()

# below line is where settings can be changed
clf = svm.SVC(kernel='rbf', degree=2, coef0=5, C=0.5)
clf.fit(trainX, trainY)

pred = clf.predict(testX)

print(classification_report(testY, pred))

'''vectorizer check-in tools'''
# print(vectorizer.vocabulary_)
# print(trainX.shape)
# print(type(trainX))
# print(trainX.toarray())
