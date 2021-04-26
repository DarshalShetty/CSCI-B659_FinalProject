# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 19:36:03 2021

@author: Midnight Blast
"""

# Imports
import csv
import re  # import regex
from sklearn.feature_extraction.text import CountVectorizer  # count vectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


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
vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 7))
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
param_grid = {'C': [0.1, 1, 10, 100],
              'degree': [2, 3, 4],
              'gamma': ['scale', 'auto'],
              'kernel': ['linear', 'poly', 'rbf']}

grid = GridSearchCV(svm.SVC(), param_grid, n_jobs=-1, refit=True, verbose=4)

grid.fit(trainX, trainY)

# print the best params after doing grid search
print(grid.best_params_)

pred = grid.predict(testX)

print('*' * 25 + "testing 2016 data" + '*' * 25)
print(classification_report(testY, pred))

# testing new data from 2021

file = open('Data/2021_test_data/2021test.txt', encoding="utf8")
fileInTest2021X = file.read()
test2021XPreSplit = cleanitup(fileInTest2021X)
test2021XSplit = test2021XPreSplit.splitlines()

test2021X = vectorizer.transform(test2021XSplit)

file = open("Data/2021_test_data/2021testgold.txt", encoding="utf8")
fileInTest2021Y = file.read()

test2021Y = fileInTest2021Y.splitlines()

pred2021 = grid.predict(test2021X)

print('*' * 25 + "testing 2021 data" + '*' * 25)
print(classification_report(test2021Y, pred2021))

# output test and predicted labels to csv
output_dir = "Data/outputs/"
output_run_name = "char_n-gram_4_to_7"

with open("{}{}_waseem.csv".format(output_dir, output_run_name), 'w') as waseem_csv:
    w = csv.writer(waseem_csv)
    w.writerow(("test_tweet", "actual_label", "predicted_label"))
    for i in range(len(testXSplit)):
        w.writerow((testXSplit[i], testY[i], pred[i]))

with open("{}{}_2021.csv".format(output_dir, output_run_name), 'w') as tweets_2021_csv:
    w = csv.writer(tweets_2021_csv)
    w.writerow(("test_tweet", "actual_label", "predicted_label"))
    for i in range(len(test2021XSplit)):
        w.writerow((test2021XSplit[i], test2021Y[i], pred2021[i]))

'''vectorizer check-in tools'''
# print(vectorizer.vocabulary_)
# print(trainX.shape)
# print(type(trainX))
# print(trainX.toarray())
