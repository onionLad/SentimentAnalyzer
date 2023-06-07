#
# bayesian_classifier.py
#   Bill Xia
#   6/7/2023
#
# This file contains the implementation of a Naive Bayes sentiment analysis
# classifier. Its purpose is to determine if reviews for various locations and
# products are positive or negative.
#

# Imports
import pandas as pd
import numpy as np
import math

# Functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Passage Cleaner
def cleanPassage(passage):
    new_passage = []
    for word in passage.split(' '):
        word = word.replace(".", "").replace(",", "").replace("!", "").lower()
        if word == 'going': word = 'go'
        if word == 'you\'re': word = 'you'
        new_passage.append(word)
    return new_passage

# Stop-word remover
def remStopWords(passage):
    stopWordList = ['the', 'was', 'i', 'it', 'had', 'a', 'at', 'if', 'to',
                    'this', 'you', 'have', 'my', 'thought', 'it\'s']
    for word in stopWordList:
        if word in passage: passage.remove(word)
    return passage

# Vectorizer
def vectorize(passage):
    pf, nf  = 0, 0
    for word in np.unique(passage):
        pf += pos_freq[word]
        nf += neg_freq[word]
    return np.array([1, pf, nf])

# Log likelihood calculator
def get_loglike(passage, log_prior, lambdas):
    return log_prior + sum([lambdas[word] for word in passage])

# Accuracy function
def accuracy(xs, ys, lambdas, log_prior):
    num_samples = len(xs)
    num_correct = 0
    for i in range(num_samples):
        log_like = get_loglike(xs[i], log_prior, lambdas)
        num_correct += ((log_like > 0) == ys[i])
    return num_correct / num_samples

# Body  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Reading in data
data  = pd.read_csv("data.csv")
train = data.loc[:79]
test  = data.loc[80:].reset_index(drop=True)

# Preprocessing data
train_x = []
for idx, row in train.iterrows():
    passage = cleanPassage(row['passage'])
    passage = remStopWords(passage)
    train_x.append(passage)
train_y = train['label']

test_x = []
for idx, row in test.iterrows():
    passage = cleanPassage(row['passage'])
    passage = remStopWords(passage)
    test_x.append(passage)
test_y = test['label']

# Building frequency dictionaries and word counts
pos_freq = {}
neg_freq = {}
for idx, passage in enumerate(train_x):

    # Adding words to the frequency dictionaries
    if train_y[idx] == 0:
        for word in passage:
            if word not in neg_freq:
                neg_freq[word] = 1
            else:
                neg_freq[word] += 1
            if word not in pos_freq:
                pos_freq[word] = 0
    else:
        for word in passage:
            if word not in pos_freq:
                pos_freq[word] = 1
            else:
                pos_freq[word] += 1
            if word not in neg_freq:
                neg_freq[word] = 0
pos_count = sum(pos_freq.values())
neg_count = sum(neg_freq.values())

# Building conditional probability dictionaries and lambda dictionary
pos_probs = {}
neg_probs = {}
lambdas   = {}
num_words = len(pos_freq.keys())
for word, freq in pos_freq.items():
    pos_probs[word] = (freq + 1) / (pos_count + num_words)
    neg_probs[word] = (neg_freq[word] + 1) / (neg_count + num_words)
    lambdas[word] = math.log(pos_probs[word] / neg_probs[word])

# Computing log prior
num_pos_docs = 50   # Hardcoded because we created our own data
num_neg_docs = 50   # Hardcoded because we created our own data
log_prior = math.log(num_pos_docs / num_neg_docs)

# Testing the classifier
print(f"Train Accuracy: {accuracy(train_x, train_y, lambdas, log_prior)}")
print(f"Test Accuracy: {accuracy(test_x, test_y, lambdas, log_prior)}")

# Feeding new inputs for further testing
print("Enter a passage: ", end="")
my_text = input()
print(my_text)
try:
    my_text = cleanPassage(my_text)
    my_text = remStopWords(my_text)
    if get_loglike(my_text, log_prior, lambdas) > 0:
        print("Positive Message")
    else:
        print("Negative Message")
except:
    print("Error: Unknown word encountered")

