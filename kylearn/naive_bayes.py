#!/Users/kylemoore/miniconda3/bin/python

"""
The :mod:`kylearn.naive_bayes` module implements Naive Bayes algorithms. These
are supervised learning methods based on applying Bayes' theorem with strong
(naive) feature independence assumptions.
"""

import numpy as np
import pandas as pd
from math import log
import sklearn.naive_bayes as nb
from sklearn.metrics import accuracy_score
class MultinomialNB(object):
    def __init__(self, alpha = 1.0):
        self.feature_log_probs_ = None
        self.class_count_ = None
        self.feature_count_ = None
        self.class_log_prior_ = None
        self.classes_ = None
        self.alpha_ = alpha

    def fit(self, X, y):
        """	Fit Naive Bayes classifier according to X, y """

        #error handling
        if X.shape[0] != y.shape[0]:
            raise Exception('X and y should have the same number of columns')

        #setting class counts and getting classes
        self.classes_, self.class_count_ = np.unique(y, return_counts = True)

        #setting class log priors
        self.class_log_prior_ = np.log(np.divide(self.class_count_, y.shape[0]))

        #setting feature counts
        self.feature_count_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        for X_i, y_i in zip(X.itertuples(index = False), y):
            X_i = list(X_i)
            class_idx = np.where( self.classes_ == y_i)[0][0]
            for j, X_i_j in enumerate(X_i):
                if X_i_j == 1:
                    self.feature_count_[class_idx][j] += 1

        #setting feature log probs
        smoothed_fc = self.feature_count_ + self.alpha_
        smoothed_cc = smoothed_fc.sum(axis = 1)
        self.feature_log_probs_ = (np.log(smoothed_fc) - np.log(smoothed_cc.reshape(-1,1)))

    def predict(self, X):
        """Perform classification on an array of test vectors X."""
        cond_probs =  np.dot(X, self.feature_log_probs_.T)
        return (cond_probs + self.class_log_prior_).argmax(axis = 1)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels."""
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)


if (__name__ == '__main__'):
    #this code just verifies that my MultinomialNB matches scikit-learn performance
    data = pd.read_csv('./datasets/spambase_binary.csv')
    X = data.drop('is_spam', axis = 1)
    y = data['is_spam']
    test_data = pd.read_csv('./datasets/spambase_test.csv')
    test_X = test_data.drop('is_spam', axis = 1)
    test_y = test_data['is_spam']
    test = nb.MultinomialNB()
    test.fit(X,y)
    clas = MultinomialNB()
    clas.fit(X,y)
    print(clas.score(X.head(), y.head()))
    print(test.score(X.head(), y.head()))
