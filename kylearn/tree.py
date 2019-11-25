#!/Users/kylemoore/miniconda3/bin/python

"""This file contains tree classes"""

#imports
from treelib import Node, Tree
import pandas as pd
import numpy as np
from math import log
import pdb
""" DecisionTreeClassifier()
    * decide based on maximum gain what to split each node on
    * implement ID3 algorithm
    * only work with categorical data (if this goes smoothly then do real too)
    * only needs to work with 2 classes to reduce complexity

    NODES:
        nodes can be either leaf nodes or internal nodes
            internal nodes --> contain attribute to be split on
            leaf nodes --> do not contain attribute to split on
        attribute_name
        pos_count
        neg_count
        data--> subset of y

"""
class DecisionTreeNode(object):
    """class for node in decision tree"""
    def __init__(self, data, attribute = None):
        self.neg_count, self.pos_count = get_class_counts(data)
        self.attribute = attribute
        self.data = data

class DecisionTreeClassifier(object):
    def  __init__(self):
        self.tree = Tree()

    def split(self, node, data):
        attr = get_best_attribute(data)
        attr_vals = get_values(data, attr)

        for val in attr_vals:
            new_data = data[data[attr] == val]
            new_node = self.tree.create_node(val,
                                             len(self.tree.all_nodes()),
                                             node.identifier,
                                             data =
                                             DecisionTreeNode(data = new_data, attribute = attr))
            if not(is_pure(new_data)):
                self.split(new_node, new_data)






#functions

def get_best_attribute(data):
    """function to return the best attribute to split on based on a dataframe where columns represent attributes. Dataframe: data must contain a one hot encoded column named 'class'
    >>> df = pd.read_csv('./datasets/test.csv').drop('Unnamed: 0', axis = 1)
    >>> get_best_attribute(df)
    'outlook'
    """
    X = data.drop('class', axis = 1)
    y = pd.DataFrame({'class': data['class']})
    num_neg, num_pos = get_class_counts(y)
    best_attr, best_gain = X.columns[0], 0

    for attr_name, attr_val in X.iteritems():
        attr_gain = gain(data, num_pos, num_neg, attr_name)
        if attr_gain > best_gain:
            best_attr, best_gain = attr_name, attr_gain

    return best_attr

def entropy(num_pos, num_neg):
    """Return the entropy given the # of positive classes and # of negative classes
    >>> entropy(3, 3)
    1.0
    >>> entropy(4,0)
    0.0
    """
    total_count = num_pos + num_neg
    p_pos = num_pos / total_count
    p_neg = num_neg / total_count

    if (p_pos == 0 or p_neg == 0):
        return 0.0  #entropy has to be zero if either of thes probabilities is 0

    return (p_pos * log(p_pos, 2) + p_neg * log(p_neg, 2)) * -1

def gain(s, num_pos, num_neg, attribute):
    """Return the expected gain after splitting on a given attribute
    >>> wind = ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak','Weak', 'Strong', 'Strong', 'Weak', 'Strong']
    >>> y = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]
    >>> s = pd.DataFrame({'wind': wind, 'class': y})
    >>> '%.3f' % gain(s, 9, 5, 'wind')
    '0.048'
    """

    prev_entropy = entropy(num_pos, num_neg)

    s_size = s.shape[0]

    values = get_values(s, attribute)

    for value in values:
        s_v = s[s[attribute] == value]
        s_v_size = s_v.shape[0]
        num_pos, num_neg = get_class_counts(s_v)
        prev_entropy -= (s_v_size / s_size) * entropy(num_pos, num_neg)

    return prev_entropy

def get_values(data, attribute):
    """gets all values from data[attribute] where data is a Pandas DataFrame
    >>> data = pd.read_csv('./datasets/credit.csv', delimiter = ',')
    >>> get_values(data, 'Housing')
    array(['own', 'free', 'rent'], dtype=object)
    """
    return data[attribute].unique()

def get_class_counts(data):
    """returns count of positive classes and negative classes from DataFrame,
    DataFrame: data needs to have class in a column named 'class'
    >>> y = [0, 0, 0, 1, 1]
    >>> data = pd.DataFrame({'class': y})
    >>> get_class_counts(data)
    (3, 2)
    """
    _, counts = np.unique(data['class'], return_counts = True)

    if (len(counts) == 1):
        counts = np.append(counts, 0)

    #TODO implement this without np.unique, it will fail on pure set

    return (counts[0], counts[1])

def is_pure(data):
    '''
    >>> y = [0,0,0]
    >>> data = pd.DataFrame({'class': y})
    >>> is_pure(data)
    True
    >>> y = [1,0,0]
    >>> data = pd.DataFrame({'class': y})
    >>> is_pure(data)
    False
    '''
    num_pos, num_neg = get_class_counts(data)
    return (num_pos == 0 or num_neg == 0)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    data = pd.read_csv('./datasets/test.csv').drop('Unnamed: 0', axis = 1)
    clas = DecisionTreeClassifier()
    node = clas.tree.create_node("Root", "root", data = DecisionTreeNode(data))
    clas.split(node, node.data.data)
    clas.tree.show()
