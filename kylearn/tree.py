#!/Users/kylemoore/miniconda3/bin/python

"""This file contains tree classes"""

#imports
from treelib import Node, Tree
import pandas as pd
import numpy as np
from math import log
""" DecisionTreeClassifier()

    * decide based on maximum gain what to split each node on
    * implement ID3 algorithm
    * only work with categorical data (if this goes smoothly then do real too)
    * only needs to work with 2 classes to reduce complexity

    NODES:
        attribute_name
        pos_count
        neg_count

"""

class DecisionTreeClassifier(object):
    def  __init__(self):
        self.tree = Tree()

    def entropy(self, num_pos, num_neg):
        """Return the entropy given the # of positive classes and # of negative classes
        >>> entropy(3, 3)
        1
        >>> entropy(4,0)
        0
        """
        total_count = num_pos + num_neg
        p_pos = num_pos / total_count
        p_neg =  num_neg / total_count
        return (p_pos * log(p_pos,2) + p_neg * log(p_neg, 2)) * -1


if __name__ == "__main__":
    clas = DecisionTreeClassifier()
    print(clas.entropy(3,3))
