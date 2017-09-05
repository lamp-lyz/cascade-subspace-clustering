'''
If we use (external) classification evalutation measures like F1 or
accuracy for clustering evaluation, problems may arise.

One way to fix is to perform label matching.

Here we performs kmeans clustering on the Iris dataset and proceed to use
the Hungarian (Munkres) algorithm to correct the mismatched labeling.
'''

import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix

from munkres import Munkres

def make_cost_matrix(c1, c2):
    """
    """
    uc1 = np.unique(c1)
    uc2 = np.unique(c2)
    l1 = uc1.size
    l2 = uc2.size
    assert(l1 == l2 and np.all(uc1 == uc2))

    m = np.ones([l1, l2])
    for i in range(l1):
        it_i = np.nonzero(c1 == uc1[i])[0]
        for j in range(l2):
            it_j = np.nonzero(c2 == uc2[j])[0]
            m_ij = np.intersect1d(it_j, it_i)
            m[i,j] =  -m_ij.size
    return m

def translate_clustering(clt, mapper):
    return np.array([ mapper[i] for i in clt ])

def accuracy(cm):
    """computes accuracy from confusion matrix"""
    return np.trace(cm, dtype=float) / np.sum(cm)

def rectify_label(labels, classes):
    """entry point"""
    num_labels = len(np.unique(classes))
    print "label nums"
    print num_labels

    cm = confusion_matrix(classes, labels, labels=range(num_labels)) # gets the confusion matrix
    print "---------------------\nold confusion matrix:\n" \
          " %s\naccuracy: %.2f" % (str(cm), accuracy(cm))

    cost_matrix = make_cost_matrix(labels, classes)

    m = Munkres()
    indexes = m.compute(cost_matrix)
    mapper = { old: new for (old, new) in indexes }

    print "---------------------\nmapping:"
    for old, new in mapper.iteritems():
        print "map: %s --> %s" %(old, new)

    new_labels = translate_clustering(labels, mapper)
    new_cm = confusion_matrix(classes, new_labels, labels=range(num_labels))
    print "---------------------\nnew confusion matrix:\n" \
          " %s\naccuracy: %.2f" % (str(new_cm), accuracy(new_cm))
    return new_labels, accuracy(new_cm)


if __name__ == "__main__":
    pass
