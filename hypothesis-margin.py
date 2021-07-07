# -*- coding: utf-8 -*-
"""
The hypothesis-margin (used in the AdaBoost meta-algorithm) is an alternative to the sample-margin (used by the SVM). 
Essentially, it is the maximum distance you can travel from a sample point before being closer to the opposite class.
It is described in Gilad-Bachrach et al. 2004 "Margin based feature selection - theory and algorithms"
"""

# %% test time complexity for SVM and hypothesis-margin

import numpy as np
import time

def dummy_data(N=1000,n_target=10):
    X = np.random.rand(N, 1)
    y = np.concatenate([np.zeros(N-n_target),np.ones(n_target)])
    return X, y

X, y = dummy_data(10000, 10)



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C = N))

start = time.time()
clf.fit(X, y)
print(time.time()-start)



def closest_neighbor(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def hypothesis_margin(X, y, target_class):
    cumulative_margin = 0
    for x in X[y == target_class]:    
        margin = 1/2*(abs(x - closest_neighbor(X[y!=target_class], x)) - abs(x - closest_neighbor(X[y==target_class], x)))
        cumulative_margin += margin
    return cumulative_margin

start = time.time()
a = hypothesis_margin(X, y, 1)
print(time.time()-start)