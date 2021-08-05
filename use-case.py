# -*- coding: utf-8 -*-

# %%

from data_loader import retrieve
from data_loader import preprocess
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from greedysearch.algorithm import GreedyShapeletSearch, fit_svm
from greedysearch.pipeline import features_transform, fit_classifier
from shapelettransform.algorithm import ShapeletTransform
from utils.dump_results import dump_object, dump_list
from utils.dump_results import load_object
from utils.plot_results import create_dataframe
from utils.plot_results import plot_swarmplot
from utils.plot_results import plot_top_shapelets


# %%
anomalies = retrieve.anormal_data('fifteen')
anomalies = [preprocess.parse(i.data) for i in anomalies]
normal = retrieve.normal_data(len(anomalies)*10)
normal = [preprocess.parse(i.data) for i in normal]


# %%
dump_object('anomalies', anomalies)
dump_object('normal', normal)


# %%

def prepare_data(imbalance, split, anomaly_type):
    anomalies = preprocess.build_array(retrieve.anormal_data(anomaly_type))
    normal = preprocess.build_array(retrieve.normal_data(anomalies.shape[0]*imbalance))
    X = np.concatenate([anomalies, normal])
    y = np.concatenate([np.ones(anomalies.shape[0]), np.zeros(normal.shape[0])])

    return train_test_split(X, y, test_size=split, random_state=42)

# %%

classifiers = [
('decision tree', DecisionTreeClassifier()), 
('1nn', KNeighborsClassifier(1)), 
('naive bayes', GaussianNB()),
('random forest', RandomForestClassifier()),
('svm', SVC(kernel='linear'))]

def evaluate_shapelets(GSS, model, X_train, y_train, X_test, y_test):
    scores = []
    for clf in classifiers:
        score = fit_classifier(GSS, X_train, y_train, X_test, y_test, clf[1], f1_score)
        scores.append((model, clf[0], score))

    dump_list(scores, 'imbalance_benchmark')


anomaly_types = ['two', 'four', 'fifteen', 'seventeen']

for a in anomaly_types:

    dump_list(a, 'imbalance_benchmark')

    X_train, X_test, y_train, y_test = prepare_data(imbalance = 20, split = 0.5, anomaly_type = a)

    ST = ShapeletTransform()
    ST.get_top_k_shapelets(X_train=X_train, y_train=y_train, n_shapelets=3, 
    shapelet_min_size=25, shapelet_max_size=45)
    dump_object(f'top3_shapelets_a_{a}_ST', ST)
    evaluate_shapelets(GSS, 'ST', X_train, y_train, X_test, y_test)

    GSS= GreedyShapeletSearch()
    GSS.get_top_k_shapelets(X_train=X_train, y_train=y_train, scoring_function=fit_svm, 
    n_shapelets=3, shapelet_min_size=25, shapelet_max_size=45)
    dump_object(f'top3_shapelets_a_{a}_GSS', GSS)
    evaluate_shapelets(GSS, 'GSS', X_train, y_train, X_test, y_test)


# %%
