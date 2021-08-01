# -*- coding: utf-8 -*-

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pyts.datasets import load_gunpoint
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from greedysearch.algorithm import GreedyShapeletSearch, fit_svm
from greedysearch.pipeline import features_transform, fit_classifier
from shapelettransform.algorithm import ShapeletTransform
from utils.dump_results import dump_object, dump_list
from utils.dump_results import load_object
from utils.plot_results import create_dataframe
from utils.plot_results import plot_swarmplot
from utils.plot_results import plot_top_shapelets





# %% extract shapelets and dump results
'''
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

ST = ShapeletTransform()
ST.get_top_k_shapelets(X_train=X_train, y_train=y_train, n_shapelets=5, 
shapelet_min_size=25, shapelet_max_size=45)
dump_object('top75_shapelets_GunPoint_ST', ST)


GSS= GreedyShapeletSearch()
GSS.get_top_k_shapelets(X_train=X_train, y_train=y_train, scoring_function=fit_svm, 
n_shapelets=75, shapelet_min_size=25, shapelet_max_size=45)
dump_object('top75_shapelets_GunPoint_GSS', GSS)
'''





# %% figure 1
'''
visualize mindist of top shapelet for GSS and ST for both training and test data
goal: demonstrate, that (soft) margin leads to more robust shapelets than IG
'''
GSS = load_object('top75_shapelets_GunPoint_GSS') 
ST = load_object('top75_shapelets_GunPoint_ST')
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

df_gss_train = create_dataframe(features_transform(X_train, GSS)[:,0], y_train, 'train', 'GSS')
df_gss_test = create_dataframe(features_transform(X_test, GSS)[:,0], y_test, 'test', 'GSS')
df_gss = pd.concat([df_gss_train, df_gss_test])
plot_swarmplot(df_gss, 'swarmplot_gss_top_shapelet')

df_st_train = create_dataframe(features_transform(X_train, ST)[:,0], y_train, 'train', 'ST')
df_st_test = create_dataframe(features_transform(X_test, ST)[:,0], y_test, 'test', 'ST')
df_st = pd.concat([df_st_train, df_st_test])
plot_swarmplot(df_st, 'swarmplot_st_top_shapelet')





# %% figure 2
'''
visualize top 5 shapelets for GSS and ST
goal: demonstrate, that GSS finds complementary shapelets
'''

GSS = load_object('top75_shapelets_GunPoint_GSS')
ST = load_object('top75_shapelets_GunPoint_ST')
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

plot_top_shapelets(ST, y_train, 'top5_shapelets_ST')
plot_top_shapelets(GSS, y_train, 'top5_shapelets_GSS')





# %% classification benchmark
'''
Repeat the benchmark in Hills et al. for GSS

all classifiers are taken from sklearn:
- decision tree classifier uses CART algorithm instead of C4.5 (very similar)
- KNN with N = 1 for 1NN classifier
- Gaussian Naive Bayes for Naive Bayes
- Bayesian network omitted
- Random forest
- Rotation forest omitted
- SVM

scaling of feature is performed within the fit_classifier function as part of the pipeline
'''

GSS = load_object('top75_shapelets_GunPoint_ST')
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

classifiers = [
('decision tree', DecisionTreeClassifier()), 
('1nn', KNeighborsClassifier(1)), 
('naive bayes', GaussianNB()),
('random forest', RandomForestClassifier()),
('svm', SVC(kernel='linear'))]

scores = []
for n in [1, 2, 3, 4, 5, 10, 75]:
    for clf in classifiers:
        score = fit_classifier(GSS, X_train, y_train, X_test, y_test, clf[1], accuracy_score)
        scores.append((n, clf[0], score))

dump_list(scores, 'classification_benchmark_ST')

















# %% imbalanced benchmark dataset

def sample_dataset(X_train, y_train):
    X_train_i = np.concatenate([X_train[y_train == 1], X_train[y_train == 2][:3]])
    y_train_i = np.concatenate([y_train[y_train == 1], y_train[y_train == 2][:3]])
    return X_train_i, y_train_i

'''
X_train_i, y_train_i = sample_dataset(X_train, y_train)

ST = ShapeletTransform()
ST.get_top_k_shapelets(X_train=X_train_i, y_train=y_train_i, n_shapelets=3, 
shapelet_min_size=25, shapelet_max_size=45)
dump_object('top3_shapelets_GunPoint_imbalanced3_ST', ST)

GSS= GreedyShapeletSearch()
GSS.get_top_k_shapelets(X_train=X_train_i, y_train=y_train_i, scoring_function=fit_svm, 
n_shapelets=3, shapelet_min_size=25, shapelet_max_size=45)
dump_object('top3_shapelets_GunPoint_imbalanced3_GSS', GSS)
'''

# %% imbalanced classification comparison GSS vs ST

GSS = load_object('top3_shapelets_GunPoint_imbalanced3_GSS')
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
X_train_i, y_train_i = sample_dataset(X_train, y_train)

classifiers = [
('decision tree', DecisionTreeClassifier()), 
('1nn', KNeighborsClassifier(1)), 
('naive bayes', GaussianNB()),
('random forest', RandomForestClassifier()),
('svm', SVC(kernel='linear'))]

scores = []
for n in [1, 2, 3]:
    for clf in classifiers:
        score = fit_classifier(GSS, X_train_i, y_train_i, X_test, y_test, clf[1], accuracy_score)
        scores.append((n, clf[0], score))

dump_list(scores, 'imbalanced_classification_benchmark_GSS')

# %%
