# -*- coding: utf-8 -*-

# %%
from greedysearch.algorithm import GreedyShapeletSearch, fit_svm
from greedysearch.pipeline import features_transform
from shapelettransform.algorithm import ShapeletTransform
from utils.dump_results import dump_object
from utils.dump_results import load_object
from utils.plot_results import create_dataframe
from utils.plot_results import plot_swarmplot
from utils.plot_results import plot_top_shapelets

from pyts.datasets import load_gunpoint
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %% extract shapelets and dump results
'''
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

ST = ShapeletTransform()
ST.get_top_k_shapelets(X_train=X_train, y_train=y_train, n_shapelets=5, 
shapelet_min_size=30, shapelet_max_size=40)
dump_object('01_top5_shapelet_GunPoint_ST', ST)

GSS= GreedyShapeletSearch()
GSS.get_top_k_shapelets(X_train=X_train, y_train=y_train, scoring_function=fit_svm, 
n_shapelets=75, shapelet_min_size=25, shapelet_max_size=45)
dump_object('01_top75_shapelet_GunPoint_GSS', GSS)
'''



# %% figure 1
'''
visualize mindist of top shapelet for GSS and ST for both training and test data
goal: demonstrate, that (soft) margin leads to more robust shapelets than IG
'''
GSS = load_object('top75_shapelets_GunPoint_GSS') 
ST = load_object('top5_shapelets_GunPoint_ST')

X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

df_gss_train = create_dataframe(features_transform(X_train, GSS)[:,0], y_train, 'train', 'GSS')
df_gss_test = create_dataframe(features_transform(X_test, GSS)[:,0], y_test, 'train', 'GSS')
df_gss = pd.concat([df_gss_train, df_gss_test])
plot_swarmplot(df_gss, 'swarmplot_gss_top_shapelet')

df_st_train = create_dataframe(features_transform(X_train, ST)[:,0], y_train, 'train', 'ST')
df_st_test = create_dataframe(features_transform(X_test, ST)[:,0], y_test, 'train', 'ST')
df_st = pd.concat([df_gss_train, df_gss_test])
plot_swarmplot(df_st, 'swarmplot_st_top_shapelet')




# %% figure 2
'''
visualize top 5 shapelets for GSS and ST
goal: demonstrate, that GSS finds complementary shapelets
'''

GSS = load_object('top75_shapelets_GunPoint_GSS') 
ST = load_object('top5_shapelets_GunPoint_ST')

X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

plot_top_shapelets(ST, y_train, 'top5_shapelets_ST')
plot_top_shapelets(GSS, y_train, 'top5_shapelets_GSS')





# %%
'''
Repeat the benchmark in from Hills et al. for GSS
'''

# Evaluation Pipeline
# Initialize SVM
clf = SVC(kernel='linear',class_weight='balanced')
# Evaluate the X_test
score = fit_classifier(GSS, X_train, y_train, X_test, y_test, clf, f1_score)

