# -*- coding: utf-8 -*-

# %%
from greedysearch.algorithm import GreedyShapeletSearch, fit_svm
from greedysearch.pipeline import features_transform
from shapelettransform.algorithm import ShapeletTransform
from utils.dump_results import dump_object
from utils.dump_results import load_object

# %%
'''
visualize mindist of top shapelet for GSS and ST for both training and test data
goal: demonstrate, that (soft) margin leads to more robust shapelets than IG
'''

from pyts.datasets import load_gunpoint
X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

GSS= GreedyShapeletSearch()
GSS.get_top_k_shapelets(X_train=X_train, y_train=y_train, scoring_function=fit_svm, 
n_shapelets=1, shapelet_min_size=30, shapelet_max_size=40)
dump_object('01_top_shapelet_GunPoint_GSS', GSS)

ST = ShapeletTransform()
ST.get_top_k_shapelets(X_train=X_train, y_train=y_train, n_shapelets=1, 
shapelet_min_size=30, shapelet_max_size=40)
dump_object('01_top_shapelet_GunPoint_ST', ST)


# %%

GSS = load_object('01_top_shapelet_GunPoint_GSS') 
ST_1 = load_object('01_top_shapelet_GunPoint_ST') 

# %% plot

GSS_1.features



# %% plot

import matplotlib.pyplot as plt
#plt.style.use('classic')
#%matplotlib inline
import numpy as np
import seaborn as sns
#sns.set()

# %%

sns.distplot(features_transform(X_train[y_train == 1], GSS))
sns.distplot(features_transform(X_train[y_train == 2], GSS))

plt.xlabel("Minimum distance")
plt.title("Feature distribution of the top shapelet for the GSS")
#plt.legend(["class 1",'class 2'])

# %%

sns.distplot(features_transform(X_test[y_test == 1], ST))
sns.distplot(features_transform(X_test[y_test == 2], ST), bins = 14)

plt.xlabel("Minimum distance")
plt.title("Feature distribution of the top shapelet for the GSS")
#plt.legend(["class 1",'class 2'])

# %%

a = features_transform(X_train[y_train == 1], ST).squeeze()
b = features_transform(X_train[y_train == 2], ST).squeeze()
c = features_transform(X_test[y_test == 1], ST).squeeze()
d = features_transform(X_test[y_test == 2], ST).squeeze()

sns.scatterplot(x=np.arange(len(a)),y=a, color = 'b', alpha=0.5)
sns.scatterplot(x=np.arange(len(b)),y=b, color = 'r', alpha = 0.5)
sns.scatterplot(x=np.arange(len(a),len(a)+len(c)),y=c, color = 'b')
sns.scatterplot(x=np.arange(len(b),len(b)+len(d)),y=d, color = 'r')


# %%

a = features_transform(X_train[y_train == 1], GSS).squeeze()
b = features_transform(X_train[y_train == 2], GSS).squeeze()
c = features_transform(X_test[y_test == 1], GSS).squeeze()
d = features_transform(X_test[y_test == 2], GSS).squeeze()

sns.scatterplot(x=np.arange(len(a)),y=a, color = 'b', alpha=0.5)
sns.scatterplot(x=np.arange(len(b)),y=b, color = 'r', alpha = 0.5)
sns.scatterplot(x=np.arange(len(a),len(a)+len(c)),y=c, color = 'b')
sns.scatterplot(x=np.arange(len(b),len(b)+len(d)),y=d, color = 'r')
# %%
