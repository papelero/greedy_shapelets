from pyts.datasets import load_gunpoint
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from shapelettransform.algorithm import ShapeletTransform, ShapeletTransformVL
from shapelettransform.pipeline import fit_classifier
from utils.dump_results import load_object

import numpy as np

if __name__ == "__main__":
    # Load data
    
    X_norm = load_object('normal')
    y_norm = [0]*len(X_norm)

    X_anom = load_object('anomalies')
    y_anom = [1]*len(X_anom)

    X_train = X_norm + X_anom
    y_train = y_norm + y_anom

    # X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

    # print(np.array(X_train).shape)

    # Greedy shapelet search
    # Initialize Shapelet transform object. NOTE: VL stands for variable length (list of variable length samples possible). THIS TAKES MUCH LONGER 
    ST = ShapeletTransformVL()
    # Retrieve a specified number of shapelets.
    ST.get_top_k_shapelets(X_train=X_train, y_train=y_train, n_shapelets=2, shapelet_min_size=30, shapelet_max_size=31)
    # ST.get_candidate_mins(X_train, shapelet_size=30)

    # Evaluation Pipeline
    # Initialize SVM
    clf = SVC(kernel='linear',class_weight='balanced')
    # Evaluate the X_test
    score = fit_classifier(ST, X_train, y_train, X_train, y_train, clf, f1_score)
    print("The following score was achieved on the test set: ", score)