from pyts.datasets import load_gunpoint
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from greedysearch.algorithm import GreedyShapeletSearch, GreedyShapeletSearchVL, fit_svm
from greedysearch.pipeline import features_transform, fit_classifier, fit_classifier_vl
from utils.dump_results import load_object

import numpy as np
if __name__ == "__main__":
    # Load data
    # X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)


    X_norm = load_object('normal')
    y_norm = [0]*len(X_norm)

    X_anom = load_object('anomalies')
    y_anom = [1]*len(X_anom)

    X_train = X_norm + X_anom
    y_train = y_norm + y_anom

    # Greedy shapelet search
    # Initialize GreedyShapeletSearch object
    # GSS = GreedyShapeletSearch()
    GSS = GreedyShapeletSearchVL()
    # Retrieve a specified number of shapelets.
    GSS.get_top_k_shapelets(X_train=X_train, y_train=np.array(y_train), scoring_function=fit_svm, n_shapelets=2, shapelet_min_size=30, shapelet_max_size=31)

    # Evaluation Pipeline
    # Initialize SVM
    clf = SVC(kernel='linear',class_weight='balanced')
    # Evaluate the X_test
    score = fit_classifier_vl(GSS, X_train, y_train, X_train, y_train, clf, f1_score)
    print("The following score was achieved on the test set: ", score)