from pyts.datasets import load_gunpoint
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from greedysearch.algorithm import GreedyShapeletSearch, fit_svm
from greedysearch.pipeline import features_transform, fit_classifier


if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)

    # Greedy shapelet search
    # Initialize GreedyShapeletSearch object
    GSS = GreedyShapeletSearch()
    # Retrieve a specified number of shapelets.
    GSS.get_top_k_shapelets(X_train=X_train, y_train=y_train, scoring_function=fit_svm, n_shapelets=1, shapelet_min_size=30, shapelet_max_size=40)

    # Evaluation Pipeline
    # Initialize SVM
    clf = SVC(kernel='linear',class_weight='balanced')
    # Evaluate the X_test
    score = fit_classifier(GSS, X_train, y_train, X_test, y_test, clf, f1_score)
    print("The following score was achieved on the test set: ", score)