"""
Functions used for the evaluation pipeline.
"""
import numpy as np

from matrixprofile.algorithms import mpx

def feature_normalization(features_train, features_test):
    """
    Normalizes features. Careful with axis.
    """
    features_train_norm = (features_train-features_train.mean(axis=0))/features_train.std(axis=0)
    features_test_norm = (features_test-features_train.mean(axis=0))/features_train.std(axis=0)
    return features_train_norm, features_test_norm



def features_transform(X, GSS):
    """
    Transformation pipeline for a new dataset X.
    Requires a dataset X (samples x time series).
    Requires a GreedyShapeletSearch object WITH top shapelets computed. (See algorithm module)
    """
    features =  []
    for _, _, _, _, shapelet_size, shapelet in GSS.top_shapelets:
        # Get starting and ending indices of windowed profile of all samples in X
        sample_indices = []
        idx = 0
        for sample in X:
            sample_indices.append((idx,idx+len(sample)-shapelet_size))
            idx += len(sample)
        # Flattens the input data for querying
        data_flat = np.concatenate(X)
        # Calculates the profile for the shapelet
        shapelet_profile = mpx(data_flat, shapelet_size, shapelet, n_jobs=1)
        # Retrieves the minima within each evaluated sample
        shapelet_minima = [min(shapelet_profile['mp'][st:ed]) for st, ed in sample_indices]

        features.append(shapelet_minima)

    return np.array(features).T

def fit_classifier(GSS, X_train, y_train, X_test, y_test, classifier, scoring_function):
    """
    Tests the performance of a classifier that is first trained  and then tested according to a specified scoring function.
    """
    # Apply the feature pipeline to the training set and testing set to get the min distances of each shapelet
    features_train = features_transform(X_train, GSS)
    features_test = features_transform(X_test, GSS)

    # Normalizing min distances
    features_train_norm, features_test_norm = feature_normalization(features_train, features_test)
    classifier.fit(features_train_norm, y_train)
    y_pred = classifier.predict(features_test_norm)
    return scoring_function(y_test, y_pred)

def train_predict(GSS, X_train, y_train, X_test, classifier):
    """
    Trains a classifier on train set and applied on test set
    """
    # Apply the feature pipeline to the training set and testing set to get the min distances of each shapelet
    features_train = features_transform(X_train, GSS)
    features_test = features_transform(X_test, GSS)

    # Normalizing min distances
    features_train_norm, features_test_norm = feature_normalization(features_train, features_test)
    classifier.fit(features_train_norm, y_train)
    return classifier.predict(features_test_norm)