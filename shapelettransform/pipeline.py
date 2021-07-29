"""
Functions used for the evaluation pipeline.
"""
import numpy as np

def features_transform(X, GSS):
    """
    Transformation pipeline for a new dataset X.
    Requires a dataset X (samples x time series).
    Requires a GreedyShapeletSearch object WITH top shapelets computed. (See algorithm module)
    """
    features =  []
    for _, _, _, _, shapelet_size, shapelet in GSS.top_shapelets:
        # Normalize shapelet
        shapelet_norm = GSS.standardize_samples_candidates(shapelet, axis=0)
        # Window the data
        windowed_test = GSS.rolling_window(X, window=shapelet_size)
        # Normalize the windowed data
        windowed_test_norm = GSS.standardize_samples_candidates(windowed_test)
        # Calculate features for shapelet and X
        shapelet_features = ((windowed_test_norm-shapelet_norm)**2).sum(axis=-1).min(axis=-1)
        features.append(shapelet_features)
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

def feature_normalization(features_train, features_test):
    """
    Normalizes features. Careful with axis.
    """
    features_train_norm = (features_train-features_train.mean(axis=0))/features_train.std(axis=0)
    features_test_norm = (features_test-features_train.mean(axis=0))/features_train.std(axis=0)
    return features_train_norm, features_test_norm
