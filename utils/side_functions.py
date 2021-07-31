# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 14:59:55 2021

"""
from greedysearch.pipeline import features_transform, feature_normalization

def flexible_fit_classifier(GSS,k, X_train, y_train, X_test, y_test, classifier, scoring_function):
    """
    Tests the performance of a classifier that is first trained  and then tested according to a specified scoring function.
    --- > IT IS FIT_CLASSIFIER FUNCTION BUT WITH THE POSSIBILITY OF SELECTING ONLY k- shapelets 
    (YOU CAN RUN GSS only ONCE WITH MAX NUMBER OF SHAPELETS)
    """
    
    if k > len(GSS.top_shapelets) : 
        raise ValueError("k cannot be bigger than the number of top shapelets of the model"
                             " (got {})."
                             .format(k))
        
    
    # Apply the feature pipeline to the training set and testing set to get the min distances of each shapelet
    features_train = features_transform(X_train, GSS)[:,:k]
    features_test = features_transform(X_test, GSS)[:,:k]
    # Normalizing min distances
    features_train_norm, features_test_norm = feature_normalization(features_train, features_test)
    classifier.fit(features_train_norm, y_train)
    y_pred = classifier.predict(features_test_norm)
    return scoring_function(y_test, y_pred)



def compute_result(shapelet_model, k, X_train, y_train, X_test, y_test,  classifier, scoring_function) :
    
    """
     gets the score for a given shapelet model, with a given number of features considered 
     with a given classifier using a given scoring function
     Stores results in a dict with appropriate name depending on these parameters
    """
    score = flexible_fit_classifier(shapelet_model,k, X_train, y_train, X_test, y_test, classifier, scoring_function)
    classifier_name = type(classifier).__name__
    shapelet_model_name = type(shapelet_model).__name__
    scoring_function_name = scoring_function.__name__
    
    key = shapelet_model_name +'_' +str(k)+'shapelets_'+ classifier_name + '_' + scoring_function_name
    res = {key : score}
    
    return res