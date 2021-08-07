"""
Contains the algorithm core and a sample 'fit_svm' scoring function.
The output of the GreedyShapeletSearch can be found in the attribute 'top_shapelets'.
"""
import time
import numpy as np
import pandas as pd
import scipy.stats as sps
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from matrixprofile.algorithms import mpx


class GreedyShapeletSearch():
    def __init__(self):
        self.shapelets = []
        self.exclusion_zone = {}
        # Contains the minimum distances of the found shapelets to the other samples
        self.features = []
        # Containes the final output. Format: (sample_idx, candidate_idx, score, margin, shapelet_size, shapelet)
        self.top_shapelets = []
        # Raw samples
        self.raw_samples = []

    @staticmethod
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def get_candidate_mins(self, sample_data, shapelet_size = 10):
        """
        Function that calculates the distance of all candidates of a given data set to all all other candidates.
        CAREFUL:
        - memory blows up quickly.
        - contains the zeros (distance of candidates to itself)
        """
        # Window the array
        windowed_data = self.rolling_window(sample_data, shapelet_size)
        # Standardize candidates
        windowed_data = self.standardize_samples_candidates(windowed_data)
        distances = []
        for sample_candidates in windowed_data:
            candidate_distances = np.array([((windowed_data - candidate)**2).sum(axis=-1).min(axis=-1) for candidate in sample_candidates])
            distances.append(candidate_distances)
        return np.stack(distances)
    
    def get_top_k_shapelets(self, X_train, y_train, scoring_function, n_shapelets=1, shapelet_min_size = 10, shapelet_max_size=20):

        for i in range(n_shapelets):
            start = time.time()
            print(f"Searching for shapelet {i}...")
            self.shapelets = []
            self.main_event_loop(X_train, y_train, scoring_function, shapelet_min_size = shapelet_min_size, shapelet_max_size = shapelet_max_size)
            print(f"Found shapelet {i} at sample: {self.top_shapelets[i][0]}, candidate: {self.top_shapelets[i][1]}.")
            print("Time taken: ", time.time()-start)

        for sample_idx, _, _, _, _, _ in self.top_shapelets:
            self.raw_samples.append(X_train[sample_idx])



    def main_event_loop(self, X_train, y_train, scoring_function, shapelet_min_size = 30, shapelet_max_size = 31):
        """
        The main event loop contains the series of steps required for the algorithm.
        """
        for shapelet_size in range(shapelet_min_size, shapelet_max_size):
            # Calculate all of the candidate minimums throughout the dataset - shape: n_samples, n_samples, n_candidate
            profiles = self.get_candidate_mins(X_train, shapelet_size)
            # Extract a shapelet for n_shapelets
            self.evaluate_candidates(profiles, y_train, shapelet_size, scoring_function)
            # Printing progress
            if ((shapelet_size-shapelet_min_size)/(shapelet_max_size-shapelet_min_size))*100 % 10 == 0:
                print(f"Finished {round(((shapelet_size-shapelet_min_size)/(shapelet_max_size-shapelet_min_size))*100,2)} percent of time step...")
        # Retrieve top shapelets
        self.retrieve_top_shapelet(X_train)
    
    def evaluate_candidates(self, profiles, y_train, shapelet_size, scoring_function):
        """
        Extracts a (greedy) optimal shapelet.
        """
        # Iterate through all samples in profiles
        for sample_idx in range(profiles.shape[0]):
            # The minimum distances of the given samples candidates to all other samples - shape: n_samples, n_candidates
            sample = profiles[sample_idx]
            # Iterate through all candidate distances of the given sample
            for candidate_idx in range(sample.shape[0]):
                # The minimum distances of a candidate to all other samples
                candidate = sample[candidate_idx,:]
                # Add features if other shapelets have been extracted
                features = self.get_features(candidate)
                # Score candidate
                score, margin = scoring_function(features, y_train)
                self.shapelets.append((sample_idx, candidate_idx, score, margin, shapelet_size, candidate))

    def get_features(self,candidate):
        """
        If shapelets have been extracted, returns the min distances of all already extracted shapelets + candidate as features
        """
        if len(self.features) == 0:
            return np.array(candidate).reshape((candidate.shape[0],1))
        return np.array([candidate]+self.features).T

    def retrieve_top_shapelet(self, X_train):
        # Sort shapelets according to info gain descending
        self.shapelets.sort(key=lambda x: (x[2],x[3]), reverse=True)

        top_shapelet_found = False
        for sample_idx, candidate_idx, score, margin, shapelet_size, candidate in self.shapelets:
            # If the correct number of shapelets was found, break out of loop
            if top_shapelet_found:
                break
            # Check if sample index of candidate in exclusion zone samples
            if sample_idx not in self.exclusion_zone.keys():
                self.exclusion_zone[sample_idx] = []
            else:
                # Otherwise check if candidate index is in exclusion zone of sample
                if candidate_idx in self.exclusion_zone[sample_idx]:
                    continue
            # Extend exclusion zone
            self.exclusion_zone[sample_idx].extend(list(range(candidate_idx - shapelet_size, candidate_idx+shapelet_size)))
            # Add profile features to features
            self.features.append(candidate)
            # Add shapelet to top shapelets
            self.top_shapelets.append((sample_idx, candidate_idx, score, margin, shapelet_size, X_train[sample_idx,candidate_idx:candidate_idx+shapelet_size]))
            # Signal shapelet found
            top_shapelet_found = True


    def calculate_infogain(self, candidate, y_train, target_class = 1):
        """
        Given a 1-d array, calculates the infogain according the labels y_train.
        """
        # Initialize decision tree classifier
        clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=1)
        # Fit decision tree
        clf.fit(candidate, y_train)
        self.clf=clf
        # Get entropy before best split
        entropy_before = clf.tree_.impurity[0]
        # Get entropy after best split
        entropy_after = clf.tree_.value.sum(-1)[1]/clf.tree_.value.sum(-1)[0] * clf.tree_.impurity[1] + \
            clf.tree_.value.sum(-1)[2]/clf.tree_.value.sum(-1)[0] * clf.tree_.impurity[2]

        # Calculate margin
        if len(set(y_train)) != 2:
            print(f"There is something wrong with the number of labels! The number o labels is {len(set(y_train))}.")
            raise ValueError
        label_avgs = [candidate[y_train==label].mean() for label in set(y_train)]
        margin = abs(label_avgs[0]-label_avgs[1])
        # Return information gain
        return entropy_before - entropy_after, margin

    @staticmethod
    def standardize_samples_candidates(samples, axis=2):
        """
        Standardized each shapelet candidate (after windowing).
        """
        return (samples-np.expand_dims(samples.mean(axis=axis),axis))/np.expand_dims(samples.std(axis=axis),axis)

def fit_svm(X,Y, target_class=1):
    """
    Fitting a SVM and returning the f1 score and the calculated margin.
    """
    # Initialize the classifier
    clf = SVC(kernel='linear', class_weight='balanced')
    # Adjust the dimensions of X if necessary
    if len(X.shape) == 1:
       X = X.reshape(-1, 1) 
    # Normalize the input data
    X_norm = (X-X.mean(axis=0))/X.std(axis=0)
    # Fit SVM
    clf.fit(X_norm, Y)
    # Calculate margin
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    # Predict
    Y_pred = clf.predict(X_norm)
    # Calculate info gain
    entropy_before = calculate_entropy(Y)
    # Start with a 'pure' entropy 
    entropy_after = 0
    # Iterating through classes
    for label in set(Y):
        # Retrieve the true labels of all instances classified as 'label'
        partial_data = Y[Y_pred == label]
        # Add the weighted entropy to the entropy after
        entropy_after += len(partial_data)/len(Y_pred) * calculate_entropy(partial_data)

    return entropy_before-entropy_after, margin

def calculate_entropy(data):
    """
    Helper function to calculate the entropy of a data set.
    """
    pd_series = pd.Series(data)
    counts = pd_series.value_counts()
    entropy = sps.entropy(counts, base=2)
    return entropy


class GreedyShapeletSearchVL():
    def __init__(self):
        self.shapelets = []
        self.exclusion_zone = {}
        # Contains the minimum distances of the found shapelets to the other samples
        self.features = []
        # Containes the final output. Format: (sample_idx, candidate_idx, score, margin, shapelet_size, shapelet)
        self.top_shapelets = []
        # Raw samples
        self.raw_samples = []

    @staticmethod
    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    
    def get_candidate_mins(self, sample_data, shapelet_size = 30):
        

        # Storing indices to only retrieve minima across specific samples
        sample_indices = []
        idx = 0
        for sample in sample_data:
            sample_indices.append((idx,idx+len(sample)-shapelet_size))
            idx += len(sample)

        # Calculating profile
        start = time.time()
        profiles = []
        for sample in sample_data:

            sample_minima = np.stack([mpx(sample, shapelet_size, data_sample, n_jobs=1)['mp'] for data_sample in sample_data]).T
            profiles.append(sample_minima)

        print("Time taken: ", time.time()-start)
        return profiles 
    
   
    def get_top_k_shapelets(self, X_train, y_train, scoring_function, n_shapelets=1, shapelet_min_size = 10, shapelet_max_size=20):

        for i in range(n_shapelets):
            start = time.time()
            print(f"Searching for shapelet {i}...")
            self.shapelets = []
            self.main_event_loop(X_train, y_train, scoring_function, shapelet_min_size = shapelet_min_size, shapelet_max_size = shapelet_max_size)
            print(f"Found shapelet {i} at sample: {self.top_shapelets[i][0]}, candidate: {self.top_shapelets[i][1]}.")
            print("Time taken: ", time.time()-start)

        for sample_idx, _, _, _, _, _ in self.top_shapelets:
            self.raw_samples.append(X_train[sample_idx])

        self.shapelets = []
        self.features = []



    def main_event_loop(self, X_train, y_train, scoring_function, shapelet_min_size = 30, shapelet_max_size = 31):
        """
        The main event loop contains the series of steps required for the algorithm.
        """
        for shapelet_size in range(shapelet_min_size, shapelet_max_size):
            # Calculate all of the candidate minimums throughout the dataset - shape: n_samples, n_samples, n_candidate
            profiles = self.get_candidate_mins(X_train, shapelet_size)
            # Extract a shapelet for n_shapelets
            self.evaluate_candidates(profiles, y_train, shapelet_size, scoring_function)
            # Printing progress
            if ((shapelet_size-shapelet_min_size)/(shapelet_max_size-shapelet_min_size))*100 % 10 == 0:
                print(f"Finished {round(((shapelet_size-shapelet_min_size)/(shapelet_max_size-shapelet_min_size))*100,2)} percent of time step...")
        # Retrieve top shapelets
        self.retrieve_top_shapelet(X_train)
    
    def evaluate_candidates(self, profiles, y_train, shapelet_size, scoring_function):
        """
        Extracts a (greedy) optimal shapelet.
        """
        # Iterate through all samples in profiles
        for sample_idx in range(len(profiles)):
            # The minimum distances of the given samples candidates to all other samples - shape: n_samples, n_candidates
            sample = profiles[sample_idx]
            # Iterate through all candidate distances of the given sample
            for candidate_idx in range(sample.shape[0]):
                # The minimum distances of a candidate to all other samples
                candidate = sample[candidate_idx,:]
                # Add features if other shapelets have been extracted
                features = self.get_features(candidate)
                # Score candidate
                try:
                    score, margin = scoring_function(features, y_train)
                except Exception as e:
                    print(f"couldnt compute score and margin for sample {sample_idx} candidate {candidate_idx}")
                    score = 0
                    margin = 0
                self.shapelets.append((sample_idx, candidate_idx, score, margin, shapelet_size, candidate))

    def get_features(self,candidate):
        """
        If shapelets have been extracted, returns the min distances of all already extracted shapelets + candidate as features
        """
        if len(self.features) == 0:
            return np.array(candidate).reshape((candidate.shape[0],1))
        return np.array([candidate]+self.features).T

    def retrieve_top_shapelet(self, X_train):
        # Sort shapelets according to info gain descending
        self.shapelets.sort(key=lambda x: (x[2],x[3]), reverse=True)

        top_shapelet_found = False
        for sample_idx, candidate_idx, score, margin, shapelet_size, candidate in self.shapelets:
            # If the correct number of shapelets was found, break out of loop
            if top_shapelet_found:
                break
            # Check if sample index of candidate in exclusion zone samples
            if sample_idx not in self.exclusion_zone.keys():
                self.exclusion_zone[sample_idx] = []
            else:
                # Otherwise check if candidate index is in exclusion zone of sample
                    continue
            # Extend exclusion zone
            self.exclusion_zone[sample_idx].extend(list(range(candidate_idx - shapelet_size, candidate_idx+shapelet_size)))
            # Add profile features to features
            self.features.append(candidate)
            # Add shapelet to top shapelets
            self.top_shapelets.append((sample_idx, candidate_idx, score, margin, shapelet_size, X_train[sample_idx][candidate_idx:candidate_idx+shapelet_size]))
            # Signal shapelet found
            top_shapelet_found = True


    def calculate_infogain(self, candidate, y_train, target_class = 1):
        """
        Given a 1-d array, calculates the infogain according the labels y_train.
        """
        # Initialize decision tree classifier
        clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=1)
        # Fit decision tree
        clf.fit(candidate, y_train)
        self.clf=clf
        # Get entropy before best split
        entropy_before = clf.tree_.impurity[0]
        # Get entropy after best split
        entropy_after = clf.tree_.value.sum(-1)[1]/clf.tree_.value.sum(-1)[0] * clf.tree_.impurity[1] + \
            clf.tree_.value.sum(-1)[2]/clf.tree_.value.sum(-1)[0] * clf.tree_.impurity[2]

        # Calculate margin
        if len(set(y_train)) != 2:
            print(f"There is something wrong with the number of labels! The number o labels is {len(set(y_train))}.")
            raise ValueError
        label_avgs = [candidate[y_train==label].mean() for label in set(y_train)]
        margin = abs(label_avgs[0]-label_avgs[1])
        # Return information gain
        return entropy_before - entropy_after, margin

    @staticmethod
    def standardize_samples_candidates(samples, axis=2):
        """
        Standardized each shapelet candidate (after windowing).
        """
        return (samples-np.expand_dims(samples.mean(axis=axis),axis))/np.expand_dims(samples.std(axis=axis),axis)

def fit_svm(X,Y, target_class=1):
    """
    Fitting a SVM and returning the f1 score and the calculated margin.
    """
    # Initialize the classifier
    clf = SVC(kernel='linear', class_weight='balanced')
    # Adjust the dimensions of X if necessary
    if len(X.shape) == 1:
       X = X.reshape(-1, 1) 
    # Normalize the input data
    # Fit SVM
    X_norm = (X-X.mean(axis=0))/X.std(axis=0)
    clf.fit(X_norm, Y)
    
    # Calculate margin
    margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    # Predict
    Y_pred = clf.predict(X_norm)
    # Calculate info gain
    entropy_before = calculate_entropy(Y)
    # Start with a 'pure' entropy 
    entropy_after = 0
    # Iterating through classes
    for label in set(Y):
        # Retrieve the true labels of all instances classified as 'label'
        partial_data = Y[Y_pred == label]
        # Add the weighted entropy to the entropy after
        entropy_after += len(partial_data)/len(Y_pred) * calculate_entropy(partial_data)
    return entropy_before-entropy_after, margin

def calculate_entropy(data):
    """
    Helper function to calculate the entropy of a data set.
    """
    pd_series = pd.Series(data)
    counts = pd_series.value_counts()
    entropy = sps.entropy(counts, base=2)
    return entropy
