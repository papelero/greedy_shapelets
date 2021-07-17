# -*- coding: utf-8 -*-
"""
The hypothesis-margin (used in the AdaBoost meta-algorithm) is an alternative to the sample-margin (used by the SVM). 
Essentially, it is the maximum distance you can travel from a sample point before being closer to the opposite class.
It is described in Gilad-Bachrach et al. 2004 "Margin based feature selection - theory and algorithms"
"""

# %% test time complexity for SVM and hypothesis-margin

import numpy as np
import time

def dummy_data(N=1000,n_target=10):
    X = np.random.rand(N, 1)
    y = np.concatenate([np.zeros(N-n_target),np.ones(n_target)])
    return X, y

X, y = dummy_data(10000, 10)



from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C = N))

start = time.time()
clf.fit(X, y)
print(time.time()-start)



def closest_neighbor(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def hypothesis_margin(X, y, target_class):
    cumulative_margin = 0
    for x in X[y == target_class]:    
        margin = 1/2*(abs(x - closest_neighbor(X[y!=target_class], x)) - abs(x - closest_neighbor(X[y==target_class], x)))
        cumulative_margin += margin
    return cumulative_margin

start = time.time()
a = hypothesis_margin(X, y, 1)
print(time.time()-start)




class GreedyShapeletSearch():
    def __init__(self):
        self.shapelets = []

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
        windowed_data = self.rolling_window(sample_data, shapelet_size)
        return np.array([abs(windowed_data - series).sum(axis=-1) for series in windowed_data])
    

    def main_event_loop(self, X_train, y_train, target_class = 1, n_shapelets=1, shapelet_size = 10):
        """
        The main event loop contains the series of steps required for the algorithm.
        """
        # Calculate all of the candidate minimums throughout the dataset - shape: n_samples, n_samples, n_candidate
        profiles = self.get_candidate_mins(X_train, shapelet_size)
        # Extract a shapelet for n_shapelets
        for _ in range(n_shapelets):
            self.extract_optimal_shapelet(profiles, y_train, target_class)


    def extract_optimal_shapelet(self, profiles, y_train, target_class):
        """
        Extracts a (greedy) optimal shapelet.
        """

        optimal_score = 0
        optimal_margin = 0
        optimal_shapelet = 0

        # Iterate through all samples in profiles
        for sample_idx in range(profiles.shape[0]):
            # The minimum distances of the given samples candidates to all other samples - shape: n_samples, n_candidates
            sample = profiles[sample_idx]
            # Iterate through all candidate distances of the given sample
            for candidate_idx in range(sample.shape[1]):
                # The minimum distances of a candidate to all other samples
                candidate = sample[:,candidate_idx]
                # Score candidate
                score, margin = self.score_candidate(candidate, sample, y_train, target_class)

                if score > optimal_score:
                    optimal_margin = margin
                    optimal_score = score
                    optimal_shapelet = (sample_idx, candidate_idx)
                elif score == optimal_score:
                    if margin > optimal_margin:
                        optimal_margin = margin
                        optimal_score = score
                        optimal_shapelet = (sample_idx, candidate_idx)
        self.shapelets.append((optimal_score, optimal_margin,optimal_shapelet))


    def score_candidate(self, candidate, y_train, target_class):
        """
        This function evaluates a candidate based on the hypothesis margin.
        """
        # Split dataset into target class and other
        candidates_target = candidate[y_train == target_class]
        candidates_other = candidate[y_train != target_class]

        return self.cumulative_hypothesis_margin(candidates_target, candidates_other)

    def cumulative_hypothesis_margin(self,candidates_target, candidates_other):
        """
        Calculate the cumulative margin.
        """
        # Compute the hypothesis margins for all candidates_target
        margins = [self.hypothesis_margin(x, np.delete(candidates_target,idx), candidates_other) for idx, x in enumerate(candidates_target)]
        # Sum of all margins
        cumulative_margin = sum(margins)
        # All margins that are positive mean that the nearest neighbor is of the same class
        score = len([margin for margin in margins if margin > 0])/len(margins) 
        return score, cumulative_margin

    def hypothesis_margin(self, x, candidates_target, candidates_other):
        """
        Calculate the hypothesis margin for a candidate.
        """
        margin_other = self.closest_neighbor(candidates_other, x)
        margin_target = self.closest_neighbor(candidates_target, x)
        margin = 1/2*(margin_other - margin_target)
        return margin

    def closest_neighbor(self, candidates, x):
        """
        Find the distance to the closest neighbor
        """
        if len(candidates.shape) == 1:
            candidates = np.expand_dims(candidates,axis=1)
        min_distance = (np.linalg.norm(candidates - x)).min()
        return min_distance





