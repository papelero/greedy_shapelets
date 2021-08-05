import time
import numpy as np
import mass_ts as mts
from sklearn.tree import DecisionTreeClassifier



class ShapeletTransform():
    def __init__(self):
        self.shapelets = []
        self.exclusion_zone = {}
        self.shapelet_size = 0
        self.top_shapelets = []
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

    def get_top_k_shapelets(self, X_train, y_train, n_shapelets=1, shapelet_min_size = 10, shapelet_max_size=20):

        for shapelet_size in range(shapelet_min_size,shapelet_max_size):
            self.main_event_loop(X_train, y_train, shapelet_size)

        # Sort shapelets according to info gain descending
        self.shapelets.sort(key=lambda x: (x[2],x[3]), reverse=True)

        # Retrieve top shapelets
        for _ in range(n_shapelets):
            self.retrieve_top_shapelet(X_train)
        
        self.top_shapelets.sort(key=lambda x: (x[2],x[3]), reverse=True)

        for sample_idx, _, _, _, _, _ in self.top_shapelets:
            self.raw_samples.append(X_train[sample_idx])
        

    def main_event_loop(self, X_train, y_train, shapelet_size = 10):
        """
        The main event loop contains the series of steps required for the algorithm.
        """
        # Store shapelet size
        self.shapelet_size = shapelet_size
        # Calculate all of the candidate minimums throughout the dataset - shape: n_samples, n_samples, n_candidate
        profiles = self.get_candidate_mins(X_train, shapelet_size)
        # Extract a shapelets
        self.extract_shapelets(profiles, y_train, shapelet_size)
     
    def extract_shapelets(self, profiles, y_train, shapelet_size):
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
                # Score candidate
                score, margin = self.calculate_infogain(candidate.reshape((candidate.shape[0],1)), y_train)
                self.shapelets.append((sample_idx, candidate_idx, score, margin, shapelet_size, candidate))
                
    def retrieve_top_shapelet(self, X_train):
        # Sort shapelets according to info gain descending
        self.shapelets.sort(key=lambda x: (x[2],x[3]), reverse=True)

        top_shapelet_found = False
        for sample_idx, candidate_idx, score, margin, shapelet_size, _ in self.shapelets:
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
            # Add shapelet to top shapelets
            self.top_shapelets.append((sample_idx, candidate_idx, score, margin, shapelet_size, X_train[sample_idx,candidate_idx:candidate_idx+shapelet_size]))
            # Signal shapelet found
            top_shapelet_found = True

    def calculate_infogain(self, candidate, y_train):
        """
        Given a 1-d array, calculates the infogain according the labels y_train.
        """
        # Initialize decision tree classifier
        clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=1)
        # Fit decision tree
        clf.fit(candidate, y_train)
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



class ShapeletTransformVL():
    def __init__(self):
        self.shapelets = []
        self.exclusion_zone = {}
        self.shapelet_size = 0
        self.top_shapelets = []
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

        sample_lengths = [len(sample) for sample in sample_data]
        samples_padded = np.stack([np.pad(sample, (0,max(sample_lengths)-len(sample)), 'constant') for sample in sample_data])
        # Window the array
        windowed_data = self.rolling_window(samples_padded, shapelet_size)
        # Standardize candidates
        windowed_data = self.standardize_samples_candidates(windowed_data)
        distances = []
        start = time.time()
        for sample_length, sample_candidates in zip(sample_lengths, windowed_data):
            # print(sample_candidates.shape)
            # return
            candidate_distances = np.stack([((windowed_data - candidate)**2).sum(axis=-1) for candidate in sample_candidates[:sample_length-shapelet_size]], axis=1)
            # candidate_distances = np.array([((windowed_data - candidate)**2).sum(axis=-1).min(axis=-1) for candidate in sample_candidates[:sample_length-shapelet_size]])
            candidate_distances_mins = np.stack([distance[:,:length-shapelet_size].min(axis=-1) for length, distance in zip(sample_lengths, candidate_distances)])
            distances.append(candidate_distances_mins.T)
        print("Time taken for candidate mins: ", time.time()-start)
        return distances

    def get_top_k_shapelets(self, X_train, y_train, n_shapelets=1, shapelet_min_size = 10, shapelet_max_size=20):

        for shapelet_size in range(shapelet_min_size,shapelet_max_size):
            self.main_event_loop(X_train, y_train, shapelet_size)

        # Sort shapelets according to info gain descending
        self.shapelets.sort(key=lambda x: (x[2],x[3]), reverse=True)

        # Retrieve top shapelets
        for _ in range(n_shapelets):
            self.retrieve_top_shapelet(X_train)
        
        self.top_shapelets.sort(key=lambda x: (x[2],x[3]), reverse=True)

        for sample_idx, _, _, _, _, _ in self.top_shapelets:
            self.raw_samples.append(X_train[sample_idx])
        

    def main_event_loop(self, X_train, y_train, shapelet_size = 10):
        """
        The main event loop contains the series of steps required for the algorithm.
        """
        # Store shapelet size
        self.shapelet_size = shapelet_size
        # Calculate all of the candidate minimums throughout the dataset - shape: n_samples, n_samples, n_candidate
        profiles = self.get_candidate_mins(X_train, shapelet_size)
        # Extract a shapelets
        self.extract_shapelets(profiles, y_train, shapelet_size)
     
    def extract_shapelets(self, profiles, y_train, shapelet_size):
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
                # Score candidate
                score, margin = self.calculate_infogain(candidate.reshape((candidate.shape[0],1)), y_train)
                self.shapelets.append((sample_idx, candidate_idx, score, margin, shapelet_size, candidate))
                
    def retrieve_top_shapelet(self, X_train):
        # Sort shapelets according to info gain descending
        self.shapelets.sort(key=lambda x: (x[2],x[3]), reverse=True)

        top_shapelet_found = False
        for sample_idx, candidate_idx, score, margin, shapelet_size, _ in self.shapelets:
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
            # Add shapelet to top shapelets
            self.top_shapelets.append((sample_idx, candidate_idx, score, margin, shapelet_size, X_train[sample_idx,candidate_idx:candidate_idx+shapelet_size]))
            # Signal shapelet found
            top_shapelet_found = True

    def calculate_infogain(self, candidate, y_train):
        """
        Given a 1-d array, calculates the infogain according the labels y_train.
        """
        # Initialize decision tree classifier
        clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=1)
        # Fit decision tree
        clf.fit(candidate, y_train)
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
