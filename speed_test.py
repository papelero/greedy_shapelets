
# %%
import numpy as np
from matrixprofile.algorithms.mass2 import mass2
import time
import matrixprofile

# %%
def mass_distance_matrix(ts, query, w):
    """
    Computes a distance matrix using mass that is used in mpdist_vector
    algorithm.
    Parameters
    ----------
    ts : array_like
        The time series to compute the matrix for.
    query : array_like
        The time series to compare against.
    w : int
        The window size.
    
    Returns
    -------
    array_like : dist_matrix
        The MASS distance matrix.
    """
    subseq_num = len(query) - w + 1
    distances = []
    
    for i in range(subseq_num):
        distances.append(np.real(mass2(ts, query[i:i + w])))
    
    return np.array(distances)

# %%

def timer(t_calc, algorithm):
    print(f'...using {algorithm}',t_calc)
    print('for 100 different window lengths and all time series',t_calc*200*100/3600,'Stunden')   


len_ts = 200
num_ts = 200
w = 20

query = np.random.rand(len_ts)
data = np.random.rand(num_ts,len_ts)

print("""time required to calculate and save the MP between all subsequences
of one TS (length 20, 200) and the dataset (200 TS)...""")

dist_profiles = []
start = time.time()
for ts in data:
    dist_profiles.append(mass_distance_matrix(ts, query, w).min(axis=1))
timer(time.time()-start, 'MASS2')

dist_profiles = []
start = time.time()
for ts in data:
    dist_profiles.append(matrixprofile.algorithms.stomp(ts, w, query, n_jobs=1))
timer(time.time()-start, 'STOMP')

dist_profiles = []
start = time.time()
for ts in data:
    dist_profiles.append(matrixprofile.algorithms.mpx(ts, w, query, n_jobs=1))
timer(time.time()-start, 'MPX')

# %%
