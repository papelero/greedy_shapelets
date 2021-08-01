# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler  


def create_dataframe(features, labels, dataset, method):
    df = pd.DataFrame(features)
    df.astype('float32').dtypes
    df.columns = [f'Feature {i}' for i in np.arange(df.shape[1])]
    df['Class label'] = labels
    df['Dataset'] = dataset
    df['method'] = 'GSS'
    return df

def plot_swarmplot(df, filename):
    fig = sns.swarmplot(x="Dataset", y="Feature 0", hue="Class label", data=df)
    plt.savefig(f'results/{filename}')

def plot_top_shapelets(GSS, y_train, filename):

    fig = plt.figure()
    ax = plt.axes()
    for i, shapelet_info in enumerate(GSS.top_shapelets[:5]):
        source_timeseries = GSS.raw_samples[i]
        shapelet_start = shapelet_info[1]
        shapelet_len = shapelet_info[4]
        shapelet = shapelet_info[5]

        if y_train[i] == 1:
            color = 'r'
        else: 
            color = 'b'

        ax.plot(source_timeseries, lw = 0.8, color = color)
        ax.plot(range(shapelet_start, shapelet_start+shapelet_len), 
        shapelet, lw=5, alpha = 0.5, color = color)

    plt.savefig(f'results/{filename}')