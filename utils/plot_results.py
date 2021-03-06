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
    pal_deep_shifted = ["#C44E52", '#55A868', "#8172B2", "#CCB974", "#64B5CD","#4C72B0"]
    fig = sns.swarmplot(x="Dataset", y="Feature 0", hue="Class label", data=df, palette=pal_deep_shifted)
    plt.savefig(f'results/{filename}.png', dpi = 300)
    plt.close()


def plot_top_shapelets_old(GSS, y_train, filename):

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







def plot_top_shapelets(GSS, ST, X_train, y_train, filename):

    fig, axs = plt.subplots(2)

    for i, curve in enumerate(X_train):
        if y_train[i] == 1:
            axs[0].plot(curve, lw = 0.2, color = '#C44E52')
        else:
            axs[1].plot(curve, lw = 0.2, color = '#55A868')

    for i, shapelet_info in enumerate(GSS.top_shapelets[:5]):
        source_timeseries = GSS.raw_samples[i]
        shapelet_start = shapelet_info[1]
        shapelet_len = shapelet_info[4]
        shapelet = shapelet_info[5]

        if y_train[i] == 1:
            axs[0].plot(source_timeseries, lw = 0.9, color = '#C44E52')
            axs[0].plot(range(shapelet_start, shapelet_start+shapelet_len), shapelet, lw=4, alpha = 0.5, color = 'b')
        else: 
            axs[1].plot(source_timeseries, lw = 0.9, color = '#55A868')
            axs[1].plot(range(shapelet_start, shapelet_start+shapelet_len), shapelet, lw=4, alpha = 0.5, color = 'b')

    for i, shapelet_info in enumerate(ST.top_shapelets[:5]):
        source_timeseries = ST.raw_samples[i]
        shapelet_start = shapelet_info[1]
        shapelet_len = shapelet_info[4]
        shapelet = shapelet_info[5]

        if y_train[i] == 1:
            axs[0].plot(source_timeseries, lw = 0.9, color = '#C44E52')
            axs[0].plot(range(shapelet_start, shapelet_start+shapelet_len), shapelet, lw=4, alpha = 0.5, color = 'm')
        else: 
            axs[1].plot(source_timeseries, lw = 0.9, color = '#55A868')
            axs[1].plot(range(shapelet_start, shapelet_start+shapelet_len), shapelet, lw=4, alpha = 0.5, color = 'm')        

    fig.set_size_inches(4, 4.5)
    fig.tight_layout()
    plt.savefig(f'results/{filename}', dpi = 300)