# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 13:32:43 2021


"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  

def plot_shapelets(obj):

    """plot all the shapelets of shap_ids contained in GSS and get their 
    parent time-series"""
    
     #sizing and ticks   
    plt.rcParams['font.size'] = '7'
    fig = plt.figure(1, figsize = (3.543,2.168))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(left = False)
    ax.set_yticklabels([])
    
    #possible colors
    colors = ['black','green','m','blue', 'orange']
    if len(shap_ids) > 5 : 
        added_colors = ['C'+str(i) for i in range(len(shap_ids)-4)]
        colors = colors + added_colors
        
        
    k = 0 # param for color choice
    for shap_id in shap_ids : 
        
        #get shapelet and parent time-series
        parent_id = GSS.top_shapelets[shap_id][0]
        parent = X[parent_id]
        shapelet_start =GSS.top_shapelets[shap_id][1]
        shap_length = GSS.top_shapelets[shap_id][4]
        shapelet_end = shapelet_start + shap_length
        shapelet = GSS.top_shapelets[shap_id][5]
        

        ax.plot(parent, '-', color=colors[k], lw = 0.8)
        ax.plot(range(shapelet_start,shapelet_end),shapelet,lw=5,
                color=colors[k],  label = 'Shapelet '+str(shap_id), alpha = 0.5)
        
        k += 1
    ax.legend()
    plt.show()
    
    return fig


def plot_results_cloud(feat,y, shap_ids = [0], scale = False, class_names = ['io', 'nio']) : 
    """plot minimal distances with respect to shapelet_ids of 
    the given transform"""
    
     #sizing and ticks   
    plt.rcParams['font.size'] = '7'
    fig = plt.figure(1, figsize = (3.543,2.168))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Minimal euclidian distance with Shapelet')
    
    
    
    reg_label = np.min(y) # param for color choice
    nio = 0 # these two are for legend purposes
    io = 0
    if scale == True:
        feat_int = feat[:,shap_ids]
        scaler = StandardScaler().fit(feat_int)
        feat_int = scaler.transform(feat_int)
    else : 
        feat_int = feat[:,shap_ids]
        
    if feat_int.shape[1] == 1 : # adapt plot to 1D --> it can also be 2D with shap_ids[0] against shap_ids[1]
        feat_int = np.concatenate((np.arange(feat_int.shape[0]).reshape(-1,1), feat_int), axis = 1)
        plt.tick_params(bottom = False)
        ax.set_xticklabels([])

    for i in range(feat.shape[0]) : 
        markers = ['.','x']
        if y[i] == reg_label : 
             c = 0
        else : 
            c = 1
        ax.scatter(feat_int[i][0],feat_int[i][1], color = 'C'+str(c), marker = markers[c])
        
        if nio == 0 and c == 1 : 
            nio = 1
            ax.scatter(feat_int[i][0],feat_int[i][1], color = 'C'+str(c),marker = markers[c], label = class_names[c])
        
        if io == 0 and c == 0 : 
            io = 1
            ax.scatter(feat_int[i][0],feat_int[i][1], color = 'C'+str(c), label = class_names[c], marker = markers[c])
    
    ax.legend()
    plt.show()
    
    return fig







def plot_mindist(feat,y, shap_ids = [0], scale = False, class_names = ['io', 'nio']) : 
    """plot minimal distances with respect to shapelet_ids of 
    the given transform"""
    
     #sizing and ticks   
    plt.rcParams['font.size'] = '7'
    fig = plt.figure(1, figsize = (3.543,2.168))
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Minimal euclidian distance with Shapelet')
    
    
    
    reg_label = np.min(y) # param for color choice
    nio = 0 # these two are for legend purposes
    io = 0
    if scale == True:
        feat_int = feat[:,shap_ids]
        scaler = StandardScaler().fit(feat_int)
        feat_int = scaler.transform(feat_int)
    else : 
        feat_int = feat[:,shap_ids]
        
    if feat_int.shape[1] == 1 : # adapt plot to 1D --> it can also be 2D with shap_ids[0] against shap_ids[1]
        feat_int = np.concatenate((np.arange(feat_int.shape[0]).reshape(-1,1), feat_int), axis = 1)
        plt.tick_params(bottom = False)
        ax.set_xticklabels([])

    for i in range(feat.shape[0]) : 
        markers = ['.','x']
        if y[i] == reg_label : 
             c = 0
        else : 
            c = 1
        ax.scatter(feat_int[i][0],feat_int[i][1], color = 'C'+str(c), marker = markers[c])
        
        if nio == 0 and c == 1 : 
            nio = 1
            ax.scatter(feat_int[i][0],feat_int[i][1], color = 'C'+str(c),marker = markers[c], label = class_names[c])
        
        if io == 0 and c == 0 : 
            io = 1
            ax.scatter(feat_int[i][0],feat_int[i][1], color = 'C'+str(c), label = class_names[c], marker = markers[c])
    
    ax.legend()
    plt.show()
    
    return fig