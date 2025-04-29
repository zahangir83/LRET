#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:02:03 2020

@author: malom
"""
import os
import matplotlib
import pdb
matplotlib.use('Agg')


import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join as join_path

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score

import seaborn as sns
import json
from json import JSONEncoder
import keras


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def plot_save_training_logs(history,training_log_saving_path):
    
    training_log = {}
    training_log["model_loss"] = history.history['loss']
    training_log["accuracy"] = history.history['accuracy']
    training_log["val_loss"] = history.history['val_loss']
    training_log["val_Accuracy"] = history.history['val_accuracy']
    
    #pdb.set_trace()
    # make experimental log saving path...
    json_file = os.path.join(training_log_saving_path,'model_training_log.json')
    with open(json_file, 'w') as file_path:
        json.dump(str(training_log), file_path, indent=4, sort_keys=True)
    
    
    plots_saving_path = os.path.join(training_log_saving_path,'Training_acc.png')
    model_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(model_acc, color="tomato", linewidth=2)
    plt.plot(val_acc, color="steelblue", linewidth=2)  
    plt.legend(["Training","Validation"],loc=4)
    plt.title("Training and validation accuracy.")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")   
    plt.grid()
    plt.savefig(plots_saving_path)

def save_all_testing_logs(y_test, y_ac_test, y_pred, log_saving_dir):
    
    y_test = np.array(y_test)
    y_ac_test = np.array(y_ac_test)
    y_pred = np.array(y_pred)
    
    name = 'y_test.npy'
    y_test_saving_path = join_path(log_saving_dir,name)
    np.save(y_test_saving_path,y_test)
    
    y_ac_test_name = 'y_ac_test.npy'
    y_ac_test_saving_path = join_path(log_saving_dir,y_ac_test_name)
    np.save(y_ac_test_saving_path,y_ac_test)
    
    y_pred_name = 'y_pred.npy'
    y_pred_saving_path = join_path(log_saving_dir,y_pred_name)
    np.save(y_pred_saving_path,y_pred)
    
    
    
def save_testing_performance_logs(y_test,labels,y_pred,log_saving_dir):
    
    #y_test = np.load('SA/y_test.npy')     # [0,5,.............]
    #y_ac_test = np.load('SA/y_ac_test.npy')
    #y_pred = np.load('SA/y_pred_dl.npy')
    #y_pred = np.argmax(y_pred, axis=-1)   # [0,4,5,2,3........]
    
    #print
    #labels = ['G3_4_I', 'G3_4_II', 'G3_4_III', 'G3_4_IV', 'G3_4_V', 'G3_4_VI', 'G3_4_VII', 'G3_4_VIII', 'SHH_alpha', 'SHH_beta', 'SHH_delta', 'SHH_gamma', 'WNT']
    
    cm = confusion_matrix(y_test, y_pred)
    print('Total number of testing sampels: ', len(y_test))
    #sns.heatmap(cm, annot=True)
    sns.heatmap(cm, annot=True, fmt='d',xticklabels=labels, yticklabels=labels)
        
    #importing accuracy_score, precision_score, recall_score, f1_score
    
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))
        
    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))
        
    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))
        
    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))
        
    
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=labels))
        
    performance_log = {}
        
    performance_log["Accuracy"] = accuracy_score(y_test, y_pred)
    performance_log["Micro Precision: "] = precision_score(y_test, y_pred, average='micro')
    performance_log["Micro Recall:"] = recall_score(y_test, y_pred, average='micro')
    performance_log["Micro F1-score:"] = f1_score(y_test, y_pred, average='micro')
            
    performance_log["Macro Precision: "] = precision_score(y_test, y_pred, average='macro')
    performance_log["Macro Recall:"] = recall_score(y_test, y_pred, average='macro')
    performance_log["Macro F1-score:"] = f1_score(y_test, y_pred, average='macro')  
        
    performance_log["Weighted Precision: "] = precision_score(y_test, y_pred, average='weighted')
    performance_log["Weighted Recall:"] = recall_score(y_test, y_pred, average='weighted')
    performance_log["Weighted F1-score:"] = f1_score(y_test, y_pred, average='weighted')
        
    #database_log_list = database_log.tolist()
    # make experimental log saving path...
    #pdb.set_trace()
    pm_saving_path = os.path.join(log_saving_dir,'DL_performance_logs.json')
    with open(pm_saving_path, 'w') as file_path:
        json.dump(performance_log, file_path, cls=NumpyArrayEncoder)  
        

def define_actual_class_name(y_test,y_ac_test):
    
    y_test =y_test.tolist()
    indexes = [y_test.index(x) for x in set(y_test)]   
    labels_for_idx = [y_test[i] for i in indexes]
    ac_labels_for_idx = [y_ac_test[i] for i in indexes]
    
    # order_labels = np.zeros(len(labels_for_idx))
    # order_ac_labels = np.zeros(len(labels_for_idx))
    
    # for k in range(len(labels_for_idx)):
        
    #     label = labels_for_idx[k]
    #     ac_label = ac_labels_for_idx[k]
        
    #     order_labels[label] = label
    #     order_ac_labels[label] = ac_label
    
    order_labels = np.array(labels_for_idx)
    order_ac_labels = np.array(ac_labels_for_idx)
    
    return order_labels,order_ac_labels
        
    
def plot_and_save_ROC_curve(y_test,y_ac_test,y_scores,testing_log_saving_path):
    
    ## Note of vairables .........<<<<<>>>>>>>>>>
    # y_test - > encoded labels [ 0, 2,3,5,1,0.....]
    # y_ac_test - > actual labels ['a','b','c',......'z']
    # y_pred .. actual prediction from DL without applying argmax......

    order_labels,order_ac_labels = define_actual_class_name(y_test,y_ac_test)    
    num_classes = len(order_labels)
    print('Number of classes :',num_classes)
    y_onehot = keras.utils.to_categorical(y_test, num_classes)
    #y_scores_rf = keras.utils.to_categorical(y_scores_rf, num_classes)  
    #print('Y_true:',y_test[:100])
    #print('Y_pred:',y_ac_test)
    #y_scores = keras.utils.to_categorical(y_scores, num_classes)
    #print('Predicted values:')
    #print(y_scores)
    #print('Number of sampels: ',y_scores.shape[0])
    #print('Number of classes:',y_scores.shape[1])
    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    #labels = ['G3_4_I', 'G3_4_II', 'G3_4_III', 'G3_4_IV', 'G3_4_V', 'G3_4_VI', 'G3_4_VII', 'G3_4_VIII', 'SHH_alpha', 'SHH_beta', 'SHH_delta', 'SHH_gamma', 'WNT']
    
    labels = order_ac_labels
    
    for i in range(y_scores.shape[1]):
        #y_true = y_onehot.iloc[:, i]
        y_true = y_onehot[:, i]
        y_score = y_scores[:, i]
    
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_score = roc_auc_score(y_true, y_score)
    
        #name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
        #fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
        name = f"{str(labels[i])} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
    
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    #fig.show()
    
    pdb.set_trace()
    final_log_sp = join_path(testing_log_saving_path,'ROC.png')
    fig.write_image(final_log_sp)
    

    # fig = go.Figure()
    # fig.add_shape(
    #     type='line', line=dict(dash='dash'),
    #     x0=0, x1=1, y0=0, y1=1
    # )
    
    # for i in range(y_scores.shape[1]):
    #     #y_true = y_onehot.iloc[:, i]
    #     y_true = y_onehot[:, i]
    #     y_scorerf = y_scores_rf[:, i]
    
    #     fpr, tpr, _ = roc_curve(y_true, y_scorerf)
    #     auc_score = roc_auc_score(y_true, y_scorerf)
    
    #     #name = f"{y_onehot.columns[i]} (AUC={auc_score:.2f})"
    #     #fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
    #     name = f"{str(labels[i])} (AUC={auc_score:.2f})"
    #     fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
    
    # fig.update_layout(
    #     xaxis_title='False Positive Rate',
    #     yaxis_title='True Positive Rate',
    #     yaxis=dict(scaleanchor="x", scaleratio=1),
    #     xaxis=dict(constrain='domain'),
    #     width=700, height=500
    # )
    # fig.show()
    
        
def plot_and_save_precision_recall_curve(y_test,y_ac_test,y_scores,testing_log_saving_path):
    
    ## Note of vairables .........<<<<<>>>>>>>>>>
    # y_test - > encoded labels [ 0, 2,3,5,1,0.....]
    # y_ac_test - > actual labels ['a','b','c',......'z']
    # y_pred .. actual prediction from DL without applying argmax......
    
    #print(y_test.shape)
    #print(y_scores.shape)
    
    order_labels,order_ac_labels = define_actual_class_name(y_test,y_ac_test)    
    num_classes = len(order_labels)
    print('Number of classes :',num_classes)
    
    y_onehot = keras.utils.to_categorical(y_test, num_classes)
    #print('Y_true:',y_test[:100])
    #print('Y_pred:',y_ac_test)
    #y_scores = keras.utils.to_categorical(y_scores, num_classes)
    #print('Predicted values:')
    #print(y_scores)
    print('Number of sampels: ',y_scores.shape[0])
    print('Number of classes:',y_scores.shape[1])
    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    #labels = ['G3_4_I', 'G3_4_II', 'G3_4_III', 'G3_4_IV', 'G3_4_V', 'G3_4_VI', 'G3_4_VII', 'G3_4_VIII', 'SHH_alpha', 'SHH_beta', 'SHH_delta', 'SHH_gamma', 'WNT']
    labels = order_ac_labels
    for i in range(y_scores.shape[1]):
        y_true = y_onehot[:, i]
        y_score = y_scores[:, i]
    
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auc_score = average_precision_score(y_true, y_score)
    
        #name = f"{y_onehot.columns[i]} (AP={auc_score:.2f})"
        name = f"{str(labels[i])} (APS={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode='lines'))
    
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    fig.show()
    pdb.set_trace()
    final_log_sp = join_path(testing_log_saving_path,'precision_recall_curve.png')
    
    pio.write_image(fig, final_log_sp)

    #fig.write_image(final_log_sp)
    
