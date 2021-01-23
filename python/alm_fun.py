import numpy as np
import pandas as pd
import smtplib
import csv
import os
import re
import gzip
import matplotlib
from cmath import inf

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.path as mpath
import matplotlib.patches as patches  
import matplotlib.collections as collections
from matplotlib.lines import Line2D  
from matplotlib.gridspec import GridSpec
from ftplib import FTP
from email.message import EmailMessage
from datetime import datetime

from sklearn import metrics
from scipy import stats
from scipy import interp
import operator
import itertools
import time
import math
import random
import codecs
import copy
import pickle

def create_ftp_object(ftp_site):   
    cur_ftp = FTP(ftp_site)
    cur_ftp.login()
    return(cur_ftp)

def download_ftp(ftp, ftp_path, ftp_file, local_file):
    try:
        if os.path.isfile(local_file):
            statinfo = os.stat(local_file)
            local_size = statinfo.st_size
            ftp.cwd(ftp_path)
            remote_size = ftp.size(ftp_path + ftp_file)
            if local_size != remote_size :
                ftp.retrbinary('RETR ' + ftp_file, open(local_file, 'wb').write)
                return('updated')
            else:
                print (local_file + ' is up to date.')
                return('up_to_date')
        else: 
            ftp.cwd(ftp_path)
            ftp.retrbinary('RETR ' + ftp_file, open(local_file, 'wb').write)
            print (local_file + ' is downloaded.')
            return('downloaded')
    except Exception as e:
#             os.remove(local_file)
        print (str(e))
        return(str(e))

def gzip_decompress(zipped_file_name, unzipped_file_name):
    inF = gzip.GzipFile(zipped_file_name, 'rb')
    s = inF.read()
    inF.close()
    outF = open(unzipped_file_name, 'wb')
    outF.write(s)
    outF.close()   

def quality_evaluation(name, path, extra_train_file, use_extra_train_data, train_file, test_file, dependent_variable, feature_engineer, ml_type, estimators, cv_split_method, cv_split_folds, verbose, onehot_features, initial_features, percent_min_feature, quality_feature, quality_feature_direction, quality_feature_cutoffs):
    cutoff_objects = []    
    # generate all quality cutoff permutation for current quality_feature_cutoffs
    for quality_feature_cutoff in itertools.product(*quality_feature_cutoffs):
        ex = alm_ml(name, path, extra_train_file, use_extra_train_data, train_file, test_file, dependent_variable, feature_engineer, ml_type, estimators, cv_split_method, cv_split_folds, verbose, onehot_features, initial_features, percent_min_feature, quality_feature, quality_feature_direction, quality_feature_cutoff)
        ex.data_filter() 
        ex.feature_evaluation()
        cutoff_objects.append(ex) 
    return  cutoff_objects
                  
def show_msg(infile, verbose, msg, with_time = 1):
    if (verbose == 1): 
        print (msg)
        if isinstance(infile,str):        
            with open(infile,'a+') as log_file:
                if with_time == 1:
                    log_file.write(str(datetime.now())[:19] +'||' + msg + '\n')
                else:
                    log_file.write( msg + '\n')
            log_file.close()
        else:
            infile.write(msg)
    else:
        print (msg)  
    
def show_msg_openlog(log,verbose,msg):
    log.write(msg)
    if (verbose == 1): print (msg)    
        
def show_start_msg(infile,verbose,class_name,fun_name):
    msg = 'Class: [' + class_name + '] Fun: [' + fun_name + '] .... start @' + str(datetime.now())
    show_msg(infile,verbose,msg)
    return(time.time()) 
        
def show_end_msg(infile,verbose,class_name,fun_name,stime):
    msg = 'Class: [' + class_name + '] Fun: [' + fun_name + '] .... done @' + str(datetime.now()) + ' time spent: %g seconds' % (time.time() - stime)
    show_msg(infile,verbose,msg)

def dependency_matrix(x, f, symmetric=1):
    n = x.shape(1)
    names = x.columns.get_values()
    dependency_matrix = pd.DataFrame(np.zeros((n, n)), columns=names)   
    score_interaction_forward.index = names
     
    for i in range(n):
        for j in range(i, n):
            dependency_matrix.loc[names[i], names[j]] = f(x[names[i]], x[names[j]]) 
            if i == j:
                break
            if symmetric == 1:
                dependency_matrix.loc[names[j], names[i]] = dependency_matrix.loc[names[i], names[j]]
            else:
                dependency_matrix.loc[names[j], names[i]] = f(x[names[j]], x[names[i]]) 
    return dependency_matrix            
                 
def pcc_cal(x, y, if_abs=False):  
    x = np.array(x,dtype = 'float')
    y = np.array(y,dtype = 'float')
    
    x_nullidx = list(np.where(np.isnan(x))[0])
    y_nullidx = list(np.where(np.isnan(y))[0])
    
    nullidx = set(x_nullidx + y_nullidx) 
    idx = list(set(range(len(x))) - nullidx)
    
    if len(idx) == 0 :
        return(np.nan)
    
    x = x[idx]
    y = y[idx]
      
    r = stats.pearsonr(x, y)
    pcc = r[0]
     
    if math.isnan(pcc):
        pcc = 0
    if if_abs:
        return (np.abs(pcc))
    else:        
        return pcc 
 
def spc_cal(x, y, if_abs=False):  
    x = np.array(x,dtype = 'float')
    y = np.array(y,dtype = 'float')
    
    x_nullidx = list(np.where(np.isnan(x))[0])
    y_nullidx = list(np.where(np.isnan(y))[0])
    
    nullidx = set(x_nullidx + y_nullidx) 
    idx = list(set(range(len(x))) - nullidx)
    
    if len(idx) == 0 :
        return(np.nan)
    
    x = x[idx]
    y = y[idx]
      
    r = stats.spearmanr(x, y)
    spc = r[0]
     
    if math.isnan(spc):
        spc = 0

    if if_abs:
        return (np.abs(spc))
    else:        
        return spc 

def linear_rmse_cal(y, y_predicted):
    y_predicted = y_predicted.reshape(y_predicted.shape[0], 1)
    es = lm.ElasticNet(alpha=1.0, l1_ratio=0.5)
    es.fit(y_predicted, y)
    y1 = es.predict(y_predicted)
    return(np.sqrt(np.sum((y - y1) ** 2) / len(y)))    
    # pearson correlation coefficient - pvalue
 
def pcc_pvalue(x, y):
    r = stats.pearsonr(x, y)
    return r[1]   

def rmse_cal(y, y_predicted):
    # if y_predicted contains nan
    y = np.array(y)
    y_predicted = np.array(y_predicted)
    notnull_idx = ~np.isnan(y_predicted)
    y_predicted = y_predicted[notnull_idx]
    y = y[notnull_idx]
    
    return(np.sqrt(np.sum((y - y_predicted) ** 2) / len(y)))    

def mse_cal(y, y_predicted):
    return(np.sum((y - y_predicted) ** 2) / len(y))    

def logloss_cal(y,y_predicted):
    logloss = 0 - (np.dot(y,np.log(y_predicted)) + np.dot(1-y,np.log(1-y_predicted)))/ len(y)
    return (logloss)

def auprc_cal(y, y_predicted):
    prc = round(metrics.average_precision_score(y, y_predicted), 4)
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_predicted)
    return [prc, precision, recall, thresholds]     

def auroc_cal(y, y_predicted):
    if len(np.unique(y)) != 1:
        roc = round(metrics.roc_auc_score(y, y_predicted), 4)
        fpr, tpr, thresholds = metrics.roc_curve(y, y_predicted)
    else:
        roc = inf
        fpr,tpr, thresholds = metrics.roc_curve(y, y_predicted)
    return [roc, fpr, tpr, thresholds]  

def roc_prc_cal(y, y_predicted):
    metric = {}
    reverse = 0
    if len(np.unique(y)) != 1:
        roc = round(metrics.roc_auc_score(y, y_predicted), 4)
    else:
        roc = inf
    if roc < 0.5:
        reverse = 1
        y_predicted = 0 - y_predicted
        if len(np.unique(y)) != 1:
            roc = round(metrics.roc_auc_score(y, y_predicted), 4)
        else:
            roc = inf
     
    prc = round(metrics.average_precision_score(y, y_predicted), 4)
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_predicted)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_predicted)
    metric['roc'] = roc
    metric['prc'] = prc
    metric['fpr'] = fpr
    metric['tpr'] = tpr
    metric['precision'] = precision
    metric['recall'] = recall
    metric['reverse'] = reverse    
    return metric

def get_classification_metrics(metrics_name, round_digits, y, y_predicted):
    if metrics_name == 'auroc':
        if len(np.unique(y)) != 1:
            result = round(metrics.roc_auc_score(y, y_predicted), round_digits)
        else:
            result = inf
     
    if metrics_name == 'auprc':
        result = round(metrics.average_precision_score(y, y_predicted), round_digits)
     
    if metrics_name == 'neg_log_loss':
        result = round(metrics.log_loss(y, y_predicted), round_digits)
    return result

def score_to_precision(y,y_predicted,cutoff):
    y = np.array(np.squeeze(y))
    y_predicted = np.array(np.squeeze(y_predicted))
    notnull_idx = ~np.isnan(y_predicted)
    y_predicted = y_predicted[notnull_idx]
    y = y[notnull_idx]
    size = len(y)  
    y_p = np.zeros(size)
    y_p[y_predicted > cutoff] = 1     
    tp = ((y == 1) & (y_p == 1)).sum() 
    fp = ((y == 0) & (y_p == 1)).sum()
    tn = ((y == 0) & (y_p == 0)).sum()
    fn = ((y == 1) & (y_p == 0)).sum()  

    prior = get_prior(y)
    precision= tp / (tp + fp)
    balanced_precision = precision*(1-prior)/(precision*(1-prior) + (1-precision)*prior)
    return(balanced_precision)

def classification_metrics(y, y_predicted, cutoff= 0.5, test_precision=0.9, test_recall=0.9,find_best_cutoff=0):    
    y_classes = np.unique(y)

    # if y_predicted contains nan, remove it.
    y = np.array(np.squeeze(y))
    y_predicted = np.array(np.squeeze(y_predicted))
    notnull_idx = ~np.isnan(y_predicted)
    y_predicted = y_predicted[notnull_idx]
    y = y[notnull_idx]
        
    #if y_predicted contains nan, then do random prediction on the Nan     
#     for i in range(len(y_predicted)):
#         if np.isnan(float(y_predicted[i])) == True :
#             y_predicted[i] = random.choice(y_classes)            
#             y_predicted[i] = np.random.uniform(np.nanmin(y_predicted),np.nanmax(y_predicted),1)[0]
#             y_predicted[i] = np.nanmax(y_predicted)
        
    # check if multi-classification
    multiclass_metrics_dict = {}

    if len(y_classes) > 2:
        for y_class in y_classes:
            y_new = y.copy()
            y_new[y == y_class] = 1
            y_new[y != y_class] = 0
            multiclass_metrics_dict[y_class] = (classification_metrics_sub(y_new, y_predicted, cutoff, test_precision, test_recall))
     
        for y_class in y_classes:
            cur_y_predicted = multiclass_metrics_dict[y_class][0]
            cur_y_predicted[cur_y_predicted == 1] = y_class
            cur_y_predicted = pd.DataFrame(cur_y_predicted, columns=[y_class])
            if 'combined_best_y_predicted_df' not in locals():
                combined_best_y_predicted_df = cur_y_predicted
            else:
                combined_best_y_predicted_df = pd.concat([combined_best_y_predicted_df, cur_y_predicted], axis=1)
                        
            cur_y_metrics = multiclass_metrics_dict[y_class][1]
            cur_y_metrics = pd.DataFrame(list(cur_y_metrics.items()), columns=['key', y_class])
            cur_y_metrics.set_index('key', drop=True, inplace=True)
            if 'combined_metrics_dict_df' not in locals():
                combined_metrics_dict_df = cur_y_metrics
            else:
                combined_metrics_dict_df = pd.concat([combined_metrics_dict_df, cur_y_metrics], axis=1)
                 
            combined_best_y_predicted = combined_best_y_predicted_df.mean(axis=1)
            combined_metrics_dict = pd.DataFrame(combined_metrics_dict_df.mean(axis=1)).T.to_dict('records')[0]
    else:        
        [combined_best_y_predicted, combined_metrics_dict] = classification_metrics_sub(y, y_predicted, cutoff, test_precision, test_recall,find_best_cutoff)
             
    return [combined_best_y_predicted, combined_metrics_dict, multiclass_metrics_dict]

def classification_metrics_sub(y, y_predicted, cutoff = 0.5, test_precision = 0.9, test_recall = 0.9, find_best_cutoff = 0,round_digits = 4):
    
    #*****************************************************************************************                
    # Metrics based on a specific value of cutoff (via input or tune for the best mcc)
    #*****************************************************************************************      
    reverse = 0
    if len(np.unique(y)) != 1:
        roc = metrics.roc_auc_score(y, y_predicted)
    else:
        roc = inf
    if roc < 0.5:
        reverse = 1
        y_predicted = 0 - y_predicted

    #*****************************************************************************************                
    # Tune the cutoff for the best MCC
    #***************************************************************************************** 
    if find_best_cutoff == 1:  
        all_cutoffs = set(y_predicted)               
        mcc = -1
        cutoff = -inf       
        for cur_cutoff in all_cutoffs:
            cur_y_p = np.zeros(len(y_predicted))
            cur_y_p[y_predicted > cur_cutoff] = 1
            cur_mcc = metrics.matthews_corrcoef(y, cur_y_p)
#             print ('cutoff:' + str(cur_cutoff) + ' mcc:' + str(cur_mcc))
            if cur_mcc >= mcc:
                cutoff = cur_cutoff
                mcc = cur_mcc
    #*****************************************************************************************                
    # Metrics based on a specific value of cutoff (via input or tune for the best mcc)
    #*****************************************************************************************
    size = len(y)  
    y_p = np.zeros(size)
    y_p[y_predicted > cutoff] = 1     
    tp = ((y == 1) & (y_p == 1)).sum() 
    fp = ((y == 0) & (y_p == 1)).sum()
    tn = ((y == 0) & (y_p == 0)).sum()
    fn = ((y == 1) & (y_p == 0)).sum()  
    
    prior = get_prior(y)
    fpt = fp / (fp + fn) # false Positive tendency?
    precision= tp / (tp + fp)
    balanced_precision = precision*(1-prior)/(precision*(1-prior) + (1-precision)*prior)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + fp) / (tn + fp + tp + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) 
    
    #*****************************************************************************************                
    # Metrics 
    # LOGLOSS
    # AUROC: Area under receiver operating characteristic (ROC) curve
    # AUPRC: Area under precision recall curve
    # AUBPRC: Area under balanced precision recall curve
    # AUTBPRC: Area under transformed balanced precision recall curve
    #               (for example incremental precisions from 0.5 to 1 for every 0.1%)
    # UP_AURPC: Area under precision recall curve but above a pre-defined precision cutoff
    #*****************************************************************************************
    logloss = logloss_cal(y,y_predicted)    
    fprs, tprs, roc_thresholds = metrics.roc_curve(y, y_predicted)  
    auroc = metrics.roc_auc_score(y, y_predicted)     
    precisions, recalls, prc_thresholds = metrics.precision_recall_curve(y, y_predicted)
    recalls = np.insert(recalls,0,1)
    precisions = np.insert(precisions,0,prior)
    
    #balanced precision and recall (prior = 0.5)
    balanced_precisions = precisions*(1-prior)/(precisions*(1-prior) + (1-precisions)*prior)
    balanced_recalls = recalls

    [auprc,up_auprc,rfp,pfr] = cal_pr_values(precisions,recalls,test_precision,test_recall)
    [aubprc,up_aubprc,brfp,bpfr] = cal_pr_values(balanced_precisions,balanced_recalls,test_precision,test_recall)

    metrics_dict = locals().copy()    

    for key in metrics_dict.keys():
        if 'float' in str(type(metrics_dict[key])):
            metrics_dict[key] = np.round(metrics_dict[key],round_digits)
    
    return([y_p, metrics_dict])

def get_interpreted_x_from_y (new_y,x,y,type = 'first_intersection'):
    new_x = np.nan

    if type == 'last_intersection':  
        x = x[::-1]
        y = y[::-1]  

    for i in range(len(x)):            
        if y[i] == new_y: 
            new_x = x[i]
            break
        if i < len(x) - 1:
            if (new_y-y[i])*(new_y-y[i+1]) < 0: 
                new_x = x[i] + (new_y- y[i])*(x[i+1] - x[i])/(y[i+1] - y[i])    
                break    
                     
 
 
    return(new_x)
     
def get_interpreted_y_from_x (new_x,x,y,type = 'first_intersection'):
    new_y = np.nan
    if type == 'last_intersection':    
        x = x[::-1]
        y = y[::-1]  

   
    for i in range(len(y)):            
        if x[i] == new_x: 
            new_y = y[i]
            break
        if i < len(y) - 1:
            if (new_x-x[i])*(new_x-x[i+1]) < 0 :
                new_y = y[i] + (new_x- x[i])*(y[i+1] - y[i])/(x[i+1] - x[i])    
                break            
    return(new_y)
    
def interp_precision(precision,precisions,recalls):
    recall = 0
    for i in range(len(precisions)):
        # this is the first point cross target precision line from left to right
        if precisions[i] >= precision:
            if i == 0:
                recall = np.max(recalls)
            else:             
                recall = recalls[i-1] - (precision-precisions[i-1])*(recalls[i-1]-recalls[i])/(precisions[i]-precisions[i-1])
            break
    return(recall)

def interp_recall(recall,precisions,recalls):
    pfr = 0
    for i in range(len(recalls)):
        if recalls[i] <= recall:
            # this is the first point cross target recall line from left to right
            pfr = precisions[i] - (recall-recalls[i])*(precisions[i]-precisions[i-1])/(recalls[i-1]-recalls[i])
            break
    return(pfr)
    
def cal_pr_values(precisions,recalls,test_precision = 0.9,test_recall = 0.9):
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    auprc = -np.sum(np.diff(recalls) * np.array(precisions)[:-1]) # average precision
    up_precisions =  precisions - test_precision    
    new_up_precisions = up_precisions[up_precisions>0]
    new_up_recalls = recalls[up_precisions>0]
    up_auprc = -np.sum(np.diff(new_up_recalls) * np.array(new_up_precisions)[:-1])
    
    rfp = get_interpreted_x_from_y(test_precision,recalls,precisions,type = 'first_intersection')
    pfr = get_interpreted_y_from_x(test_recall,recalls,precisions,type = 'last_intersection')
    
    return [ auprc,up_auprc,rfp,pfr]

def plot_prc(y, y_predicted, plot_name, fig_w, fig_h, cutoff=np.nan, test_precision=0.9, test_recall=0.9, title_name='AUPRC'):
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_auprc = plt.subplot()   
    ax_auprc.margins(1,1)
    color_lst = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000','#666633']
    
    
    if y_predicted.shape[1] > 1:
        if y_predicted.shape[1] > 8:
#             lst_colors = list(matplotlib.colors.CSS4_COLORS.keys())
            lst_colors = color_lst
        else:
            lst_colors = color_lst
#             lst_colors = list(matplotlib.colors.BASE_COLORS.keys())
        
        for i in range(y_predicted.shape[1]):
            metrics_dict = classification_metrics(y, y_predicted[y_predicted.columns[i]], cutoff, test_precision, test_recall)[1]
#             metrics_dict['precisions'][-1] = metrics_dict['precisions'][-2]                          
#             ax_auprc.plot(metrics_dict['recalls'] + i*0.005 ,metrics_dict['precisions'] + i*0.005,marker='o',markersize=5,color = lst_colors[i],label = y_predicted.columns[i] + '(' + str(metrics_dict['auprc']) + ')')
            ax_auprc.plot(metrics_dict['recalls'], metrics_dict['precisions'], marker='o', markersize=2, color=lst_colors[i], label = y_predicted.columns[i] + ' (' + str(metrics_dict['auprc']) + ') (' + str(metrics_dict['recall_fixed_precision']) +')')
            print (i)
        # Now add the legend with some customizations.
#         legend = ax_auprc.legend(loc='upper left',bbox_to_anchor=(1, 1),shadow=True)        
        legend = ax_auprc.legend(loc='lower left', shadow=True ,prop={'size': 30}) 
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        
        # Set the fontsize
#         for label in legend.get_texts():
#             label.set_fontsize('large')
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width    
    else:        
        metrics_dict = classification_metrics(y, y_predicted, cutoff, test_precision, test_recall)[1]    
        fig = plt.figure() 
        ax_auprc = plt.subplot()   
#         metrics_dict['precisions'][-1] = metrics_dict['precisions'][-2]          
        ax_auprc.plot(metrics_dict['recalls'], metrics_dict['precisions'], marker='o', markersize=2, color='black', label='AUPRC: (' + str(metrics_dict['auprc']) + ')')
        # Now add the legend with some customizations.
        legend = ax_auprc.legend(loc='lower left', shadow=True ,prop={'size': 20}) 
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        
#         for label in legend.get_texts():
#             label.set_fontsize(25)
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width   
    pass

    ax_auprc.plot([0, 1], [test_precision, test_precision], linestyle = '--', color = 'black', lw=2)
    
    ax_auprc.set_title(title_name, size=30)
    ax_auprc.set_xlabel('Recall',size = 20)
    ax_auprc.set_ylabel('Precision',size = 20) 
    ax_auprc.set_xlim(0, 1)
    ax_auprc.set_ylim(0, 1.05) 
    fig.tight_layout()
    plt.savefig(plot_name)   
    return(ax_auprc)

def plot_roc(y, y_predicted, plot_name, fig_w, fig_h, cutoff=np.nan, test_precision=0.9, test_recall=0.9, title_name='AUROC'):
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_auroc = plt.subplot()   
    ax_auroc.margins(1,1)
    color_lst = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000','#666633']

    if y_predicted.shape[1] > 1:
        if y_predicted.shape[1] > 8:
            lst_colors = color_lst
        else:
            lst_colors = color_lst
#             lst_colors = list(matplotlib.colors.BASE_COLORS.keys())
        
        for i in range(y_predicted.shape[1]):
            metrics_dict = classification_metrics(y, y_predicted[y_predicted.columns[i]], cutoff, test_precision, test_recall)[1]  
            ax_auroc.plot(metrics_dict['fprs'], metrics_dict['tprs'], marker='o', markersize=2, color=lst_colors[i], label=y_predicted.columns[i] + '(' + str(metrics_dict['auroc']) + ')')
            
        # Now add the legend with some customizations.
#         legend = ax_auroc.legend(loc='upper left',bbox_to_anchor=(1, 1),shadow=True ,prop={'size': 20})  
        legend = ax_auroc.legend(loc='lower right',shadow=True ,prop={'size': 20})
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        
        # Set the fontsize
#         for label in legend.get_texts():
#             label.set_fontsize(25)
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width    
    else:
        
        metrics_dict = classification_metrics(y, y_predicted, cutoff, test_precision, test_recall)    
        fig = plt.figure() 
        ax_auroc = plt.subplot()            
        ax_auroc.plot(metrics_dict['fprs'], metrics_dict['tprs'], marker='o', markersize=2, color='black', label='AUROC:(' + str(metrics_dict['auroc']) + ')')
    pass
    ax_auroc.set_title(title_name, size=30)
    ax_auroc.set_xlabel('False Positive Rate',size = 20)
    ax_auroc.set_ylabel('True Positive Rate',size = 20) 
    ax_auroc.set_xlim(0, 1)
    ax_auroc.set_ylim(0, 1.05)
    fig.tight_layout()
    plt.savefig(plot_name)   
    return(ax_auroc)

def plot_fpt_ax(y, y_predicted, ax_aufpt, fig_w, fig_h, cutoff=np.nan, test_precision=0.9, test_recall=0.9, title_name='AUFPT'):  
    ax_aufpt.margins(1,1)
    color_lst = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000','#666633']
    
    if y_predicted.shape[1] > 1:

        lst_colors = color_lst
        
        for i in range(y_predicted.shape[1]):
            metrics_dict = classification_metrics(y, y_predicted[y_predicted.columns[i]], cutoff, test_precision, test_recall)[1]
#             ax_aufpt.plot(metrics_dict['fpts'], metrics_dict['precisions'], marker='o', markersize=2, color=lst_colors[i], label = y_predicted.columns[i] + ' (' + str(metrics_dict['aufpt']) + ') (' + str(metrics_dict['recall_fixed_precision']) +')')
            ax_aufpt.plot(metrics_dict['pcrs'], metrics_dict['pcr_fpts'], marker='o', markersize=2, color=lst_colors[i], label = y_predicted.columns[i])
            print (i)
        # Now add the legend with some customizations.
#         legend = ax_aufpt.legend(loc='upper left',bbox_to_anchor=(1, 1),shadow=True)        
        legend = ax_aufpt.legend(loc='lower right', shadow=True ,prop={'size': 30}) 
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        
        # Set the fontsize
#         for label in legend.get_texts():
#             label.set_fontsize('large')
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width    
    pass



    ax_aufpt.plot([0, 1], [0, 1], linestyle = '--', color = 'black', lw=2)
    
    ax_aufpt.set_title(title_name, size=40)
    ax_aufpt.set_xlabel('Positive prediction count ratio',size = 35)
    ax_aufpt.set_ylabel('False Positive Trend (FP/FP+FN)',size = 35) 
    ax_aufpt.tick_params(labelsize=30)
    ax_aufpt.set_xlim(0, 1)
    ax_aufpt.set_ylim(0, 1.05) 
    return(ax_aufpt)

def plot_fnt_ax(y, y_predicted, ax_aufnt, fig_w, fig_h, cutoff=np.nan, test_precision=0.9, test_recall=0.9, title_name='AUFNT'):  
    ax_aufnt.margins(1,1)
    color_lst = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000','#666633']
    
    if y_predicted.shape[1] > 1:

        lst_colors = color_lst
        
        for i in range(y_predicted.shape[1]):
            metrics_dict = classification_metrics(y, y_predicted[y_predicted.columns[i]], cutoff, test_precision, test_recall)[1]
#             ax_aufnt.plot(metrics_dict['fnts'], metrics_dict['precisions'], marker='o', markersize=2, color=lst_colors[i], label = y_predicted.columns[i] + ' (' + str(metrics_dict['aufnt']) + ') (' + str(metrics_dict['recall_fixed_precision']) +')')
            ax_aufnt.plot(metrics_dict['pcrs'], metrics_dict['pcr_fnts'], marker='o', markersize=2, color=lst_colors[i], label = y_predicted.columns[i])
            print (i)
        # Now add the legend with some customizations.
#         legend = ax_aufnt.legend(loc='upper left',bbox_to_anchor=(1, 1),shadow=True)        
        legend = ax_aufnt.legend(loc='lower right', shadow=True ,prop={'size': 30}) 
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        
        # Set the fontsize
#         for label in legend.get_texts():
#             label.set_fontsize('large')
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width    
    pass



    ax_aufnt.plot([0, 1], [1, 0], linestyle = '--', color = 'black', lw=2)
    
    ax_aufnt.set_title(title_name, size=40)
    ax_aufnt.set_xlabel('Positive count ratio',size = 35)
    ax_aufnt.set_ylabel('False Positive Trend (FN/FP+FN)',size = 35) 
    ax_aufnt.tick_params(labelsize=30)
    ax_aufnt.set_xlim(0, 1)
    ax_aufnt.set_ylim(0, 1.05) 
    return(ax_aufnt)

def plot_cv_prc_ax(mean_interp_recalls,mean_interp_precisions, interp_recalls_upper,interp_recalls_lower, ax, test_precision=0.9, test_recall=0.9,color = 'black',size_factor = 1):  
    ax.margins(1,1)    
#     print (str(list(mean_interp_recalls)[list(mean_interp_precisions).index(0.9)]))
    ax.plot(mean_interp_recalls,mean_interp_precisions, color =  color,lw=3, alpha=.8)
    ax.fill_betweenx(mean_interp_precisions, interp_recalls_lower, interp_recalls_upper, color='lightgrey', alpha=.2)               
    ax.plot([0, 1], [test_precision, test_precision], linestyle = '--', color = 'black', lw=2)   
    ax.set_xlabel('Recall',size = 40*size_factor,labelpad = 10*size_factor)
    ax.set_ylabel('Precision',size = 40*size_factor,labelpad = 10*size_factor)
    ax.tick_params(labelsize=35*size_factor,pad = 20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)     
    ax.set_xticklabels(['0','0.2','0.4','0.6','0.8','1'])     
    ax.set_yticklabels(['0','0.2','0.4','0.6','0.8','1'])    
    return(ax)

def plot_cv_roc_ax(mean_interp_fprs,mean_interp_tprs, interp_fprs_upper,interp_fprs_lower, ax, color = 'black',size_factor = 1):  
    ax.margins(1,1)
    ax.plot(mean_interp_fprs,mean_interp_tprs, color =  color,lw=3, alpha=.8)
    ax.fill_betweenx(mean_interp_tprs, interp_fprs_lower, interp_fprs_upper, color='lightgrey', alpha=.2)               
    ax.set_xlabel('False Positive Rate',size = 40*size_factor,labelpad = 10*size_factor)
    ax.set_ylabel('True Positive Rate',size = 40*size_factor,labelpad = 10*size_factor) 
    ax.tick_params(labelsize=35*size_factor,pad = 20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(['0','0.2','0.4','0.6','0.8','1'])     
    ax.set_yticklabels(['0','0.2','0.4','0.6','0.8','1'])
    return(ax)

def plot_prc_ax(y, y_predicted, ax_auprc, fig_w, fig_h, cutoff=0.5, test_precision=0.9, test_recall=0.9, title_name='AUPRC'):  
    ax_auprc.margins(1,1)
    color_lst = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000','#666633']
    
    if y_predicted.shape[1] > 1:
        if y_predicted.shape[1] > 8:
#             lst_colors = list(matplotlib.colors.CSS4_COLORS.keys())
            lst_colors = color_lst
        else:
            lst_colors = color_lst
#             lst_colors = list(matplotlib.colors.BASE_COLORS.keys())
        
        for i in range(y_predicted.shape[1]):
            metrics_dict = classification_metrics(y, y_predicted[y_predicted.columns[i]], cutoff, test_precision, test_recall)[1]
#             metrics_dict['precisions'][-1] = metrics_dict['precisions'][-2]                          
            ax_auprc.plot(metrics_dict['recalls'] + i*0.005 ,metrics_dict['precisions'] + i*0.005,marker='o',markersize=5,color = lst_colors[i],label = y_predicted.columns[i] + '(' + str(metrics_dict['auprc']) + ')')
            ax_auprc.plot(metrics_dict['balanced_recalls'], metrics_dict['balanced_precisions'], marker='o', markersize=5, color=lst_colors[i], label = y_predicted.columns[i] + ' (' + str(metrics_dict['aubprc']) + ') (' + str(metrics_dict['brfp']) +')')
            print (i)
        # Now add the legend with some customizations.
#         legend = ax_auprc.legend(loc='upper left',bbox_to_anchor=(1, 1),shadow=True)        
        legend = ax_auprc.legend(loc='lower left', shadow=True ,prop={'size': 30}) 
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        
        # Set the fontsize
#         for label in legend.get_texts():
#             label.set_fontsize('large')
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width    
    else:        
        metrics_dict = classification_metrics(y, y_predicted, cutoff, test_precision, test_recall)[1]    
        fig = plt.figure() 
        ax_auprc = plt.subplot()   
#         metrics_dict['precisions'][-1] = metrics_dict['precisions'][-2]          
#         ax_auprc.plot(metrics_dict['balanced_recalls'], metrics_dict['balanced_precisions'], marker='o', markersize=2, color='black', label='AUBPRC: (' + str(metrics_dict['aubprc']) + ')')
        
#         ax_auprc.plot(metrics_dict['recalls'], metrics_dict['precisions'], marker='o', markersize=2, color='black', label='AUPRC: (' + str(metrics_dict['auprc']) + ')')
        
        
        
        # Now add the legend with some customizations.
        legend = ax_auprc.legend(loc='lower left', shadow=True ,prop={'size': 20}) 
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        
#         for label in legend.get_texts():
#             label.set_fontsize(25)
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width   
    pass



    ax_auprc.plot([0, 1], [test_precision, test_precision], linestyle = '--', color = 'black', lw=2)
    
    ax_auprc.set_title(title_name, size=40)
    ax_auprc.set_xlabel('Recall',size = 35)
    ax_auprc.set_ylabel('Precision',size = 35) 
    ax_auprc.tick_params(labelsize=30)
    ax_auprc.set_xlim(0, 1)
    ax_auprc.set_ylim(0, 1.05) 
    return(ax_auprc)

def plot_roc_ax(y, y_predicted, ax_auroc, fig_w, fig_h, cutoff=np.nan, test_precision=0.9, test_recall=0.9, title_name='AUROC'):
    ax_auroc.margins(1,1)
    color_lst = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000','#666633']

    if y_predicted.shape[1] > 1:
        if y_predicted.shape[1] > 8:
            lst_colors = color_lst
        else:
            lst_colors = color_lst
#             lst_colors = list(matplotlib.colors.BASE_COLORS.keys())
        
        for i in range(y_predicted.shape[1]):
            metrics_dict = classification_metrics(y, y_predicted[y_predicted.columns[i]], cutoff, test_precision, test_recall)[1]  
            ax_auroc.plot(metrics_dict['fprs'], metrics_dict['tprs'], marker='o', markersize=2, color=lst_colors[i], label=y_predicted.columns[i] + '(' + str(metrics_dict['auroc']) + ')')
            
        # Now add the legend with some customizations.
#         legend = ax_auroc.legend(loc='upper left',bbox_to_anchor=(1, 1),shadow=True ,prop={'size': 20})  
        legend = ax_auroc.legend(loc='lower right',shadow=True ,prop={'size': 20})
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        
        # Set the fontsize
#         for label in legend.get_texts():
#             label.set_fontsize(25)
        
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width    
    else:
        
        metrics_dict = classification_metrics(y, y_predicted, cutoff, test_precision, test_recall)    
        fig = plt.figure() 
        ax_auroc = plt.subplot()            
        ax_auroc.plot(metrics_dict['fprs'], metrics_dict['tprs'], marker='o', markersize=2, color='black', label='AUROC:(' + str(metrics_dict['auroc']) + ')')
    pass
    ax_auroc.set_title(title_name, size=30)
    ax_auroc.set_xlabel('False Positive Rate',size = 20)
    ax_auroc.set_ylabel('True Positive Rate',size = 20) 
    ax_auroc.set_xlim(0, 1)
    ax_auroc.set_ylim(0, 1.05)
    return(ax_auroc)

def plot_prc_backup(metrics_dict):
    fig = plt.figure() 
    ax_auprc = plt.subplot()            
    ax_auprc.scatter(metrics_dict['recalls'], metrics_dict['precisions'], s=10)    
     
    diff = abs(metrics_dict['prc_thresholds'] - metrics_dict['best_cutoff'])
    diff_min = min(diff)
    cutoff_ind = list(diff).index(diff_min)
     
    cutoff_recall = round(metrics_dict['recalls'][cutoff_ind], 3)
    cutoff_precision = round(metrics_dict['precisions'][cutoff_ind], 3)
    best_cutoff = round(metrics_dict['best_cutoff'], 3)
     
    ax_auprc.scatter(cutoff_recall, cutoff_precision, color='red')  
    ax_auprc.annotate('[x: ' + str(cutoff_recall) + ' y: ' + str(cutoff_precision) + ' cutoff: ' + str(best_cutoff) + ']',
                xy=(cutoff_recall, cutoff_precision), xytext=(cutoff_recall - 0.1, cutoff_recall - 0.1), arrowprops=dict(facecolor='black', shrink=0.05))
       
    if metrics_dict['reverse'] == 1:      
        ax_auprc.set_title(' AUPRC: ' + str(metrics_dict['auprc']) + ' [reverse]', size=15)
    else:
        ax_auprc.set_title(' AUPRC: ' + str(metrics_dict['auprc']), size=15)
    pass
    ax_auprc.set_xlabel('Recall')
    ax_auprc.set_ylabel('Precision') 
    ax_auprc.set_xlim(0, 1)
    ax_auprc.set_ylim(0, 1)
    fig.tight_layout()
    plt.savefig(plot_name)   
    
def plot_roc_backup(metrics_dict):
    fig = plt.figure()    
    ax_auroc = plt.subplot()            
    ax_auroc.scatter(metrics_dict['fprs'], metrics_dict['tprs'], s=10)   
     
    diff = abs(metrics_dict['roc_thresholds'] - metrics_dict['best_cutoff'])
    diff_min = min(diff)
    cutoff_ind = list(diff).index(diff_min)
     
    cutoff_tpr = round(metrics_dict['tprs'][cutoff_ind], 3)
    cutoff_fpr = round(metrics_dict['fprs'][cutoff_ind], 3)
    best_cutoff = round(metrics_dict['best_cutoff'], 3)
     
    ax_auroc.scatter(cutoff_fpr, cutoff_tpr, color='red')  
    ax_auroc.annotate('[x: ' + str(cutoff_fpr) + ' y: ' + str(cutoff_tpr) + ' cutoff: ' + str(best_cutoff) + ']',
                xy=(cutoff_fpr, cutoff_tpr), xytext=(cutoff_fpr + 0.1, cutoff_tpr - 0.1),)
       
    if metrics_dict['reverse'] == 1:      
        ax_auroc.set_title(' AUROC: ' + str(metrics_dict['auroc']) + ' [reverse]', size=15)
    else:
        ax_auroc.set_title(' AUROC: ' + str(metrics_dict['auroc']), size=15)
    pass
    ax_auroc.set_xlabel('False Positive Rate')
    ax_auroc.set_ylabel('Ture Positive Rate') 
    ax_auroc.set_xlim(0, 1)
    ax_auroc.set_ylim(0, 1)         
             
def get_prior(y):
    return (y == 1).sum() / len(y)
    
def label_vector(y):
    y_unique = np.unique(y)
    y_vector = np.zeros([len(y), len(y_unique)])
    for i in range(len(y_unique)):
        y_vector[:, i] = (y == y_unique[i]).astype(int)
    return y_vector

def down_sampling(x):   
    x_p = x.loc[x.label == 1, :]
    x_n = x.loc[x.label == 0, :]
    if x_p.shape[0] > x_n.shape[0]:
        x_p = x_p.loc[np.random.permutation(x_p.index)[range(x_n.shape[0])], :]
    else:
        x_n = x_n.loc[np.random.permutation(x_n.index)[range(x_p.shape[0])], :]        
    x = pd.concat([x_p, x_n])
    return(x)

def sentence_count(text_string):
    return len(text_string.split('. ')) + 1
    
def word_count(text_string):
    return len(text_string.split()) + 1

def word_frequency(text_string, words, reg, normalization=0):
    text_frequency = dict(Counter(re.findall(reg, text_string)))
    if len(words) == 0:
        init_frequency = text_frequency
    else:
        init_frequency = {k: text_frequency.get(k, 0) for k in words}   
 
    if normalization == 0:
        frequency_normalized = init_frequency
     
    if normalization == 1:            
        n = sentence_count(text_string)
        frequency_normalized = dict((k, v / n) for k, v in init_frequency.items())
     
    if normalization == 2:            
        n = word_count(text_string)
        frequency_normalized = dict((k, v / n) for k, v in init_frequency.items())
     
    return frequency_normalized

def double_word_frequency(text_string, words):
    n_words = len(words)
    s_wf = np.empty((0, len(words)), int)
    sentences = text_string.split('. ')
    for s in sentences:                
        s_wf = np.vstack((s_wf, list(word_frequency(s, words, 0, 0).values())))
    pass
    x = np.matmul(np.transpose(s_wf), s_wf)
    idx_tri = np.triu_indices(n_words)    
    return list(x[idx_tri])

def multiclass_to_vectors(x):
    x_classes = x.unique()
    x_classes = sorted(x_classes)
    x_vectors = np.zeros([len(x), len(x_classes)])
     
    for i in range(len(x_classes)):
        x_vectors[x == x_classes[i], i] = 1
    return x_vectors

def bin_data(x, n_bins):
    bins = np.linspace(x.min(), x.max(), num=n_bins)    
    return np.digitize(x, bins) 



def normalize_data(x):
    x_max = np.nanmax(x)
    x_min  = np.nanmin(x)
    
    x = (x - x_min)/(x_max -x_min)
    return(x)    

def plot_stacked_barplot(data, fig_w, fig_h, title, title_size, x_label, y_label, label_size, tick_size, ylim_min, ylim_max, plot_name,ci_available = 0):
    plt.rcParams["font.family"] = "Helvetica"    
    fig = plt.figure(figsize=(fig_w, fig_h))
    # data_plot = data.stack().reset_index(inplace = True)
    
    data = data.loc[~data.isnull().any(axis=1), :]
    data_plot = data.copy()
    data_plot['x'] = data_plot.index
    # data_plot = data.reset_index(inplace = True)
    
#     if ci_available == 1:
#         data_plot.columns = ['y', 'ci', 'x']
#         list_ci = list(data_plot['ci'])
#     else:
#         data_plot.columns = ['total','bottom' 'x']
        
    ax = sns.barplot(x='x', y='total', data=data_plot,color = 'tomato')
    ax = sns.barplot(x='x', y='bottom', data=data_plot,color = 'seagreen')
    
    ax.set_xlabel(x_label, size=label_size)
    ax.set_ylabel(y_label, size=label_size)
    ax.tick_params(labelsize=tick_size)
    if (ylim_min is not None) & (ylim_max is not None):
        ax.set_ylim(ylim_min, ylim_max)
    ttl = ax.set_title(title, size=title_size)
    ttl.set_position([.5, 1.05])
    
    i = 1
    for p in ax.patches:        
        if i <= data_plot.shape[0]:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2, height + 500, str('{:,}'.format(int(height))), ha="center", size=tick_size,color = 'black')
        i += 1    
        
    labels = ['Pathogenic','Benign']        
    red_patch = patches.Patch(color='tomato', label='Pathogenic')
    green_patch = patches.Patch(color='seagreen', label='Benign')
    ax.legend(handles=[red_patch,green_patch],loc = 2,fontsize = 35)
    
        
    pass
    fig.tight_layout()
    plt.savefig(plot_name)   

def plot_stacked_barplot_ax(ax,data,title, title_size, x_label, y_label, label_size, tick_size, ylim_min, ylim_max,legend_size):
    data = data.loc[~data.isnull().any(axis=1), :]
    data_plot = data.copy()
    data_plot['x'] = data_plot.index
    # data_plot = data.reset_index(inplace = True)
    ax.ticklabel_format(axis = 'y', style = 'sci')
    
    ax.bar(data_plot['x'],data_plot['total'],color = '#C0504D')
    ax.bar(data_plot['x'],data_plot['bottom'],color = '#158066')

    ax.set_xlabel(x_label, size=label_size,labelpad = 20)
    ax.set_ylabel(y_label, size=label_size,labelpad = 20)
    ax.tick_params(labelsize=tick_size)

    if (ylim_min is not None) & (ylim_max is not None):
        ax.set_ylim(ylim_min, ylim_max)
    ttl = ax.set_title(title, size=title_size)
    ttl.set_position([.5, 1.05])
    
    i = 1
    for p in ax.patches:        
        if i <= data_plot.shape[0]:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2, height + ylim_max*0.02, str('{:,}'.format(int(height))), ha="center", size=tick_size,color = 'black')
        i += 1    
        
    labels = ['Putatively Pathogenic','Putatively Benign']        
    red_patch = patches.Patch(color='#C0504D', label='Putatively Pathogenic')
    green_patch = patches.Patch(color='#158066', label='Putatively Benign')
    ax.legend(handles=[red_patch,green_patch],loc = 2,fontsize = legend_size)
    
def plot_logistic (data,k,L,x0,plot_name):
    plt.rcParams["font.family"] = "Helvetica"    
    fig = plt.figure(figsize=(20,30))
    
#     data = data  * 100
#     x0 = x0 *100
    ax = plt.subplot() 

    for k in range(100):    
        ax.scatter (data, L/(1+np.exp(0-k*(data-x0))),color='black')
    fig.tight_layout()
    plt.savefig(plot_name)          
             
        
    
def plot_barplot(data, fig_w, fig_h, title, title_size, x_label, y_label, label_size, tick_size, ylim_min, ylim_max, plot_name,ci_available = 0):
    plt.rcParams["font.family"] = "Helvetica"    
    fig = plt.figure(figsize=(fig_w, fig_h))
    # data_plot = data.stack().reset_index(inplace = True)
    
    data = data.loc[~data.isnull().any(axis=1), :]
    data_plot = data.copy()
    data_plot['x'] = data_plot.index
    # data_plot = data.reset_index(inplace = True)
    
    if ci_available == 1:
        data_plot.columns = ['y', 'ci', 'x']
        list_ci = list(data_plot['ci'])
    else:
        data_plot.columns = ['y', 'x']
    ax = sns.barplot(x='x', y='y', data=data_plot)
    ax.set_xlabel(x_label, size=label_size)
    ax.set_ylabel(y_label, size=label_size)
    ax.tick_params(labelsize=tick_size)
    if (ylim_min is not None) & (ylim_max is not None):
        ax.set_ylim(ylim_min, ylim_max)
    ttl = ax.set_title(title, size=title_size)
    ttl.set_position([.5, 1.05])
    i = 0
    for p in ax.patches:
        
        height = p.get_height()
        if ci_available == 1:
            # plot error bar
            line_x = [p.get_x() + p.get_width() / 2, p.get_x() + p.get_width() / 2]
            line_y = [height - list_ci[i], height + list_ci[i]]
            ax.plot(line_x, line_y, 'k-', color='black' , linewidth=1) 
            # plot caps
            line_x = [p.get_x() + p.get_width() / 2 - p.get_width() / 20 , p.get_x() + p.get_width() / 2 + p.get_width() / 20]
            line_y = [height - list_ci[i], height - list_ci[i]]
            ax.plot(line_x, line_y, 'k-', color='black' , linewidth=1) 
            line_x = [p.get_x() + p.get_width() / 2 - p.get_width() / 20 , p.get_x() + p.get_width() / 2 + p.get_width() / 20]
            line_y = [height + list_ci[i], height + list_ci[i]]
            ax.plot(line_x, line_y, 'k-', color='black', linewidth=1) 
            # plot value   
            # ax.text(p.get_x()+p.get_width()/2,height + list_ci[i] + 0.01,str(np.round(height,4)) + '±' + str(list_ci[i]),ha="center",size = tick_size) 
            ax.text(p.get_x() + p.get_width() / 2, height + list_ci[i] + 0.01, str(np.round(height, 4)), ha="center", size=tick_size)
#         ax.text(p.get_x() + p.get_width() / 2, height + 1000, str(np.round(height, 0)), ha="center", size=tick_size)
        ax.text(p.get_x() + p.get_width() / 2, height/2, str('{:,}'.format(int(height))), ha="center", size=tick_size,color = 'white')
#         ax.text(p.get_x() + p.get_width() / 2, height/2, str(round(height*100,3)) + '%', ha="center", size=tick_size,color = 'white')
        i += 1    
    pass
    fig.tight_layout()
    plt.savefig(plot_name)   
    
def plot_scatter(fig_w, fig_h, x, y, x_test_name, y_test_name, hue, hue_name, title_extra, marker_size, plot_name):    
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.subplot()    
    if hue is None:
        ax.scatter(x, y, cmap='Blues', s=marker_size)
    else:
        ax.scatter(x, y, c=hue, cmap='Blues', s=marker_size)
 
    ax.set_title(x_test_name + ' VS ' + y_test_name + ' [pcc:' + str(round(pcc_cal(x, y), 3)) + 
                     '][spc:' + str(round(spc_cal(x, y), 3)) + '][color: ' + hue_name + '] ' + title_extra, size=25)
    ax.set_ylabel(y_test_name, size=20)
    ax.set_xlabel(x_test_name, size=20) 
    ax.tick_params(size=20)
    ax.legend()
    fig.tight_layout()
    plt.savefig(plot_name)  
    
def plot_sns_scatter(fig_w, fig_h, x, y, x_test_name, y_test_name, hue, hue_name, title_extra, marker_size, plot_name,normalization = 1):
    
    if normalization == 1:
        score_spc = spc_cal(x,y)        
        if score_spc < 0:
            x = 0-x        
        x = normalize_data(x)
        y = normalize_data(y)
        
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.subplot()   
    if hue is None:
        sns.scatterplot(x, y, ax = ax, s=marker_size)
    else:
        sns.scatterplot(x, y, hue=hue,ax = ax, s=marker_size)
 
    ax.set_title(x_test_name + ' VS ' + y_test_name + ' [pcc:' + str(round(pcc_cal(x, y), 3)) + 
                     '][spc:' + str(round(spc_cal(x, y), 3)) + '][color: ' + hue_name + '] ' + title_extra, size=25)
    ax.set_ylabel(y_test_name, size=20)
    ax.set_xlabel(x_test_name, size=20) 
    ax.tick_params(size=20)
    ax.legend()
    fig.tight_layout()
    plt.savefig(plot_name)  
       
def plot_color_gradients(vmax, vmin, vcenter, max_color, min_color, center_color, max_step, min_step, fig_width, fig_height, orientation, legend_name, fig_savepath):
    [lst_max_colors, lst_min_colors] = create_color_gradients(vmax, vmin, vcenter, max_color, min_color, center_color, max_step, min_step)                           
    fig = plt.figure(figsize=(fig_width, fig_height)) 
    ax = plt.subplot()
    
    if lst_min_colors is not None:
        lst_colors_new = lst_max_colors + lst_min_colors[1:]
    else:
        lst_colors_new = lst_max_colors
    n_tracks = 1
    
    
    if orientation == 'H':
        x = [0] + [0] * (n_tracks + 1) + list(range(0, len(lst_colors_new) + 1))
        y = [-1] + list(range(0, n_tracks + 1)) + [0] * (len(lst_colors_new) + 1)
        
    if orientation == 'V':
        y = [0] + [0] * (n_tracks + 1) + list(range(0, len(lst_colors_new) + 1))
        x = [-1] + list(range(0, n_tracks + 1)) + [0] * (len(lst_colors_new) + 1)

    ax.plot(x, y, alpha=0)
#     ax.plot(x, y)

    # add rectangles
    for i in range(len(lst_colors_new)):
        xy_color = lst_colors_new[len(lst_colors_new) - i - 1] 
        if orientation == 'V':
            rect = patches.Rectangle((0, i), 1, 1, linewidth=1, edgecolor=xy_color, facecolor=xy_color, fill=True, clip_on=False)
        if orientation == 'H':
            rect = patches.Rectangle((i, 0), 1, 1, linewidth=1, edgecolor=xy_color, facecolor=xy_color, fill=True, clip_on=False)
             
        ax.add_patch(rect)
    pass
 
    # add lines
    min_pos = 0.5
    max_pos = len(lst_colors_new) - 0.5
    center_pos = (len(lst_colors_new) - 1) / 2 + 0.5
    tick_offset = -0.15
    lines = []    
    if orientation == 'H':
        lines.append(((min_pos, tick_offset), (max_pos, tick_offset)))
        lines.append(((min_pos, tick_offset), (min_pos, tick_offset * 2)))
        if vmin != vcenter :
            lines.append(((center_pos, tick_offset), (center_pos, tick_offset * 2)))
        lines.append(((max_pos, tick_offset), (max_pos, tick_offset * 2)))
    if orientation == 'V':
        lines.append((tick_offset, min_pos), (tick_offset, max_pos))
        lines.append((tick_offset, min_pos), (tick_offset * 2, min_pos))
        if vmin != vcenter :
            lines.append((tick_offset, center_pos), (tick_offset * 2, center_pos))
        lines.append((tick_offset, max_pos), (tick_offset * 2, max_pos))        
    lc = collections.LineCollection(lines, linewidth=2, color='black', clip_on=False)
    ax.add_collection(lc) 
     
    if orientation == 'H':
        ax.text(min_pos, tick_offset * 4, vmin, rotation=360, fontsize=10, ha='center', weight='bold')
        ax.text(max_pos, tick_offset * 4, vmax, rotation=360, fontsize=10, ha='center', weight='bold')
        if vmin != vcenter :
            ax.text(center_pos, tick_offset * 4, vcenter, rotation=360, fontsize=10, ha='center', weight='bold')
        ax.text(center_pos, tick_offset * 6, legend_name, rotation=360, fontsize=12, ha='center', weight='bold')
    if orientation == 'V':
        ax.text(tick_offset * 4, min_pos, vmin, rotation=360, fontsize=10, va='center', weight='bold')
        ax.text(tick_offset * 4, max_pos, vmax, rotation=360, fontsize=10, va='center', weight='bold')
        if vmin != vcenter :
            ax.text(tick_offset * 4, vcenter, center_pos, rotation=360, fontsize=10, va='center', weight='bold')
        ax.text(tick_offset * 6, center_pos, legend_name, rotation=360, fontsize=12, va='center', weight='bold')
     

    plt.axis('off')
    plt.savefig(fig_savepath, format='png', dpi=300, transparent=True)

def create_color_gradients(vmax, vmin, vcenter, max_color, min_color, center_color, max_step, min_step):
    if vmax > vcenter:
        lst_max_colors = color_gradient(center_color, max_color, max_step)
    else:
        lst_max_colors = None
    if vmin < vcenter:
        lst_min_colors = color_gradient(min_color, center_color, min_step)
    else:
        lst_min_colors = None
    return [lst_max_colors, lst_min_colors]
   
def get_colorcode(value, vmax, vmin, vcenter, max_step, min_step, lst_max_colors, lst_min_colors):
    if value > vmax: value = vmax  
    if value < vmin: value = vmin        
    if np.isnan(value): 
        colorcode = '#C0C0C0'
    else:
        if value >= vcenter:            
            colorcode = lst_max_colors[(len(lst_max_colors) - 1) - int(round((value - vcenter) * max_step / (vmax - vcenter)))]
        else:
            colorcode = lst_min_colors[(len(lst_min_colors) - 1) - int(round((value - vmin) * min_step / (vcenter - vmin)))]
    colorcode = colorcode.replace('x', '0')
    return colorcode

def color_arrayMultiply(array, c):
    return [element * c for element in array]

def color_arraySum(a, b):
    return list(map(sum, zip(a, b)))

def color_intermediate(a, b, ratio):
    aComponent = color_arrayMultiply(a, ratio)
    bComponent = color_arrayMultiply(b, 1 - ratio)
    decimal_color = color_arraySum(aComponent, bComponent)
    hex_color = '#' + str(hex(int(decimal_color[0])))[-2:] + str(hex(int(decimal_color[1])))[-2:] + str(hex(int(decimal_color[2])))[-2:]
    hex_color = hex_color.replace('x', '0')
    return hex_color

def color_gradient(a, b, steps):
    lst_gradient_colors = []
    start_color = [ int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)]
    end_color = [ int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)]
    steps = [n / float(steps) for n in range(steps)]
    for step in steps:
        hex_color = color_intermediate(start_color, end_color, step)
        lst_gradient_colors.append(hex_color)
    lst_gradient_colors = lst_gradient_colors + [a]
    return lst_gradient_colors

def send_email(server_address,server_port,login_user,login_password,from_address,to_address, subject, msg_content):
    server = smtplib.SMTP('smtp-relay.gmail.com', 587)
    server.starttls()
    server.login(login_user, login_password)  
    msg = EmailMessage()       

    msg.set_content(msg_content)
    msg['Subject'] = subject
    msg['From'] = from_address
    msg['To'] = to_address
    server.sendmail(from_address, to_address, msg.as_string())
    server.quit()
           
def error_propagation_fun(value1, value1_err, inFun):
    if inFun == 'log':
        value = np.log(value1)
        value_err = np.abs(value1_err / value1)
        
    if inFun == 'log10':
        value = np.log(value1)
        value_err = np.abs(value1_err / (value1 * log(10)))

    return pd.Series([value, value_err])   

def error_propagation_operation(value1, value1_err, value2, value2_err, inOperation):        
    if inOperation == 'addition':
        value = value1 + value2
        value_err = np.sqrt(value1_err ** 2 + value2_err ** 2)            
    if inOperation == 'subtraction':
        value = value1 - value2
        value_err = np.sqrt(value1_err ** 2 + value2_err ** 2)                 
    if inOperation == 'division':
        value = value1 / value2
        value_err = np.abs(value) * np.sqrt((value1_err / value1) ** 2 + (value2_err / value2) ** 2)   
    if inOperation == 'multiplication':
        value = value1 / value2
        value_err = np.abs(value) * np.sqrt((value1_err / value1) ** 2 + (value2_err / value2) ** 2)   
    
    return pd.Series([value, value_err])

def scatter_plots(ax, x, y, x_target_name, y_target_name, hue =  None, hue_name = '', title_extra = '', color = 'royalblue', color_map = 'Blues', marker_size=160, title_show_r = 0):
    if hue is None:
        ax.scatter(x, y, c = color, s=marker_size)
    else:
        ax.scatter(x, y, c=hue, cmap= color_map, s=marker_size)
   
    if title_show_r == 1:
        ax.set_title(x_target_name + ' VS ' + y_target_name + ' [pcc:' + str(round(alm_fun.pcc_cal(x, y), 3)) + '][spc:' + str(round(alm_fun.spc_cal(x, y), 3)) + '][color: ' + hue_name + '] ' + title_extra, size=20)
    else:
        ax.set_title(x_target_name + ' VS ' + y_target_name +  title_extra, size=20)

    ax.set_ylabel(y_target_name, size=15)
    ax.set_xlabel(x_target_name, size=15) 
    ax.tick_params(size=20)
    return(ax)

def mse_xgb_obj(y_true,y_pred):
    y = y_true
    x = y_pred
    n = len(x)
    
    grad = 2*(x-y)
    hess = np.array(n*[2])
    
#     print(alm_fun.mse_cal(y_true,y_pred))
    return (grad,hess)

def modified_mse_xgb_obj(y_true,y_pred):
    y = y_true
    x = y_pred
    n = len(x)
    
#     theta = 0
#     gamma = np.array([1]*(n-38758) +[0.5]*38758)
        
#     if np.unique(y_pred)[0] == 0.5:
#         x = x + np.random.uniform(0,0.01,n)

    sd_x = np.std(x)*np.sqrt(n)
    mean_x = np.mean(x)
    sd_y = np.std(y)*np.sqrt(n)
    mean_y = np.mean(y)
    cov_xy = np.cov(x,y)[0][1]*n   

    mse_grad = 2*(x-y)*gamma
    mse_hess = np.array(n*[2])*gamma
    
    pcc_grad = ((sd_x**2)*(y-mean_y) - cov_xy*(x-mean_x)) / ((sd_x**3)*sd_y)
    pcc_hess = (2*(sd_x**2)*(x-mean_x)*(y-mean_y) + cov_xy*((3*(x-mean_x)**2) - ((n-1)/n)*(sd_x**2)))/((sd_x**5)*sd_y)
    
    if np.unique(y_pred)[0] == 0.5:
        pcc_grad = np.array(n*[0])
        pcc_hess = np.array(n*[0])

    grad = mse_grad - theta*pcc_grad
    hess = mse_hess - theta*pcc_hess
    
#     print('pcc:' + str(round(alm_fun.pcc_cal(y_true,y_pred),4)) + ' mse:' + str(round(alm_fun.mse_cal(y_true,y_pred),4)))
          
    return (grad,hess)

def pcc_xgb_obj(y_true,y_pred):    
    y = y_true
    x = y_pred
    n = len(x)
    
    if np.unique(y_pred)[0] == 0.5:
        x = x + np.random.uniform(0,0.01,n)

    sd_x = np.std(x)*np.sqrt(n)
    mean_x = np.mean(x)
    sd_y = np.std(y)*np.sqrt(n)
    mean_y = np.mean(y)
    cov_xy = np.cov(x,y)[0][1]*n
        
    grad = ((sd_x**2)*(y-mean_y) - cov_xy*(x-mean_x)) / ((sd_x**3)*sd_y)
    hess = (2*(sd_x**2)*(x-mean_x)*(y-mean_y) + cov_xy*((3*(x-mean_x)**2) - ((n-1)/n)*(sd_x**2)))/((sd_x**5)*sd_y)
    
    grad = 0-grad
    hess = 0-hess
    
#     print(alm_fun.pcc_cal(y_true,y_pred))  
    print('pcc:' + str(round(alm_fun.pcc_cal(y_true,y_pred),4)) + ' mse:' + str(round(alm_fun.mse_cal(y_true,y_pred),4)))
      
    return(grad,hess)

def pcc_mse_xgb_obj(y_true,y_pred): 
    theta = 50
    y = y_true
    x = y_pred
    n = len(x)
    
    if np.unique(y_pred)[0] == 0.5:
        x = x + np.random.uniform(0,0.01,n)

    sd_x = np.std(x)*np.sqrt(n)
    mean_x = np.mean(x)
    sd_y = np.std(y)*np.sqrt(n)
    mean_y = np.mean(y)
    cov_xy = np.cov(x,y)[0][1]*n

    mse_grad = 2*(x-y)
    mse_hess = np.array(n*[2])
    
    pcc_grad = ((sd_x**2)*(y-mean_y) - cov_xy*(x-mean_x)) / ((sd_x**3)*sd_y)
    pcc_hess = (2*(sd_x**2)*(x-mean_x)*(y-mean_y) + cov_xy*((3*(x-mean_x)**2) - ((n-1)/n)*(sd_x**2)))/((sd_x**5)*sd_y)
    
    grad = mse_grad - theta*pcc_grad
    hess = mse_hess - theta*pcc_hess
    
    print('pcc:' + str(round(alm_fun.pcc_cal(y_true,y_pred),4)) + ' mse:' + str(round(alm_fun.mse_cal(y_true,y_pred),4)))
    
    return(grad,hess)

def fdr_cutoff(p_values, alpha = 0.05):
    p_values = sorted(list(p_values))
    m = len(p_values)
    for i in range(m):
        if p_values[i] > (i+1)*alpha/m:
            if i == 0:
                return(0)
            else:
                return(p_values[i-1])
    
def hamming_distance(s1, s2):
    assert len(s1) == len(s2)
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))   

def get_codon_titv(ref_codon, alt_codon):
        ref_codon_list = list(ref_codon)
        alt_codon_list = list(alt_codon)
        ref_nt = ''
        alt_nt = ''
        for i in range(len(ref_codon_list)):
            if ref_codon_list[i] != alt_codon_list[i]:
                ref_nt = ref_codon_list[i]
                alt_nt = alt_codon_list[i]
                break
        if ((ref_nt == 'A') & (alt_nt == 'G')) | ((ref_nt == 'G') & (alt_nt == 'A')) | ((ref_nt == 'C') & (alt_nt == 'T')) | ((ref_nt == 'T') & (alt_nt == 'C')):
            return ('ti')
        else:
            return ('tv')
    
def get_nt_titv(ref_nt, alt_nt):
        if ((ref_nt == 'A') & (alt_nt == 'G')) | ((ref_nt == 'G') & (alt_nt == 'A')) | ((ref_nt == 'C') & (alt_nt == 'T')) | ((ref_nt == 'T') & (alt_nt == 'C')):
            return ('ti')
        else:
            return ('tv')
           
def get_random_id(length):
    id = ''
    s = '0123456789ABCDdEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(length):
        id = id + random.choice(s)
    return(str(id))