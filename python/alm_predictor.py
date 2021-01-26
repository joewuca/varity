#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import pandas as pd
import csv
import os
import re
import operator
import itertools
import time
import math
import random
import codecs
import copy
import pickle
import hyperopt
from datetime import datetime

from sklearn import model_selection as ms
import alm_fun
    
class alm_predictor:

    def __init__(self, predictor_init_params):        
        for key in predictor_init_params:
            setattr(self, key, predictor_init_params[key])
            
        #parameters that are not open for configuration yet
        self.rfp_cutoff = 0.9
        self.pfr_cutoff = 0.9
        self.init_weights = None
        self.target_as_source = None
        self.fill_na_type = None
        self.tune_tree_nums_before_test = 0
        self.tune_tree_nums_during_cv = 0
        self.shuffle_features =[]
        self.eval_obj = 'auc'
        self.trials_mv_step = 0
        self.use_extra_train_data = 2
        self.nofit = 0
        self.if_feature_engineer = 0

        alm_fun.show_msg (self.log,self.verbose,'Class: [alm_predictor] [__init__] ' + self.name + ' ...... @' + str(datetime.now()))

        if self.type == 1: # VARITY predictors      
#             if self.hp_tune_type == 'hyperopt':
#                 self.create_hyperopt_hps() # create hp_config_dict and hyperopt_hps
#             if self.hp_tune_type == 'hyperopt_logistic':
            self.create_hyperopt_logistic_hps()
        

    def create_hyperopt_logistic_hps(self):
        self.hyperopt_hps = {} 
        self.hp_default = {}
        extra_data = self.data_instance.extra_train_data_df_lst[0]
        core_data =  self.data_instance.train_data_index_df
        for cur_hp_name in self.hyperparameter.keys():            
            cur_hp_dict =  self.hyperparameter[cur_hp_name]
#             print (cur_hp_name)  
            cur_hp_dict['enable'] = 1              
            if cur_hp_dict['hp_type'] in [1,2]:
                if self.name not in self.qip[cur_hp_dict['qip']]['predictors']:  # The qip is not in use for the current predictor
                    cur_hp_dict['enable'] = 0         
                if self.qip[cur_hp_dict['qip']]['enable'] == 0 :  # The qip is not enabled
                    cur_hp_dict['enable'] = 0                                             
                      
            if cur_hp_dict['enable'] == 1:                    
                if ((cur_hp_dict['from'] == 'None') | (cur_hp_dict['to'] == 'None')) & (cur_hp_dict['hp_type'] == 2): #determine range of the weight function hyperparameters
                    
                    ####order the affect extra data set by informative property and its index to make sure the rank of the order is identical very time
                    cur_qip = cur_hp_dict['qip']
                    cur_set_type = self.qip[cur_qip]['set_type']
                    cur_set_list = self.qip[cur_qip]['set_list']
                    cur_qip_col = self.qip[cur_qip]['qip_col']
                    cur_qip_normalized_col = cur_qip + '_normalized'
                    cur_qip_direction = self.qip[cur_qip]['direction']
                    
                    if cur_set_type == 'core':
                        data = core_data
                    if cur_set_type == 'addon':
                        data = extra_data
                    
                    cur_qip_indices = data['set_name'].isin(cur_set_list)                                                            
                    data[cur_qip_normalized_col] = np.nan                                    
                    cur_max_value = np.nanmax(data.loc[cur_qip_indices,cur_qip_col])
                    cur_min_value = np.nanmin(data.loc[cur_qip_indices,cur_qip_col])
                                    
                    data.loc[cur_qip_indices,cur_qip_normalized_col] = (data.loc[cur_qip_indices,cur_qip_col] - cur_min_value)/(cur_max_value-cur_min_value)                    
                    cur_data = data.loc[cur_qip_indices,:]                    
                    cur_data['org_index'] = cur_data.index
                    cur_data = cur_data.sort_values([cur_qip_col,'org_index'])
                    cur_data = cur_data.reset_index(drop = True )
                    #### extract candidate midpoints 
                    mid_points = []
                    cur_data_interval = cur_hp_dict['data_interval']
                    max_index = cur_data.index.max()
                    for idx in range(0,max_index,cur_data_interval):
                        mid_points.append(cur_data.loc[idx,cur_qip_normalized_col])
                    
                    #### add two possible midpoints out of range 0 to 1, to take care the case that the smallest and largest midpoint values are constraint to be L/2 when they are picked as x0     
                    mid_points.append(-1.0)
                    mid_points.append(2.0)                              
                    cur_hp_dict['values'] = mid_points                  
                else:                                    
                    cur_hp_dict['values'] = list(np.arange(cur_hp_dict['from'], cur_hp_dict['to'], cur_hp_dict['step'],dtype=cur_hp_dict['data_type']))
                    cur_hp_dict['values'].append(cur_hp_dict['to']) # np.arrange doesn't include stop point, so we manully add to it
                    
                
                # remove duplicates and sort the values
                cur_hp_dict['values'] = list(set(cur_hp_dict['values']))
                cur_hp_dict['values'].sort()   
                                    
                # round to significant_digits
                if str(cur_hp_dict['significant_digits']) != 'None':
                    cur_hp_dict['values'] = np.round(cur_hp_dict['values'],cur_hp_dict['significant_digits']) 
                                          
                self.hyperopt_hps[cur_hp_name] = hyperopt.hp.choice(cur_hp_name,cur_hp_dict['values'])     
                self.hp_default[cur_hp_name] = cur_hp_dict['default']
                        
    def create_hyperopt_hps(self):    
        hyperopt_hps = {}               
        extra_data = self.data_instance.extra_train_data_df_lst[0]
        self.hp_parameters = {} 
        self.hp_parameters['all'] = []   
        self.hp_parameters['hyperopt'] = []
        self.hp_parameters['sp'] = []
        self.hp_default = {}     
        self.hp_values = {}  
        self.hp_range_start = {}
        self.hp_range_end = {}
        self.hp_indices = {}      
        self.hp_rest_indices  = {}      
        self.hp_directions = {}     
        
        self.hp_mv_values = {}
        self.hp_mv_range_start = {}
        self.hp_mv_range_end = {}
        self.hp_mv_indices = {}      
        self.hp_mv_rest_indices  = {}      
        
        self.hps = {}
              
        create_new_hp_config = 0                 
        hp_config_file = self.project_path + '/output/npy/' + self.session_id + '_' + self.name + '_hp_config_dict.npy'
        if os.path.isfile(hp_config_file):
            if self.init_hp_config == 1: 
                create_new_hp_config = 1
        else:
            create_new_hp_config = 1
            
        if create_new_hp_config == 0:        
            hp_config_dict = np.load(hp_config_file).item()
            self.hp_directions = hp_config_dict['hp_directions']
            self.hp_parameters = hp_config_dict['hp_parameters']             
            self.hp_values = hp_config_dict['hp_values'] 
            self.hp_range_start = hp_config_dict['hp_range_start'] 
            self.hp_range_end = hp_config_dict['hp_range_end']
            self.hp_indices = hp_config_dict['hp_indices']
            self.hp_rest_indices = hp_config_dict['hp_rest_indices']  
            
            
            if self.old_system == 1:
#                 self.hp_directions['extra_gnomad_af'] = 1
                #the old hp_config_file didn't include hp_default and moving window analysis config so we add it manually             
                #for hp evaluation config
                for cur_hp in self.hyperparameter.keys():            
                    hp = self.hyperparameter[cur_hp]
                    #parameters that are not open for configuration yet
                    hp['filter_type'] = 3 # filtering method
                    hp['mv_type'] = 0 # moving analysis method
                    hp['mv_size'] = 0    
                    if self.name not in hp['predictor']:
                        continue
                    else:
                        self.hps[cur_hp] = hp                    
                                    
                    if hp['hp_type'] == 1: #filtering parameters
                        extra_data_df = extra_data.loc[extra_data['set_name'].isin(hp['source']),:]            
                        self.hp_parameters['sp'].append(cur_hp)
                        [pivots,pivot_indices,pivot_rest_indices,pivot_values_range_start,pivot_values_range_end] = self.pivot_points(extra_data_df,hp,1)
                        self.hp_mv_values[cur_hp] = range(len(pivots))
                        self.hp_mv_indices[cur_hp] = pivot_indices
                        self.hp_mv_rest_indices[cur_hp] = pivot_rest_indices
                        self.hp_mv_range_start[cur_hp] = pivot_values_range_start
                        self.hp_mv_range_end[cur_hp] = pivot_values_range_end    
                
                #for hp_default
                self.hp_default = hp_config_dict['hp_default']
                for cur_hp in self.hyperparameter.keys():            
                    hp = self.hyperparameter[cur_hp]            
                    self.hp_default[cur_hp] = hp['default']         

                hp_config_dict['hp_mv_values'] = self.hp_mv_values
                hp_config_dict['hp_mv_range_start'] = self.hp_mv_range_start
                hp_config_dict['hp_mv_range_end'] = self.hp_mv_range_end
                hp_config_dict['hp_mv_indices'] = self.hp_mv_indices
                hp_config_dict['hp_mv_rest_indices'] = self.hp_mv_rest_indices
                
                hp_config_dict['hp_default'] = self.hp_default
                hp_config_dict['hps'] = self.hps
                                    
                np.save(hp_config_file,hp_config_dict)
                print ('old system hp_config_dict converted.....')  
            else:
                self.hp_mv_values = hp_config_dict['hp_mv_values']
                self.hp_mv_indices = hp_config_dict['hp_mv_indices']
                self.hp_mv_rest_indices = hp_config_dict['hp_mv_rest_indices']
                self.hp_mv_range_start = hp_config_dict['hp_mv_range_start']
                self.hp_mv_range_end = hp_config_dict['hp_mv_range_end']
                self.hp_default = hp_config_dict['hp_default']
                self.hps = hp_config_dict['hps']
                                               
            alm_fun.show_msg (self.log,self.verbose,'Saved hp config dict loaded.' )  
        else:
            for cur_hp in self.hyperparameter.keys():                            
                hp = self.hyperparameter[cur_hp]
                #parameters that are not open for configuration yet
                hp['filter_type'] = 3 # filtering method
                hp['mv_type'] = 0 # moving analysis method
                hp['mv_size'] = 0
                                
                if self.name not in hp['predictor']:
                    continue
                else:
                    self.hps[cur_hp] = hp

                if hp['hp_type'] == 1: #filtering parameters
                    self.hp_default[cur_hp] = hp['default']
                    self.hp_directions[cur_hp] = hp['direction']
                    if hp['enable'] == 1:
                        self.hp_parameters['hyperopt'].append(cur_hp)                    
                    extra_data_df = extra_data.loc[extra_data['set_name'].isin(hp['source']),:]
                    [pivots,pivot_indices,pivot_rest_indices,pivot_values_range_start,pivot_values_range_end] = self.pivot_points(extra_data_df,hp,0)                         
                    self.hp_values[cur_hp] = range(len(pivots))
                    self.hp_indices[cur_hp] = pivot_indices
                    self.hp_rest_indices[cur_hp] = pivot_rest_indices
                    self.hp_range_start[cur_hp] = pivot_values_range_start
                    self.hp_range_end[cur_hp] = pivot_values_range_end        
                    
                    #for moving widow analysis
                    self.hp_parameters['sp'].append(cur_hp)
                    [pivots,pivot_indices,pivot_rest_indices,pivot_values_range_start,pivot_values_range_end] = self.pivot_points(extra_data_df,hp,1)
                    self.hp_mv_values[cur_hp] = range(len(pivots))
                    self.hp_mv_indices[cur_hp] = pivot_indices
                    self.hp_mv_rest_indices[cur_hp] = pivot_rest_indices
                    self.hp_mv_range_start[cur_hp] = pivot_values_range_start
                    self.hp_mv_range_end[cur_hp] = pivot_values_range_end                         
                    
                if hp['hp_type'] == 2: #weight parameters                                              
                    self.hp_default[cur_hp] = hp['default']
                    self.hp_parameters['hyperopt'].append(cur_hp)
#                     cur_weights = np.linspace(start = hp['from'], stop = hp['to'], num = 101)
                    cur_weights = np.round(np.arange( hp['from'],  hp['to'] + hp['step'], hp['step']),2)
                    self.hp_values[cur_hp] =  cur_weights
                    self.hp_indices[cur_hp] = {} 
                    self.hp_rest_indices[cur_hp] = {}
                    self.hp_range_start[cur_hp] = {} 
                    self.hp_range_end[cur_hp] = {}
                    for weight in cur_weights:
                        self.hp_indices[cur_hp][weight] = extra_data_df.index   
                        self.hp_rest_indices[cur_hp][weight] = []
                        self.hp_range_start[cur_hp][weight] = np.nan
                        self.hp_range_end[cur_hp][weight] = np.nan    
                            
             #save current hyper parameter configurations        
            hp_config_dict = {}                
            hp_config_dict['hp_parameters'] = self.hp_parameters            
            hp_config_dict['hp_values'] = self.hp_values
            hp_config_dict['hp_range_start'] = self.hp_range_start
            hp_config_dict['hp_range_end'] = self.hp_range_end
            hp_config_dict['hp_indices'] = self.hp_indices
            hp_config_dict['hp_rest_indices'] = self.hp_rest_indices
            hp_config_dict['hp_directions'] = self.hp_directions     
            
            hp_config_dict['hp_mv_values'] = self.hp_mv_values
            hp_config_dict['hp_mv_range_start'] = self.hp_mv_range_start
            hp_config_dict['hp_mv_range_end'] = self.hp_mv_range_end
            hp_config_dict['hp_mv_indices'] = self.hp_mv_indices
            hp_config_dict['hp_mv_rest_indices'] = self.hp_mv_rest_indices
            
            hp_config_dict['hp_default'] = self.hp_default
            hp_config_dict['hps'] = self.hps
                       
            np.save(hp_config_file,hp_config_dict)                   
            alm_fun.show_msg (self.log,self.verbose,'Hyperparameter config dictionary for ' + self.name + ' saved.' )    
            
                        
        for hp_parameter in self.hp_parameters['hyperopt']:                
            hyperopt_hps[hp_parameter] = hyperopt.hp.choice(hp_parameter, self.hp_values[hp_parameter])
        pass  
        
        for cur_hp in self.hyperparameter.keys():
            hp = self.hyperparameter[cur_hp]
            if self.name not in hp['predictor']:
                continue

            if hp['hp_type'] == 3: #algorithm level hyper-parameters      
                self.hp_default[cur_hp] =  hp['default']             
                if hp['type'] == 'real':
                    hyperopt_hps[cur_hp] = hyperopt.hp.quniform(cur_hp, hp['from'], hp['to'], hp['step'])                
                if (hp['type'] == 'int') | (hp['type'] == 'category'):
                    hyperopt_hps[cur_hp] = hyperopt.hp.choice(cur_hp, np.arange(hp['from'], hp['to'], dtype=int))        
        
#         np.save(self.project_path + '/output/npy/' + self.session_id + '_hyperopt_hps.npy',hyperopt_hps)
        return(hyperopt_hps)

    def pivot_points (self,data_df,hp,hp_evaluation):                            
        pivots = [0]
        pivot_indices = {}
        pivot_indices[0] = []            
        pivot_rest_indices = {}
        pivot_rest_indices[0] = list(data_df.index)       
        pivot_values_range_start = {}
        pivot_values_range_start[0] = np.nan
        pivot_values_range_end = {}
        pivot_values_range_end[0] = np.nan
        
        if data_df.shape[0] ==0:
            return ([pivots,pivot_indices,pivot_rest_indices,pivot_values_range_start,pivot_values_range_end])
        
        max_value = data_df[hp['orderby']].max()
        min_value = data_df[hp['orderby']].min()

        if hp['direction'] == -1:
            data_df = data_df.loc[np.random.permutation(data_df.index),:]  
        if hp['direction'] == 0:
                data_df = data_df.sort_values(hp['orderby'],ascending = False)
        if hp['direction'] == 1:
                data_df = data_df.sort_values(hp['orderby'],ascending = True)

        if hp_evaluation == 1:    #For Moving Window Analysis                            
            if hp['mv_type'] == 0:  # use mv_size_percent to determine window size
                hp['mv_size'] = int(data_df.shape[0] * hp['mv_size_percent'] /100)
                moving_length = int((data_df.shape[0]-hp['mv_size'])/(hp['mv_data_points']-1))
                                
            if hp['mv_type'] == 1: # use mv_size to determine window size                  
                moving_length = int((data_df.shape[0]-hp['mv_size'])/(hp['mv_data_points']-1))
                                         
            for i in range(hp['mv_data_points']):
                af_index_low = i*(moving_length)
                af_index_high = af_index_low + hp['mv_size']
                if i == hp['mv_data_points']-1:
                    af_index_high = data_df.shape[0]                    
                pivots.append(af_index_high)
                cur_pivot_indices = list(data_df.index[af_index_low:af_index_high])
                cur_pivot_rest_indices = list(data_df.index[:af_index_low]) + list(data_df.index[af_index_high:]) 

                pivot_indices[len(pivots)-1] = cur_pivot_indices
                pivot_rest_indices[len(pivots)-1] = cur_pivot_rest_indices
                pivot_values_range_start[len(pivots)-1] = data_df.loc[data_df.index[af_index_low],hp['orderby']]
                pivot_values_range_end[len(pivots)-1] = data_df.loc[data_df.index[af_index_high-1],hp['orderby']]
                
            pivots.append (data_df.shape[0])
            pivot_indices[len(pivots)-1] = list(data_df.index)
            pivot_rest_indices[len(pivots)-1] = []
            pivot_values_range_start[len(pivots)-1] = min_value
            pivot_values_range_end[len(pivots)-1] = max_value
        
        else:            
            if hp['filter_type'] == 1:  #For Filtering: data points are splited by value cutoffs         
                cutoffs = np.linspace(start = min_value, stop = max_value, num = hp['filter_data_points']+1)                              
                for i in range(hp['filter_data_points']):  
                    if i == hp['filter_data_points'] - 1:
                        range_length = data_df.loc[(data_df[hp['orderby']] >= cutoffs[i]) & (data_df[hp['orderby']] <= cutoffs[i+1])].shape[0]
                    else:
                        range_length = data_df.loc[(data_df[hp['orderby']] >= cutoffs[i]) & (data_df[hp['orderby']] < cutoffs[i+1])].shape[0]     
                    af_index_low = 0
                    af_index_high = range_length + pivots[len(pivots)-1]
                    pivots.append(af_index_high)                         
                    cur_pivot_indices = list(data_df.index[af_index_low:af_index_high])
                    cur_pivot_rest_indices = list(data_df.index[:af_index_low]) + list(data_df.index[af_index_high:]) 
                    pivot_indices[len(pivots)-1] = cur_pivot_indices
                    pivot_rest_indices[len(pivots)-1] = cur_pivot_rest_indices
                    pivot_values_range_start[len(pivots)-1] = data_df.loc[data_df.index[af_index_low],hp['orderby']]
                    pivot_values_range_end[len(pivots)-1] = data_df.loc[data_df.index[af_index_high-1],hp['orderby']]      
    
            if hp['filter_type'] == 2: # #For Filtering: data points are splited into equal chunks by input number of windows                            
                range_length =   int(data_df.shape[0]/hp['filter_data_points'])
                for i in range(hp['filter_data_points']):  
                    af_index_low = 0
                    af_index_high = range_length + pivots[len(pivots)-1]
                    if i == hp['filter_data_points']-1:
                        af_index_high = data_df.shape[0]  
                    pivots.append(af_index_high)                         
                    cur_pivot_indices = list(data_df.index[af_index_low:af_index_high])
                    cur_pivot_rest_indices = list(data_df.index[:af_index_low]) + list(data_df.index[af_index_high:]) 
                    pivot_indices[len(pivots)-1] = cur_pivot_indices
                    pivot_rest_indices[len(pivots)-1] = cur_pivot_rest_indices
                    pivot_values_range_start[len(pivots)-1] = data_df.loc[data_df.index[af_index_low],hp['orderby']]
                    pivot_values_range_end[len(pivots)-1] = data_df.loc[data_df.index[af_index_high-1],hp['orderby']]    
                
            if hp['filter_type'] == 3: # #For Filtering: data points are splited into equal chunks by input chunk size                            
                range_length =  hp['step']
                hp['filter_data_points'] = int(np.ceil(data_df.shape[0]/range_length))
                for i in range(hp['filter_data_points']):  
                    af_index_low = 0
                    af_index_high = range_length + pivots[len(pivots)-1]
                    if i == hp['filter_data_points']-1:
                        af_index_high = data_df.shape[0]  
                    pivots.append(af_index_high)                         
                    cur_pivot_indices = list(data_df.index[af_index_low:af_index_high])
                    cur_pivot_rest_indices = list(data_df.index[:af_index_low]) + list(data_df.index[af_index_high:]) 
                    pivot_indices[len(pivots)-1] = cur_pivot_indices
                    pivot_rest_indices[len(pivots)-1] = cur_pivot_rest_indices
                    pivot_values_range_start[len(pivots)-1] = data_df.loc[data_df.index[af_index_low],hp['orderby']]
                    pivot_values_range_end[len(pivots)-1] = data_df.loc[data_df.index[af_index_high-1],hp['orderby']]    
    
            if hp['filter_type'] == 4:  #Use for check the best window size for the dataset
                data_sizes = np.linspace(start = 0, stop = data_df.shape[0], num = self.num_sizes + 2,dtype = int)[1:-1]
                for windows_size in data_sizes:             
                    for i in range(hp['filter_data_points']): 
                        pivots.append(np.nan)                       
                        cur_pivot_indices = list(np.random.choice(data_df.index,windows_size,replace = False))
                        cur_pivot_rest_indices = list(set(data_df.index) - set(cur_pivot_indices)) 
                        pivot_indices[len(pivots)-1] = cur_pivot_indices
                        pivot_rest_indices[len(pivots)-1] = cur_pivot_rest_indices
                        pivot_values_range_start[len(pivots)-1] = np.nan
                        pivot_values_range_end[len(pivots)-1] = np.nan   
                pivots.append (np.nan)
                pivot_indices[len(pivots)-1] = list(data_df.index)
                pivot_rest_indices[len(pivots)-1] = []
                pivot_values_range_start[len(pivots)-1] = np.nan
                pivot_values_range_end[len(pivots)-1] = np.nan                                                        
        return ([pivots,pivot_indices,pivot_rest_indices,pivot_values_range_start,pivot_values_range_end])    
       