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
from datetime import datetime

from sklearn import model_selection as ms
import alm_fun
    
class alm_data:

    def __init__(self, data_init_params):
        
        for key in data_init_params:
            setattr(self, key, data_init_params[key])
            
        #parameters that are not open for configuration yet
        self.filter_train = 0
        self.filter_test = 0
        self.filter_target = 0
        self.filter_validation = 0
        self.if_gradient = 0
        self.verbose = 1
        self.dependent_variable = 'label'
        self.independent_testset = 0
        self.validation_from_testset = 0
        self.if_engineer = 0
        self.use_extra_data = 1
        self.extra_data_index = 0
            
        alm_fun.show_msg (self.log,self.verbose,'Class: [alm_data] [__init__] ' + self.name + ' ...... @' + str(datetime.now()))
 
    def refresh_data(self): 

#         self.verbose = verbose
        
        if (self.load_from_disk == 0) | (not os.path.isfile(self.project_path + '/output/npy/' + self.session_id + '_' + self.name + '_savedata.npy')) :                    
            # load data (set initial features, handel onehot features,remove samples without valid dependent variable)
            self.load_data()
            msg = "Data loading......\n" + self.data_msg(split=0)
            alm_fun.show_msg(self.log, self.verbose, msg)                           
            # slice data
            self.preprocess_data()            
            msg = "Data preprocessing......\n" + self.data_msg(split=0)
            alm_fun.show_msg(self.log, self.verbose, msg)  
            # filter data
            self.filter_data()
            msg = "Data filtering......\n" + self.data_msg(split=0)
            alm_fun.show_msg(self.log, self.verbose, msg)  
            
            #split data
            self.split_data()
            msg = "Data spliting.....\n" + self.data_msg()
            alm_fun.show_msg(self.log, self.verbose, msg)           
 
            # gradient reshape
            if self.if_gradient == 1:   
                self.gradient_data()
                msg = "[gradient_data]\n" + self.data_msg()
                alm_fun.show_msg(self.log, self.verbose, msg)  
                        
            # engineer data            
            if self.if_engineer == 1:
                self.engineer_data()
                msg = "[egineer_data]\n" + self.data_msg()
                alm_fun.show_msg(self.log, self.verbose, msg)    
            
            if self.save_to_disk == 1:                    
                self.save_data()
                     
        else:
            self.dict_savedata = np.load(self.project_path + '/output/npy/' + self.session_id + '_' + self.name + '_savedata.npy')
            self.train_data_original_df = self.dict_savedata.get('train_data_original_df',None)
            self.extra_train_data_df_lst = self.dict_savedata['extra_train_data_df_lst']
            self.train_data_df = self.dict_savedata['train_data_df'] 
            self.test_data_df = self.dict_savedata['test_data_df'] 
            self.target_data_df = self.dict_savedata['target_data_df'] 
            
            self.train_data_index_df = self.dict_savedata['train_data_index_df'] 
            self.validation_data_index_df = self.dict_savedata['validation_data_index_df']
            self.test_data_index_df = self.dict_savedata['test_data_index_df'] 
            self.target_data_index_df = self.dict_savedata['target_data_index_df'] 
            
            self.train_data_for_target_df = self.dict_savedata['train_data_for_target_df'] 
            self.target_data_for_target_df = self.dict_savedata['target_data_for_target_df'] 
            self.validation_data_for_target_df = self.dict_savedata['validation_data_for_target_df']                 
            
            self.train_splits_df = self.dict_savedata['train_splits_df'] 
            self.test_splits_df = self.dict_savedata['test_splits_df'] 
            
            self.train_cv_splits_df = self.dict_savedata['train_cv_splits_df'] 
            self.validation_cv_splits_df = self.dict_savedata['validation_cv_splits_df']  
            
            self.train_data_index_df['weight'] = 1
            self.extra_train_data_df_lst[0]['weight'] = 1
            
#             print (str(self.train_data_index_df.loc[self.train_data_index_df['mpc_obs_exp'].notnull(),:].shape))
            
            alm_fun.show_msg (self.log,self.verbose, str(self.train_data_index_df['set_name'].value_counts().sort_index()))
            alm_fun.show_msg (self.log,self.verbose,str(self.extra_train_data_df_lst[0]['set_name'].value_counts().sort_index()))
            
            if self.if_gradient:
                self.gradients = self.dict_savedata['gradients']
            
            if self.if_engineer == 1:
                self.dict_savedata_engineered = np.load(self.project_path + '/output/npy/'  + self.session_id + '_' + self.name + '_savedata_engineered.npy')
                self.train_data_for_target_engineered_df = self.dict_savedata_engineered['train_data_for_target_engineered_df'] 
                self.target_data_for_target_engineered_df = self.dict_savedata_engineered['target_data_for_target_engineered_df'] 
                self.validation_data_for_target_engineered_df  = self.dict_savedata_engineered['validation_data_for_target_engineered_df']   
                
                self.train_splits_engineered_df = self.dict_savedata_engineered['train_splits_engineered_df'] 
                self.test_splits_engineered_df = self.dict_savedata_engineered['test_splits_engineered_df'] 
                
                self.train_cv_splits_engineered_df = self.dict_savedata_engineered['train_cv_splits_engineered_df'] 
                self.validation_cv_splits_engineered_df = self.dict_savedata_engineered['validation_cv_splits_engineered_df'] 
                

    def reload_data(self, ctuoff=np.nan):
        # Read training (extra training) and test data from files
        self.read_data()        
        # refresh data
        self.refresh_data()
  
    def convert_old_data(self,data_df):
        
        data_df['set_name'] = np.nan
        data_df.loc[(data_df['train_clinvar_source'] == 1) & (data_df['label'] == 0) & (data_df['gnomAD_exomes_AF'] > 1e-06) & (data_df['gnomAD_exomes_AF'] <0.005),'set_name'] = 'extra_clinvar_0_low'
        data_df.loc[(data_df['train_clinvar_source'] == 1) & (data_df['label'] == 0) & (data_df['gnomAD_exomes_AF'] >=0.005),'set_name'] = 'extra_clinvar_0_high'                
        data_df.loc[(data_df['train_clinvar_source'] == 1) & (data_df['label'] == 1) & (data_df['gnomAD_exomes_AF'] > 1e-06),'set_name']= 'extra_clinvar_1'
                
        data_df.loc[(data_df['train_humsavar_source'] == 1) & (data_df['label'] == 0) & (data_df['gnomAD_exomes_AF'] <0.005),'set_name'] = 'extra_humsavar_0_low'
        data_df.loc[(data_df['train_humsavar_source'] == 1) & (data_df['label'] == 0) & (data_df['gnomAD_exomes_AF'] >=0.005),'set_name'] = 'extra_humsavar_0_high'                
        data_df.loc[(data_df['train_humsavar_source'] == 1) & (data_df['label'] == 1),'set_name']= 'extra_humsavar_1'
        
        data_df.loc[(data_df['train_gnomad_source'] == 1) & (data_df['label'] == 0) & (data_df['gnomAD_exomes_AF'] <0.005),'set_name'] = 'extra_gnomad_low'
        data_df.loc[(data_df['train_gnomad_source'] == 1) & (data_df['label'] == 0) & (data_df['gnomAD_exomes_AF'] >=0.005),'set_name'] = 'extra_gnomad_high'     
        data_df.loc[(data_df['train_hgmd_source'] == 1) & (data_df['label'] == 1),'set_name']= 'extra_hgmd'
        
        data_df.loc[(data_df['train_mave_source'] == 1) & (data_df['label'] == 0),'set_name']= 'extra_mave_0'     
        data_df.loc[(data_df['train_mave_source'] == 1) & (data_df['label'] == 1),'set_name']= 'extra_mave_1'        

        return(data_df)
    
    def add_new_predictor_scores(self,data_df):
#         data_df['merge_flag'] = 1
#         x = data_df.groupby(['chr','nt_pos','nt_ref','nt_alt'])['aa_pos'].agg('min').reset_index()
#         print (str(data_df.shape[0]) + '-' + str(x.shape[0]) + '=' + str(data_df.shape[0]-x.shape[0]))
#         self.train_data_df.loc[(self.train_data_df['nt_pos'] == 48935717) & (self.train_data_df['nt_ref'] == 'C') & (self.train_data_df['nt_alt'] == 'G'),['p_vid','aa_pos','aa_ref','aa_alt']]            
#         mistic_df = pd.read_csv(self.db_path + '/mistic/org/MISTIC_GRCh37.tsv', sep = '\t')
#         mistic_df.columns = ['chr','nt_pos','nt_ref','nt_alt','mistic_score','mistic_pred']        
#         new_mistic_df = mistic_df.groupby(['chr','nt_pos','nt_ref','nt_alt'])['mistic_score'].agg('min').reset_index()
#         print(new_mistic_df.loc[(new_mistic_df['nt_pos'] == 150659434) & (new_mistic_df['nt_ref'] == 'G') & (new_mistic_df['nt_alt'] == 'T'),['chr','nt_pos','nt_ref','nt_alt','mistic_score','mistic_pred']])
#         y = data_merge_df.groupby(['chr','nt_pos','nt_ref','nt_alt'])['aa_pos'].count().reset_index()
#         y.loc[y['aa_pos']>1,:]
#         data_merge_df.loc[(data_merge_df['nt_pos'] == 150659434) & (data_merge_df['nt_ref'] == 'G') & (data_merge_df['nt_alt'] == 'T'),['p_vid','aa_pos','aa_ref','aa_alt','mistic_score','mistic_pred']]

        # add mistic and mpc        
#       
        mpc_df = pd.read_csv(self.db_path + '/mpc/all/mpc_values_v2_avg_duplicated_scores.csv') 
        data_merge_df = pd.merge(data_df,mpc_df,how = 'left')
        mistic_df = pd.read_csv(self.db_path + '/mistic/all/MISTIC_GRCh37_avg_duplicated_scores.csv')                            
        data_merge_df = pd.merge(data_merge_df,mistic_df,how = 'left')
        #         print (str(data_merge_df.shape[0]) + '-' + str(data_df.shape[0]) + '=' + str(data_merge_df.shape[0]-data_df.shape[0]))                                
        print (str(data_merge_df.shape[0]) + '-' + str(data_df.shape[0]) + '=' + str(data_merge_df.shape[0]-data_df.shape[0]))                 
        return(data_merge_df)
        
    def save_data(self):
        
        self.dict_savedata = {}
        self.dict_savedata['train_data_original_df'] = self.train_data_original_df
        self.dict_savedata['extra_train_data_df_lst'] = self.extra_train_data_df_lst
        self.dict_savedata['train_data_df'] = self.train_data_df
        self.dict_savedata['test_data_df'] = self.test_data_df
        self.dict_savedata['target_data_df'] = self.target_data_df
        
        self.dict_savedata['train_data_index_df'] = self.train_data_index_df
        self.dict_savedata['validation_data_index_df'] = self.validation_data_index_df                
        self.dict_savedata['test_data_index_df'] = self.test_data_index_df
        self.dict_savedata['target_data_index_df'] = self.target_data_index_df
        
        self.dict_savedata['train_data_for_target_df'] = self.train_data_for_target_df
        self.dict_savedata['target_data_for_target_df'] = self.target_data_for_target_df
        self.dict_savedata['validation_data_for_target_df'] = self.validation_data_for_target_df                
        
        self.dict_savedata['train_splits_df'] = self.train_splits_df
        self.dict_savedata['test_splits_df'] = self.test_splits_df
        
        self.dict_savedata['train_cv_splits_df'] = self.train_cv_splits_df
        self.dict_savedata['validation_cv_splits_df'] = self.validation_cv_splits_df
        
        if self.if_gradient:
            self.dict_savedata['gradients'] = self.gradients
        
        pickle_out = open(self.project_path + '/output/npy/' + self.session_id + '_' +self.name + '_savedata.npy', 'wb')
        pickle.dump(self.dict_savedata, pickle_out) 
        pickle_out.close()        
                
        if self.if_engineer:
            self.dict_savedata_engineered = {}
            self.dict_savedata_engineered['train_data_for_target_engineered_df'] = self.train_data_for_target_engineered_df
            self.dict_savedata_engineered['target_data_for_target_engineered_df'] = self.target_data_for_target_engineered_df
            self.dict_savedata_engineered['validation_data_for_target_engineered_df'] = self.validation_data_for_target_engineered_df    
            
            self.dict_savedata_engineered['train_splits_engineered_df'] = self.train_splits_engineered_df
            self.dict_savedata_engineered['test_splits_engineered_df'] = self.test_splits_engineered_df
            
            self.dict_savedata_engineered['train_cv_splits_engineered_df'] = self.train_cv_splits_engineered_df
            self.dict_savedata_engineered['validation_cv_splits_engineered_df'] = self.validation_cv_splits_engineered_df
            
            if self.if_gradient:
                self.dict_savedata_engineered['gradients'] = self.gradients                
            
            pickle_out = open(self.project_path + '/output/npy/'  + self.session_id + '_' + self.name + '_savedata_engineered.npy','wb')
            pickle.dump(self.dict_savedata_engineered, pickle_out) 
            pickle_out.close()     
            
    def read_data(self): 
        self.train_data_original_df = pd.read_csv(self.project_path + self.train_file)
        self.test_data_original_df = pd.read_csv(self.project_path + self.test_file) 
        self.target_data_original_df = pd.read_csv(self.project_path + self.target_file)
        if self.use_extra_data != 0:
            self.extra_train_data_original_df = pd.read_csv(self.project_path + self.extra_train_file) 
        else:
            self.extra_train_data_original_df = None
                             
    def load_data(self):                  
        # loading original training data , add random feature , remove the ones without label   
        self.train_data_working_df = self.train_data_original_df.copy()        
        self.train_data_working_df['random_feature'] = np.random.uniform(0, 1, self.train_data_working_df.shape[0])
        self.train_data_working_df = self.train_data_working_df.loc[self.train_data_working_df[self.dependent_variable].notnull(), :]
        
        # loading original test data, if there is no dependent variable than add one, otherwise remove the records without label
        self.test_data_working_df = self.test_data_original_df.copy()
        if self.test_data_original_df.shape[0] != 0 :
            self.test_data_working_df['random_feature'] = np.random.uniform(0, 1, self.test_data_working_df.shape[0])
            self.test_data_working_df = self.test_data_working_df.loc[self.test_data_working_df[self.dependent_variable].notnull(), :]

        # loading original target data
        self.target_data_working_df = self.target_data_original_df.copy()
        if self.target_data_original_df.shape[0] != 0 :
            self.target_data_working_df['random_feature'] = np.random.uniform(0, 1, self.target_data_working_df.shape[0])
#             self.target_data_working_df = self.target_data_working_df.loc[self.target_data_working_df[self.dependent_variable].notnull(), :]
        
        self.extra_train_data_working_df_lst = [x for x in self.extra_train_data_original_df_lst]
        for i in range(len(self.extra_train_data_working_df_lst)):     
            if self.extra_train_data_working_df_lst[i].shape[0] != 0:
                self.extra_train_data_working_df_lst[i]['random_feature'] = np.random.uniform(0, 1, self.extra_train_data_working_df_lst[i].shape[0])
                self.extra_train_data_working_df_lst[i] = self.extra_train_data_working_df_lst[i].loc[self.extra_train_data_working_df_lst[i][self.dependent_variable].notnull(), :]


        self.train_data_df = self.train_data_working_df.copy()
        self.test_data_df = self.test_data_working_df.copy()  
        self.target_data_df = self.target_data_working_df.copy()
        self.extra_train_data_df_lst = [x for x in self.extra_train_data_working_df_lst]
                       
        self.n_features = self.train_data_df.shape[1] - 1
        self.feature_names = self.train_data_df.columns.get_values()
        
        self.train_counts = self.train_data_df.shape[0]
        self.test_counts = self.test_data_df.shape[0]
        self.target_counts = self.target_data_df.shape[0]
        
        self.train_data_for_target_df = None
        self.train_cv_splits_df = None
        self.validation_cv_splits_df = None
          
    def preprocess_data(self):
        [self.target_data_df, self.train_data_df, self.test_data_df, self.extra_train_data_df_lst] = self.data_preprocess(self.name, self.target_data_df, self.train_data_df, self.test_data_df, self.extra_train_data_df_lst)        
        
    def engineer_data(self):              

        self.train_data_for_target_engineered_df = copy.deepcopy(self.train_data_for_target_df)
        self.target_data_for_target_engineered_df = copy.deepcopy(self.target_data_for_target_df)
        if (self.independent_testset == 1) & (self.validation_from_testset == 1):
            self.validation_data_for_target_engineered_df = copy.deepcopy(self.validation_data_for_target_df)
        else:
            self.validation_data_for_target_engineered_df = None
        
        self.train_splits_engineered_df = copy.deepcopy(self.train_splits_df)
        self.test_splits_engineered_df = copy.deepcopy(self.test_splits_df)
        
        self.train_cv_splits_engineered_df = copy.deepcopy(self.train_cv_splits_df)
        self.validation_cv_splits_engineered_df = copy.deepcopy(self.validation_cv_splits_df)
                   
        # take care of the train_data_for_target_df, target_data_for_target_df, validation_data_for_target_df
        for key in self.train_data_for_target_df.keys():
            [self.train_data_for_target_engineered_df[key],self.target_data_for_target_engineered_df[key]] = \
            self.feature_engineer(self.train_data_index_df.loc[self.train_data_for_target_df[key],:],self.target_data_index_df.loc[self.target_data_for_target_df[key],:])
        
        if (self.independent_testset == 1) & (self.validation_from_testset == 1):
            for key in self.train_data_for_target_df.keys():
                [self.train_data_for_target_engineered_df[key],self.validation_data_for_target_engineered_df[key]] = \
                self.feature_engineer(self.train_data_index_df.loc[self.train_data_for_target_df[key],:],self.validation_data_index_df.loc[self.validation_data_for_target_df[key],:])
            
        # take care of train_splits_df and test_splits_df
        for i in range(self.test_split_folds):
            for key in self.train_splits_df[i].keys():                    
                print('i:' + str(i) + '-' + key)
                [self.train_splits_engineered_df[i][key],self.test_splits_engineered_df[i][key]] = \
                self.feature_engineer(self.train_data_index_df.loc[self.train_splits_df[i][key],:],self.test_data_index_df.loc[self.test_splits_df[i][key],:])
                
        #take care of train_cv_splits_df and validation_cv_splites_df
        for i in range(self.test_split_folds):
            for j in range(self.cv_split_folds):
                for key in self.train_cv_splits_engineered_df[i][j].keys():
                    [self.train_cv_splits_engineered_df[i][j][key],self.validation_cv_splits_engineered_df[i][j][key]] = \
                    self.feature_engineer(self.train_data_index_df.loc[self.train_cv_splits_df[i][j][key],:],self.validation_data_index_df.loc[self.validation_cv_splits_df[i][j][key],:])

    def filter_data(self):
        if self.filter_train == 1:   
            train_null_idx = self.train_data_df[self.train_features].isnull().any(axis=1)
            train_null_count = train_null_idx[train_null_idx].shape[0]
            self.train_data_df = self.train_data_df[-train_null_idx]
            
#             if self.train_data_for_target_df is not None:
#                 train_null_idx = self.train_data_for_target_df.isnull().any(axis=1)
#                 train_null_count = train_null_idx[train_null_idx].shape[0]
#                 self.train_data_for_target_df = self.train_data_for_target_df[-train_null_idx]
#             
            if self.use_extra_data != 0:
                for i in range(len(self.extra_train_data_df_lst)):
                    extra_train_null_idx = self.extra_train_data_df_lst[i][self.train_features].isnull().any(axis=1)
                    extra_train_null_count = extra_train_null_idx[extra_train_null_idx].shape[0]
                    self.extra_train_data_df_lst[i] = self.extra_train_data_df_lst[i][-extra_train_null_idx]
                  
        if self.filter_test == 1:     
            test_null_idx = self.test_data_df[self.train_features].isnull().any(axis=1)
            test_null_count = test_null_idx[test_null_idx].shape[0]
            self.test_data_df = self.test_data_df[-test_null_idx]    
            
        if self.filter_target == 1:     
            target_null_idx = self.target_data_df[self.train_features].isnull().any(axis=1)
            target_null_count = target_null_idx[target_null_idx].shape[0]
            self.target_data_df = self.target_data_df[-target_null_idx]    
    
#         for fold_id in range(self.cv_split_folds):
#             if self.filter_train == 1:  
#                 train_null_idx = self.train_cv_splits_df[fold_id].isnull().any(axis=1)
#                 train_null_count = train_null_idx[train_null_idx].shape[0]
#                 self.train_cv_splits_df[fold_id] = self.train_cv_splits_df[fold_id][-train_null_idx]
#             if self.filter_validation == 1:
#                 validation_null_idx = self.validation_cv_splits_df[fold_id].isnull().any(axis=1)
#                 validation_null_count = validation_null_idx[validation_null_idx].shape[0]
#                 self.validation_cv_splits_df[fold_id] = self.validation_cv_splits_df[fold_id][-validation_null_idx]

    def gradient_data(self):             
        #first to get the graidents 
        self.gradients = self.setup_gradients(self.train_data_df)
                   
        #take care of the train_data_for_target_df, target_data_for_target_df, validation_data_for_target_df
        train_dict = self.gradient_reshape(self.train_data_index_df.loc[self.train_data_for_target_df['no_gradient'],:],self.gradients)
        self.train_data_for_target_df.update(train_dict)
        target_dict = {x:self.target_data_for_target_df['no_gradient'] for x in self.gradients}
        self.target_data_for_target_df.update(target_dict)
         
        if (self.independent_testset == 1) & (self.validation_from_testset == 1):
            validation_dict = {x:self.validation_data_for_target_df['no_gradient'] for x in self.gradients}
            self.validation_data_for_target_df.update(validation_dict)
        
        #take care of train_splits_df and test_splits_df
        for i in range(self.test_split_folds):
            train_splits_dict = self.gradient_reshape(self.train_data_index_df.loc[self.train_splits_df[i]['no_gradient'],:],self.gradients)
            self.train_splits_df[i].update(train_splits_dict)
            test_dict = {x:self.test_splits_df[i]['no_gradient'] for x in self.gradients}
            self.test_splits_df[i].update(test_dict)
            
        #take care of train_cv_splits_df and validation_cv_splites_df
        for i in range(self.test_split_folds):
            for j in range(self.cv_split_folds):
                train_cv_splits_dict = self.gradient_reshape(self.train_data_index_df.loc[self.train_cv_splits_df[i][j]['no_gradient'],:],self.gradients)
                self.train_cv_splits_df[i][j].update(train_cv_splits_dict)
                validation_dict = {x:self.validation_cv_splits_df[i][j]['no_gradient'] for x in self.gradients}
                self.validation_cv_splits_df[i][j].update(validation_dict)
                                
    def split_data(self):
        self.train_data_for_target_df = {}        
        self.train_data_for_target_df['no_gradient'] = self.train_data_df.index        
        self.target_data_for_target_df = {}        
        self.target_data_for_target_df['no_gradient'] = self.target_data_df.index
        
        self.target_data_index_df = self.target_data_df
        self.train_data_index_df = self.train_data_df
        
        if self.independent_testset == 1:
            self.test_data_index_df = self.test_data_df
        else:
            self.test_data_index_df = self.train_data_df
            
        if self.validation_from_testset == 1:
            self.validation_data_index_df = self.test_data_df
        else:
            self.validation_data_index_df = self.train_data_df                
        
        if (self.independent_testset == 1) & (self.validation_from_testset == 1):            
            self.validation_data_for_target_df = {}        
            self.validation_data_for_target_df['no_gradient'] = self.test_data_df.index
        else:
            self.validation_data_for_target_df = None
            
        #split test, validation, training dataset
        #case1: all test, validation, training from one original training set 
        if self.independent_testset == 0:
            #Split the original training set into test_split_folds folds.  (training - test)
            #output two list train_splits_df and test_splits_df
            self.train_splits_df = [{} for i in range(self.test_split_folds)]      
            self.test_splits_df = [{} for i in range(self.test_split_folds)]
            self.train_cv_splits_df = [[{} for i in range(self.cv_split_folds)] for j in range(self.test_split_folds)]
            self.validation_cv_splits_df = [[{} for i in range(self.cv_split_folds)] for j in range(self.test_split_folds)]
             
            if self.test_split_method == 0 :   
                if self.test_split_folds == 1:
                    kf_list = []
                    if self.test_split_ratio == 0: 
                        kf_list.append((range(len(self.train_data_df.index)),[]))
                    else:
                        kf_folds = int(1/self.test_split_ratio)
                        kf = ms.KFold(n_splits=kf_folds, shuffle=True) 
                        kf_list.append(list(kf.split(self.train_data_df))[0])
                else:
                    kf = ms.KFold(n_splits=self.test_split_folds, shuffle=True) 
                    kf_list = list(kf.split(self.train_data_df))  
            # stratified split (keep prior)
            if self.test_split_method == 1 :   
                if self.test_split_folds == 1:
                    kf_list = []                    
                    if self.test_split_ratio == 0: 
                        kf_list.append((range(len(self.train_data_df.index)),[]))
                    else:
                        kf_folds = int(1/self.test_split_ratio)
                        kf = ms.StratifiedKFold(n_splits=kf_folds, shuffle=True) 
                        kf_list.append(list(kf.split(self.train_data_df,self.train_data_df[self.dependent_variable]))[0])
                else:                
                    kf = ms.StratifiedKFold(n_splits=self.test_split_folds, shuffle=True)   
                    kf_list = list(kf.split(self.train_data_df,self.train_data_df[self.dependent_variable]))
            # customized split   
            if self.test_split_method == 2 :
                kf_list = self.test_split(self.name,self.train_data_df,self.sametest_as_data_name) 
                 
            test_split_fold_id = 0 
            for train_index_split, test_index_split in kf_list:
                train_index = self.train_data_df.index[train_index_split]
                self.train_splits_df[test_split_fold_id]['no_gradient'] = train_index
                if len(test_index_split) ==  0:
                    test_index = []
                else:
                    test_index = self.train_data_df.index[test_index_split] 
                self.test_splits_df[test_split_fold_id]['no_gradient'] = test_index                                    
                test_split_fold_id += 1 
                
            #Split each training set into cv_split_folds folds  (training - validation)
            for i in range(self.test_split_folds):
                cur_train_data_df = self.train_data_df.loc[self.train_splits_df[i]['no_gradient'],:]
                if self.cv_split_method == 0 : 
                    if self.cv_split_folds == 1:
                        kf_folds = int(1/self.cv_split_ratio)
                        kf = ms.KFold(n_splits=kf_folds, shuffle=True) 
                        kf_list = []
                        kf_list.append(list(kf.split(cur_train_data_df))[0])
                    else:  
                        kf = ms.KFold(n_splits=self.cv_split_folds, shuffle=True) 
                        kf_list = list(kf.split(cur_train_data_df))
                # stratified split (keep prior)
                if self.cv_split_method == 1 : 
                    if self.cv_split_folds == 1:
                        kf_folds = int(1/self.cv_split_ratio)
                        kf = ms.StratifiedKFold(n_splits=kf_folds, shuffle=True) 
                        kf_list = []
                        kf_list.append(list(kf.split(cur_train_data_df,cur_train_data_df[self.dependent_variable]))[0])
                    else:    
                        kf = ms.StratifiedKFold(n_splits=self.cv_split_folds, shuffle=True)   
                        kf_list = list(kf.split(cur_train_data_df,cur_train_data_df[self.dependent_variable]))
                # customized split   
                if self.cv_split_method == 2 :
                    kf_list = self.cv_split(self.name,cur_train_data_df)  
                    
                cv_split_fold_id = 0 
                for train_index_split, validation_index_split in kf_list:
                    train_index = cur_train_data_df.index[train_index_split]
                    validation_index = cur_train_data_df.index[validation_index_split]
                    self.train_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = train_index
                    self.validation_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = validation_index                   
#                     self.train_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = self.train_splits_df[i].loc[train_index, :]
#                     self.validation_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = self.train_splits_df[i].loc[validation_index, :] 
                    cv_split_fold_id += 1 
                    
        
        #case2: training from one set, validation and test from another set (independent_testset and validation_from_testset parameters) 
        if self.independent_testset == 1:                                       
            self.train_splits_df = [{} for i in range(self.test_split_folds)]      
            self.test_splits_df = [{} for i in range(self.test_split_folds)]
            self.train_cv_splits_df = [[{} for i in range(self.cv_split_folds)] for j in range(self.test_split_folds)]
            self.validation_cv_splits_df = [[{} for i in range(self.cv_split_folds)] for j in range(self.test_split_folds)]
            
            #split testset to test_split_folds folds (validation - test)
            if self.validation_from_testset:                 
                if self.validation_equal_testset:
                    self.test_split_folds = 1 #special case that validation set is the same as test set
                    kf_list = [(range(self.test_data_df.shape[0]),range(self.test_data_df.shape[0]))]
                else:                
                    if self.test_split_method == 0 :  
                        if self.test_split_folds == 1:
                            kf_folds = int(1/self.test_split_ratio)
                            kf = ms.KFold(n_splits=kf_folds, shuffle=True) 
                            kf_list = []
                            kf_list.append(list(kf.split(self.test_data_df))[0])
                        else:                                         
                            kf = ms.KFold(n_splits=self.test_split_folds, shuffle=True) 
                            kf_list = list(kf.split(self.test_data_df))  
                    # stratified split (keep prior)
                    if self.test_split_method == 1 : 
                        if self.test_split_folds == 1:
                            kf_folds = int(1/self.test_split_ratio)
                            kf = ms.StratifiedKFold(n_splits=kf_folds, shuffle=True) 
                            kf_list = []
                            kf_list.append(list(kf.split(self.test_data_df,self.test_data_df[self.dependent_variable]))[0])
                        else:   
                            kf = ms.StratifiedKFold(n_splits=self.test_split_folds, shuffle=True)   
                            kf_list = list(kf.split(self.test_data_df,self.test_data_df[self.dependent_variable]))
                    # customized split   
                    if self.test_split_method == 2 :
                        kf_list = self.test_split(self.test_data_df.copy())

                test_split_fold_id = 0 
                for validation_index_split , test_index_split in kf_list:
                    validation_index = self.test_data_df.index[validation_index_split]
                    test_index = self.test_data_df.index[test_index_split]  
                    self.train_splits_df[test_split_fold_id]['no_gradient'] = self.train_data_df.index
                    self.test_splits_df[test_split_fold_id]['no_gradient'] = test_index
                    
                    cv_validation_index = np.array_split(validation_index,self.cv_split_folds)
                    
                    for j in range(self.cv_split_folds):

                        self.train_cv_splits_df[test_split_fold_id][j]['no_gradient'] = self.train_data_df.index
                        self.validation_cv_splits_df[test_split_fold_id][j]['no_gradient'] = cv_validation_index[j]                                                          
    #                     self.train_splits_df[test_split_fold_id]['no_gradient'] = self.train_data_df
    #                     self.test_splits_df[test_split_fold_id]['no_gradient'] = self.test_data_df.loc[test_index, :]
    #                     self.train_cv_splits_df[test_split_fold_id][0]['no_gradient'] = self.train_data_df
    #                     self.validation_cv_splits_df[test_split_fold_id][0]['no_gradient'] = self.test_data_df.loc[validation_index, :] 
                    test_split_fold_id += 1 
                print('done')
            else:
                self.train_splits_df[0]['no_gradient'] = self.train_data_df.index
                self.test_splits_df[0]['no_gradient'] = self.test_data_df.index
                
                #Split each training set into cv_split_folds folds  (training - validation)
                for i in range(self.test_split_folds):
                    cur_train_data_df = self.train_data_df.loc[self.train_splits_df[i]['no_gradient'],:]
                    if self.cv_split_method == 0 : 
                        if self.cv_split_folds == 1:
                            kf_folds = int(1/self.cv_split_ratio)
                            kf = ms.KFold(n_splits=kf_folds, shuffle=True) 
                            kf_list = []
                            kf_list.append(list(kf.split(cur_train_data_df))[0])
                        else:  
                            kf = ms.KFold(n_splits=self.cv_split_folds, shuffle=True) 
                            kf_list = list(kf.split(cur_train_data_df))
                    # stratified split (keep prior)
                    if self.cv_split_method == 1 : 
                        if self.cv_split_folds == 1:
                            kf_folds = int(1/self.cv_split_ratio)
                            kf = ms.StratifiedKFold(n_splits=kf_folds, shuffle=True) 
                            kf_list = []
                            kf_list.append(list(kf.split(cur_train_data_df,cur_train_data_df[self.dependent_variable]))[0])
                        else:    
                            kf = ms.StratifiedKFold(n_splits=self.cv_split_folds, shuffle=True)   
                            kf_list = list(kf.split(cur_train_data_df,cur_train_data_df[self.dependent_variable]))
                    # customized split   
                    if self.cv_split_method == 2 :
                        kf_list = self.cv_split(self.name,cur_train_data_df)  
                        
                    cv_split_fold_id = 0 
                    for train_index_split, validation_index_split in kf_list:
                        train_index = cur_train_data_df.index[train_index_split]
                        validation_index = cur_train_data_df.index[validation_index_split]
                        self.train_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = train_index
                        self.validation_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = validation_index                     
#                         self.train_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = self.train_splits_df[i].loc[train_index, :]
#                         self.validation_cv_splits_df[i][cv_split_fold_id]['no_gradient'] = self.train_splits_df[i].loc[validation_index, :] 
                        cv_split_fold_id += 1 
  
    def data_msg (self,split = 1):
        msg = 'Core training data shape: ' + ' [' + str(self.train_data_df.shape[0]) + ',' + str(self.train_data_df.shape[1]) + ']\n'
        
#               'test_data' + ' [' + str(self.test_data_df.shape[0]) + ',' + str(self.test_data_df.shape[1] - 1) + ']\n' + \
#               'target_data' + ' [' + str(self.target_data_df.shape[0]) + ',' + str(self.target_data_df.shape[1] - 1) + ']\n' 
        if self.use_extra_data != 0 :
            for i in range(len(self.extra_train_data_df_lst)):
#                 msg += 'extra_train_data_lst' + '[' + str(i) + '] [' + str(self.extra_train_data_df_lst[i].shape[0]) + ',' + str(self.extra_train_data_df_lst[i].shape[1] - 1) + ']\n'
                msg += 'Add-on training data shape:'  + ' [' + str(self.extra_train_data_df_lst[i].shape[0]) + ',' + str(self.extra_train_data_df_lst[i].shape[1]) + ']\n'
        
#         if self.train_data_for_target_df is not None:
#             msg += 'train_data_for_target' + ' [' + str(self.train_data_for_target_df['no_gradient'].shape[0]) + ',' + str(self.train_data_index_df.shape[1]) + ']\n'

        if split == 1:            
            for outer_fold_id in range(self.test_split_folds):
                msg += '**outer loop fold ' + str(outer_fold_id) + ': [train: ' + str(len(self.train_splits_df[outer_fold_id]['no_gradient'])) + ', test: ' + str(len(self.test_splits_df[outer_fold_id]['no_gradient'])) + ']\n'          
                for inner_fold_id in range(self.cv_split_folds):
                    msg += 'inner loop fold ' + str(inner_fold_id) + ': [train: ' + str(len(self.train_cv_splits_df[outer_fold_id][inner_fold_id]['no_gradient'] )) + ', validation: ' + str(len(self.validation_cv_splits_df[outer_fold_id][inner_fold_id]['no_gradient'])) + ']\n'        
        return (msg)     


 