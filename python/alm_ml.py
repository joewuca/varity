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
import alm_fun
import warnings
import string
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.patches import Rectangle
import matplotlib.path as mpath
import matplotlib.patches as patches  
from matplotlib.lines import Line2D  
from matplotlib.gridspec import GridSpec
import matplotlib.collections as collections
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 1000)

class alm_ml:
    def __init__(self, ml_init_params):
        for key in ml_init_params:
            setattr(self, key, ml_init_params[key])
        msg = "Class: [alm_ml] [__init__]......done @" + str(datetime.now())
        
        #parameters that are not open for configuration yet
        self.verbose = 0
        self.run_grid_search = 0
        fs_start_feature = None
        fs_type = 'local search'
        fs_T = 0.001
        fs_alpha = 0.8
        fs_K = 100
        fs_epsilon = 0.00001
        
        alm_fun.show_msg(self.log, self.verbose, msg)    

    def prefix(self,predictor,runtime, folder, cv_flag = 0):        
        alm_predictor = self.proj.predictor[predictor]
        predictor = alm_predictor.name
        data_name =  alm_predictor.data_instance.name        
        session_id = alm_predictor.session_id
        tune_obj = alm_predictor.tune_obj     
        if (cv_flag == 1) | (runtime['cur_test_fold'] == -1):                   
#             prefix = self.project_path +'/output/' + folder + '/' + session_id + '_' + predictor +  '_' +  data_name + '_' + tune_obj 
            prefix = self.project_path +'/output/' + folder + '/' + session_id + '_' + predictor 
        else:
#             prefix = self.project_path +'/output/' + folder + '/' + session_id + '_' + predictor +  '_' +  data_name + '_' + tune_obj +   '_tf' + str(runtime['cur_test_fold']) 
            prefix = self.project_path +'/output/' + folder + '/' + session_id + '_' + predictor + '_tf' + str(runtime['cur_test_fold'])
        return(prefix)    

    def weights_opt_hyperopt(self,predictor,runtime):        
        
        alm_predictor = self.proj.predictor[predictor]
        self.hyperopt_predictor = alm_predictor
        self.hyperopt_runtime = runtime
                        
        alm_fun.show_msg (self.log,1,"***************************************************************")
        alm_fun.show_msg (self.log,1,"Hyperopt")
        alm_fun.show_msg (self.log,1,"Predictor: " + alm_predictor.name)
        alm_fun.show_msg (self.log,1,"Fold: " + str(runtime['cur_test_fold']))
        alm_fun.show_msg (self.log,1,"Tune obj: " + alm_predictor.tune_obj)
        alm_fun.show_msg (self.log,1,"Session id: " + runtime['session_id'])
        alm_fun.show_msg (self.log,1,"Data name: " + alm_predictor.data)
        alm_fun.show_msg (self.log,1,"Start Time: " + str(datetime.now()))        
        alm_fun.show_msg (self.log,1,"***************************************************************")

        #********************************************************************************************
        # Define hyperopt search space
        #********************************************************************************************
        hyperopt_hps = alm_predictor.hyperopt_hps        
        available_trials_result_file = self.prefix(predictor,runtime,'npy') + '_trials.pkl'
        if  os.path.isfile(available_trials_result_file):                        
            [cur_trials_result,cur_trials_df,X] = self.get_trials(available_trials_result_file)
            alm_fun.show_msg (self.log,1,"Previous trials have been loaded.")            
            if cur_trials_df.shape[0] == 0:
                cur_trials_result = hyperopt.Trials()
                new_max_evals = alm_predictor.hyperopt_trials
            else:
                cur_max_trials_num = cur_trials_df['trial'].max() 
                cur_max_tid_num =   len(cur_trials_result)
                new_max_evals = alm_predictor.hyperopt_trials - cur_max_trials_num + cur_max_tid_num 
                alm_fun.show_msg (self.log,1,"total effective trials: " + str(cur_max_trials_num)   + " total ran trials : " + str(cur_max_tid_num) + ' new max evals: '  + str(new_max_evals))
                 
        else:
            cur_trials_result = hyperopt.Trials()
            new_max_evals = alm_predictor.hyperopt_trials
            
        self.cur_trials_result = cur_trials_result
        best_hyperopt = hyperopt.fmin(self.fun_validation_cv_prediction_hyperopt,hyperopt_hps,algo = hyperopt.tpe.suggest,show_progressbar= False,max_evals = new_max_evals,trials = self.cur_trials_result)
        pickle.dump(cur_trials_result, open(trial_file, "wb"))
        
        self.save_best_hp_dict_from_trials(predictor,runtime)
                   
        alm_fun.show_msg (self.log,1,"End Time: " + str(datetime.now()))        
        alm_fun.show_msg (self.log,1,"***************************************************************")
    
    def weights_opt_sp(self,predictor,runtime):
        alm_predictor = self.proj.predictor[predictor]
        filtering_hp_values = alm_predictor.hp_mv_values[runtime['filtering_hp']]

        for filtering_hp_value in filtering_hp_values:
            new_runtime = runtime.copy()
            new_runtime['filtering_hp_value'] = filtering_hp_value
            alm_fun.show_msg (self.log,self.verbose,'Running moving analysis on ' + runtime['filtering_hp'] + ' at moving window ' + str(filtering_hp_value) + '......')            
            self.fun_validation_cv_prediction_sp(predictor, new_runtime)
        
    def fun_test_cv_prediction(self,predictor,runtime):        
        alm_predictor = self.proj.predictor[predictor]
        test_split_folds = alm_predictor.data_instance.test_split_folds
        #**********************************************************************
        # Fire parallel jobs for all test folds
        #**********************************************************************
        cur_jobs= {}                   
        if runtime['batch_id'] =='':                           
            runtime['batch_id'] = alm_fun.get_random_id(10)
        alm_fun.show_msg (self.log,1,'Start to run test cv prediction, batch id: ' + runtime['batch_id'] + '......' )        
        for cur_test_fold in range(test_split_folds): 
            new_runtime = runtime.copy()                          
            new_runtime['cur_test_fold']  = cur_test_fold
            new_runtime['single_fold_type']  = 'test'
            new_runtime['hp_tune_type'] = 'hyperopt'      
            new_runtime['run_on_node'] = 1    
            new_runtime['action'] = 'single_fold_prediction'                         
            new_runtime['job_name'] = alm_fun.get_random_id(8)                
            new_runtime['cur_fold_result'] = self.prefix(predictor, new_runtime, 'npy_temp')+  '_' + new_runtime['batch_id'] + '_hp_test_single_fold_result.npy'                
            new_runtime['hp_dict_file'] = self.prefix(predictor, new_runtime, 'npy') + '_hp_dict.npy'
            if not os.path.isfile(new_runtime['cur_fold_result']):
                alm_fun.show_msg (self.log,1, 'Run prediction on test fold '  + str(cur_test_fold) + '......')                                                 
                if (runtime['cluster'] == 1) :                                      
                    [job_id,job_name] = self.varity_obj.varity_action(new_runtime)
                    cur_jobs[job_name] = []
                    cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                    cur_jobs[job_name].append(job_id)
                    cur_jobs[job_name].append(new_runtime)                     
                else:                             
                    self.fun_single_fold_prediction(predictor,new_runtime)       
                    
        if runtime['cluster'] == 1:
            batch_log = self.prefix(predictor,new_runtime,'log') + '_' + runtime['batch_id'] + '.log'
            if self.varity_obj.fun_monitor_jobs(cur_jobs,batch_log,runtime) == 1:
                alm_fun.show_msg (self.log,1, 'Batch: '  +  runtime['batch_id'] + ' all results are done,start to gathering results......')                              

        #**********************************************************************
        # Collect  the results of all parallel jobs and retrieve the best value
        #**********************************************************************                          
        train_y_predicted_dict = {}
        train_y_truth_dict = {}       
        test_y_predicted_dict = {}        
        test_y_truth_dict = {}
        test_y_index_dict = {}
        if alm_predictor.type == 1:
            hp_dict_df = pd.DataFrame(columns = alm_predictor.hp_default.keys())
            
        for cur_test_fold in range(test_split_folds):                   
            runtime['cur_test_fold'] = cur_test_fold
            cur_test_result = self.prefix(predictor, runtime, 'npy_temp')+  '_' + new_runtime['batch_id'] + '_hp_test_single_fold_result.npy'
            r = np.load(cur_test_result).item()
 
            train_y_fold_predicted = r['train_y_predicted']
            train_y_fold_truth = r['train_y_truth']
            train_score_df = r['train_score_df']
            train_score_df['tuned_tree_num'] = r['tuned_tree_num']
             
            test_y_fold_predicted = r['test_y_predicted']
            test_y_fold_truth = r['test_y_truth']
            test_y_fold_index = r['test_y_index']
            test_score_df = r['test_score_df']
            test_score_df['tuned_tree_num'] = r['tuned_tree_num']
            
            feature_importance = r['feature_importance']
            
            if alm_predictor.type == 1:            
                cur_fold_hp_dict = r['hp_dict']
                hp_dict_df.loc[cur_test_fold,:] = [cur_fold_hp_dict[x] for x in hp_dict_df.columns]
            
            if 'train_y_predicted' not in locals():
                train_y_predicted = train_y_fold_predicted
            else:
                train_y_predicted = np.hstack([train_y_predicted, train_y_fold_predicted])
                                
            if 'train_y_truth' not in locals():
                train_y_truth = train_y_fold_truth
            else:
                train_y_truth = np.hstack([train_y_truth, train_y_fold_truth])

            if 'test_y_predicted' not in locals():
                test_y_predicted = test_y_fold_predicted
            else:
                test_y_predicted = np.hstack([test_y_predicted, test_y_fold_predicted])
                                
            if 'test_y_truth' not in locals():
                test_y_truth = test_y_fold_truth
            else:
                test_y_truth = np.hstack([test_y_truth, test_y_fold_truth])
                                            
            train_y_predicted_dict[cur_test_fold] = train_y_fold_predicted                
            train_y_truth_dict[cur_test_fold] = train_y_fold_truth                         
            test_y_predicted_dict[cur_test_fold] = test_y_fold_predicted
            test_y_truth_dict[cur_test_fold] = test_y_fold_truth 
            test_y_index_dict[cur_test_fold] = test_y_fold_index
                 
            if 'train_cv_results' not in locals():
                train_cv_results = train_score_df   
            else:
                train_cv_results = pd.concat([train_cv_results, train_score_df])
              
            if 'test_cv_results' not in locals():
                test_cv_results = test_score_df
            else:
                test_cv_results = pd.concat([test_cv_results, test_score_df])
      
            feature_importance.reset_index(inplace = True)
            feature_importance.columns = ['feature',str(cur_test_fold)]       
      
            if 'cv_feature_importances' not in locals():
                cv_feature_importances = feature_importance
            else:                
                cv_feature_importances = pd.merge(cv_feature_importances, feature_importance)
                 
        train_cv_results = train_cv_results.reset_index(drop=True)  
        train_cv_result_mean = train_cv_results.mean(axis=0)
        train_cv_result_mean.index = ['macro_cv_' + x for x in train_cv_result_mean.index]        
        train_cv_result_ste = train_cv_results.std(axis=0) / np.sqrt(test_split_folds)
        train_cv_result_ste.index = ['macro_cv_' + x + '_ste' for x in train_cv_result_ste.index]
        train_cv_result = pd.DataFrame(pd.concat([train_cv_result_mean, train_cv_result_ste], axis=0)).transpose()
         
        test_cv_results = test_cv_results.reset_index(drop=True)  
        test_cv_result_mean = test_cv_results.mean(axis=0)
        test_cv_result_mean.index = ['macro_cv_' + x for x in test_cv_result_mean.index]
        test_cv_result_ste = test_cv_results.std(axis=0) / np.sqrt(test_split_folds)
        test_cv_result_ste.index = ['macro_cv_' + x + '_ste' for x in test_cv_result_ste.index]
        test_cv_result = pd.DataFrame(pd.concat([test_cv_result_mean, test_cv_result_ste], axis=0)).transpose()
 
        fis = cv_feature_importances['feature']
        cv_feature_importances = cv_feature_importances.drop(columns = {'feature'})
         
        cv_feature_importances = cv_feature_importances.transpose()
        cv_feature_importances.columns = fis
         
        cv_feature_importances = cv_feature_importances.reset_index(drop=True)
        cv_feature_importance = cv_feature_importances.mean(axis=0)
        cv_feature_importance = cv_feature_importance.sort_values(ascending=False)
          
        if 'classification' in alm_predictor.ml_type: 
            [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(test_y_truth, test_y_predicted)
            test_cv_result['micro_cv_aubprc'] = metric['aubprc']
            test_cv_result['micro_cv_brfp'] = metric['brfp']
            
            test_cv_result['micro_cv_auprc'] = metric['auprc']
            test_cv_result['micro_cv_up_auprc'] = metric['up_auprc']
            test_cv_result['micro_cv_auroc'] = metric['auroc']
            test_cv_result['micro_cv_logloss'] = metric['logloss']            
            test_cv_result['micro_cv_pfr'] = metric['pfr']
            test_cv_result['micro_cv_rfp'] = metric['rfp']
            test_cv_result['micro_cv_prior'] = metric['prior']
            test_cv_result['micro_cv_size'] = metric['size']
 
        if 'regression' in alm_predictor.ml_type:
            test_cv_result['micro_cv_pcc'] = alm_fun.pcc_cal(test_y_truth,test_y_predicted)
            test_cv_result['micro_cv_spc'] = alm_fun.spc_cal(test_y_truth,test_y_predicted)        
            train_cv_result['micro_cv_pcc'] = alm_fun.pcc_cal(train_y_truth,train_y_predicted)
            train_cv_result['micro_cv_spc'] = alm_fun.spc_cal(train_y_truth,train_y_predicted)
                     
        return_dict = {}        

        return_dict['train_y_predicted_dict'] = train_y_predicted_dict
        return_dict['train_y_truth_dict'] = train_y_truth_dict
        
        return_dict['train_cv_results'] = train_cv_results        
        return_dict['train_cv_result'] = train_cv_result  

        return_dict['test_y_predicted_dict'] = test_y_predicted_dict
        return_dict['test_y_truth_dict'] = test_y_truth_dict
        return_dict['test_y_index_dict'] = test_y_index_dict
        return_dict['test_cv_results'] = test_cv_results        
        return_dict['test_cv_result'] = test_cv_result  
        
        return_dict['cv_feature_importance'] = cv_feature_importance
        return_dict['cv_feature_importances'] = cv_feature_importances
         
        if alm_predictor.tune_obj == 'macro_cv_auprc':
            cur_hp_performance = test_cv_result['macro_cv_auprc'].get_values()[0]  
            cur_hp_performance_ste= test_cv_result['macro_cv_auprc_ste'].get_values()[0]
        if alm_predictor.tune_obj == 'macro_cv_rfp':
            cur_hp_performance = test_cv_result['macro_cv_rfp'].get_values()[0]                                                   
            cur_hp_performance_ste = test_cv_result['macro_cv_rfp_ste'].get_values()[0]
        if alm_predictor.tune_obj == 'macro_cv_auroc':
            cur_hp_performance = test_cv_result['macro_cv_auroc'].get_values()[0]
            cur_hp_performance_ste = test_cv_result['macro_cv_auroc_ste'].get_values()[0]
        if alm_predictor.tune_obj == 'macro_cv_aubprc':
            cur_hp_performance = test_cv_result['macro_cv_aubprc'].get_values()[0]
            cur_hp_performance_ste = test_cv_result['macro_cv_aubprc_ste'].get_values()[0]
        if alm_predictor.tune_obj == 'macro_cv_brfp':
            cur_hp_performance = test_cv_result['macro_cv_brfp'].get_values()[0]            
            cur_hp_performance_ste = test_cv_result['macro_cv_brfp_ste'].get_values()[0]
        if alm_predictor.tune_obj == 'micro_cv_auprc':
            cur_hp_performance = test_cv_result['micro_cv_auprc'].get_values()[0]
            cur_hp_performance_ste = test_cv_result['micro_cv_auprc_ste'].get_values()[0]
        if alm_predictor.tune_obj == 'micro_cv_auroc':
            cur_hp_performance = test_cv_result['micro_cv_auroc'].get_values()[0]            
            cur_hp_performance_ste = test_cv_result['micro_cv_auroc_ste'].get_values()[0]
                 
        alm_fun.show_msg (self.log,1,alm_predictor.tune_obj + ': ' + str(round(cur_hp_performance,4)) + '±' + str(round(cur_hp_performance_ste,4)))
        train_cv_result.to_csv(self.prefix(predictor, runtime, 'csv',1) + '_hp_train_cv_result.csv')
        train_cv_results.to_csv(self.prefix(predictor, runtime, 'csv',1) + '_hp_train_cv_results.csv')
        test_cv_result.to_csv(self.prefix(predictor, runtime, 'csv',1) +'_hp_test_cv_result.csv')
        test_cv_results.to_csv(self.prefix(predictor, runtime, 'csv',1) + '_hp_test_cv_results.csv')
        
        if alm_predictor.type == 1:   
            hp_dict_df.to_csv(self.prefix(predictor, runtime, 'csv',1) + '_hp_dict_results.csv')
            
        np.save(self.prefix(predictor, runtime, 'npy',1)+ '_test_cv_results.npy',return_dict)
        
        alm_fun.show_msg (self.log,1,"End Time: " + str(datetime.now()))
        alm_fun.show_msg (self.log,1,"***************************************************************")    
     
    def fun_target_prediction(self,predictor,runtime):        
        alm_predictor = self.proj.predictor[predictor]
        alm_dataset = alm_predictor.data_instance        
        features = alm_predictor.features        
        cur_target_df = pd.read_csv(runtime['target_file'],low_memory = False).loc[:,features + runtime['key_cols']]
        runtime['hp_dict_file'] = self.prefix(predictor, runtime, 'npy') + '_hp_dict.npy'

        if alm_predictor.type == 0: # no loo predition for 
            alm_fun.show_msg (self.log,self.verbose,'No LOO prediction for Non-VARITY models, making predictions without LOO......' )
            runtime['loo'] == 0
            
        if runtime['loo'] == 1:                        
            loo_dict = self.get_loo_dict(predictor, runtime)
            alm_fun.show_msg (self.log,self.verbose,'Runing prediction for ' + str(len(loo_dict.keys())) + ' records that exist in training data......' )  
            #**********************************************************************
            # Fire parallel jobs for loo predictions
            #**********************************************************************
            cur_jobs= {}                   
            if runtime['batch_id'] =='':                           
                runtime['batch_id'] = alm_fun.get_random_id(10)
                
            alm_fun.show_msg (self.log,1,'Start to run target loo predictions, batch id: ' + runtime['batch_id'] + '......' )        
            for target_index in loo_dict.keys(): 
                new_runtime = runtime.copy()                                             
                new_runtime['cur_target_fold']  = loo_dict[target_index]
                new_runtime['single_fold_type']  = 'target'
                new_runtime['hp_tune_type'] = 'hyperopt'      
                new_runtime['run_on_node'] = 1    
                new_runtime['action'] = 'single_fold_prediction'                                         
                new_runtime['job_name'] = alm_fun.get_random_id(8)                
                new_runtime['cur_fold_result'] = self.prefix(predictor, new_runtime, 'npy_temp')+ '_loo_' + str(target_index) + '_' + new_runtime['batch_id'] + '_hp_target_single_fold_result.npy'                                                
                                                
                if not os.path.isfile(new_runtime['cur_fold_result']):                                                 
                    if (runtime['cluster'] == 1) :               
                        alm_fun.show_msg (self.log,1, 'Run prediction on target fold '  + '[target_index: ' + str(target_index) + '-' + str(new_runtime['cur_target_fold']) + ']......')
                        [job_id,job_name] = self.varity_obj.varity_action(new_runtime)
                        cur_jobs[job_name] = []
                        cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                        cur_jobs[job_name].append(job_id)
                        cur_jobs[job_name].append(new_runtime)                                             
                    else:                                                    
                        self.fun_single_fold_prediction(predictor,new_runtime)
                else:
                    alm_fun.show_msg (self.log,1, 'Result is avaiable on target fold '  + '[target_index: ' + str(target_index) + '-' + str(new_runtime['cur_target_fold']) + ']......')       
                        
            if runtime['cluster'] == 1:
                batch_log = self.prefix(predictor,new_runtime,'log') + '_' + runtime['batch_id'] + '.log'
                if self.varity_obj.fun_monitor_jobs(cur_jobs,batch_log,runtime) == 1:
                    alm_fun.show_msg (self.log,1, 'Batch: '  +  runtime['batch_id'] + ' all results are done,start to gathering results......')       
                    
            #**********************************************************************
            # Collect results for all parallel loo predictions
            #**********************************************************************
            for target_index in loo_dict.keys():                
                cur_fold_result = self.prefix(predictor, new_runtime, 'npy_temp')+ '_loo_' + str(target_index) + '_' + runtime['batch_id'] + '_hp_target_single_fold_result.npy'                 
                r_loo = np.load(cur_fold_result).item()
                cur_target_df.loc[target_index,alm_predictor.name + '_LOO'] = r_loo['test_y_predicted'].values[0]
             
            save_prediction_file = self.prefix(predictor, runtime, 'csv') + '_target_loo_predicted.csv'
            print (cur_target_df.columns)
            cur_target_df.to_csv(save_prediction_file)
        else:            
            runtime['cur_fold_result'] = self.prefix(predictor, runtime, 'npy') + '_target_result.npy'
            runtime['single_fold_type'] = 'target'
            self.fun_single_fold_prediction(predictor,runtime)                                        
            r = np.load(runtime['cur_fold_result']).item()            
            cur_target_df[alm_predictor.name] = r['test_y_predicted']
            save_prediction_file = self.prefix(predictor, runtime, 'csv') + '_target_predicted.csv'
            cur_target_df.to_csv(save_prediction_file)
            
            shap_output_target_interaction  = r['shap_output_test_interaction']
            shap_output_target_interaction_file = self.prefix(predictor, runtime, 'csv') + '_shap_output_target_interaction.csv'
            if shap_output_target_interaction is not None:
                np.save(shap_output_target_interaction_file,shap_output_target_interaction)   
            shap_output_train_interaction  = r['shap_output_train_interaction']
            shap_output_train_interaction_file = self.prefix(predictor, runtime, 'csv') + '_shap_output_train_interaction.csv'
            if shap_output_train_interaction is not None:
                np.save(shap_output_train_interaction_file,shap_output_train_interaction)    
                                    
    def fun_validation_cv_prediction_sp(self,predictor,runtime):
        alm_predictor = self.proj.predictor[predictor]
        cv_split_folds = alm_predictor.data_instance.cv_split_folds   
                     
        #save the current hp_dict                
        cur_hp_dict = alm_predictor.hp_default
        cur_hp_dict[runtime['filtering_hp']] = runtime['filtering_hp_value']
        cur_hp_dict_file = self.prefix(predictor, runtime, 'npy_temp')+   '_' + runtime['filtering_hp'] + '_' + str(runtime['filtering_hp_value']) + '_hp_dict.npy'
        np.save(cur_hp_dict_file,cur_hp_dict)
        
        #**********************************************************************
        # Fire parallel jobs for all test folds
        #**********************************************************************
        cur_jobs= {}                   
        if runtime['batch_id'] =='':                           
            runtime['batch_id'] = alm_fun.get_random_id(10)
        alm_fun.show_msg (self.log,1,'Start to run validation cv prediction, batch id: ' + runtime['batch_id'] + '......' )        
        for cur_validation_fold in range(cv_split_folds): 
            cur_validation_fold_result = self.prefix(predictor, runtime, 'npy_temp') +  '_' + runtime['filtering_hp'] + '_' + str(runtime['filtering_hp_value'])+ '_' + 'vf' + str(cur_validation_fold) + '_' + runtime['batch_id'] + '_hp_validation_single_fold_result.npy'
            new_runtime = runtime.copy()                                      
            new_runtime['cur_validation_fold']  = cur_validation_fold
            new_runtime['single_fold_type']  = 'validation'
            new_runtime['hp_tune_type'] = 'sp'      
            new_runtime['run_on_node'] = 1    
            new_runtime['action'] = 'single_fold_prediction'                         
            new_runtime['job_name'] = alm_fun.get_random_id(8)                
            new_runtime['cur_fold_result'] = cur_validation_fold_result            
            new_runtime['hp_dict_file'] = cur_hp_dict_file
            if not os.path.isfile(new_runtime['cur_fold_result']):
                alm_fun.show_msg (self.log,1, 'Run prediction on validation fold '  + str(cur_validation_fold) + '......')                                                 
                if (runtime['cluster'] == 1) :                                      
                    [job_id,job_name] = self.varity_obj.varity_action(new_runtime)
                    cur_jobs[job_name] = []
                    cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                    cur_jobs[job_name].append(job_id)
                    cur_jobs[job_name].append(new_runtime)                     
                else:                             
                    self.fun_single_fold_prediction(predictor,new_runtime)       
                    
        if runtime['cluster'] == 1:
            batch_log = self.prefix(predictor,new_runtime,'log') + '_' + runtime['batch_id'] + '.log'
            if self.varity_obj.fun_monitor_jobs(cur_jobs,batch_log,runtime) == 1:
                alm_fun.show_msg (self.log,1, 'Batch: '  +  runtime['batch_id'] + ' all results are done,start to gathering results......')                       
            
        #**********************************************************************
        # Collect  the results of all parallel jobs and retrieve the best value
        #**********************************************************************                          
        alm_fun.show_msg (self.log,self.verbose,'All parallel jobs are finished. Collecting results......' )
        for cur_validation_fold in range(cv_split_folds):                               
            cur_validation_result = self.prefix(predictor, new_runtime, 'npy_temp') +  '_' + runtime['filtering_hp'] + '_' + str(runtime['filtering_hp_value'])+ '_' + 'vf' + str(cur_validation_fold) + '_' + runtime['batch_id'] + '_hp_validation_single_fold_result.npy'
            #in the case the result file does exist, but not compelete while loading (causing pickle loading error)
            successful_load = 0
            while successful_load  == 0:
                try:
                    r = np.load(cur_validation_result).item()
                    successful_load = 1
                except: 
                    time.sleep(10)
            train_y_fold_predicted = r['train_y_predicted']
            train_y_fold_truth = r['train_y_truth']
            train_score_df = r['train_score_df']
            train_score_df['tuned_tree_num'] = r['tuned_tree_num']
             
            validation_y_fold_predicted = r['test_y_predicted']
            validation_y_fold_truth = r['test_y_truth']
            validation_score_df = r['test_score_df']
            validation_score_df['tuned_tree_num'] = r['tuned_tree_num']
             
            feature_importance = r['feature_importance']
                                                  
            if 'train_y_predicted' not in locals():
                train_y_predicted = train_y_fold_predicted
            else:
                train_y_predicted = np.hstack([train_y_predicted, train_y_fold_predicted])                      
              
            if 'train_y_truth' not in locals():
                train_y_truth = train_y_fold_truth
            else:
                train_y_truth = np.hstack([train_y_truth, train_y_fold_truth])                
 
            if 'validation_y_predicted' not in locals():
                validation_y_predicted = validation_y_fold_predicted
            else:
                validation_y_predicted = np.hstack([validation_y_predicted, validation_y_fold_predicted]) 
              
            if 'validation_y_truth' not in locals():
                validation_y_truth = validation_y_fold_truth
            else:
                validation_y_truth = np.hstack([validation_y_truth, validation_y_fold_truth])
                 
            if 'train_cv_results' not in locals():
                train_cv_results = train_score_df   
            else:
                train_cv_results = pd.concat([train_cv_results, train_score_df])
              
            if 'validation_cv_results' not in locals():
                validation_cv_results = validation_score_df   
            else:
                validation_cv_results = pd.concat([validation_cv_results, validation_score_df])
      
            feature_importance.reset_index(inplace = True)
            feature_importance.columns = ['feature',str(cur_validation_fold)]       
      
            if 'cv_feature_importances' not in locals():
                cv_feature_importances = feature_importance
            else:                
                cv_feature_importances = pd.merge(cv_feature_importances, feature_importance)
                 
        train_cv_results = train_cv_results.reset_index(drop=True)  
        train_cv_result_mean = train_cv_results.mean(axis=0)
        train_cv_result_mean.index = ['macro_cv_' + x for x in train_cv_result_mean.index]        
        train_cv_result_ste = train_cv_results.std(axis=0) / np.sqrt(cv_split_folds)
        train_cv_result_ste.index = ['macro_cv_' + x + '_ste' for x in train_cv_result_ste.index]
        train_cv_result = pd.DataFrame(pd.concat([train_cv_result_mean, train_cv_result_ste], axis=0)).transpose()
         
        validation_cv_results = validation_cv_results.reset_index(drop=True)  
        validation_cv_result_mean = validation_cv_results.mean(axis=0)
        validation_cv_result_mean.index = ['macro_cv_' + x for x in validation_cv_result_mean.index]
        validation_cv_result_ste = validation_cv_results.std(axis=0) / np.sqrt(cv_split_folds)
        validation_cv_result_ste.index = ['macro_cv_' + x + '_ste' for x in validation_cv_result_ste.index]
        validation_cv_result = pd.DataFrame(pd.concat([validation_cv_result_mean, validation_cv_result_ste], axis=0)).transpose()
 
        fis = cv_feature_importances['feature']
        cv_feature_importances = cv_feature_importances.drop(columns = {'feature'})
         
        cv_feature_importances = cv_feature_importances.transpose()
        cv_feature_importances.columns = fis
         
        cv_feature_importances = cv_feature_importances.reset_index(drop=True)
        cv_feature_importance = cv_feature_importances.mean(axis=0)
        cv_feature_importance = cv_feature_importance.sort_values(ascending=False)
          
        if 'classification' in alm_predictor.ml_type:
            [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(validation_y_truth, validation_y_predicted)
            
            validation_cv_result['micro_cv_aubprc'] = metric['aubprc']
            validation_cv_result['micro_cv_aurfp'] = metric['brfp']
            
            validation_cv_result['micro_cv_auprc'] = metric['aubprc']
            validation_cv_result['micro_cv_up_auprc'] = metric['up_auprc']
            validation_cv_result['micro_cv_auroc'] = metric['auroc']
            validation_cv_result['micro_cv_logloss'] = metric['logloss']            
            validation_cv_result['micro_cv_pfr'] = metric['pfr']
            validation_cv_result['micro_cv_rfp'] = metric['rfp']
            validation_cv_result['micro_cv_prior'] = metric['prior']
            validation_cv_result['micro_cv_size'] = metric['size']
            
            [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(train_y_truth, train_y_predicted)
            
            train_cv_result['micro_cv_aubprc'] = metric['aubprc']
            train_cv_result['micro_cv_aurfp'] = metric['brfp']
            
            train_cv_result['micro_cv_auprc'] = metric['aubprc']
            train_cv_result['micro_cv_up_auprc'] = metric['up_auprc']
            train_cv_result['micro_cv_auroc'] = metric['auroc']
            train_cv_result['micro_cv_logloss'] = metric['logloss']            
            train_cv_result['micro_cv_pfr'] = metric['pfr']
            train_cv_result['micro_cv_rfp'] = metric['rfp']
            train_cv_result['micro_cv_prior'] = metric['prior']
            train_cv_result['micro_cv_size'] = metric['size']
            
 
        if 'regression' in alm_predictor.ml_type:
            validation_cv_result['micro_cv_pcc'] = alm_fun.pcc_cal(validation_y_truth,validation_y_predicted)
            validation_cv_result['micro_cv_spc'] = alm_fun.spc_cal(validation_y_truth,validation_y_predicted)        
            train_cv_result['micro_cv_pcc'] = alm_fun.pcc_cal(train_y_truth,train_y_predicted)
            train_cv_result['micro_cv_spc'] = alm_fun.spc_cal(train_y_truth,train_y_predicted)
                                 
        cur_hp_validation_performance = validation_cv_result[alm_predictor.tune_obj].get_values()[0]  
        cur_hp_validation_performance_ste= validation_cv_result[alm_predictor.tune_obj + '_ste'].get_values()[0] 
        cur_hp_train_performance = train_cv_result[alm_predictor.tune_obj].get_values()[0]  
        cur_hp_train_performance_ste= train_cv_result[alm_predictor.tune_obj + '_ste'].get_values()[0]                      
                 
        alm_fun.show_msg (self.log,1,alm_predictor.tune_obj + ' on validation set:' + str(round(cur_hp_validation_performance,4)) + '±' + str(round(cur_hp_validation_performance_ste,4)))         
        alm_fun.show_msg (self.log,1,alm_predictor.tune_obj + ' on training set: ' + str(round(cur_hp_train_performance,4)) + '±' + str(round(cur_hp_train_performance_ste,4)))
        
        spvalue_result_file = self.prefix(predictor, runtime, 'csv',1) + '_' + runtime['filtering_hp'] + '_spvalue_results.txt'
        cur_spvalue_result = [str(runtime['filtering_hp_value']),str(cur_hp_train_performance),str(cur_hp_train_performance_ste),str(cur_hp_validation_performance),str(cur_hp_validation_performance_ste)] 
        alm_fun.show_msg(spvalue_result_file,1,'\t'.join(cur_spvalue_result),with_time = 0)
        return  (True)                            
    
    def fun_validation_cv_prediction_hyperopt(self,cur_hyperopt_hp):
                
        alm_predictor =  self.hyperopt_predictor
        runtime = self.hyperopt_runtime         
        cur_trial = len(self.cur_trials_result)-1        
        cv_split_folds = alm_predictor.data_instance.cv_split_folds        
        #save the compelete trails so far
        pickle.dump(self.cur_trials_result, open(self.prefix(alm_predictor.name,runtime, 'npy') + '_trials.pkl', "wb"))
        #save the current hp_dict                
        cur_hp_dict = alm_predictor.hp_default
        for key in cur_hyperopt_hp.keys():
            cur_hp_dict[key] = cur_hyperopt_hp[key]
        # save current  cur_hp_dict
        cur_hp_dict_file = self.prefix(alm_predictor.name, runtime, 'npy_temp')+ '_' + 'tr' +  str(cur_trial) + '_hp_dict.npy'
        np.save(cur_hp_dict_file,cur_hp_dict)

        alm_fun.show_msg (self.log,1,"***************************************************************")
        alm_fun.show_msg (self.log,1,"fun_validation_cv_prediction_hyperopt")
        alm_fun.show_msg (self.log,1,"Predictor: " + alm_predictor.name)
        alm_fun.show_msg (self.log,1,"Test fold: " + str(runtime['cur_test_fold']))
        alm_fun.show_msg (self.log,1,"Tune obj: " + alm_predictor.tune_obj)
        alm_fun.show_msg (self.log,1,"Data name: " + alm_predictor.data_instance.name)  
        alm_fun.show_msg (self.log,1,"Session ID: " + runtime['session_id'])
        alm_fun.show_msg (self.log,1,"Start Time: " + str(datetime.now()))
        alm_fun.show_msg (self.log,1,"***************************************************************")
        alm_fun.show_msg (self.log,1,"Trial: " + str(cur_trial))
        alm_fun.show_msg (self.log,1,"Hyper-parameters:")
        for key in cur_hyperopt_hp.keys():
            alm_fun.show_msg (self.log,1,key + ': ' + str(cur_hyperopt_hp[key]))
        alm_fun.show_msg (self.log,1,"***************************************************************")

        #**********************************************************************
        # Fire parallel jobs for all test folds
        #**********************************************************************
        cur_jobs= {}                   
        if runtime['batch_id'] =='':                           
            runtime['batch_id'] = alm_fun.get_random_id(10)
        alm_fun.show_msg (self.log,1,'Start to run validation cv prediction, batch id: ' + runtime['batch_id'] + '......' )        
        for cur_validation_fold in range(cv_split_folds):
            cur_validation_fold_result = self.prefix(alm_predictor.name, runtime, 'npy_temp') + '_' + 'vf' + str(cur_validation_fold) + '_' + 'tr' + str(cur_trial) + '_' + runtime['batch_id'] + '_hp_validation_single_fold_result.npy' 
            new_runtime = runtime.copy()              
            new_runtime['cur_trial'] = cur_trial            
            new_runtime['cur_validation_fold']  = cur_validation_fold
            new_runtime['single_fold_type']  = 'validation'
            new_runtime['hp_tune_type'] = 'hyperopt'      
            new_runtime['run_on_node'] = 1    
            new_runtime['action'] = 'single_fold_prediction'                         
            new_runtime['job_name'] = alm_fun.get_random_id(8)                
            new_runtime['cur_fold_result'] =  cur_validation_fold_result           
            new_runtime['hp_dict_file'] = cur_hp_dict_file
            if not os.path.isfile(new_runtime['cur_fold_result']):        
                alm_fun.show_msg (self.log,1, 'Run prediction on test fold '  + str(cur_validation_fold) + '......')                                         
                if (runtime['cluster'] == 1) :                  
                    [job_id,job_name] = self.varity_obj.varity_action(new_runtime)
                    cur_jobs[job_name] = []
                    cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                    cur_jobs[job_name].append(job_id)
                    cur_jobs[job_name].append(new_runtime)                     
                else:                             
                    self.fun_single_fold_prediction(alm_predictor.name,new_runtime)       
                        
        if runtime['cluster'] == 1:
            batch_log = self.prefix(alm_predictor.name,new_runtime,'log') + '_' + runtime['batch_id'] + '.log'
            if self.varity_obj.fun_monitor_jobs(cur_jobs,batch_log,runtime) == 1:
                alm_fun.show_msg (self.log,1, 'Batch: '  +  runtime['batch_id'] + ' all results are done,start to gathering results......')                       
            
        #**********************************************************************
        # Collect  the results of all parallel jobs
        #**********************************************************************                          
        alm_fun.show_msg (self.log,1,'All parallel jobs are finished. Collecting results......' )
        for cur_validation_fold in range(cv_split_folds):                               
            cur_validation_result = self.prefix(alm_predictor.name, runtime, 'npy_temp') + '_' + 'vf' + str(cur_validation_fold) + '_' + 'tr' + str(cur_trial) + '_' + runtime['batch_id'] + '_hp_validation_single_fold_result.npy'
        
            #in the case the result file does exist, but not compelete while loading (causing pickle loading error)
            successful_load = 0
            while successful_load  == 0:
                try:
                    r = np.load(cur_validation_result).item()
                    successful_load = 1
                except: 
                    time.sleep(10)
            train_y_fold_predicted = r['train_y_predicted']
            train_y_fold_truth = r['train_y_truth']
            train_score_df = r['train_score_df']
            train_score_df['tuned_tree_num'] = r['tuned_tree_num']
             
            validation_y_fold_predicted = r['test_y_predicted']
            validation_y_fold_truth = r['test_y_truth']
            validation_score_df = r['test_score_df']
            validation_score_df['tuned_tree_num'] = r['tuned_tree_num']
             
            feature_importance = r['feature_importance']
                                                  
            if 'train_y_predicted' not in locals():
                train_y_predicted = train_y_fold_predicted
            else:
                train_y_predicted = np.hstack([train_y_predicted, train_y_fold_predicted])                      
              
            if 'train_y_truth' not in locals():
                train_y_truth = train_y_fold_truth
            else:
                train_y_truth = np.hstack([train_y_truth, train_y_fold_truth])                
 
            if 'validation_y_predicted' not in locals():
                validation_y_predicted = validation_y_fold_predicted
            else:
                validation_y_predicted = np.hstack([validation_y_predicted, validation_y_fold_predicted]) 
              
            if 'validation_y_truth' not in locals():
                validation_y_truth = validation_y_fold_truth
            else:
                validation_y_truth = np.hstack([validation_y_truth, validation_y_fold_truth])
                 
            if 'train_cv_results' not in locals():
                train_cv_results = train_score_df   
            else:
                train_cv_results = pd.concat([train_cv_results, train_score_df])
              
            if 'validation_cv_results' not in locals():
                validation_cv_results = validation_score_df   
            else:
                validation_cv_results = pd.concat([validation_cv_results, validation_score_df])
      
            feature_importance.reset_index(inplace = True)
            feature_importance.columns = ['feature',str(cur_validation_fold)]       
      
            if 'cv_feature_importances' not in locals():
                cv_feature_importances = feature_importance
            else:                
                cv_feature_importances = pd.merge(cv_feature_importances, feature_importance)
                 
        train_cv_results = train_cv_results.reset_index(drop=True)  
        train_cv_result_mean = train_cv_results.mean(axis=0)
        train_cv_result_mean.index = ['macro_cv_' + x for x in train_cv_result_mean.index]        
        train_cv_result_ste = train_cv_results.std(axis=0) / np.sqrt(cv_split_folds)
        train_cv_result_ste.index = ['macro_cv_' + x + '_ste' for x in train_cv_result_ste.index]
        train_cv_result = pd.DataFrame(pd.concat([train_cv_result_mean, train_cv_result_ste], axis=0)).transpose()
         
        validation_cv_results = validation_cv_results.reset_index(drop=True)  
        validation_cv_result_mean = validation_cv_results.mean(axis=0)
        validation_cv_result_mean.index = ['macro_cv_' + x for x in validation_cv_result_mean.index]
        validation_cv_result_ste = validation_cv_results.std(axis=0) / np.sqrt(cv_split_folds)
        validation_cv_result_ste.index = ['macro_cv_' + x + '_ste' for x in validation_cv_result_ste.index]
        validation_cv_result = pd.DataFrame(pd.concat([validation_cv_result_mean, validation_cv_result_ste], axis=0)).transpose()
 
        fis = cv_feature_importances['feature']
        cv_feature_importances = cv_feature_importances.drop(columns = {'feature'})
         
        cv_feature_importances = cv_feature_importances.transpose()
        cv_feature_importances.columns = fis
         
        cv_feature_importances = cv_feature_importances.reset_index(drop=True)
        cv_feature_importance = cv_feature_importances.mean(axis=0)
        cv_feature_importance = cv_feature_importance.sort_values(ascending=False)
          
        if 'classification' in alm_predictor.ml_type: 
            [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(validation_y_truth, validation_y_predicted)
            
            validation_cv_result['micro_cv_aubprc'] = metric['aubprc']
            validation_cv_result['micro_cv_aurfp'] = metric['brfp']
            
            validation_cv_result['micro_cv_auprc'] = metric['aubprc']
            validation_cv_result['micro_cv_up_auprc'] = metric['up_auprc']
            validation_cv_result['micro_cv_auroc'] = metric['auroc']
            validation_cv_result['micro_cv_logloss'] = metric['logloss']            
            validation_cv_result['micro_cv_pfr'] = metric['pfr']
            validation_cv_result['micro_cv_rfp'] = metric['rfp']
            validation_cv_result['micro_cv_prior'] = metric['prior']
            validation_cv_result['micro_cv_size'] = metric['size']
            
            [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(train_y_truth, train_y_predicted)
            
            train_cv_result['micro_cv_aubprc'] = metric['aubprc']
            train_cv_result['micro_cv_aurfp'] = metric['brfp']
            
            train_cv_result['micro_cv_auprc'] = metric['aubprc']
            train_cv_result['micro_cv_up_auprc'] = metric['up_auprc']
            train_cv_result['micro_cv_auroc'] = metric['auroc']
            train_cv_result['micro_cv_logloss'] = metric['logloss']            
            train_cv_result['micro_cv_pfr'] = metric['pfr']
            train_cv_result['micro_cv_rfp'] = metric['rfp']
            train_cv_result['micro_cv_prior'] = metric['prior']
            train_cv_result['micro_cv_size'] = metric['size']
            
 
        if 'regression' in alm_predictor.ml_type:
            validation_cv_result['micro_cv_pcc'] = alm_fun.pcc_cal(validation_y_truth,validation_y_predicted)
            validation_cv_result['micro_cv_spc'] = alm_fun.spc_cal(validation_y_truth,validation_y_predicted)        
            train_cv_result['micro_cv_pcc'] = alm_fun.pcc_cal(train_y_truth,train_y_predicted)
            train_cv_result['micro_cv_spc'] = alm_fun.spc_cal(train_y_truth,train_y_predicted)

        cur_hp_validation_performance = validation_cv_result[alm_predictor.tune_obj].get_values()[0]  
        cur_hp_validation_performance_ste= validation_cv_result[alm_predictor.tune_obj + '_ste'].get_values()[0] 
        cur_hp_train_performance = train_cv_result[alm_predictor.tune_obj].get_values()[0]  
        cur_hp_train_performance_ste= train_cv_result[alm_predictor.tune_obj + '_ste'].get_values()[0]                      
                 
        alm_fun.show_msg (self.log,1,alm_predictor.tune_obj + ' on validation set:' + str(round(cur_hp_validation_performance,4)) + '±' + str(round(cur_hp_validation_performance_ste,4)))         
        alm_fun.show_msg (self.log,1,alm_predictor.tune_obj + ' on training set: ' + str(round(cur_hp_train_performance,4)) + '±' + str(round(cur_hp_train_performance_ste,4)))
        alm_fun.show_msg (self.log,1,"End Time: " + str(datetime.now()))
        alm_fun.show_msg (self.log,1,"***************************************************************")        
                
        trial_result_file = self.prefix(alm_predictor.name, runtime, 'csv') + '_trial_results.txt'
        cur_trial_result = [str(cur_trial),str(cur_hp_train_performance),str(cur_hp_train_performance_ste),str(cur_hp_validation_performance),str(cur_hp_validation_performance_ste)] + \
                                        [str(cur_hyperopt_hp[key]) for key in cur_hyperopt_hp.keys()] 
        alm_fun.show_msg(trial_result_file,1,'\t'.join(cur_trial_result),with_time = 0)
        return    ({'loss':1-cur_hp_validation_performance,'status': hyperopt.STATUS_OK})  
    
    def fun_single_fold_prediction(self,predictor,runtime):
        alm_predictor = self.proj.predictor[predictor]
        alm_estimator = alm_predictor.es_instance
        alm_dataset = alm_predictor.data_instance
        features = alm_predictor.features

        if alm_predictor.type == 1:                    
            #load current tuned hyper-parameter dictionary            
            cur_hp_dict = self.load_cur_hp_dict(runtime['hp_dict_file'])
            if cur_hp_dict is None:
                cur_hp_dict = alm_predictor.hp_default                      
            #update the weight hyper-parameter for each extra training example
            alpha = self.update_sample_weights(cur_hp_dict,predictor,runtime)
            
            #update alogrithm-level hyperparameter
            for cur_hp in alm_predictor.hps.keys():
                if alm_predictor.hps[cur_hp]['hp_type'] == 3:                    
                    if (alm_predictor.hps[cur_hp]['type'] == 'int') | (cur_hp == 'n_estimators'):
                        cur_hp_dict[cur_hp] = int(cur_hp_dict[cur_hp])                                        
                    setattr(alm_estimator.estimator,cur_hp,cur_hp_dict[cur_hp])        
        
        #prepare the extra_data_df
        extra_data_df = None
        if len(alm_dataset.extra_train_data_df_lst) != 0:    
            extra_data_df = alm_dataset.extra_train_data_df_lst[alm_dataset.extra_data_index].copy()
            
        #if predicting on test set 
        if runtime['single_fold_type'] == 'test':            
            cur_train_df = alm_dataset.train_data_index_df.loc[alm_dataset.train_splits_df[runtime['cur_test_fold']][runtime['cur_gradient_key']],:]
            cur_test_df = alm_dataset.test_data_index_df.loc[alm_dataset.test_splits_df[runtime['cur_test_fold']][runtime['cur_gradient_key']],:]
        
        if runtime['single_fold_type'] == 'validation':
            train_splits_indices = alm_dataset.train_cv_splits_df[runtime['cur_test_fold']][runtime['cur_validation_fold']][runtime['cur_gradient_key']]
            validation_splits_indices = alm_dataset.validation_cv_splits_df[runtime['cur_test_fold']][runtime['cur_validation_fold']][runtime['cur_gradient_key']]        
            cur_train_df = alm_dataset.train_data_index_df.loc[train_splits_indices,:]
            cur_test_df = alm_dataset.validation_data_index_df.loc[validation_splits_indices,:]             

        if runtime['single_fold_type'] == 'target':                
            cur_train_df = alm_dataset.train_data_index_df.loc[alm_dataset.train_data_for_target_df[runtime['cur_gradient_key']],:]
            cur_test_df = pd.read_csv(runtime['target_file'])
            
            if runtime['loo'] == 1:
                cur_remove_data = runtime['cur_target_fold'].split(':')[0]
                cur_remove_index = int(runtime['cur_target_fold'].split(':')[1])
                if cur_remove_data == 'extra':
                    cur_test_df = extra_data_df.loc[[cur_remove_index],:]
                    
                if cur_remove_data =='core':                    
                    cur_test_df = cur_train_df.loc[[cur_remove_index],:]     
                                   
                extra_data_df = extra_data_df.loc[set(extra_data_df.index) - set([cur_remove_index]),:]
                cur_train_df = cur_train_df.loc[set(cur_train_df.index) - set([cur_remove_index]),:]
#                     extra_data_df = extra_data_df.drop(int(cur_remove_index))              
#                     cur_train_df = cur_train_df.drop(int(cur_remove_index))

                    
            cur_test_df[alm_dataset.dependent_variable] = np.random.random_integers(0,1,cur_test_df.shape[0])
        r = alm_estimator.run(features, alm_dataset.dependent_variable, alm_predictor.ml_type, cur_train_df, cur_test_df ,extra_data_df,None,alm_predictor)
        
        if alm_predictor.type == 1:           
            r['hp_dict'] = cur_hp_dict                                    
            if (r['model'] is not None) & (runtime['single_fold_type'] in ['target','test']):
                r['model'].save_model(self.prefix(predictor, runtime, 'npy')+  '.model')
                    
        print (r['test_score_df'])   
#         print (r['test_y_predicted'])                 
        np.save(runtime['cur_fold_result'],r)         
                        
    def get_loo_dict(self,predictor,runtime):
        loo_dict = {}
        alm_predictor = self.proj.predictor[predictor]
        alm_dataset = alm_predictor.data_instance        
        features = alm_predictor.features        
        if len(runtime['key_cols']) == 0:
            runtime['key_cols'] = features
        
        cur_target_df = pd.read_csv(runtime['target_file'],low_memory = False)
        cur_target_df['target_index'] = cur_target_df.index
        cur_target_df = cur_target_df[runtime['key_cols'] + ['target_index'] ]

        if alm_predictor.type == 1:                    
            #load current tuned hyper-parameter dictionary            
            cur_hp_dict = self.load_cur_hp_dict(runtime['hp_dict_file'])
            if cur_hp_dict is None:
                cur_hp_dict = alm_predictor.hp_default                      
            #update the weight hyper-parameter for each extra training example
            alpha = self.update_sample_weights(cur_hp_dict,predictor,runtime)
                                      
        extra_data_df = None
        if len(alm_dataset.extra_train_data_df_lst) != 0:    
            extra_data_df = alm_dataset.extra_train_data_df_lst[alm_dataset.extra_data_index].copy()        
        pass      
        cur_train_df = alm_dataset.train_data_index_df.loc[alm_dataset.train_data_for_target_df[runtime['cur_gradient_key']],:]
        #disable the shapley value for loo prediction
        alm_predictor.shap_train_interaction = 0
        alm_predictor.shap_test_interaction = 0
        
        valid_extra_data_df = extra_data_df.loc[extra_data_df['weight'] != 0, runtime['key_cols']]
        valid_extra_data_df['type'] = 'extra'
        valid_train_df = cur_train_df[runtime['key_cols']]
        valid_train_df['type'] = 'core'
                    
        final_train_df = pd.concat([valid_extra_data_df,valid_train_df])
        final_train_df['loo'] = 1
        final_train_df['train_index'] = final_train_df.index
        
        predicted_target_df = pd.merge(cur_target_df,final_train_df,how = 'left')  
        predicted_target_df[alm_predictor.name + '_LOO'] = np.nan                  
        noloo_indices = predicted_target_df.loc[predicted_target_df['loo'].isnull(),'target_index']
        loo_indices = predicted_target_df.loc[predicted_target_df['loo'].notnull(),'target_index']     
        
        for loo_index in loo_indices:            
            cur_remove_index = int(predicted_target_df.loc[predicted_target_df['target_index'] == loo_index,'train_index'].values[0])
            cur_remove_data = predicted_target_df.loc[predicted_target_df['target_index'] == loo_index,'type'].values[0]
            loo_dict[loo_index] = cur_remove_data + ':' + str(cur_remove_index)
    
        return(loo_dict)
    
    def update_sample_weights(self,hp_dict,predictor,runtime): 
        alm_predictor = self.proj.predictor[predictor]
        extra_data_index = alm_predictor.data_instance.extra_data_index  
        alpha = self.get_sample_weights(hp_dict,predictor,runtime)
        print('extra weight:' + str(alpha.sum()))
        alm_predictor.data_instance.extra_train_data_df_lst[extra_data_index]['weight'] = alpha
        return(alpha)   
    
    def get_sample_weights(self,hp_dict,predictor,runtime):
        alm_predictor = self.proj.predictor[predictor]
        extra_data_index = alm_predictor.data_instance.extra_data_index        
        extra_data = alm_predictor.data_instance.extra_train_data_df_lst[extra_data_index].copy()    
        if type == 'from_hp_npy':
            if os.path.isfile(self.prefix(predictor, runtime,'npy') + '_hp_weights.npy'):
                alpha = np.load(self.prefix(predictor, runtime,'npy') + '_hp_weights.npy')
                extra_data['weight'] = alpha
                
#         if type == 'from_nn_npy':    
#             if os.path.isfile(self.prefix(predictor,runtime,'npy') + '_' + str(init_weights) +'_' + str(target_as_source)  +'_' + self.session_id + '_nn_weights.npy'):
#                 alpha = np.load(self.prefix(predictor,runtime,'npy') + '_' + str(init_weights) +'_' + str(target_as_source)  +'_' + self.session_id + '_nn_weights.npy')
#                 extra_data['weight'] = alpha
                
        if runtime['hp_tune_type'] == 'sp':                           
            extra_data['weight'] = 0
            extra_data.loc[alm_predictor.hp_mv_indices[runtime['filtering_hp']][hp_dict[runtime['filtering_hp']]],'weight'] = 1
    
        if runtime['hp_tune_type'] == 'cd':                           
            extra_data['weight'] = 0
            for hp_parameter in alm_predictor.hp_parameters['cd']:
                if 'weight' in hp_parameter:   
                     extra_data.loc[alm_predictor.hp_indices[hp_parameter][0],'weight'] = 1
                else:
                    extra_data.loc[alm_predictor.hp_rest_indices[hp_parameter][0],'weight'] = 1
                    
            for hp_parameter in alm_predictor.hp_parameters['all']:  
                if 'weight' in hp_parameter:                    
                    extra_data.loc[alm_predictor.hp_indices[hp_parameter][hp_dict[hp_parameter]],'weight'] = extra_data.loc[alm_predictor.hp_indices[hp_parameter][hp_dict[hp_parameter]],'weight']*hp_dict[hp_parameter]
                else:
                    extra_data.loc[alm_predictor.hp_rest_indices[hp_parameter][hp_dict[hp_parameter]],'weight'] = 0
    
        if runtime['hp_tune_type'] == 'hyperopt':      
            extra_data['weight'] = 0
            for hp_parameter in alm_predictor.hp_parameters['hyperopt']:                     
                if 'weight' in hp_parameter:   
                    extra_data.loc[alm_predictor.hp_indices[hp_parameter][0],'weight'] = 1
                else:
                    extra_data.loc[alm_predictor.hp_rest_indices[hp_parameter][0],'weight'] = 1
                    
            for hp_parameter in alm_predictor.hp_parameters['hyperopt']:                                                       
                if 'weight' in hp_parameter:                    
                    extra_data.loc[alm_predictor.hp_indices[hp_parameter][hp_dict[hp_parameter]],'weight'] = extra_data.loc[alm_predictor.hp_indices[hp_parameter][hp_dict[hp_parameter]],'weight']*hp_dict[hp_parameter]
                else:
                    extra_data.loc[alm_predictor.hp_rest_indices[hp_parameter][hp_dict[hp_parameter]],'weight'] = 0

        ###*********************************************************
        #Apply predictor weight constraints   
        ###*********************************************************      

        if 'm0' in predictor:
            extra_data.loc[extra_data['train_mave_source'] == 1,'weight'] = 0
            
        if 'm1' in predictor:
            extra_data.loc[extra_data['train_mave_source'] == 1,'weight'] = 1

        if 'm2' in predictor: 
            extra_data.loc[extra_data['train_mave_source'] == 1,'weight'] = 0
            extra_data.loc[(extra_data['train_mave_source'] == 1) & (extra_data['accessibility'] > 0),'weight'] = 1    
            
        if 'm3' in predictor: 
            extra_data.loc[extra_data['train_mave_source'] == 1,'weight'] = 0
            extra_data.loc[(extra_data['train_mave_source'] == 1) & (extra_data['accessibility'] == 0),'weight'] = 1       
                        
        if 'h0' in predictor:            
            extra_data.loc[extra_data['train_gnomad_source'] == 1,'weight'] = 0  
            
        if 'h1' in predictor:            
            extra_data.loc[extra_data['train_gnomad_source'] == 1,'weight'] = 1
            
        if 'v0' in predictor:
            extra_data.loc[extra_data['train_humsavar_source'] == 1,'weight'] = 0
            
        if 'v1' in predictor:
            extra_data.loc[extra_data['train_humsavar_source'] == 1,'weight'] = 1
            
        if 'c0' in predictor:
            extra_data.loc[extra_data['train_clinvar_source'] == 1,'weight'] = 0
            
        if 'c1' in predictor:
            extra_data.loc[extra_data['train_clinvar_source'] == 1,'weight'] = 1     
            
        if 'd0' in predictor:
            extra_data.loc[extra_data['train_hgmd_source'] == 1,'weight'] = 0
            
        if 'd1' in predictor:
            extra_data.loc[extra_data['train_hgmd_source'] == 1,'weight'] = 1       
                                                                                        
        return(np.array(extra_data['weight']))
         
    def get_trials(self,trial_file,result_file = None,test_result_file = None,source = 'trials',candidate_offset = 0.0005,trials_cutoff = 1000):
        cur_trials = pickle.load(open(trial_file, "rb"))
        if source == 'trials':
            cur_trials_test_df = None
            cur_trials_df = pd.DataFrame(columns = ['tid','trial','cv_validation_result','increase_flag','candidate_flag'])
            best_trial_performance = 0
            new_tid = 0
            for i in range(len(cur_trials)):
                if cur_trials.trials[i]['result'].get('loss',-1) != -1 :
                    new_tid += 1
                    if new_tid > trials_cutoff:
                        break
                    else:
                        cur_performance  = 1-cur_trials.trials[i]['result']['loss']
                        if cur_performance > best_trial_performance:                        
                            cur_trials_df.loc[i,:] = [i,new_tid,cur_performance,1,0]
                            best_trial_performance = cur_performance
                        else:
                            cur_trials_df.loc[i,:] = [i,new_tid,cur_performance,0,0]
            pass
            cur_trials_df.loc[cur_trials_df['cv_validation_result'] >= best_trial_performance - candidate_offset,'candidate_flag'] = 1            
            total_trials = cur_trials_df.shape[0]  
            
        if source == 'result':      
            cur_trials_df = pd.DataFrame(columns = ['tid','trial','cv_train_result','cv_validation_result','increase_flag','candidate_flag'])
            best_trial_performance = 0              
            new_tid = 0    
            for line in  open(result_file,'r'):
                line_list = line.split('\t')
                new_tid += 1
                if new_tid > trials_cutoff:
                    break
                else:
                    cur_validation_performance  = np.float(line_list[3])
                    cur_train_performance = np.float(line_list[1])
                    cur_trial = int(line_list[0])
                    if cur_validation_performance > best_trial_performance:
                        cur_trials_df.loc[new_tid,:] = [cur_trial,new_tid,cur_train_performance,cur_validation_performance,1,0]                                    
                        best_trial_performance = cur_validation_performance
                    else:
                        cur_trials_df.loc[new_tid,:] = [cur_trial,new_tid,cur_train_performance,cur_validation_performance,0,0]
            cur_trials_df = cur_trials_df.loc[~cur_trials_df['trial'].isnull(),:]            
            cur_trials_df.loc[cur_trials_df['cv_validation_result'] >= best_trial_performance - candidate_offset,'candidate_flag'] = 1
            
            if test_result_file is not None:
                if os.path.isfile(test_result_file):
                    cur_trials_test_df = pd.read_csv(test_result_file)
                    cur_trials_test_df = cur_trials_test_df.loc[cur_trials_test_df['trial'] <= trials_cutoff,:]
                    best_performance = cur_trials_test_df['cv_validation_result'].max()
                    cur_trials_test_df['candidate_flag'] = 0
                    cur_trials_test_df.loc[cur_trials_test_df['cv_validation_result'] >= best_performance -candidate_offset,'candidate_flag'] = 1
                else:
                    cur_trials_test_df = None
            else:
                cur_trials_test_df = None            
            total_trials = cur_trials_df.shape[0]              
        return([cur_trials,cur_trials_df,cur_trials_test_df])         
                          
    def load_cur_hp_dict(self,hp_dict_npy_file):                
        if os.path.isfile(hp_dict_npy_file):
            print('non-default hp_dict loaded!')
            cur_hp_dict =  np.load(hp_dict_npy_file).item()
        else:
            cur_hp_dict = None
            print('default hp_dict loaded!')
        return(cur_hp_dict)
    
    def save_best_hp_dict_from_trials(self,predictor,runtime) :
        alm_predictor = self.proj.predictor[predictor]                
        mv_size = alm_predictor.trials_mv_size    
        trial_file = self.prefix(predictor,runtime,'npy') +'_trials.pkl'
        trial_result_file = self.prefix(predictor,runtime,'csv') + '_trial_results.txt'
        best_hp_dict_file = self.prefix(predictor,runtime,'npy') + '_hp_dict.npy'
        best_hp_df_file = self.prefix(predictor,runtime,'csv') + '_best_hps.csv'

        #********************************************************************************************
        # Save the best trial for each fold
        #********************************************************************************************
        all_hp_df = pd.DataFrame(columns = ['trial','aubprc'] + list(alm_predictor.hp_default.keys()))
        candidate_offset = 0
        select_strategy = 'first_descent_mv_validation_selected_index'
        [cur_trials,cur_trials_df,x] = self.get_trials(trial_file,trial_result_file,None,source = 'result')   
        #**********************************************************
        # first_descent_mv_validation_selected_index
        #**********************************************************
        if select_strategy == 'first_descent_mv_validation_selected_index':      
            sort_type = 'cv_train_result'
            cur_trials_df = cur_trials_df.sort_values([sort_type])
            
            cur_trials_df = cur_trials_df.reset_index()
            cur_trials_df['index'] = cur_trials_df.index
            
            cur_trials_df['mv_cv_train_avg'] = np.nan
            cur_trials_df['mv_cv_validation_avg'] = np.nan
            cur_trials_df['mv_train_avg'] = np.nan
            cur_trials_df['mv_test_avg'] = np.nan
            
            mv_index = 0
            mv_step = mv_size
            
            while 1==1:
                if mv_index + mv_step > cur_trials_df.shape[0] - 1:
                    mv_index = cur_trials_df.shape[0] - 1
                else:
                    mv_index = mv_index + mv_step                       
                    
                mv_low_index = mv_index - mv_size
                cur_mv_cv_train_avg = np.round(cur_trials_df.loc[(cur_trials_df.index < mv_index) & (cur_trials_df.index >= mv_low_index) ,'cv_train_result'].mean(),5)
                cur_mv_cv_validation_avg = np.round(cur_trials_df.loc[(cur_trials_df.index < mv_index) & (cur_trials_df.index >= mv_low_index) ,'cv_validation_result'].mean(),5)                    
                cur_trials_df.loc[mv_index,'mv_cv_train_avg'] = cur_mv_cv_train_avg    
                cur_trials_df.loc[mv_index,'mv_cv_validation_avg'] = cur_mv_cv_validation_avg
                
                if mv_index ==cur_trials_df.shape[0] - 1:
                    break

            cur_trials_mv_df = cur_trials_df.loc[cur_trials_df['mv_cv_train_avg'].notnull(),:]
            cur_trials_mv_df['delta_mv_cv_validation_avg'] = 0            
            cur_trials_mv_df.loc[cur_trials_mv_df.index[:-1],'delta_mv_cv_validation_avg'] =  np.array(cur_trials_mv_df.loc[cur_trials_mv_df.index[1:],'mv_cv_validation_avg']) - np.array(cur_trials_mv_df.loc[cur_trials_mv_df.index[:-1],'mv_cv_validation_avg'])                        
            first_descent_mv_validation_index = np.min(list(cur_trials_mv_df.loc[cur_trials_mv_df['delta_mv_cv_validation_avg'] <= 0,: ].index))
            cur_trials_selected_window_df = cur_trials_df.loc[range((first_descent_mv_validation_index - mv_size),first_descent_mv_validation_index),:]
            first_descent_mv_validation_selected_index= cur_trials_selected_window_df.sort_values(['cv_validation_result']).index[-1]
            
            best_trial_num = cur_trials_df.loc[first_descent_mv_validation_selected_index,'trial']
            best_trial_index = cur_trials_df.loc[first_descent_mv_validation_selected_index,'tid']
            
            
            #******************************************************************************************
            #Plot trials result
            #******************************************************************************************
            fig = plt.figure(figsize=(runtime['fig_x'], runtime['fig_y']))
            ax = plt.subplot()
            marker_offset = 0.001                    
            ax1, = ax.plot(cur_trials_df['index'],cur_trials_df['cv_train_result'],linewidth=3,marker='o', markersize=0,color = 'black')                      
            ax2, = ax.plot(cur_trials_df['index'],cur_trials_df['cv_validation_result'],linewidth=3,marker='o', markersize=0,color = '#558ED5')
            ax3,=ax.plot(cur_trials_mv_df['index'],cur_trials_mv_df['mv_cv_validation_avg'],linewidth=8,marker='o', markersize=22,color = 'darkblue')
            highest_validation_index = cur_trials_df.sort_values(['cv_validation_result']).index[-1]
            ax4, = ax.plot(highest_validation_index,cur_trials_df.loc[highest_validation_index,'cv_validation_result'] + marker_offset + 0.001,linewidth=0,marker='v', markersize=30,color = 'orangered')
            ax5, = ax.plot(first_descent_mv_validation_selected_index,cur_trials_df.loc[first_descent_mv_validation_selected_index,'cv_validation_result'] + marker_offset,linewidth=0,marker='v', markersize=30,color = 'limegreen')
            ax.set_xlim(-10,cur_trials_df.shape[0] + 100)               
            ax.set_title(alm_predictor.name + ' hyperparameter optimization' ,size = 35,pad = 20)
            ax.set_xlabel('Trials', size=30,labelpad = 20)
            ax.set_ylabel('10 Folds ' +  alm_predictor.tune_obj, size=30,labelpad = 20)             
            ax.tick_params(labelsize=25)
            ax.legend([ax4,ax5,ax1,ax2,ax3],['Trial with highest CV  performance on validation sets','Trail picked by VARITY','CV  performance on training sets','CV  performance on validation sets','CV  performance on validation sets \n(moving window average)'], frameon = False,loc = 'upper left',labelspacing = 0.5 ,markerscale = 0.6,fontsize = 20)
            fig.tight_layout()        
            plot_name = self.prefix(predictor, runtime, 'img') + '_hp_selection.png'    
            plt.savefig(plot_name)   
        #**********************************************************
        # pick the best cv  valdaiton performance 
        #**********************************************************
        if select_strategy == 'Best cv performance':              
            best_trial_num = cur_trials_df.loc[cur_trials_df['increase_flag'] == 1,'trial'].max()
            best_trial_index = cur_trials_df.loc[cur_trials_df['trial'] == best_trial_num,'tid'].get_values()[0]            
        #********************************************************************************************************************
        # pick the the ones with lowest training performance in candidates (small difference from the best performance)
        #********************************************************************************************************************     
        if select_strategy == 'Lowest training performance':            
            min_train_performance = cur_trials_df.loc[cur_trials_df['candidate_flag'] == 1,'cv_train_result'].min()
            best_trial_num = cur_trials_df.loc[cur_trials_df['cv_train_result'] == min_train_performance,'trial'].max()
            best_trial_index = cur_trials_df.loc[cur_trials_df['trial'] == best_trial_num,'tid'].get_values()[0]

        cur_hp_dict = self.get_hp_dict_from_trial(best_trial_index,cur_trials,alm_predictor.hyperopt_hps,alm_predictor.hp_default)
        np.save(best_hp_dict_file,cur_hp_dict)
            
        cur_hp_values = [str(cur_hp_dict[x]) + '|' + str(alm_predictor.hp_default[x]) for x in cur_hp_dict.keys()]                        
        best_trial_performance = cur_trials_df.loc[cur_trials_df['tid'] == best_trial_index,'cv_validation_result'].get_values()[0]            
        total_trials = cur_trials_df.shape[0]             
        print ('Selected hyper-paramter point -- Trial: ' + str(best_trial_num) + '/' + str(total_trials) + ', Tid: ' + str(best_trial_index) + ', Performance: ' + str(best_trial_performance))
        all_hp_df.loc[runtime['cur_test_fold'],:] = [str(best_trial_num) + '|' + str(total_trials),str(best_trial_performance)]  + cur_hp_values                            
        all_hp_df.transpose().to_csv(best_hp_df_file)                
        
    def get_hp_dict_from_trial(self,cur_trial_index,cur_trials,hyperopt_hps,hp_default):
        cur_trial = cur_trials.trials[cur_trial_index]                  
        cur_trial_values = cur_trial['misc']['vals'].values()
        cur_trial_values = [x[0] for x in cur_trial_values] 
        cur_trial_keys = list(cur_trial['misc']['vals'].keys())
        cur_trial_for_eval= {cur_trial_keys[i]:cur_trial_values[i] for i in range(len(cur_trial_values))}
        cur_trial_eval = hyperopt.space_eval(hyperopt_hps,cur_trial_for_eval)            
        cur_hp_dict = hp_default.copy()     
        for key in cur_trial_eval.keys():            
            cur_hp_dict[key] = cur_trial_eval[key]       
                         
        return(cur_hp_dict)    

    def score_filter(self,input_df,predictor,runtime):
        self.alm_predictor = self.proj.predictor[predictor]
        if runtime['filter_test_score'] != 0 : # filter out any record has missing value in any of the predictors  when filter_socores is not 0
            for plot_predictor in runtime['compare_predictors']:
                alm_plot_predictor = self.proj.predictor[plot_predictor]
                score_name = alm_plot_predictor.features[0]
                input_df = input_df.loc[input_df[score_name].notnull(), :]
            
        if runtime['filter_test_score'] == 2: # customized filtering 
            input_df = alm_predictor.filter_test(input_df)
        
        return (input_df)        
    
    def plot_classification_curve(self,type,predictors,compare_predictor, cv_folds, filter_indices_dict,test_result_dict,output_file,size_factor = 1,title = '',dpi = 96,table_scale_factor = 3,show_size = 0,legend_bbox = [0.01,0.05,0.8,0.5],extra_info_x = 0.2,extra_info_y = 0.01,fig_x = 30, fig_y = 20):
        plot_score_type = type.split('_')[1]
        plot_score_metric = type.split('_')[0]
        if len(predictors) <= 10 :            
            size_factor = 2 * size_factor
                                
        fig = plt.figure(figsize=(fig_x, fig_y),dpi = dpi)
        plt.clf()
        plt.rcParams["font.family"] = "Helvetica"    
        ax = plt.subplot()
        
        if len(predictors) <= 10 :
            color_lst = ['orangered', 'darkgreen', 'darkblue', 'darkorange', 'darkmagenta', 'darkcyan', 'saddlebrown', 'darkgoldenrod','deeppink','dodgerblue']
        else:
            color_lst = ['#ff0000', '#40e0d0', '#bada55', '#ff80ed', '#696969', '#133337', '#065535', '#5ac18e', '#f7347a', 
                             '#000000', '#420420', '#008080', '#ffd700', '#ff7373', '#ffa500', '#0000ff', '#003366', '#fa8072', 
                             '#800000', '#800080', '#333333', '#4ca3dd','#ff00ff','#008000','#0e2f44','#daa520']
            
        all_dict = {}    
        t_value = 1.833113    
        two_sided = 0 # 0 : one-sided  1: two-sided , for both p value and confidence interval 
        score_types = ['interp','org'] # orginal score or interpreted score
        score_point_metrics = ['precisions','recalls','fprs','tprs']
        score_metrics = ['aubprc','brfp','auroc']
        score_statistics = ['df','mean','se','effect_size','effect_size_se','ci','pvalue','display']

        precision_pivots = list(np.linspace(0.5, 1, 101))               
        tpr_pivots = list(np.linspace(0, 1, 101))     
        color_index = 0
        
        #************************************************************#
        # Data and result for each predictor and each fold
        # Plot curves for each predictor
        #************************************************************#
        
        for predictor in predictors:
            all_dict[predictor] = {}
            all_dict[predictor]['data'] = {}
            all_dict[predictor]['data']['size'] = 0
            all_dict[predictor]['data']['positive_size'] = 0
            all_dict[predictor]['data']['negative_size'] = 0
            all_dict[predictor]['result'] = {}
            all_dict[predictor]['color'] = color_lst[color_index]
            
            for cur_fold in range(cv_folds): 
                all_dict[predictor]['data'][cur_fold] = {}
                all_dict[predictor]['data'][cur_fold]['truth'] = test_result_dict[predictor]['test_y_truth_dict'][cur_fold][filter_indices_dict[cur_fold]]
                all_dict[predictor]['data'][cur_fold]['predicted'] = test_result_dict[predictor]['test_y_predicted_dict'][cur_fold][filter_indices_dict[cur_fold]]
                all_dict[predictor]['data'][cur_fold]['size'] = (~np.isnan(all_dict[predictor]['data'][cur_fold]['predicted'])).sum()
                all_dict[predictor]['data'][cur_fold]['positive_size'] = (~np.isnan(all_dict[predictor]['data'][cur_fold]['predicted']) & (all_dict[predictor]['data'][cur_fold]['truth']  == 1)).sum()
                all_dict[predictor]['data'][cur_fold]['negative_size'] = (~np.isnan(all_dict[predictor]['data'][cur_fold]['predicted']) & (all_dict[predictor]['data'][cur_fold]['truth']  == 0)).sum()
                
                all_dict[predictor]['data']['positive_size'] = all_dict[predictor]['data']['positive_size'] + all_dict[predictor]['data'][cur_fold]['positive_size']
                all_dict[predictor]['data']['negative_size'] = all_dict[predictor]['data']['negative_size'] + all_dict[predictor]['data'][cur_fold]['negative_size']
                all_dict[predictor]['data']['size'] = all_dict[predictor]['data']['size'] + all_dict[predictor]['data'][cur_fold]['size']
                
                
                
                all_dict[predictor]['result'][cur_fold] = {}
                for score_type in score_types:
                    for score_metric in score_point_metrics + score_metrics:
                        all_dict[predictor]['result'][cur_fold][score_type + '_' + score_metric] = np.nan
                
                #interplated precision recall curve 
                metrics_dict = alm_fun.classification_metrics(all_dict[predictor]['data'][cur_fold]['truth'], all_dict[predictor]['data'][cur_fold]['predicted'])[1]                
                cur_fold_interp_recalls =  [alm_fun.get_interpreted_x_from_y(new_precision,metrics_dict['balanced_recalls'],metrics_dict['balanced_precisions'],type = 'last_intersection') for new_precision in precision_pivots]
                cur_fold_interp_recalls.append(0)                
                cur_fold_interp_precisions = precision_pivots
                cur_fold_interp_precisions = cur_fold_interp_precisions + [1]     
                [cur_fold_interp_aubprc,x,cur_fold_interp_brfp,y] = alm_fun.cal_pr_values(cur_fold_interp_precisions,cur_fold_interp_recalls)
                #orginal precision recall curve
                cur_fold_org_recalls = metrics_dict['balanced_recalls']
                cur_fold_org_precisions = metrics_dict['balanced_precisions']
                [cur_fold_org_aubprc,x,cur_fold_org_brfp,y] = alm_fun.cal_pr_values(metrics_dict['balanced_precisions'],metrics_dict['balanced_recalls'])                                
                #interplated roc curve 
                cur_fold_interp_fprs =  [alm_fun.get_interpreted_x_from_y(new_tpr,metrics_dict['fprs'],metrics_dict['tprs'],type = 'last_intersection') for new_tpr in tpr_pivots]
                cur_fold_interp_tprs = tpr_pivots  
                cur_fold_interp_auroc = metrics_dict['auroc']
                #original roc curve
                cur_fold_org_fprs = metrics_dict['fprs']
                cur_fold_org_tprs = metrics_dict['tprs']  
                cur_fold_org_auroc = metrics_dict['auroc']     
                
                all_dict[predictor]['result'][cur_fold]['interp_precisions'] = cur_fold_interp_precisions
                all_dict[predictor]['result'][cur_fold]['interp_recalls'] = cur_fold_interp_recalls
                all_dict[predictor]['result'][cur_fold]['interp_aubprc'] = cur_fold_interp_aubprc
                all_dict[predictor]['result'][cur_fold]['interp_brfp'] = cur_fold_interp_brfp
                
                all_dict[predictor]['result'][cur_fold]['org_precisions'] = cur_fold_org_precisions
                all_dict[predictor]['result'][cur_fold]['org_recalls'] = cur_fold_org_recalls
                all_dict[predictor]['result'][cur_fold]['org_aubprc'] = cur_fold_org_aubprc
                all_dict[predictor]['result'][cur_fold]['org_brfp'] = cur_fold_org_brfp
                                    
                all_dict[predictor]['result'][cur_fold]['interp_fprs'] = cur_fold_interp_fprs
                all_dict[predictor]['result'][cur_fold]['interp_tprs'] = cur_fold_interp_tprs
                all_dict[predictor]['result'][cur_fold]['interp_auroc'] = cur_fold_interp_auroc
                
                all_dict[predictor]['result'][cur_fold]['org_fprs'] = cur_fold_org_fprs
                all_dict[predictor]['result'][cur_fold]['org_tprs'] = cur_fold_org_tprs
                all_dict[predictor]['result'][cur_fold]['org_auroc'] = cur_fold_org_auroc

            # data points for the curve 
            for score_type in score_types:
                for score_point_metric in score_point_metrics:                     
                    cur_score_list = []
                    for cur_fold in range(cv_folds): 
                        if (cv_folds > 1) & (score_type == 'org'):
                            cur_score_list.append(all_dict[predictor]['result'][cur_fold]['interp_' + score_point_metric])  
                        else:
                            cur_score_list.append(all_dict[predictor]['result'][cur_fold][score_type + '_' + score_point_metric])
                                              
                    all_dict[predictor]['result'][score_type + '_' + score_point_metric]= np.mean(cur_score_list,axis= 0)
                    all_dict[predictor]['result'][score_type + '_' + score_point_metric +'_se']= np.std(cur_score_list,axis= 0)/np.sqrt(cv_folds)
                    if score_type + '_' + score_point_metric in  ['interp_recalls','interp_fprs']:
                        all_dict[predictor]['result'][score_type + '_' + score_point_metric +'_upper']= np.minimum(all_dict[predictor]['result'][score_type + '_' + score_point_metric] + all_dict[predictor]['result'][score_type + '_' + score_point_metric +'_se'], 1)
                        all_dict[predictor]['result'][score_type + '_' + score_point_metric +'_lower']= np.maximum(all_dict[predictor]['result'][score_type + '_' + score_point_metric] - all_dict[predictor]['result'][score_type + '_' + score_point_metric +'_se'], 0)
                        
            #**************************************************************
            # 10 folds CV AUBPRC plot with interpreted precision recall values
            #**************************************************************
            if plot_score_metric == 'auroc':
                ax = alm_fun.plot_cv_roc_ax(all_dict[predictor]['result']['interp_fprs'],all_dict[predictor]['result']['interp_tprs'],all_dict[predictor]['result']['interp_fprs_upper'],all_dict[predictor]['result']['interp_fprs_lower'],ax,color = all_dict[predictor]['color'],size_factor = size_factor)
            if plot_score_metric == 'aubprc':
                ax = alm_fun.plot_cv_prc_ax(all_dict[predictor]['result']['interp_recalls'],all_dict[predictor]['result']['interp_precisions'],all_dict[predictor]['result']['interp_recalls_upper'],all_dict[predictor]['result']['interp_recalls_lower'],ax,color = all_dict[predictor]['color'],size_factor = size_factor)
            color_index += 1     
        
        #*******************************************************************
        # Get statistics for each predictor when compare to compare_predictor
        #*******************************************************************                
        for predictor in predictors:
            for score_type in score_types:
                for score_metric in score_metrics:
                    compare_score_list = []
                    cur_score_list = []                    
                    for cur_fold in range(cv_folds):   
                        compare_score_list.append(all_dict[compare_predictor]['result'][cur_fold][score_type + '_' + score_metric])                            
                        cur_score_list.append(all_dict[predictor]['result'][cur_fold][score_type + '_' + score_metric])                                     
                    all_dict[predictor]['result'][score_type + '_' + score_metric + '_df']= cv_folds -1                        
                    all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean']= np.mean(cur_score_list)                    
                    all_dict[predictor]['result'][score_type + '_' + score_metric +'_se']= np.std(cur_score_list)/np.sqrt(cv_folds)
                    if two_sided == 0:                        
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_pvalue']= stats.ttest_rel(compare_score_list,cur_score_list)[1]/2 # one sided test
                    else:
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_pvalue']= stats.ttest_rel(compare_score_list,cur_score_list)[1] # one sided test
                    all_dict[predictor]['result'][score_type + '_' + score_metric +'_effect_size'] = np.mean(np.array(compare_score_list) - np.array(cur_score_list))
                    all_dict[predictor]['result'][score_type + '_' + score_metric +'_effect_size_se'] = np.std(np.array(compare_score_list) - np.array(cur_score_list))/np.sqrt(cv_folds)
                    all_dict[predictor]['result'][score_type + '_' + score_metric +'_ci'] = "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric +'_effect_size'] - t_value*all_dict[predictor]['result'][score_type + '_' + score_metric +'_effect_size_se']) + ' ~ inf'
                    
                    if predictor == compare_predictor:
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_display'] =  "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean']) + '±' + "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_se'])
                    else:                    
#                         all_dict[predictor]['result'][score_type + '_' + score_metric +'_display'] =  "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean']) + '±' + "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_se']) + ' (' + all_dict[predictor]['result'][score_type + '_' + score_metric +'_ci'] +',' + "{:.2e}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_pvalue']) +')'                                    
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_display'] =  "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean']) + '±' + "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_se']) + ' [' + "{:.2e}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_pvalue']) +']'

                    if cv_folds == 1:
                         all_dict[predictor]['result'][score_type + '_' + score_metric +'_display'] =  "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean'])
        #**************************************************************************************************************************                        
        # a special case that CADD and VARITY_R_LOO have same AUROC when round to 3 digits, sort as if VARITY_R_LOO is bigger than CADD          
        #**************************************************************************************************************************  
#         if ('VARITY_R_LOO' in predictors) & ('CADD_raw' in predictors):
#             if np.round(mean_aurocs[predictors.index('VARITY_R_LOO')],3)== np.round(mean_aurocs[predictors.index('CADD_raw')],3):
#                 mean_aurocs[predictors.index('CADD_raw')] = mean_aurocs[predictors.index('CADD_raw')] - 0.0003
                
                
        #**************************************************************************************************************************                        
        # Sort the predictor with AUBPRC       
        #**************************************************************************************************************************
        sort_metrics = 'interp_aubprc_mean'
        sort_metric_list = []
        for predictor in predictors:
            sort_metric_list.append(all_dict[predictor]['result'][sort_metrics])                                                                                                                            
        sort_indices = list(np.argsort(sort_metric_list))
        sort_indices.reverse()    
        predictors = [predictors[i] for i in sort_indices]    
#         colors =   [colors[i] for i in sort_indices]   
        
        #**************************************************************************************************************************                        
        # Create Figure Legend   (output AUBUPRC and AUROC)
        #**************************************************************************************************************************        
        legend_data = []
        columns = 0
        if len(predictors) > 10 :
            n_rows = int(len(predictors)/2)
            columns = 2       
        else:
            n_rows = len(predictors)
            columns = 1
     
        # set legend table data             
        for i in range(n_rows):
            cur_legend_info = []
            legend_col_label = []            
            for column in range(columns):                        
                if plot_score_metric == 'auroc':
                    legend_col_label.append('Method')
                    cur_legend_info.append(predictors[i+column*n_rows])
                    legend_col_label.append('AUROC')
                    cur_legend_info.append(all_dict[predictors[i+ column*n_rows]]['result'][plot_score_type + '_' + plot_score_metric +'_display'])                                                                                                                                                                                                                                                                                 
                if plot_score_metric == 'aubprc':
                    legend_col_label.append('Method')
                    cur_legend_info.append(predictors[i+column*n_rows])
                    legend_col_label.append('AUBPRC')
                    cur_legend_info.append(all_dict[predictors[i+column*n_rows]]['result'][plot_score_type + '_' + plot_score_metric +'_display'])
                    legend_col_label.append('R90BP')
                    cur_legend_info.append(all_dict[predictors[i+column*n_rows]]['result'][plot_score_type + '_brfp_display'])
                if show_size == 1:
                    legend_col_label.append('Size')
                    cur_legend_info.append('[' +  str(all_dict[predictors[i+column*n_rows]]['data']['size']) + ',' + 'P: ' + str(all_dict[predictors[i+column*n_rows]]['data']['positive_size']) + ' N: ' + str(all_dict[predictors[i+column*n_rows]]['data']['negative_size']) + ']')         
            legend_data.append(cur_legend_info)
                          
        if plot_score_metric == 'auroc':
#             legend_table = ax.table(cellText = legend_data,colLabels = legend_col_label,bbox = legend_bbox)
            legend_table = ax.table(cellText = legend_data,colLabels = legend_col_label,loc ='lower right')            
        if plot_score_metric == 'aubprc':
            legend_table = ax.table(cellText = legend_data,colLabels = legend_col_label,loc ='lower left')
#             legend_table = ax.table(cellText = legend_data,colLabels = legend_col_label,bbox =legend_bbox)
             
        # set legend table text color   
        for i in range(n_rows):            
            for column in range(columns):                                                                                                                         
                if plot_score_metric == 'auroc':         
                    legend_table._cells[(i+1,column*(2+show_size))].get_text().set_color(all_dict[predictors[i + column*n_rows]]['color'])
                if plot_score_metric == 'aubprc':
                    legend_table._cells[(i+1,column*(3+show_size))].get_text().set_color(all_dict[predictors[i + column*n_rows]]['color'])      
                    
        # set legend table scale and text font size 
        legend_table.set_fontsize(25*size_factor)
        legend_table.auto_set_font_size(False)
        legend_table.scale(1, table_scale_factor*size_factor)    
        for i in range(len(legend_col_label)):  
            legend_table._cells[(0,i)].set_fontsize(30*size_factor)        
        
        # set legend table column width (automatic)   
        for i in range(len(legend_col_label)):    
            legend_table.auto_set_column_width(i)
               
        # set legend table aligment                
        for i in range(1,n_rows+1):
            for j in range(len(legend_col_label)):
                   legend_table._cells[(i,j)]._loc ='left'       
        
        # set plot title                       
        if title == '':
            if plot_score_metric == 'aubprc':
                    title = 'Balanced Precision Recall Curve \n' + compare_predictor
            if plot_score_metric == 'auroc':
                    title = 'ROC Curve \n' + compare_predictor                  


#         # shwo extra_info (such size of the test space) 
#         extra_info = 'Number of Variants Tested: ' + f"{all_dict[predictors[0]]['data']['size']:,}" + ' [Postive: ' + f"{all_dict[predictors[0]]['data']['positive_size']:,}" + ', Negative: ' + f"{all_dict[predictors[0]]['data']['negative_size']:,}"  + ']'       
#         ax.text(extra_info_x,extra_info_y,extra_info,fontsize =25*size_factor)
                                
        ax.set_title(title, size=40*size_factor, pad = 20)
        fig.tight_layout()
        plt.savefig(output_file)  
        
        output_cols = ['predictor']
        for score_type in score_types:
            for score_metric in score_metrics:
                for score_statistic in score_statistics:
                    output_cols.append(score_type + '_' + score_metric + '_' + score_statistic)
        score_output = pd.DataFrame(columns = output_cols)      
          
        for predictor in predictors:
            cur_score_list = []     
            for score_type in score_types:
                for score_metric in score_metrics:
                    for score_statistic in score_statistics:    
                        cur_score_list.append(all_dict[predictor]['result'][score_type + '_' + score_metric + '_' + score_statistic])                                                        
            score_output.loc[predictor] = [predictor] + cur_score_list                                           
        score_output.to_csv(output_file[:-4].replace('img','csv') + '.csv')

    def plot_test_result(self,predictor,runtime): 
        alm_predictor = self.proj.predictor[predictor]
        if runtime['cur_test_fold'] == -1:
            test_folds = list(range(alm_predictor.data_instance.test_split_folds))
        else:
            test_folds = [runtime['cur_test_fold']]        
                    
        # Get the filter_indices for each test fold
        filter_indices_dict = {}
        filter_predictors = runtime['compare_predictors']        

        for cur_fold in test_folds:                    
            #apply test set filtering                                
            cur_test_fold_df = alm_predictor.data_instance.train_data_df.loc[alm_predictor.data_instance.test_splits_df[cur_fold]['no_gradient'],:]
            cur_filtered_test_fold_df = self.score_filter(cur_test_fold_df,predictor,runtime)
            filter_indices_dict[cur_fold] = cur_filtered_test_fold_df.index
                    
        # Get the test result for each predictor 
        test_result_dict = {}
        for plot_predictor in runtime['compare_predictors'] + [runtime['predictor']]:                                                                                                    
            cur_test_result_dict_file = self.prefix(plot_predictor, runtime, 'npy',1) + '_test_cv_results.npy'
            cur_test_result_dict = np.load(cur_test_result_dict_file).item()      
            test_result_dict[plot_predictor] = cur_test_result_dict
            

        for plot_metric in runtime['plot_metric']:
            # Define the the output file name
            if len(runtime['compare_predictors']) > 10 :            
                output_file = self.prefix(predictor, runtime, 'img')+ '_filter' + '_' + str(runtime['filter_test_score'])  +  '_' + plot_metric + '.png'  
            else:
                output_file = self.prefix(predictor, runtime, 'img') + '_filter' + '_' + str(runtime['filter_test_score'])  +  '_' + plot_metric + '.png'
                
            if len(runtime['compare_predictors']) == 3 :
                output_file = self.prefix(predictor, runtime, 'img') + '_filter' + '_' + + str(runtime['filter_test_score'])  +  '_' + plot_metric + '.png'
                
                
            self.plot_classification_curve(plot_metric,runtime['compare_predictors'] + [predictor],predictor,len(test_folds), filter_indices_dict,test_result_dict,output_file,size_factor = 0.95,table_scale_factor= 3, show_size = 0,fig_x = runtime['fig_x'],fig_y = runtime['fig_y'] )

    def plot_sp_result(self,predictor,runtime):       
         
        alm_predictor = self.proj.predictor[predictor]
        cols = ['window','train_' + alm_predictor.tune_obj,'train_' + alm_predictor.tune_obj + '_ste','validation_' + alm_predictor.tune_obj ,'validation' + alm_predictor.tune_obj]
        cur_metric = 'validation_' + alm_predictor.tune_obj
        data_name_sp = alm_predictor.data
        cur_parameter = runtime['filtering_hp']                
        k = 0

        fig = plt.figure(figsize=(runtime['fig_x'],runtime['fig_y']))
        plt.rcParams["font.family"] = "Helvetica"  
        plt.clf()
                    
        sorted_result_file = self.prefix(predictor, runtime, 'csv',1)  + '_' + cur_parameter + '_spvalue_results.txt'
        cur_sorted_hp_df = pd.read_csv(sorted_result_file,sep = '\t',header = None)
        cur_sorted_hp_df.columns = cols
        cur_sorted_hp_df['range_start'] = np.nan
        cur_sorted_hp_df['range_end'] = np.nan

        property = alm_predictor.hps[cur_parameter]['orderby']
        if   len(alm_predictor.hps[cur_parameter]['source']) == 1:                                       
            addon_set_name = alm_predictor.hps[cur_parameter]['source'][0]
        else:
            addon_set_name = '&'.join(alm_predictor.hps[cur_parameter]['source'])
        mv_size_percent = alm_predictor.hps[cur_parameter]['mv_size_percent']
        mv_data_points =  alm_predictor.hps[cur_parameter]['mv_data_points']                 
        cur_direction = alm_predictor.hp_directions[cur_parameter]            
        window_size = int(len(alm_predictor.hp_mv_indices[cur_parameter][mv_data_points+1])*mv_size_percent/100)                
        if cur_direction == 0:
            direction = 'high to low'
        if cur_direction == 1:
            direction = 'low to high'
                        
        varity_0_score_df = cur_sorted_hp_df.loc[cur_sorted_hp_df.index[0],:]
        varity_1_score_df = cur_sorted_hp_df.loc[cur_sorted_hp_df.index[-1],:]
        varity_0_score = varity_0_score_df[cur_metric]
        varity_1_score = varity_1_score_df[cur_metric]
        cur_sorted_hp_df = cur_sorted_hp_df.loc[cur_sorted_hp_df.index[1:-1],:]
                                
        sorted_score_mean = cur_sorted_hp_df[cur_metric].mean()
        sorted_score_se = cur_sorted_hp_df[cur_metric].std()/np.sqrt(cur_sorted_hp_df.shape[0])
        sorted_ci_plus = 1.96*sorted_score_se
        sorted_ci_minus = 0 - 1.96*sorted_score_se                    

        cur_sorted_hp_df[cur_metric + '_diff_base'] = cur_sorted_hp_df[cur_metric] - varity_0_score            
        
        ax = plt.subplot()
        ax.plot(cur_sorted_hp_df['window'],cur_sorted_hp_df[cur_metric],linewidth=6,marker='o', markersize=10,color = '#558ED5')
                                
        max_index = cur_sorted_hp_df.loc[cur_sorted_hp_df[cur_metric] == cur_sorted_hp_df[cur_metric].max(),:].index[0]
        sorted_spc = alm_fun.spc_cal (cur_sorted_hp_df['window'],cur_sorted_hp_df[cur_metric])            
        sorted_low_spc = alm_fun.spc_cal (cur_sorted_hp_df.loc[1:max_index,'window'],cur_sorted_hp_df.loc[1:max_index,cur_metric])            
        sorted_high_spc = alm_fun.spc_cal (cur_sorted_hp_df.loc[max_index:mv_data_points,'window'],cur_sorted_hp_df.loc[max_index:mv_data_points,cur_metric])
                                                                
#             ax.set_xticks([1,20,40,60,80,100])
        ax.plot([cur_sorted_hp_df['window'].min(),cur_sorted_hp_df['window'].max()],[sorted_score_mean,sorted_score_mean],color = 'black')
        ax.plot([cur_sorted_hp_df['window'].min(),cur_sorted_hp_df['window'].max()],[sorted_score_mean + sorted_ci_plus,sorted_score_mean + sorted_ci_plus],ls = '-.',color = 'black')
        ax.plot([cur_sorted_hp_df['window'].min(),cur_sorted_hp_df['window'].max()],[sorted_score_mean + sorted_ci_minus,sorted_score_mean + sorted_ci_minus],ls = '-.',color = 'black')     
        ax.set_ylabel('10 Folds AUBPRC',size = 28,labelpad = 10)
        ax.set_xlabel('Moving Windows (Window size: ' + str('{:,}'.format(window_size))  +', PCC: ' + str('{:.3f}'.format(sorted_spc)) +  ')', size = 28,labelpad = 10)
        ax.set_title('Add-on set: ' + addon_set_name + '\n (Examples ordered by ' + property + ' ' + direction + ')' ,size = 32,pad = 20)
        ax.tick_params(labelsize=20)
        
        fig.tight_layout(pad = 3)
        plt.savefig(self.prefix(predictor, runtime, 'img',1)  + '_' + cur_parameter + '_mv_result.png')

        print ('OK')
                  
    def plot_extra_data(self,predictor,runtime):
        runtime['hp_tune_type'] = 'hyperopt'
        alm_predictor = self.proj.predictor[predictor]        
        cur_hp_npy = self.prefix(predictor,runtime,'npy')+ '_hp_dict.npy'       
        cur_hp_dict = np.load(cur_hp_npy).item()
        alpha = self.update_sample_weights(cur_hp_dict,predictor,runtime)            
        extra_data = alm_predictor.data_instance.extra_train_data_df_lst[0].copy() 

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.summer_r)
        extra_data['weight_color'] = np.nan
        for weight in extra_data['weight'].unique():
            weight_color = mapper.to_rgba(weight)
            extra_data.loc[extra_data['weight'] == weight,'weight_color'] =  matplotlib.colors.rgb2hex(weight_color)
        color_full_weight = matplotlib.colors.rgb2hex(mapper.to_rgba(1))

        hp_parameter = runtime['filtering_hp']
        hp_full_length_key = list(alm_predictor.hp_indices[hp_parameter].keys())[-1]
        data_indices = alm_predictor.hp_indices[hp_parameter][hp_full_length_key]
        
        if alm_predictor.hp_directions[hp_parameter] == 0:
            data_indices = list(data_indices)[::-1]            
        extra_data_df = extra_data.loc[data_indices,:]             

        fig = plt.figure(figsize=(runtime['fig_x'], runtime['fig_y']))           
        ax = plt.subplot()

        extra_data_df['height'] = 1
        orderby = alm_predictor.hps[hp_parameter]['orderby']
        source = alm_predictor.hps[hp_parameter]['source']
        if len(source) == 1:                
            addon_set = source[0]
        else:
            addon_set = '&'.join(source)
            
        ax.bar(range(1,extra_data_df.shape[0]+1), extra_data_df[orderby],color = extra_data_df['weight_color'],width = 1)
        ax.set_ylabel(orderby,size = 25,labelpad = 15, fontweight='bold') 
        ax.set_xlabel('Rank of each ordered variant',size = 25,labelpad = 15,fontweight='bold')                                    
        ax.set_title('Add-on set: ' + addon_set + ' order by ' + orderby,size = 32,pad = 20,fontweight='bold')                
        ax.tick_params(labelsize=25)    
        fig.tight_layout(pad = 3)
        fig.subplots_adjust(right = 0.88)                              
        cbar_ax = fig.add_axes([0.90, 0.2, 0.02,0.6])        
        cb = fig.colorbar(mapper, cax=cbar_ax)     
        cb.set_label('Weight',size = 30,fontweight='bold',labelpad = 15)
        cb.ax.tick_params(labelsize=25)
        plt.savefig(self.prefix(predictor,runtime,'img') + '_' + hp_parameter + '_hp_weight.png')  
                                             

