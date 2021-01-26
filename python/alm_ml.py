#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import pandas as pd
import csv
import os
import re
import glob
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
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sklearn
from matplotlib import colors, ticker
# from matplotlib.patches import Rectangle
# import matplotlib.path as mpath
# import matplotlib.patches as patches
# from matplotlib.lines import Line2D
# from matplotlib.gridspec import GridSpec
# import matplotlib.collections as collections
# from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 1000)

class alm_ml:
    def __init__(self, ml_init_params):
        for key in ml_init_params:
            setattr(self, key, ml_init_params[key])
        msg = "Class: [alm_ml] [__init__]......done @" + str(datetime.now())
        
        #parameters that are not open for configuration yet
        self.verbose = 1
        self.run_grid_search = 0
        fs_start_feature = None
        fs_type = 'local search'
        fs_T = 0.001
        fs_alpha = 0.8
        fs_K = 100
        fs_epsilon = 0.00001
        
        alm_fun.show_msg(self.log, self.verbose, msg)    

    def run_batch_id_jobs(self,runtime):
        #### Check how many Uniprot IDs need to run        
        exist_ids = []
        all_ids = list(pd.read_csv(runtime['batch_id_file'])[runtime['batch_id_name']])
        for exist_file in glob.glob(os.path.join(runtime['batch_id_exist_files'])):
            if os.stat(exist_file).st_size != 0:
                exist_ids.append(exist_file.split('/')[-1].split('.')[0].split('_')[0])
        run_ids = list(set(all_ids) - set(exist_ids))
        varity_run_ids_df = self.fun_id_batches(run_ids, runtime)
        varity_batch_ids = list(varity_run_ids_df['varity_batch_id'].unique())
        
        ##### Fire parallel jobs 
        cur_jobs= {}                                                  
        cur_log = self.project_path + 'output/log/run_batch_id_jobs_' + runtime['varity_action'] + '_' + runtime['batch_id'] + '.log'
        alm_fun.show_msg (cur_log,1, '# of ids: ' + str(len(run_ids)) + ', # of batches: ' + str(len(varity_batch_ids)))                   
        for varity_batch_id in varity_batch_ids:              
            new_runtime = runtime.copy()            
#             new_runtime[runtime['batch_id_name'] + 's'] = str(list(varity_run_ids_df.loc[varity_run_ids_df['varity_batch_id'] == varity_batch_id,runtime['batch_id_name']].unique())).replace("'","")                        
            new_runtime['varity_batch_id'] = varity_batch_id                                    
            new_runtime['run_on_node'] = 1
            new_runtime['hp_dict_file'] = self.fun_perfix(new_runtime, 'npy',target_action = 'hp_tuning') + '_hp_dict.npy'   
            new_runtime['action'] = runtime['varity_action']                        
            new_runtime['job_name'] = runtime['varity_action']  + '_' + new_runtime['batch_id'] +  '_' + varity_batch_id                
            new_runtime['cur_result'] = self.project_path + 'output/log/' + runtime['varity_action'] + '_' + varity_batch_id + '_done.log'                        
            if not os.path.isfile(new_runtime['cur_result']):
                alm_fun.show_msg (cur_log,1, 'Processing batch'  + str(varity_batch_id) + '......')                                                                                      
                [job_id,job_name,result_dict] = self.varity_obj.varity_action(new_runtime)
                alm_fun.show_msg (cur_log,1, 'Job ID:' + str(job_id) + ', Job Name: ' + job_name)
                cur_jobs[job_name] = []
                cur_jobs[job_name].append(new_runtime['cur_result'])
                cur_jobs[job_name].append(job_id)
                cur_jobs[job_name].append(new_runtime)  
            else:
                alm_fun.show_msg (cur_log,1, 'Varity genes batch ' + str(varity_batch_id) + ' result is available.')
         
        if self.varity_obj.fun_monitor_jobs(cur_jobs,cur_log,runtime) == 1:
            alm_fun.show_msg (cur_log,1, 'Batch: '  +  runtime['batch_id'] + 'all varity batch ids are processed.')
            
    def fun_id_batches(self,ids,runtime):   
        
        #************************************************************************************************************************************************************************
        ### arrange batch so that can run on cluster (parallel jobs)
        #************************************************************************************************************************************************************************                
        parallel_batches = runtime['parallel_batches']
        varity_batchid_df = pd.DataFrame()
        varity_batchid_df[runtime['batch_id_name']] = ids        
        varity_batchid_df['varity_batch_id'] = np.nan
        
        #### total length of all the ids
        varity_batch = str(alm_fun.get_random_id(4))  
        total_ids_num = len(ids)
        linspaces = np.linspace(0,total_ids_num,parallel_batches+1,dtype = int)
        indices = varity_batchid_df.index 
        for i in range(len(linspaces)-1):
            varity_batchid_df.loc[indices[linspaces[i]:linspaces[i+1]],'varity_batch_id'] = varity_batch + '_' + str(i)
        pass
#         varity_batchid_df['varity_batch_id'].value_counts()
#         avg_batch_ids_num = int(total_ids_num/parallel_batches)
#         
#       
#         cur_varity_batch_id = 1
#         cur_total_batch_num = 0
#         for cur_index in varity_batchid_df.index:            
#             varity_batchid_df.loc[cur_index,'varity_batch_id'] = varity_batch + '_' + str(cur_varity_batch_id)
#             cur_total_batch_num= cur_total_batch_num + 1
#             if cur_total_batch_num >= avg_batch_ids_num :
#                 cur_total_batch_num = 0
#                 cur_varity_batch_id = cur_varity_batch_id + 1   
                     
        varity_batchid_df[[runtime['batch_id_name'],'varity_batch_id']].to_csv(self.project_path + 'output/csv/' + runtime['varity_action'] + '_' +  runtime['batch_id'] + '_varity_batch_id.csv',index = False)        
        return(varity_batchid_df)                  

    def fun_perfix(self,runtime, folder, cv_flag = 0, with_path = 1, target_action = '',ignore_trails_mv_size = 0):
        
        if target_action  == '':
            target_action = runtime['action']
            
        if (runtime['trials_mv_size'] == -1) | (ignore_trails_mv_size == 1) :
            predictor_name = runtime['predictor'] 
        else:
            predictor_name = runtime['predictor'] + '_mv_' + str(runtime['trials_mv_size'])
            
#         if runtime['hp_tune_type'] == 'hyperopt_logistic':                    
        if (cv_flag == 1) | (runtime['cur_test_fold'] == -1):                    
            prefix = self.project_path +'/output/' + folder + '/' + runtime['session_id'] + '_' + target_action + '_' + predictor_name
        else:
            prefix = self.project_path +'/output/' + folder + '/' + runtime['session_id'] + '_' + target_action + '_' + predictor_name + '_tf' + str(runtime['cur_test_fold'])
            
#         else:            
#             if (cv_flag == 1) | (runtime['cur_test_fold'] == -1):                    
#                 prefix = self.project_path +'/output/' + folder + '/' + runtime['session_id'] + '_' + predictor_name
#             else:
#                 prefix = self.project_path +'/output/' + folder + '/' + runtime['session_id'] + '_' + predictor_name + '_tf' + str(runtime['cur_test_fold'])
#                             
#                 
        if with_path == 0:
            prefix = prefix.split('/')[-1]
        return(prefix)    

    def weights_opt_hyperopt(self,runtime):        
        
        alm_predictor = self.proj.predictor[runtime['predictor']]
        self.hyperopt_predictor = alm_predictor
        self.hyperopt_runtime = runtime             
        alm_fun.show_msg (runtime['log'],1,"***************************************************************")
        alm_fun.show_msg (runtime['log'],1,"Hyperopt")
        alm_fun.show_msg (runtime['log'],1,"Predictor: " + alm_predictor.name)
        alm_fun.show_msg (runtime['log'],1,"Fold: " + str(runtime['cur_test_fold']))
        alm_fun.show_msg (runtime['log'],1,"Tune obj: " + alm_predictor.tune_obj)
        alm_fun.show_msg (runtime['log'],1,"Session id: " + runtime['session_id'])
        alm_fun.show_msg (runtime['log'],1,"Data name: " + alm_predictor.data)
        alm_fun.show_msg (runtime['log'],1,"Start Time: " + str(datetime.now()))        
        alm_fun.show_msg (runtime['log'],1,"***************************************************************")

        #********************************************************************************************
        # Define hyperopt search space
        #********************************************************************************************
        hyperopt_hps = alm_predictor.hyperopt_hps 
        available_trials_file = self.fun_perfix(runtime,'npy') + '_trials.pkl'
        available_trials_result_file = self.fun_perfix(runtime,'csv') + '_trial_results.txt'
        if  os.path.isfile(available_trials_file):                        
            [cur_trials_result,cur_trials_df,X] = self.get_trials(available_trials_file,available_trials_result_file)
            alm_fun.show_msg (runtime['log'],1,"Previous trials have been loaded.")            
            if cur_trials_df is None:
                cur_trials_result = hyperopt.Trials()
                new_max_evals = alm_predictor.hyperopt_trials
            else:
                cur_max_trial_num = cur_trials_df['trial'].max() 
                cur_max_tid_num =   len(cur_trials_result)
                new_max_evals = alm_predictor.hyperopt_trials - cur_max_trial_num + cur_max_tid_num 
                alm_fun.show_msg (runtime['log'],1,"total effective trials: " + str(cur_max_trial_num)   + " total ran trials : " + str(cur_max_tid_num) + ' new max evals: '  + str(new_max_evals))
                 
        else:
            cur_trials_result = hyperopt.Trials()
            new_max_evals = alm_predictor.hyperopt_trials
            
        self.cur_trials_result = cur_trials_result            
        best_hyperopt = hyperopt.fmin(self.fun_validation_cv_prediction_hyperopt,hyperopt_hps,algo = hyperopt.tpe.suggest,show_progressbar= False,max_evals = new_max_evals,trials = self.cur_trials_result)
        pickle.dump(cur_trials_result, open(trial_file, "wb"))
        
        self.save_best_hp_dict_from_trials(runtime)
                   
        alm_fun.show_msg (runtime['log'],1,"End Time: " + str(datetime.now()))        
        alm_fun.show_msg (runtime['log'],1,"***************************************************************")
    
    
    
    def weights_opt_sp(self,runtime):
        alm_predictor = self.proj.predictor[runtime['predictor']]
        filtering_hp_values = alm_predictor.hp_mv_values[runtime['filtering_hp']]

        for filtering_hp_value in filtering_hp_values:
            new_runtime = runtime.copy()
            new_runtime['filtering_hp_value'] = filtering_hp_value
            alm_fun.show_msg (runtime['log'],self.verbose,'Running moving analysis on ' + runtime['filtering_hp'] + ' at moving window ' + str(filtering_hp_value) + '......')            
            self.fun_validation_cv_prediction_sp(new_runtime)
            
                
    def fun_mv_analysis(self,runtime):
        alm_predictor = self.proj.predictor[runtime['predictor']]        
        num_mv_windows = alm_predictor.qip[runtime['mv_qip']]['mv_data_points']
        cur_log = cur_log = self.project_path + 'output/log/fun_mv_analysis' + runtime['batch_id'] + '.log'
        cur_jobs= {}
        for mv_id in ['none'] + list(range(num_mv_windows)) + ['all']:
            new_runtime = runtime.copy()                                                  
            new_runtime['hp_tune_type'] = 'mv_analysis'  
            new_runtime['mv_id'] = mv_id
            new_runtime['run_on_node'] = 1    
            new_runtime['action'] = 'test_cv_prediction'                                                       
            new_runtime['cur_fold_result'] = self.fun_perfix(new_runtime, 'csv',1) + '_hp_test_cv_result_' + runtime['mv_qip'] + '_mv_' + str(mv_id) + '.csv'                
            new_runtime['hp_dict_file'] = 'na'
            new_runtime['batch_id'] =  str(alm_fun.get_random_id(10))  
            new_runtime['job_name'] = self.varity_obj.set_job_name(new_runtime)            
#             if not os.path.isfile(new_runtime['cur_fold_result']):
            alm_fun.show_msg (cur_log,1, 'Run prediction on moving window ID: '  + str(mv_id) + '......')                                                 
            if (runtime['cluster'] == 1) :                                      
                [job_id,job_name,result_dict] = self.varity_obj.varity_action(new_runtime)
                cur_jobs[job_name] = []
                cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                cur_jobs[job_name].append(job_id)
                cur_jobs[job_name].append(new_runtime)                     
            else:                             
                self.fun_test_cv_prediction(new_runtime)       
                
        if runtime['cluster'] == 1:
            if self.varity_obj.fun_monitor_jobs(cur_jobs,cur_log,runtime) == 1:
                alm_fun.show_msg (cur_log,1, 'Batch: '  +  runtime['batch_id'] + ' all results are done,start to gathering results......')                              
        
        mv_analysis_result_file = self.fun_perfix(runtime, 'csv',1) + '_' + runtime['mv_qip'] + '.csv'
        for mv_id in ['none'] + list(range(num_mv_windows)) + ['all']:  
            cur_mv_result = pd.read_csv(self.fun_perfix(runtime, 'csv',1,target_action = 'test_cv_prediction') + '_hp_test_cv_result_' + runtime['mv_qip'] + '_mv_' + str(mv_id) + '.csv')
            if mv_id == 'none':
                cur_mv_result.to_csv(mv_analysis_result_file,index = False)                                
            else:
                cur_mv_result.to_csv(mv_analysis_result_file,mode = 'a', header = False ,index = False)
            
        
    def fun_test_cv_prediction(self,runtime):        
        alm_predictor = self.proj.predictor[runtime['predictor']]
        test_split_folds = alm_predictor.data_instance.test_split_folds
                              
        #**********************************************************************
        # Fire parallel jobs for all test folds
        #**********************************************************************
        cur_jobs= {}                   
        alm_fun.show_msg (runtime['log'],1,'Start to run test cv prediction, batch id: ' + runtime['batch_id'] + '......' )        
        for cur_test_fold in range(test_split_folds): 
            new_runtime = runtime.copy()                          
            new_runtime['cur_test_fold']  = cur_test_fold
            new_runtime['single_fold_type']  = 'test'
#             new_runtime['hp_tune_type'] = 'hyperopt'      
            new_runtime['run_on_node'] = 1    
            new_runtime['action'] = 'single_fold_prediction'                                                       
            new_runtime['cur_fold_result'] = self.fun_perfix(new_runtime, 'npy_temp')+  '_' + new_runtime['batch_id'] + '_hp_test_single_fold_result.npy'                
            new_runtime['hp_dict_file'] = self.fun_perfix(new_runtime, 'npy',target_action = 'hp_tuning') + '_hp_dict.npy'
            new_runtime['job_name'] = self.varity_obj.set_job_name(new_runtime)            
            if not os.path.isfile(new_runtime['cur_fold_result']):
                alm_fun.show_msg (runtime['log'],1, 'Run prediction on test fold '  + str(cur_test_fold) + '......')                                                 
                if (runtime['cluster'] == 1) :                                      
                    [job_id,job_name,result_dict] = self.varity_obj.varity_action(new_runtime)
                    cur_jobs[job_name] = []
                    cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                    cur_jobs[job_name].append(job_id)
                    cur_jobs[job_name].append(new_runtime)                     
                else:                             
                    self.fun_single_fold_prediction(new_runtime)       
                    
        if runtime['cluster'] == 1:
            if self.varity_obj.fun_monitor_jobs(cur_jobs,runtime['log'],runtime) == 1:
                alm_fun.show_msg (runtime['log'],1, 'Batch: '  +  runtime['batch_id'] + ' all results are done,start to gathering results......')                              

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
            cur_test_result = self.fun_perfix(runtime, 'npy_temp',target_action = 'single_fold_prediction')+  '_' + runtime['batch_id'] + '_hp_test_single_fold_result.npy'
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
            cur_hp_performance = test_cv_result['macro_cv_auprc'].values[0]  
            cur_hp_performance_ste= test_cv_result['macro_cv_auprc_ste'].values[0]
        if alm_predictor.tune_obj == 'macro_cv_rfp':
            cur_hp_performance = test_cv_result['macro_cv_rfp'].values[0]                                                   
            cur_hp_performance_ste = test_cv_result['macro_cv_rfp_ste'].values[0]
        if alm_predictor.tune_obj == 'macro_cv_auroc':
            cur_hp_performance = test_cv_result['macro_cv_auroc'].values[0]
            cur_hp_performance_ste = test_cv_result['macro_cv_auroc_ste'].values[0]
        if alm_predictor.tune_obj == 'macro_cv_aubprc':
            cur_hp_performance = test_cv_result['macro_cv_aubprc'].values[0]
            cur_hp_performance_ste = test_cv_result['macro_cv_aubprc_ste'].values[0]
        if alm_predictor.tune_obj == 'macro_cv_brfp':
            cur_hp_performance = test_cv_result['macro_cv_brfp'].values[0]            
            cur_hp_performance_ste = test_cv_result['macro_cv_brfp_ste'].values[0]
        if alm_predictor.tune_obj == 'micro_cv_auprc':
            cur_hp_performance = test_cv_result['micro_cv_auprc'].values[0]
            cur_hp_performance_ste = test_cv_result['micro_cv_auprc_ste'].values[0]
        if alm_predictor.tune_obj == 'micro_cv_auroc':
            cur_hp_performance = test_cv_result['micro_cv_auroc'].values[0]            
            cur_hp_performance_ste = test_cv_result['micro_cv_auroc_ste'].values[0]
                 
        alm_fun.show_msg (runtime['log'],1,alm_predictor.tune_obj + ': ' + str(round(cur_hp_performance,4)) + '±' + str(round(cur_hp_performance_ste,4)))
        
        if runtime['hp_tune_type'] == 'mv_analysis':
            test_cv_result['mv_id'] = str(runtime['mv_id'])
            test_cv_result.to_csv(self.fun_perfix(runtime, 'csv',1) + '_hp_test_cv_result_' + runtime['mv_qip'] + '_mv_' + str(runtime['mv_id']) + '.csv',index = False)
        else:
            train_cv_result.to_csv(self.fun_perfix(runtime, 'csv',1) + '_hp_train_cv_result.csv')
            train_cv_results.to_csv(self.fun_perfix(runtime, 'csv',1) + '_hp_train_cv_results.csv')
            test_cv_result.to_csv(self.fun_perfix(runtime, 'csv',1) +'_hp_test_cv_result.csv')
            test_cv_results.to_csv(self.fun_perfix(runtime, 'csv',1) + '_hp_test_cv_results.csv')
            
            if alm_predictor.type == 1:   
                hp_dict_df.to_csv(self.fun_perfix(runtime, 'csv',1) + '_hp_dict_results.csv')
                
            np.save(self.fun_perfix(runtime, 'npy',1)+ '_test_cv_results.npy',return_dict)
        
        alm_fun.show_msg (runtime['log'],1,"End Time: " + str(datetime.now()))
        alm_fun.show_msg (runtime['log'],1,"***************************************************************")  
        
    def fun_target_predictions(self,runtime):
        alm_predictor = self.proj.predictor[runtime['predictor']]
        alm_dataset = alm_predictor.data_instance   

        cur_jobs= {}                                   
        alm_fun.show_msg (runtime['log'],1,'Start to run target predictions, batch id: ' + runtime['batch_id'] + '......' )        
        for target_file in runtime['target_files']: 
            new_runtime = runtime.copy()   
            new_runtime['target_file'] = target_file                                                
            new_runtime['run_on_node'] = 1    
            new_runtime['action'] = 'target_prediction'                                         
            target_file_name = new_runtime['target_file'].split('/')[-1].split('.')[0]
            new_runtime['job_name'] = target_file_name + '_predition_' + new_runtime['batch_id']  
            if runtime['loo'] == 1:
                new_runtime['cur_fold_result'] = self.fun_perfix(runtime, 'csv') + '_' + target_file_name + '_loo_predicted.csv'                                                
            else:
                new_runtime['cur_fold_result'] = self.fun_perfix(runtime, 'csv') + '_' + target_file_name + '_predicted.csv'
                                            
            if not os.path.isfile(new_runtime['cur_fold_result']):                                                 
                if (runtime['cluster'] == 1) :               
                    alm_fun.show_msg (runtime['log'],1, 'Run prediction on ' + target_file_name + '......')
                    [job_id,job_name,result_dict] = self.varity_obj.varity_action(new_runtime)
                    cur_jobs[job_name] = []
                    cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                    cur_jobs[job_name].append(job_id)
                    cur_jobs[job_name].append(new_runtime)                                             
                else:                                                    
                    self.fun_single_fold_prediction(new_runtime)
            else:
                alm_fun.show_msg (runtime['log'],1, 'Result is avaiable on ' + target_file_name +  '......')       
                    
        if runtime['cluster'] == 1:
            batch_log = self.fun_perfix(runtime,'log') + '_' + runtime['batch_id'] + '.log'
            if self.varity_obj.fun_monitor_jobs(cur_jobs,batch_log,runtime) == 1:
                alm_fun.show_msg (runtime['log'],1, 'Batch: '  +  runtime['batch_id'] + ' all target predictions are done.')
                               
    def fun_test_hyperopt(self,runtime):
        alm_fun.show_msg (runtime['log'],1, 'fun_test_hyperopt started......')
        alm_predictor = self.proj.predictor[runtime['predictor']]                
        trial_file = self.fun_perfix(runtime,'npy',target_action = 'hp_tuning') +'_trials.pkl'
        trial_result_file = self.fun_perfix(runtime,'csv',target_action = 'hp_tuning') + '_trial_results.txt'
        [cur_trials,cur_trials_df,x] = self.get_trials(trial_file,trial_result_file,None)
        cur_trials_df = cur_trials_df.loc[cur_trials_df['trial'] < runtime['trials_max_num'],:]
        test_hyperopt_df = None 
        ignore_trails_mv_size = 0
        
        if runtime['test_hyperopt_type'] == 'all':
            tids =  cur_trials_df['tid']
            
        if runtime['test_hyperopt_type'] == 'mv':
            ignore_trails_mv_size = 1
            tids = []
            for mv in runtime['test_hyperopt_mvs']:                
                runtime['trials_mv_size']= mv
                [best_trial_tid,best_trial_num] = self.save_best_hp_dict_from_trials(runtime)
                tids.append(best_trial_tid)
                print ('mv:' + str(mv) + ' tid:' + str(best_trial_tid))
        cur_jobs = {}   
        
                       
        for cur_tid in tids:                
#         for cur_tid in tids:                    
            cur_hp_dict = self.get_hp_dict_from_trial_results(cur_tid,cur_trials_df,alm_predictor.hp_default)
            cur_hp_dict_file = self.fun_perfix(runtime,'npy_temp',target_action = 'hp_tuning',ignore_trails_mv_size = ignore_trails_mv_size) + '_tid_' + str(cur_tid) + '_hp_dict.npy'
            np.save(cur_hp_dict_file,cur_hp_dict)
            new_runtime = runtime.copy()
            new_runtime['hp_dict_file'] = cur_hp_dict_file                                                                                                  
            new_runtime['single_fold_type']  = 'target'#                     
            new_runtime['run_on_node'] = 1    
            new_runtime['mem'] = 10240
            new_runtime['action'] = 'single_fold_prediction'                                         
            new_runtime['job_name'] = self.fun_perfix(new_runtime,'npy_temp',with_path = 0,ignore_trails_mv_size = ignore_trails_mv_size) + '_target_tid_' + str(cur_tid) + '_' + runtime['batch_id']                            
            new_runtime['cur_fold_result'] = self.fun_perfix(new_runtime,'npy_temp',ignore_trails_mv_size = ignore_trails_mv_size) + '_target_tid_' + str(cur_tid) + '_' + runtime['batch_id'] + '.npy'                                               
                                            
            if not os.path.isfile(new_runtime['cur_fold_result']):                                                 
                if (runtime['cluster'] == 1) :               
                    alm_fun.show_msg (runtime['log'],1, 'Run prediction on target file, using hyper-parameter from trial - ' + str(cur_tid))
                    [job_id,job_name,result_dict] = self.varity_obj.varity_action(new_runtime)
                    cur_jobs[job_name] = []
                    cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                    cur_jobs[job_name].append(job_id)
                    cur_jobs[job_name].append(new_runtime)                                             
                else:                                                    
                    self.fun_single_fold_prediction(new_runtime)
            else:
                alm_fun.show_msg (runtime['log'],1, 'Result is avaiable for trial  - ' + str(cur_tid))     
                    
        if runtime['cluster'] == 1:
            if self.varity_obj.fun_monitor_jobs(cur_jobs,runtime['log'],runtime) == 1:
                alm_fun.show_msg (runtime['log'],1, 'Batch: '  +  runtime['batch_id'] + ' all results are done,start to gathering results......')       
        
        i = 0  
        for cur_tid in tids:   
            cur_result_file = self.fun_perfix(new_runtime,'npy_temp',ignore_trails_mv_size = ignore_trails_mv_size) + '_target_tid_' + str(cur_tid) + '_' + runtime['batch_id'] + '.npy'
            cur_result_dict = np.load(cur_result_file).item()
            
            cur_test_hyperopt_df = cur_trials_df.loc[cur_trials_df['tid'] == cur_tid,:]
            cur_test_hyperopt_df.index = [cur_tid]
            if runtime['test_hyperopt_type'] == 'mv':
                cur_test_hyperopt_df['mv'] = runtime['test_hyperopt_mvs'][i]
                i = i + 1            
            cur_result_dict['test_score_df'].index = [cur_tid]
            cur_test_hyperopt_df = pd.concat([cur_test_hyperopt_df,cur_result_dict['test_score_df']],axis = 1,join = 'inner')
            
            if test_hyperopt_df is None:
                test_hyperopt_df = cur_test_hyperopt_df
            else:
                test_hyperopt_df = pd.concat([test_hyperopt_df,cur_test_hyperopt_df])
                
        target_name = runtime['target_file'].split('/')[-1].split('.')[0]
        test_hyperopt_df.to_csv(self.fun_perfix(runtime,'csv',ignore_trails_mv_size = ignore_trails_mv_size) + '_test_hyperopt_' + runtime['test_hyperopt_type'] + '_' + target_name + '.csv',index = False)
            
    def fun_merge_target_prediction(self,runtime):
        ####load original target files 
        if runtime['target_type'] == 'file':                                  
            merged_target_df = pd.read_csv(runtime['target_file'],low_memory = False)        
        if runtime['target_type'] == 'dataframe':            
            merged_target_df = runtime['target_dataframe']

        alm_fun.show_msg(runtime['log'],self.verbose, 'Target size : ' + str(merged_target_df.shape[0]) )                    
        #### merge other predictions                 
        for predicted_file in runtime['target_predicted_files']:            
            cur_predicted_df = pd.read_csv(runtime['project_path'] + 'output/csv/' + predicted_file,low_memory = False)
            merged_target_df = pd.merge(merged_target_df,cur_predicted_df,how = 'left')
            
            alm_fun.show_msg(runtime['log'],self.verbose, 'After merging ' + predicted_file + ' target size : ' + str(merged_target_df.shape[0]) )
                        
        target_file_name = runtime['session_id'] + '_' + runtime['target_file'].split('/')[-1].split('.')[0]                        
        merged_target_df.to_csv(runtime['project_path'] + 'output/csv/' + target_file_name +  '_predicted.csv',index = False)             
        
    def fun_target_prediction(self,runtime):        
        alm_predictor = self.proj.predictor[runtime['predictor']]
        alm_dataset = alm_predictor.data_instance        
        features = alm_predictor.features 
        
        prediction_ouput_cols = runtime['prediction_ouput_cols'] + [runtime['target_dependent_variable']]     
        
        if runtime['target_type'] == 'file':                                  
            cur_target_df = pd.read_csv(runtime['target_file'],low_memory = False)    
            target_name = runtime['session_id'] + '_' + runtime['target_file'].split('/')[-1].split('.')[0]    
        if runtime['target_type'] == 'dataframe':            
            cur_target_df = runtime['target_dataframe']
            target_name = runtime['target_dataframe_name']
        
        if runtime['prediction_ouput_with_input_cols'] == 1:
            output_cols = list(cur_target_df.columns)
        else:
            if runtime['prediction_ouput_with_features'] == 1:
                output_cols = prediction_ouput_cols + features
            else:
                output_cols = prediction_ouput_cols

        shap_output_target = None
        shap_output_train = None   
        test_score_df = None         
                
        runtime['hp_dict_file'] = self.fun_perfix(runtime, 'npy',target_action = 'hp_tuning') + '_hp_dict.npy'
                            
        if runtime['trials_mv_size'] == -1:  
            predictor_name = alm_predictor.name
        else:
            predictor_name = alm_predictor.name + '_mv_' + str(runtime['trials_mv_size'])
        
        
        print ('target_name: ' + target_name)
        
        if runtime['save_target_csv_name'] != '': 
            csv_output_file = runtime['save_target_csv_name']
        else:      
            if runtime['loo'] == 1:      
                csv_output_file = runtime['project_path'] + 'output/csv/' + target_name + '_' + predictor_name + '_loo.csv'
            else:
                csv_output_file = runtime['project_path'] + 'output/csv/' + target_name + '_' + predictor_name + '.csv'
                
        if runtime['save_target_npy_name'] != '': 
            npy_output_file = runtime['save_target_npy_name']
        else:            
            npy_output_file = runtime['project_path'] + 'output/npy/' + target_name + '_' + predictor_name + '.npy'
                

        if alm_predictor.type == 0: # no loo prediction for 
            alm_fun.show_msg (runtime['log'],self.verbose,'No LOO prediction for Non-VARITY models, making predictions without LOO......' )
            runtime['loo'] == 0
            
        if runtime['loo'] == 1:                        
            loo_dict = self.get_loo_dict(runtime)
            alm_fun.show_msg (runtime['log'],self.verbose,'Runing prediction for ' + str(len(loo_dict.keys())) + ' records that exist in training data......' )  
            #**********************************************************************
            # Fire parallel jobs for loo predictions
            #**********************************************************************
            cur_jobs= {}                                   
            alm_fun.show_msg (runtime['log'],1,'Start to run target loo predictions, batch id: ' + runtime['batch_id'] + '......' )        
            for target_index in loo_dict.keys(): 
                new_runtime = runtime.copy()                                             
                new_runtime['cur_target_fold']  = loo_dict[target_index]
                new_runtime['single_fold_type']  = 'target'
#                 new_runtime['hp_tune_type'] = 'hyperopt'      
                new_runtime['run_on_node'] = 1    
                new_runtime['mem'] = 10240
                new_runtime['action'] = 'single_fold_prediction'                                         
                new_runtime['job_name'] = alm_fun.get_random_id(8)                
                new_runtime['cur_fold_result'] = self.fun_perfix(new_runtime, 'npy_temp')+ '_loo_' + target_name + '_' + str(target_index) + '_' + new_runtime['batch_id'] + '.npy'                                                
                                                
                if not os.path.isfile(new_runtime['cur_fold_result']):                                                 
                    if (runtime['cluster'] == 1) :               
                        alm_fun.show_msg (runtime['log'],1, 'Run prediction on target fold '  + '[target_index: ' + str(target_index) + '-' + str(new_runtime['cur_target_fold']) + ']......')
                        [job_id,job_name,result_dict] = self.varity_obj.varity_action(new_runtime)
                        cur_jobs[job_name] = []
                        cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                        cur_jobs[job_name].append(job_id)
                        cur_jobs[job_name].append(new_runtime)                                             
                    else:                                                    
                        self.fun_single_fold_prediction(new_runtime)
                else:
                    alm_fun.show_msg (runtime['log'],1, 'Result is avaiable on target fold '  + '[target_index: ' + str(target_index) + '-' + str(new_runtime['cur_target_fold']) + ']......')       
                        
            if runtime['cluster'] == 1:
                batch_log = self.fun_perfix(new_runtime,'log') + '_' + runtime['batch_id'] + '.log'
                if self.varity_obj.fun_monitor_jobs(cur_jobs,batch_log,runtime) == 1:
                    alm_fun.show_msg (runtime['log'],1, 'Batch: '  +  runtime['batch_id'] + ' all results are done,start to gathering results......')       
                    
            #**********************************************************************
            # Collect results for all parallel loo predictions
            #**********************************************************************
            for target_index in loo_dict.keys():                
                cur_fold_result = self.fun_perfix(new_runtime, 'npy_temp')+ '_loo_' + target_name + '_' + str(target_index) + '_' + runtime['batch_id'] + '.npy'                 
                r_loo = np.load(cur_fold_result).item()
                cur_target_df.loc[target_index,predictor_name + '_LOO'] = r_loo['test_y_predicted'].values[0]
             
#             save_prediction_file = self.fun_perfix(runtime, 'csv') +  '_' + target_name + '_loo_predicted.csv'
            output_cols = output_cols + [predictor_name + '_LOO']
            cur_target_df[output_cols].to_csv(csv_output_file,index = False)
        else:            
            runtime['cur_fold_result'] = self.fun_perfix(runtime, 'npy') + '_' + target_name + '_' +  runtime['batch_id'] + '.npy' 
            runtime['single_fold_type'] = 'target'
            self.fun_single_fold_prediction(runtime)                                        
            r = np.load(runtime['cur_fold_result']).item()            
            cur_target_df[predictor_name] = r['test_y_predicted']
            cur_target_df = cur_target_df.reset_index(drop = True)
            test_score_df = r['test_score_df']
            print(r['feature_importance'])
#             save_prediction_file = self.fun_perfix(runtime, 'csv') + '_' + target_name + '_predicted.csv'
            output_cols = output_cols + [predictor_name]
            cur_target_df[output_cols].to_csv(csv_output_file,index = False)
            
            shap_output_target_interaction  = r['shap_output_test_interaction']
            if shap_output_target_interaction is not None:
                np.save(self.fun_perfix(runtime, 'npy') + '_' + target_name + '_target_interaction_shap.npy',shap_output_target_interaction)             
                shap_output_target = pd.DataFrame(np.sum(shap_output_target_interaction[:,:,:],axis = 1),columns = [x +'_shap' for x in features]+['base_shap'])
                shap_output_target['total_shap'] = shap_output_target.sum(axis=1)
                shap_output_target = pd.concat([cur_target_df[output_cols],shap_output_target],axis = 1)                
                shap_output_target.to_csv(self.fun_perfix(runtime, 'csv') + '_' + target_name + '_target_shap.csv',index = False)
  
            shap_output_train_interaction  = r['shap_output_train_interaction']
            all_train_indices = r['all_train_indices']            
            if shap_output_train_interaction is not None:
                np.save(self.fun_perfix(runtime, 'npy') + '_' + target_name + '_train_interaction_shap.npy',shap_output_train_interaction)    
                shap_output_train = pd.DataFrame(np.sum(shap_output_train_interaction[:,:,:],axis = 1),columns = [x +'_shap' for x in features]+['base_shap'])                
                shap_output_train['total_shap'] = shap_output_train.sum(axis=1)
                shap_output_train.index = all_train_indices                    
                shap_output_train.to_csv(self.fun_perfix(runtime, 'csv') + '_' + target_name + '_train_shap.csv',index = False)    
                
        result_dict = {}
        result_dict['target_predictions'] = cur_target_df[output_cols]
        result_dict['shap_output_target'] = shap_output_target
        result_dict['shap_output_train'] = shap_output_train
        result_dict['test_score_df'] = test_score_df
        np.save(npy_output_file,result_dict)
        
        return(result_dict)
    
                            
    def fun_validation_cv_prediction_sp(self,runtime):
        alm_predictor = self.proj.predictor[runtime['predictor']]
        cv_split_folds = alm_predictor.data_instance.cv_split_folds   
                             
        #**********************************************************************
        # Fire parallel jobs for all test folds
        #**********************************************************************
        cur_jobs= {}                   
        if runtime['batch_id'] =='':                           
            runtime['batch_id'] = alm_fun.get_random_id(10)
        alm_fun.show_msg (runtime['log'],1,'Start to run validation cv prediction, batch id: ' + runtime['batch_id'] + '......' )        
        for cur_validation_fold in range(cv_split_folds): 
            cur_validation_fold_result = self.fun_perfix(runtime, 'npy_temp') +  '_' + runtime['filtering_hp'] + '_' + str(runtime['filtering_hp_value'])+ '_' + 'vf' + str(cur_validation_fold) + '_' + runtime['batch_id'] + '_hp_validation_single_fold_result.npy'
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
                alm_fun.show_msg (runtime['log'],1, 'Run prediction on validation fold '  + str(cur_validation_fold) + '......')                                                 
                if (runtime['cluster'] == 1) :                                      
                    [job_id,job_name,result_dict] = self.varity_obj.varity_action(new_runtime)
                    cur_jobs[job_name] = []
                    cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                    cur_jobs[job_name].append(job_id)
                    cur_jobs[job_name].append(new_runtime)                     
                else:                             
                    self.fun_single_fold_prediction(new_runtime)       
                    
        if runtime['cluster'] == 1:
            batch_log = self.fun_perfix(new_runtime,'log') + '_' + runtime['batch_id'] + '.log'
            if self.varity_obj.fun_monitor_jobs(cur_jobs,batch_log,runtime) == 1:
                alm_fun.show_msg (runtime['log'],1, 'Batch: '  +  runtime['batch_id'] + ' all results are done,start to gathering results......')                       
            
        #**********************************************************************
        # Collect  the results of all parallel jobs and retrieve the best value
        #**********************************************************************                          
        alm_fun.show_msg (runtime['log'],self.verbose,'All parallel jobs are finished. Collecting results......' )
        for cur_validation_fold in range(cv_split_folds):                               
            cur_validation_result = self.fun_perfix(new_runtime, 'npy_temp') +  '_' + runtime['filtering_hp'] + '_' + str(runtime['filtering_hp_value'])+ '_' + 'vf' + str(cur_validation_fold) + '_' + runtime['batch_id'] + '_hp_validation_single_fold_result.npy'
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
                                 
        cur_hp_validation_performance = validation_cv_result[alm_predictor.tune_obj].values[0]  
        cur_hp_validation_performance_ste= validation_cv_result[alm_predictor.tune_obj + '_ste'].values[0] 
        cur_hp_train_performance = train_cv_result[alm_predictor.tune_obj].values[0]  
        cur_hp_train_performance_ste= train_cv_result[alm_predictor.tune_obj + '_ste'].values[0]                      
                 
        alm_fun.show_msg (runtime['log'],1,alm_predictor.tune_obj + ' on validation set:' + str(round(cur_hp_validation_performance,4)) + '±' + str(round(cur_hp_validation_performance_ste,4)))         
        alm_fun.show_msg (runtime['log'],1,alm_predictor.tune_obj + ' on training set: ' + str(round(cur_hp_train_performance,4)) + '±' + str(round(cur_hp_train_performance_ste,4)))
        
        spvalue_result_file = self.fun_perfix(runtime, 'csv',1) + '_' + runtime['filtering_hp'] + '_spvalue_results.txt'
        cur_spvalue_result = [str(runtime['filtering_hp_value']),str(cur_hp_train_performance),str(cur_hp_train_performance_ste),str(cur_hp_validation_performance),str(cur_hp_validation_performance_ste)] 
        alm_fun.show_msg(spvalue_result_file,1,'\t'.join(cur_spvalue_result),with_time = 0)
        return  (True)                            
    
    def fun_validation_cv_prediction_hyperopt(self,cur_hyperopt_hp):
                
        alm_predictor =  self.hyperopt_predictor
        runtime = self.hyperopt_runtime         
        cur_trial = len(self.cur_trials_result)-1        
        cv_split_folds = alm_predictor.data_instance.cv_split_folds        
        #save the compelete trials so far
        pickle.dump(self.cur_trials_result, open(self.fun_perfix(runtime, 'npy') + '_trials.pkl', "wb"))
        #save the current hp_dict                
        cur_hp_dict = alm_predictor.hp_default
        for key in cur_hyperopt_hp.keys():
            cur_hp_dict[key] = cur_hyperopt_hp[key]
        # save current  cur_hp_dict
        cur_hp_dict_file = self.fun_perfix(runtime, 'npy_temp')+ '_' + 'tr' +  str(cur_trial) + '_hp_dict.npy'
        np.save(cur_hp_dict_file,cur_hp_dict)

        alm_fun.show_msg (runtime['log'],1,"***************************************************************")
        alm_fun.show_msg (runtime['log'],1,"fun_validation_cv_prediction_hyperopt")
        alm_fun.show_msg (runtime['log'],1,"Predictor: " + alm_predictor.name)
        alm_fun.show_msg (runtime['log'],1,"Test fold: " + str(runtime['cur_test_fold']))
        alm_fun.show_msg (runtime['log'],1,"Tune obj: " + alm_predictor.tune_obj)
        alm_fun.show_msg (runtime['log'],1,"Data name: " + alm_predictor.data_instance.name)  
        alm_fun.show_msg (runtime['log'],1,"Session ID: " + runtime['session_id'])
        alm_fun.show_msg (runtime['log'],1,"Start Time: " + str(datetime.now()))
        alm_fun.show_msg (runtime['log'],1,"***************************************************************")
        alm_fun.show_msg (runtime['log'],1,"Trial: " + str(cur_trial))
        alm_fun.show_msg (runtime['log'],1,"Hyper-parameters:")
        for key in cur_hyperopt_hp.keys():
            alm_fun.show_msg (runtime['log'],1,key + ': ' + str(cur_hyperopt_hp[key]))
        alm_fun.show_msg (runtime['log'],1,"***************************************************************")

        #**********************************************************************
        # Fire parallel jobs for all test folds
        #**********************************************************************
        cur_jobs= {}                   
        if runtime['batch_id'] =='':                           
            runtime['batch_id'] = alm_fun.get_random_id(10)
        alm_fun.show_msg (runtime['log'],1,'Start to run validation cv prediction, batch id: ' + runtime['batch_id'] + '......' )        
        for cur_validation_fold in range(cv_split_folds):
            cur_validation_fold_result = self.fun_perfix(runtime, 'npy_temp') + '_vf' + str(cur_validation_fold) + '_tr' + str(cur_trial) + '_' + runtime['batch_id'] + '_hp_validation_single_fold_result.npy' 
            new_runtime = runtime.copy()              
            new_runtime['cur_trial'] = cur_trial            
            new_runtime['cur_validation_fold']  = cur_validation_fold
            new_runtime['single_fold_type']  = 'validation'                
            new_runtime['run_on_node'] = 1
            new_runtime['mem'] = 10240
            new_runtime['action'] = 'single_fold_prediction'                        
            new_runtime['job_name'] = self.fun_perfix(new_runtime,'npy_temp',with_path = 0) + '_vf' + str(cur_validation_fold) + '_tr' + str(cur_trial) + '_' + runtime['batch_id']           
            new_runtime['cur_fold_result'] =  cur_validation_fold_result           
            new_runtime['hp_dict_file'] = cur_hp_dict_file
            if not os.path.isfile(new_runtime['cur_fold_result']):        
                alm_fun.show_msg (runtime['log'],1, 'Run prediction on test fold '  + str(cur_validation_fold) + '......')                                         
                if (runtime['cluster'] == 1) :                  
                    [job_id,job_name,result_dict] = self.varity_obj.varity_action(new_runtime)
                    cur_jobs[job_name] = []
                    cur_jobs[job_name].append(new_runtime['cur_fold_result'])
                    cur_jobs[job_name].append(job_id)
                    cur_jobs[job_name].append(new_runtime)                     
                else:                             
                    self.fun_single_fold_prediction(new_runtime)       
                        
        if runtime['cluster'] == 1:
            if self.varity_obj.fun_monitor_jobs(cur_jobs,runtime['log'],runtime) == 1:
                alm_fun.show_msg (runtime['log'],1, 'Batch: '  +  runtime['batch_id'] + ' all results are done,start to gathering results......')                       
            
        #**********************************************************************
        # Collect  the results of all parallel jobs
        #**********************************************************************                          
        alm_fun.show_msg (runtime['log'],1,'All parallel jobs are finished. Collecting results......' )
        for cur_validation_fold in range(cv_split_folds):                               
            cur_validation_result = self.fun_perfix(runtime, 'npy_temp') + '_vf' + str(cur_validation_fold) + '_tr' + str(cur_trial) + '_' + runtime['batch_id'] + '_hp_validation_single_fold_result.npy'
        
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
            if len(np.unique(validation_y_truth)) ==  1: 
                validation_cv_result['micro_cv_aubprc'] = np.nan
                validation_cv_result['micro_cv_aurfp'] = np.nan                
                validation_cv_result['micro_cv_auprc'] = np.nan
                validation_cv_result['micro_cv_up_auprc'] = np.nan
                validation_cv_result['micro_cv_auroc'] = np.nan
                validation_cv_result['micro_cv_logloss'] = np.nan            
                validation_cv_result['micro_cv_pfr'] = np.nan
                validation_cv_result['micro_cv_rfp'] = np.nan
                validation_cv_result['micro_cv_prior'] = np.nan
                validation_cv_result['micro_cv_size'] = len(validation_y_truth)      
            else:                         
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
                
            if len(np.unique(train_y_truth)) ==  1: 
                train_cv_result['micro_cv_aubprc'] = np.nan
                train_cv_result['micro_cv_aurfp'] = np.nan                
                train_cv_result['micro_cv_auprc'] = np.nan
                train_cv_result['micro_cv_up_auprc'] = np.nan
                train_cv_result['micro_cv_auroc'] = np.nan
                train_cv_result['micro_cv_logloss'] = np.nan            
                train_cv_result['micro_cv_pfr'] = np.nan
                train_cv_result['micro_cv_rfp'] = np.nan
                train_cv_result['micro_cv_prior'] = np.nan
                train_cv_result['micro_cv_size'] = len(train_y_truth)      
            else:                         
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

        cur_hp_validation_performance = validation_cv_result[alm_predictor.tune_obj].values[0]  
        cur_hp_validation_performance_ste= validation_cv_result[alm_predictor.tune_obj + '_ste'].values[0] 
        cur_hp_train_performance = train_cv_result[alm_predictor.tune_obj].values[0]  
        cur_hp_train_performance_ste= train_cv_result[alm_predictor.tune_obj + '_ste'].values[0]                      
                 
        alm_fun.show_msg (runtime['log'],1,alm_predictor.tune_obj + ' on validation set:' + str(round(cur_hp_validation_performance,4)) + '±' + str(round(cur_hp_validation_performance_ste,4)))         
        alm_fun.show_msg (runtime['log'],1,alm_predictor.tune_obj + ' on training set: ' + str(round(cur_hp_train_performance,4)) + '±' + str(round(cur_hp_train_performance_ste,4)))
        alm_fun.show_msg (runtime['log'],1,"End Time: " + str(datetime.now()))
        alm_fun.show_msg (runtime['log'],1,"***************************************************************")        
                
        trial_result_file = self.fun_perfix(runtime, 'csv') + '_trial_results.txt'
        
        cur_trial_cols = ['tid','train_' + alm_predictor.tune_obj, 'train_' + alm_predictor.tune_obj + '_ste', 'validation_' + alm_predictor.tune_obj, 'validation_' + alm_predictor.tune_obj + '_ste'] + \
                         [key for key in cur_hyperopt_hp.keys()]
        
        cur_trial_result = [str(cur_trial),str(cur_hp_train_performance),str(cur_hp_train_performance_ste),str(cur_hp_validation_performance),str(cur_hp_validation_performance_ste)] + \
                                        [str(cur_hyperopt_hp[key]) for key in cur_hyperopt_hp.keys()]
        if not os.path.isfile(trial_result_file):
            alm_fun.show_msg(trial_result_file,1,'\t'.join(cur_trial_cols),with_time = 0)
        alm_fun.show_msg(trial_result_file,1,'\t'.join(cur_trial_result),with_time = 0)
                    
        return    ({'loss':1-cur_hp_validation_performance,'status': hyperopt.STATUS_OK})  
    
    def fun_combine_loo_predictions(self,runtime):
        alm_predictor = self.proj.predictor[runtime['predictor']]        
        alm_dataset = alm_predictor.data_instance
                
        #combine loo predictions
        
#         combined_loo_file = open(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_' + runtime['predictor'] + '_loo_predictions.txt','w')
#         varity_indices_df = pd.read_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_' + runtime['predictor'] + '_loo_indices.csv')        
#         for target_index in varity_indices_df['target_index']:
#             print (target_index)
#             cur_target_file = runtime['project_path'] + 'output/loo_temp/' + target_index + '_' + runtime['predictor'] + '.txt'
#             for line in  open(cur_target_file,'r'):
#                 combined_loo_file.write(line)
#         combined_loo_file.close()      
#         
#         
        cur_hp_npy = self.fun_perfix(runtime,'npy',target_action = 'hp_tuning')+ '_hp_dict.npy'       
        cur_hp_dict = np.load(cur_hp_npy).item()
        [alpha,beta] = self.update_sample_weights(cur_hp_dict,runtime)            
        extra_data = alm_predictor.data_instance.extra_train_data_df_lst[0].copy()
        core_data = alm_predictor.data_instance.train_data_index_df.copy()               
        all_data = pd.concat([core_data,extra_data])
        weight_scale = 1/all_data['weight'].max()
        all_data[runtime['predictor'] + '_weight'] = all_data['weight'] * weight_scale
        all_data = all_data[['p_vid','aa_pos','aa_ref','aa_alt',runtime['predictor'] + '_weight']]
        all_data['target_index'] = all_data.index
        
#         all_nonzero_data = all_data.loc[all_data['weight'] !=0,['p_vid','aa_pos','aa_ref','aa_alt','weight']]

        loo_predictions = pd.read_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_' + runtime['predictor'] + '_loo_predictions.txt',sep = '\t',header = None)
        loo_predictions.columns = ['target_index',alm_predictor.name + '_LOO']  
                                            
        loo_predictions_df = pd.merge(all_data,loo_predictions,how = 'left')                    
        loo_predictions_df.to_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_' + runtime['predictor'] + '_loo_predictions_with_keycols.csv',index =False)
        
        print (runtime['predictor'] + ' loo predictions are combined!')
        
    def fun_loo_predictions(self,runtime):        
        alm_predictor = self.proj.predictor[runtime['predictor']]
        alm_estimator = alm_predictor.es_instance
        alm_dataset = alm_predictor.data_instance
        features = alm_predictor.features + runtime['additional_features']
        
        cur_log = self.project_path + 'output/log/fun_loo_predictions_' + runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/fun_loo_predictions_' + runtime['varity_batch_id'] + '_done.log'  

        if alm_predictor.type == 1:                    
            #load current tuned hyper-parameter dictionary            
            cur_hp_dict = self.load_cur_hp_dict(runtime['hp_dict_file'])
            if cur_hp_dict is None:
                cur_hp_dict = alm_predictor.hp_default                      
            #update the weight hyper-parameter for each extra training example
            [alpha,beta] = self.update_sample_weights(cur_hp_dict,runtime)
            
            #update alogrithm-level hyperparameter
            for cur_hp in alm_predictor.hyperparameter.keys():
                if alm_predictor.hyperparameter[cur_hp]['hp_type'] == 3:                                                           
                    setattr(alm_estimator.estimator,cur_hp,cur_hp_dict[cur_hp])        
        
        #prepare the training data
        cur_train_df = alm_dataset.train_data_index_df
        extra_data_df = None
        if len(alm_dataset.extra_train_data_df_lst) != 0:    
            extra_data_df = alm_dataset.extra_train_data_df_lst[alm_dataset.extra_data_index].copy()
            
        target_ids_df = pd.read_csv(self.project_path + 'output/csv/' + runtime['varity_action'] + '_' +  runtime['batch_id'] + '_varity_batch_id.csv')                    
        target_ids = list(target_ids_df.loc[target_ids_df['varity_batch_id'] == runtime['varity_batch_id'],'target_index'])
        
        for target_id in target_ids:
            cur_remove_data = target_id.split('-')[0]
            cur_remove_index = int(target_id.split('-')[1])
            if cur_remove_data == 'extra':
                cur_loo_test_df = extra_data_df.loc[[cur_remove_index],:]
                
            if cur_remove_data =='core':                    
                cur_loo_test_df = cur_train_df.loc[[cur_remove_index],:]     
                               
            extra_loo_data_df = extra_data_df.loc[set(extra_data_df.index) - set([cur_remove_index]),:]
            cur_loo_train_df = cur_train_df.loc[set(cur_train_df.index) - set([cur_remove_index]),:]
        
            model_file = self.fun_perfix(runtime, 'npy')+  '.model'           
            r = alm_estimator.run(features, alm_dataset.dependent_variable, alm_predictor.ml_type, cur_loo_train_df, cur_loo_test_df ,extra_loo_data_df,None,alm_predictor,model_file)
            
            target_id_file = open(runtime['project_path'] + 'output/loo_temp/' + target_id + '_' + alm_predictor.name + '.txt','w')            
            alm_fun.show_msg (cur_log,1,str(cur_remove_index) + '\t' + str(r['test_y_predicted'].values[0]) + '\n')            
            target_id_file.write(str(cur_remove_index) + '\t' + str(r['test_y_predicted'].values[0]) + '\n')
            target_id_file.close()               
        
        alm_fun.show_msg(cur_done_log,1,runtime['varity_batch_id'] + ' is done.')
                                
    def fun_single_fold_prediction(self,runtime):                
        alm_predictor = self.proj.predictor[runtime['predictor']]
        alm_estimator = alm_predictor.es_instance
        alm_dataset = alm_predictor.data_instance
        features = alm_predictor.features + runtime['additional_features']

#         alm_fun.show_msg(runtime['log'],1, runtime['job_name'] + ' started......')

        if alm_predictor.type == 1:                    
            #load current tuned hyper-parameter dictionary            
            cur_hp_dict = self.load_cur_hp_dict(runtime['hp_dict_file'])
            if cur_hp_dict is None:
                cur_hp_dict = alm_predictor.hp_default                      
            #update the weight hyper-parameter for each extra training example
            [alpha,beta] = self.update_sample_weights(cur_hp_dict,runtime)
            
            #update alogrithm-level hyperparameter
            for cur_hp in alm_predictor.hyperparameter.keys():
                if alm_predictor.hyperparameter[cur_hp]['hp_type'] == 3:                                                           
                    setattr(alm_estimator.estimator,cur_hp,cur_hp_dict[cur_hp])        
        
        #prepare the extra_data_df
        extra_data_df = None
        if len(alm_dataset.extra_train_data_df_lst) != 0:    
            extra_data_df = alm_dataset.extra_train_data_df_lst[alm_dataset.extra_data_index].copy()
            
        #if predicting on test set 
        if runtime['single_fold_type'] == 'test':            
            cur_train_df = alm_dataset.train_data_index_df.loc[alm_dataset.train_splits_df[runtime['cur_test_fold']][runtime['cur_gradient_key']],:]
            if runtime['add_test_data'] == 1:                
                cur_test_df = self.varity_obj.add_test(alm_dataset,runtime)
                alm_fun.show_msg(runtime['log'],1,'# of records in test set: ' + str(cur_test_df.shape[0]))
            else:
                cur_test_df = alm_dataset.train_data_index_df.loc[alm_dataset.test_splits_df[runtime['cur_test_fold']][runtime['cur_gradient_key']],:]
            
            if runtime['remove_structural_features'] == 1:
                structural_cols = ['asa_mean','aa_psipred_E','aa_psipred_H','aa_psipred_C','bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_abs_max']
                for col in structural_cols:
                    cur_test_df[col] = np.nan
            
        if runtime['single_fold_type'] == 'validation':
            train_splits_df = alm_dataset.train_cv_splits_df[runtime['cur_test_fold']][runtime['cur_validation_fold']][runtime['cur_gradient_key']]
            validation_splits_df = alm_dataset.validation_cv_splits_df[runtime['cur_test_fold']][runtime['cur_validation_fold']][runtime['cur_gradient_key']]        
            cur_train_df = alm_dataset.train_data_index_df.loc[train_splits_df,:]
            cur_test_df = alm_dataset.train_data_index_df.loc[validation_splits_df,:]             

        if runtime['single_fold_type'] == 'target':                
            cur_train_df = alm_dataset.train_data_index_df
            
            if runtime['target_type'] == 'file':                                  
                cur_target_df = pd.read_csv(runtime['target_file'],low_memory = False)        
            if runtime['target_type'] == 'dataframe':            
                cur_target_df = runtime['target_dataframe']
                                                        
            cur_test_df = cur_target_df
            
            if  runtime['target_dependent_variable'] in cur_test_df.columns: 
                cur_test_df[runtime['dependent_variable']] = cur_test_df[runtime['target_dependent_variable']]
            else:                                                 
                cur_test_df[runtime['dependent_variable']] = np.random.random_integers(0,1,cur_test_df.shape[0])
                
            cur_test_df = cur_test_df.loc[:,list(set(features + runtime['loo_key_cols'] +runtime['prediction_ouput_cols'])) + [runtime['dependent_variable']]] 
            
            if runtime['loo'] == 1:
                cur_remove_data = runtime['cur_target_fold'].split('-')[0]
                cur_remove_index = int(runtime['cur_target_fold'].split('-')[1])
                if cur_remove_data == 'extra':
                    cur_test_df = extra_data_df.loc[[cur_remove_index],:]
                    
                if cur_remove_data =='core':                    
                    cur_test_df = cur_train_df.loc[[cur_remove_index],:]     
                                   
                extra_data_df = extra_data_df.loc[set(extra_data_df.index) - set([cur_remove_index]),:]
                cur_train_df = cur_train_df.loc[set(cur_train_df.index) - set([cur_remove_index]),:]

            print (cur_test_df.dtypes)
            print (str(cur_test_df.shape))
            
        model_file = self.fun_perfix(runtime, 'npy')+  '.model'
#         alm_fun.show_msg(runtime['log'],1, runtime['job_name'] + ' started to predict......')
        r = alm_estimator.run(features, alm_dataset.dependent_variable, alm_predictor.ml_type, cur_train_df, cur_test_df ,extra_data_df,None,alm_predictor,model_file)
        
        if alm_predictor.type == 1:           
            r['hp_dict'] = cur_hp_dict                                    
            if (r['model'] is not None) & (runtime['single_fold_type'] in ['target','test']):
                r['model'].save_model(self.fun_perfix(runtime, 'npy')+  '.model')
                    
        print (r['test_score_df'])   
#         print (r['test_y_predicted'])                 
        np.save(runtime['cur_fold_result'],r)         
#         alm_fun.show_msg(runtime['log'],1, runtime['job_name'] + ' done......')
                        
    def get_loo_dict(self,runtime):
        loo_dict = {}
        alm_predictor = self.proj.predictor[runtime['predictor']]
        alm_dataset = alm_predictor.data_instance        
        features = alm_predictor.features        
        if len(runtime['loo_key_cols']) == 0:
            runtime['loo_key_cols'] = features
        
        cur_target_df = pd.read_csv(runtime['target_file'],low_memory = False)
        cur_target_df['target_index'] = cur_target_df.index
        cur_target_df = cur_target_df[runtime['loo_key_cols'] + ['target_index'] ]

        if alm_predictor.type == 1:                    
            #load current tuned hyper-parameter dictionary            
            cur_hp_dict = self.load_cur_hp_dict(runtime['hp_dict_file'])
            if cur_hp_dict is None:
                cur_hp_dict = alm_predictor.hp_default                      
            #update the weight hyper-parameter for each extra training example
            [alpha,beta] = self.update_sample_weights(cur_hp_dict,runtime)
                                      
        extra_data_df = None
        if len(alm_dataset.extra_train_data_df_lst) != 0:    
            extra_data_df = alm_dataset.extra_train_data_df_lst[alm_dataset.extra_data_index].copy()        
        pass      
        cur_train_df = alm_dataset.train_data_index_df.copy()
        #disable the shapley value for loo prediction
        alm_predictor.shap_train_interaction = 0
        alm_predictor.shap_test_interaction = 0
        
        valid_extra_data_df = extra_data_df.loc[extra_data_df['weight'] != 0, runtime['loo_key_cols']]
        valid_extra_data_df['type'] = 'extra'
        valid_train_df = cur_train_df[runtime['loo_key_cols']]
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
            loo_dict[loo_index] = cur_remove_data + '-' + str(cur_remove_index)
    
        return(loo_dict)
    
    def update_sample_weights(self,hp_dict,runtime): 
        alm_predictor = self.proj.predictor[runtime['predictor']]
        extra_data_index = alm_predictor.data_instance.extra_data_index  
        [alpha,beta] = self.get_sample_weights(hp_dict,runtime)
        print('Extra set examples: ' + str(len(alpha)) + ', weights:' + str(alpha.sum()))
        print('core set examples: ' + str(len(beta)) + ', weights:' + str(beta.sum()))        
        alm_predictor.data_instance.extra_train_data_df_lst[extra_data_index]['weight'] = alpha
        alm_predictor.data_instance.train_data_index_df['weight'] = beta
        
#         extra_weight_df = alm_predictor.data_instance.extra_train_data_df_lst[extra_data_index][['chr','nt_pos','nt_ref','nt_alt','aa_pos','aa_ref','aa_alt','p_vid','gnomAD_exomes_AF','gnomAD_exomes_nhomalt','set_name','weight']]
#         core_weight_df = alm_predictor.data_instance.train_data_index_df[['chr','nt_pos','nt_ref','nt_alt','aa_pos','aa_ref','aa_alt','p_vid','gnomAD_exomes_AF','gnomAD_exomes_nhomalt','set_name','weight']]
#                                                                                                
#         extra_weight_df.to_csv(runtime['project_path'] + 'output/csv/' + runtime['predictor'] + '_extra_weight.csv')
#         core_weight_df.to_csv(runtime['project_path'] + 'output/csv/' + runtime['predictor'] + '_core_weight.csv')
#
        return([alpha,beta])   
    
    def get_sample_weights(self,hp_dict,runtime):
        alm_predictor = self.proj.predictor[runtime['predictor']]
        extra_data_index = alm_predictor.data_instance.extra_data_index        
        extra_data = alm_predictor.data_instance.extra_train_data_df_lst[extra_data_index].copy()   
        core_data =  alm_predictor.data_instance.train_data_index_df.copy()
        if type == 'from_hp_npy':
            if os.path.isfile(self.fun_perfix(runtime,'npy') + '_hp_weights.npy'):
                alpha = np.load(self.fun_perfix(runtime,'npy') + '_hp_weights.npy')
                extra_data['weight'] = alpha
                
#         if type == 'from_nn_npy':    
#             if os.path.isfile(self.fun_perfix(runtime,'npy') + '_' + str(init_weights) +'_' + str(target_as_source)  +'_' + self.session_id + '_nn_weights.npy'):
#                 alpha = np.load(self.fun_perfix(runtime,'npy') + '_' + str(init_weights) +'_' + str(target_as_source)  +'_' + self.session_id + '_nn_weights.npy')
#                 extra_data['weight'] = alpha
                
        if runtime['hp_tune_type'] == 'mv_analysis':
            core_data['weight'] = 1                           
            extra_data['weight'] = 0
            
            cur_qip_dict = alm_predictor.qip[runtime['mv_qip']]
            
            cur_extra_data = extra_data.loc[extra_data['set_name'].isin(cur_qip_dict['set_list'])]
            cur_extra_data['index_for_order'] = cur_extra_data.index
            
            if cur_qip_dict['direction'] == 0:
                cur_extra_data  = cur_extra_data.sort_values([cur_qip_dict['qip_col'],'index_for_order'])
            else:
                cur_extra_data  = cur_extra_data.sort_values([cur_qip_dict['qip_col'],'index_for_order'],ascending = False)

            n = int(cur_qip_dict['mv_data_points'])
            r = cur_qip_dict['mv_size_percent']/100
            m = cur_extra_data.shape[0]            
            x= int(m*(1-r)/(n-1))
            mv_size = int(r*m)
            
            if (runtime['mv_id'] != 'none') & (runtime['mv_id'] != 'all'):
                cur_mv_start = int(runtime['mv_id']) * x
                cur_mv_end =   cur_mv_start + mv_size
                if int(runtime['mv_id']) == n - 1:
                    cur_mv_end = m                
                cur_mv_data_indices = cur_extra_data.index[cur_mv_start:cur_mv_end]    
                            
            if runtime['mv_id'] == 'none':
                cur_mv_data_indices = []
                
            if runtime['mv_id'] == 'all':
                cur_mv_data_indices = cur_extra_data.index                   
            
            extra_data.loc[cur_mv_data_indices,'weight'] = 1
    
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
            core_data['weight'] = 1   
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
                    print (hp_parameter)
                           
        if runtime['hp_tune_type'] == 'hyperopt_logistic':
            core_data['weight'] = 1
            extra_data['weight'] = 1
            extra_data['weight_inuse'] = 0
            for cur_qip_name in alm_predictor.qip.keys():
                cur_qip_dict = alm_predictor.qip[cur_qip_name]
                if (alm_predictor.name in cur_qip_dict['predictors']) & (cur_qip_dict['enable'] == 1):
                    if cur_qip_dict['weight_function'] == 'logistic':
                        k = hp_dict[cur_qip_dict['hyperparameters'][0]]                        
                        L = hp_dict[cur_qip_dict['hyperparameters'][1]]
                        x0 = hp_dict[cur_qip_dict['hyperparameters'][2]]
                        set_type = cur_qip_dict['set_type']
                        set_list = cur_qip_dict['set_list']
                        qip_col = cur_qip_dict['qip_col']
                        qip_normalized_col = cur_qip_name + '_normalized'
                        print (cur_qip_name + '|' + set_type + '|' + str(set_list) + '|' + qip_col + '|' + qip_normalized_col + '|k:' + str(k) + '|L:' + str(L) + '|x0:' + str(x0))
                        
                        if set_type == 'addon':    
                            cur_extra_data = extra_data.loc[extra_data['set_name'].isin(set_list),[qip_normalized_col,'weight']]
                            cur_extra_weight = cur_extra_data['weight'] * L/(1+np.exp(0-k*(cur_extra_data[qip_normalized_col]-x0)))                            
                            extra_data.loc[cur_extra_weight.index,'weight'] = cur_extra_weight     
                            extra_data.loc[cur_extra_weight.index,'weight_inuse'] = 1                               
#                             extra_weight = extra_data.loc[extra_data['set_name'].isin(set_list),[qip_normalized_col,'weight']].apply(lambda x: x['weight']*L/(1+np.exp(0-k*(x[qip_normalized_col]-x0))),axis = 1)
#                             extra_data.loc[extra_data['set_name'].isin(set_list),'weight_inuse'] = 1
#                             extra_data.loc[extra_data['set_name'].isin(set_list),'weight'] = extra_weight
                        if set_type == 'core':
#                             x = core_data.loc[core_data['set_name'].isin(set_list),[qip_col,qip_normalized_col,'weight']]
                            cur_core_data = core_data.loc[core_data['set_name'].isin(set_list),[qip_normalized_col,'weight']]
                            cur_core_weight = cur_core_data['weight'] * L/(1+np.exp(0-k*(cur_core_data[qip_normalized_col]-x0)))                                                                                                        
                            core_data.loc[cur_core_weight.index,'weight'] = cur_core_weight
#                             core_weight = core_data.loc[core_data['set_name'].isin(set_list),[qip_normalized_col,'weight']].apply(lambda x: x['weight']*L/(1+np.exp(0-k*(x[qip_normalized_col]-x0))),axis = 1)
#                             core_data.loc[core_data['set_name'].isin(set_list),'weight'] = core_weight
            extra_data.loc[extra_data['weight_inuse'] == 0 ,'weight'] = 0

        return([np.array(extra_data['weight']),np.array(core_data['weight'])])

    def get_trials(self,trial_file,result_file = None,test_result_file = None):
        cur_trials = pickle.load(open(trial_file, "rb"))        
        if result_file is None:
            cur_trials_df = None
        else:    
            if os.path.isfile(result_file):        
                cur_trials_df = pd.read_csv(result_file,sep = '\t')
                cur_trials_df = cur_trials_df.sort_values(['tid'])
                cur_trials_df = cur_trials_df.reset_index(drop = True)
                cur_trials_df['trial'] = cur_trials_df.index  
            else:
                cur_trials_df = None  

        if test_result_file is None:
            cur_trials_test_df = None
        else:
            cur_trials_test_df = pd.read_csv(test_result_file)    
        return([cur_trials,cur_trials_df,cur_trials_test_df]) 
    
    def get_trials_old(self,trial_file,result_file = None,test_result_file = None,source = 'trials',candidate_offset = 0.0005,trials_cutoff = 1000):
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
            new_tid = -1 
            first_line = 1  
            for line in  open(result_file,'r'):
                if first_line == 1:
                    first_line = 0
                    continue
                
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
    
    def save_best_hp_dict_from_trials(self,runtime) :
        alm_predictor = self.proj.predictor[runtime['predictor']] 
        
        if runtime['trials_mv_size'] == -1:                
            mv_size = alm_predictor.trials_mv_size
        else:
            mv_size = runtime['trials_mv_size']
            
        mv_size = int(mv_size)
                    
        trial_file = self.fun_perfix(runtime,'npy',target_action = 'hp_tuning',ignore_trails_mv_size=1) +'_trials.pkl'
        trial_result_file = self.fun_perfix(runtime,'csv',target_action = 'hp_tuning',ignore_trails_mv_size=1) + '_trial_results.txt'
                
        best_hp_dict_file = self.fun_perfix(runtime,'npy',target_action = 'hp_tuning') + '_hp_dict.npy'
        best_hp_df_file = self.fun_perfix(runtime,'csv',target_action = 'hp_tuning') + '_best_hps.csv'
        best_hp_plot_file = self.fun_perfix(runtime, 'img') + '_hp_selection.png'   
        

        #********************************************************************************************
        # Save the best trial for each fold
        #********************************************************************************************
        all_hp_df = pd.DataFrame(columns = ['trial','tid','aubprc'] + list(alm_predictor.hp_default.keys()))
        candidate_offset = 0
        select_strategy = runtime['hp_select_strategy'] 
        [cur_trials,cur_trials_df,x] = self.get_trials(trial_file,trial_result_file,None)  
        
        cur_trials_df = cur_trials_df.loc[cur_trials_df['trial'] < runtime['trials_max_num'],:]
        
        if select_strategy == 'highest_validtion_low_overfit':
            cur_trials_df['overfit'] = cur_trials_df['train_' + alm_predictor.tune_obj] - cur_trials_df['validation_' + alm_predictor.tune_obj]
            cur_trials_df = cur_trials_df.sort_values('validation_' + alm_predictor.tune_obj)
            cur_trials_df = cur_trials_df.reset_index()
            cur_trials_df['index'] = cur_trials_df.index
            cur_trials_low_overfit_df = cur_trials_df.loc[cur_trials_df['overfit'] <= runtime['overfit_epsilon'],: ]            
            cur_trials_low_overfit_df = cur_trials_low_overfit_df.sort_values('validation_' + alm_predictor.tune_obj)              
            highest_validtion_low_overfit_selected_index =  cur_trials_low_overfit_df.index[-1]       
            best_trial_num = cur_trials_low_overfit_df.loc[highest_validtion_low_overfit_selected_index,'trial']
            best_trial_tid = cur_trials_low_overfit_df.loc[highest_validtion_low_overfit_selected_index,'tid']
            
            #******************************************************************************************
            #Plot trials result
            #******************************************************************************************
            fig = plt.figure(figsize=(runtime['fig_x'], runtime['fig_y']))
            ax = plt.subplot()
            marker_offset = 0.001                    
            ax1, = ax.plot(cur_trials_df['index'],cur_trials_df['train_' + alm_predictor.tune_obj],linewidth=3,marker='o', markersize=0,color = 'black')                      
            ax2, = ax.plot(cur_trials_df['index'],cur_trials_df['validation_' + alm_predictor.tune_obj],linewidth=3,marker='o', markersize=0,color = '#558ED5')
#             ax3,=ax.plot(cur_trials_mv_df['index'],cur_trials_mv_df['mv_cv_validation_avg'],linewidth=8,marker='o', markersize=22,color = 'darkblue')
            highest_validation_index = cur_trials_df.sort_values(['validation_' + alm_predictor.tune_obj]).index[-1]
            ax4, = ax.plot(highest_validation_index,cur_trials_df.loc[highest_validation_index,'validation_' + alm_predictor.tune_obj] + marker_offset + 0.001,linewidth=0,marker='v', markersize=30,color = 'orangered')
            ax5, = ax.plot(highest_validtion_low_overfit_selected_index,cur_trials_df.loc[highest_validtion_low_overfit_selected_index,'validation_' + alm_predictor.tune_obj] + marker_offset,linewidth=0,marker='v', markersize=30,color = 'limegreen')
            ax.set_xlim(-10,cur_trials_df.shape[0] + 100)               
            ax.set_title(alm_predictor.name + ' hyperparameter optimization' ,size = 35,pad = 20)
            ax.set_xlabel('Trials', size=30,labelpad = 20)
            ax.set_ylabel('10 Folds ' +  alm_predictor.tune_obj, size=30,labelpad = 20)             
            ax.tick_params(labelsize=25)
            ax.legend([ax4,ax5,ax1,ax2],['Trial with highest CV  performance on validation sets','trial picked by VARITY','CV  performance on training sets','CV  performance on validation sets'], frameon = False,loc = 'upper left',labelspacing = 0.5 ,markerscale = 0.6,fontsize = 20)
            fig.tight_layout()                     
            plt.savefig(best_hp_plot_file)   

        #**********************************************************
        # first_descent_mv_validation_selected_index
        #**********************************************************
        if select_strategy == 'selected_tid': 
            print(str(runtime['selected_tid']))           
            selected_tid_index = list(cur_trials_df.loc[cur_trials_df['tid'] == runtime['selected_tid'],:].index)[0]
            best_trial_tid = cur_trials_df.loc[selected_tid_index,'tid']           
            best_trial_num = cur_trials_df.loc[selected_tid_index,'trial']
            
        if select_strategy == 'first_descent_mv_validation_selected_index':      
            sort_type = 'train_' + alm_predictor.tune_obj
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
                cur_mv_cv_train_avg = np.round(cur_trials_df.loc[(cur_trials_df.index < mv_index) & (cur_trials_df.index >= mv_low_index) ,'train_' + alm_predictor.tune_obj].mean(),5)
                cur_mv_cv_validation_avg = np.round(cur_trials_df.loc[(cur_trials_df.index < mv_index) & (cur_trials_df.index >= mv_low_index) ,'validation_' + alm_predictor.tune_obj].mean(),5)                    
                cur_trials_df.loc[mv_index,'mv_cv_train_avg'] = cur_mv_cv_train_avg    
                cur_trials_df.loc[mv_index,'mv_cv_validation_avg'] = cur_mv_cv_validation_avg
                
                if mv_index ==cur_trials_df.shape[0] - 1:
                    break

            cur_trials_mv_df = cur_trials_df.loc[cur_trials_df['mv_cv_train_avg'].notnull(),:]
            cur_trials_mv_df['delta_mv_cv_validation_avg'] = 0            
            cur_trials_mv_df.loc[cur_trials_mv_df.index[:-1],'delta_mv_cv_validation_avg'] =  np.array(cur_trials_mv_df.loc[cur_trials_mv_df.index[1:],'mv_cv_validation_avg']) - np.array(cur_trials_mv_df.loc[cur_trials_mv_df.index[:-1],'mv_cv_validation_avg'])                        
            first_descent_mv_validation_index = np.min(list(cur_trials_mv_df.loc[cur_trials_mv_df['delta_mv_cv_validation_avg'] <= 0,: ].index))
#             cur_trials_selected_window_df = cur_trials_df.loc[range((first_descent_mv_validation_index - mv_size),first_descent_mv_validation_index),:]
            cur_trials_selected_window_df = cur_trials_df.loc[range(0,first_descent_mv_validation_index),:]
            first_descent_mv_validation_selected_index= cur_trials_selected_window_df.sort_values(['validation_' + alm_predictor.tune_obj]).index[-1]
            
            best_trial_num = cur_trials_df.loc[first_descent_mv_validation_selected_index,'trial']
            best_trial_tid = cur_trials_df.loc[first_descent_mv_validation_selected_index,'tid']
            
            
            #******************************************************************************************
            #Plot trials result
            #******************************************************************************************
            fig = plt.figure(figsize=(runtime['fig_x']*0.65, runtime['fig_y']*0.65))
            plt.clf()
            plt.rcParams["font.family"] = "Helvetica"  
            ax = plt.subplot()
            marker_offset = 0.007                    
            ax1, = ax.plot(cur_trials_df['index'],cur_trials_df['train_' + alm_predictor.tune_obj],linewidth=3,marker='o', markersize=0,color = 'black')                      
            ax2, = ax.plot(cur_trials_df['index'],cur_trials_df['validation_' + alm_predictor.tune_obj],linewidth=3,marker='o', markersize=0,color = '#558ED5')
            ax3,=ax.plot(cur_trials_mv_df['index'],cur_trials_mv_df['mv_cv_validation_avg'],linewidth=8,marker='o', markersize=22,color = 'darkblue')
            highest_validation_index = cur_trials_df.sort_values(['validation_' + alm_predictor.tune_obj]).index[-1]
            ax4, = ax.plot(highest_validation_index,cur_trials_df.loc[highest_validation_index,'validation_' + alm_predictor.tune_obj] + marker_offset +0.002,linewidth=0,marker='v', markersize=30,color = 'orangered')
            ax5, = ax.plot(first_descent_mv_validation_selected_index,cur_trials_df.loc[first_descent_mv_validation_selected_index,'validation_' + alm_predictor.tune_obj] + marker_offset,linewidth=0,marker='v', markersize=30,color = 'limegreen')
            ax.set_xlim(-10,cur_trials_df.shape[0]+30)               
            ax.set_title(alm_predictor.name + ' hyperparameter optimization (300 trials)' ,size = 35,pad = 20)
            ax.set_xlabel('Trials', size=30,labelpad = 20)
            ax.set_ylabel('10 Folds AUBPRC', size=30,labelpad = 20)             
            ax.tick_params(labelsize=25)
            ax.legend([ax4,ax5,ax1,ax2,ax3],['Trial with highest CV  performance on validation sets','Trial picked by VARITY','CV  performance on training sets','CV  performance on validation sets','CV  performance on validation sets \n(moving window average)'], frameon = False,loc = 'upper left',labelspacing = 0.5 ,markerscale = 0.6,fontsize = 20)
            fig.tight_layout()                     
            plt.savefig(best_hp_plot_file,dpi = 300)   
        #**********************************************************
        # pick the best cv  valdaiton performance 
        #**********************************************************
        if select_strategy == 'best_cv_performance':     
            cur_trials_df = cur_trials_df.sort_values(['validation_' + alm_predictor.tune_obj])                  
            best_trial_num = cur_trials_df.loc[cur_trials_df.index[-1],'trial']
            best_trial_tid = cur_trials_df.loc[cur_trials_df.index[-1],'tid']    
        #********************************************************************************************************************
        # pick the the ones with lowest training performance in candidates (small difference from the best performance)
        #********************************************************************************************************************     
        if select_strategy == 'lowest_training_performance':            
            min_train_performance = cur_trials_df.loc[cur_trials_df['candidate_flag'] == 1,'train_' + alm_predictor.tune_obj].min()
            best_trial_num = cur_trials_df.loc[cur_trials_df['train_' + alm_predictor.tune_obj] == min_train_performance,'trial'].max()
            best_trial_tid = cur_trials_df.loc[cur_trials_df['trial'] == best_trial_num,'tid'].values[0]

        cur_hp_dict = self.get_hp_dict_from_trial_results(best_trial_tid,cur_trials_df,alm_predictor.hp_default)
        np.save(best_hp_dict_file,cur_hp_dict)
            
        cur_hp_values = [str(cur_hp_dict[x]) + '|' + str(alm_predictor.hp_default[x]) for x in cur_hp_dict.keys()]                        
        best_trial_performance = cur_trials_df.loc[cur_trials_df['tid'] == best_trial_tid,'validation_' + alm_predictor.tune_obj].values[0]
        total_trials = cur_trials_df.shape[0]             
        print ('Selected hyper-paramter point -- Trial: ' + str(best_trial_num) + '/' + str(total_trials) + ', Tid: ' + str(best_trial_tid) + ', Performance: ' + str(best_trial_performance))
        all_hp_df.loc[runtime['cur_test_fold'],:] = [str(best_trial_num) + '|' + str(total_trials), str(best_trial_tid), str(best_trial_performance)]  + cur_hp_values                            
        all_hp_df.transpose().to_csv(best_hp_df_file)  
        
        return([best_trial_tid,best_trial_num])              
        
    def get_hp_dict_from_trial_results(self,best_trial_tid,cur_trials_df,hp_default):
        cur_hp_dict = hp_default.copy()
        for key in cur_hp_dict.keys():
            cur_hp_dict[key] = cur_trials_df.loc[cur_trials_df['tid'] == best_trial_tid,key].values[0]
            
        return(cur_hp_dict) 
                   
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

    def filter_test(self,input_df,runtime):
        self.alm_predictor = self.proj.predictor[runtime['predictor']]
        filter_test_indices = self.varity_obj.filter_test(self.alm_predictor.data_instance,runtime['filter_test_score'])   
        return (filter_test_indices)        
    
    def plot_classification_curve(self,runtime,plot_metrics,plot_metric_orders,predictors,compare_predictor, cv_folds, remain_indices_dict,test_result_dict,output_file,size_factor = 1,title = '',run_bootstrapping = 0, dpi = 96,table_scale_factor = 3,show_size = 0,legend_bbox = [0.01,0.05,0.8,0.5],extra_info_x = 0.2,extra_info_y = 0.01,fig_x = 30, fig_y = 20):

        if len(predictors) <= 10 :            
            size_factor = 1.5 * size_factor
                                
#         if len(predictors) <= 10 :
#             color_lst = ['orangered', 'darkgreen', 'darkblue', 'darkorange', 'darkmagenta', 'darkcyan', 'saddlebrown', 'darkgoldenrod','deeppink','dodgerblue']
#         else:
#             color_lst = ['#ff0000', '#40e0d0', '#bada55', '#ff80ed', '#696969', '#133337', '#065535', '#5ac18e', '#f7347a', 
#                              '#000000', '#420420', '#008080', '#ffd700', '#ff7373', '#ffa500', '#0000ff', '#003366', '#fa8072', 
#                              '#800000', '#800080', '#333333', '#4ca3dd','#ff00ff','#008000','#0e2f44','#daa520','#444444','#555555']
        color_dict = {}
        color_dict['VARITY_R'] = '#ff0000'
        color_dict['VARITY_ER'] = '#40e0d0'
        color_dict['SIFT'] = '#bada55'
        color_dict['Polyphen2_HDIV'] = '#ff80ed'
        color_dict['Polyphen2_HVAR'] = '#696969'
        color_dict['PROVEAN'] = '#133337'
        color_dict['CADD'] = '#065535'
        color_dict['PrimateAI'] = '#5ac18e'
        color_dict['Eigen'] = '#f7347a'
        color_dict['REVEL'] = '#000000'
        color_dict['M-CAP'] = '#420420'
        color_dict['LRT'] = '#008080'
        color_dict['MutationTaster'] = '#ffd700'
        color_dict['MutationAssessor'] = '#ff7373'
        color_dict['FATHMM'] = '#ffa500'
        color_dict['MetaSVM'] = '#0000ff'
        color_dict['MetaLR'] = '#003366'
        color_dict['GenoCanyon'] = '#fa8072'
        color_dict['DANN'] = '#800000'
        color_dict['GERP++'] = '#800080'
        color_dict['phyloP'] = '#ff0000'
        color_dict['PhastCons'] = '#333333'
        color_dict['SiPhy'] = '#4ca3dd'
        color_dict['fitCons'] = '#008000'
        color_dict['MISTIC'] = '#0e2f44'
        color_dict['MPC'] = '#daa520'
        color_dict['DeepSequence'] = '#444444'
        color_dict['EVMutation'] = '#555555'
        color_dict['VARITY_LOO'] = '#666666'
        
        
        predictor_dict = {}
        predictor_dict['names'] = {}
        predictor_dict['colors'] = {}
        for predictor in predictors:            
            if ('VARITY' in predictor) | ('Polyphen2' in predictor):
                predictor_name = predictor.split('_')[0] + '_' + predictor.split('_')[1] 
            else:                      
                predictor_name = predictor.split('_')[0]
                
            predictor_dict['colors'][predictor] = color_dict[predictor_name]
             
            if predictor.split('_')[0] in runtime['nucleotide_predictors']:
#                 predictor_name = predictor_name+ '(Δ)'
                predictor_name = predictor_name+ '(•)'
                
                
                
            predictor_dict['names'][predictor] = predictor_name

        all_dict = {}    
            
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
        #************************************************************#
        
        if runtime['plot_test_with_saved_data'] == 1:
            all_dict = np.load(output_file.replace('img','npy') + '.npy').item()
        else:
            for predictor in predictors:
                all_dict[predictor] = {}
                all_dict[predictor]['data'] = {}
                all_dict[predictor]['data']['size'] = 0
                all_dict[predictor]['data']['positive_size'] = 0
                all_dict[predictor]['data']['negative_size'] = 0
                all_dict[predictor]['result'] = {}
#                 all_dict[predictor]['color'] = color_lst[color_index]
                color_index = color_index + 1
                
                for cur_fold in range(cv_folds):                 
                    all_dict[predictor]['data'][cur_fold] = {}
                    all_dict[predictor]['data'][cur_fold]['truth'] = test_result_dict[predictor]['test_y_truth_dict'][cur_fold][remain_indices_dict[cur_fold]]
                    all_dict[predictor]['data'][cur_fold]['predicted'] = test_result_dict[predictor]['test_y_predicted_dict'][cur_fold][remain_indices_dict[cur_fold]]
                    all_dict[predictor]['data'][cur_fold]['size'] = (~np.isnan(all_dict[predictor]['data'][cur_fold]['predicted'])).sum()
                    all_dict[predictor]['data'][cur_fold]['positive_size'] = (~np.isnan(all_dict[predictor]['data'][cur_fold]['predicted']) & (all_dict[predictor]['data'][cur_fold]['truth']  == 1)).sum()
                    all_dict[predictor]['data'][cur_fold]['negative_size'] = (~np.isnan(all_dict[predictor]['data'][cur_fold]['predicted']) & (all_dict[predictor]['data'][cur_fold]['truth']  == 0)).sum()
                    
                    all_dict[predictor]['data']['positive_size'] = all_dict[predictor]['data']['positive_size'] + all_dict[predictor]['data'][cur_fold]['positive_size']
                    all_dict[predictor]['data']['negative_size'] = all_dict[predictor]['data']['negative_size'] + all_dict[predictor]['data'][cur_fold]['negative_size']
                    all_dict[predictor]['data']['size'] = all_dict[predictor]['data']['size'] + all_dict[predictor]['data'][cur_fold]['size']
                    
                    
    #                 print (predictor + " - fold: " + str(cur_fold) + "[P:" + str(all_dict[predictor]['data'][cur_fold]['positive_size']) + ",N:" + str(all_dict[predictor]['data'][cur_fold]['negative_size']) + "]")
                                                    
                    all_dict[predictor]['result'][cur_fold] = {}
                    for score_type in score_types:
                        for score_metric in score_point_metrics + score_metrics:
                            all_dict[predictor]['result'][cur_fold][score_type + '_' + score_metric] = np.nan
                    
                    #interplated precision recall curve 
                    metrics_dict = alm_fun.classification_metrics(all_dict[predictor]['data'][cur_fold]['truth'], all_dict[predictor]['data'][cur_fold]['predicted'])[1]                
                    cur_fold_interp_recalls =  [alm_fun.get_interpreted_x_from_y(new_precision,metrics_dict['balanced_recalls'],metrics_dict['balanced_precisions']) for new_precision in precision_pivots]
                    cur_fold_interp_recalls.append(0)                
                    cur_fold_interp_precisions = precision_pivots
                    cur_fold_interp_precisions = cur_fold_interp_precisions + [1]     
                    [cur_fold_interp_aubprc,x,cur_fold_interp_brfp,y] = alm_fun.cal_pr_values(cur_fold_interp_precisions,cur_fold_interp_recalls)
                    #orginal precision recall curve
                    cur_fold_org_recalls = metrics_dict['balanced_recalls']
                    cur_fold_org_precisions = metrics_dict['balanced_precisions']
                    [cur_fold_org_aubprc,x,cur_fold_org_brfp,y] = alm_fun.cal_pr_values(metrics_dict['balanced_precisions'],metrics_dict['balanced_recalls'])                                
                    #interplated roc curve 
                    cur_fold_interp_fprs =  [alm_fun.get_interpreted_x_from_y(new_tpr,metrics_dict['fprs'],metrics_dict['tprs']) for new_tpr in tpr_pivots]
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
                    
            np.save(output_file.replace('img','npy') + '.npy',all_dict)
                                        
        for predictor in predictors:
            for score_type in score_types:
                # data points for the curve 
                for score_point_metric in score_point_metrics:                     
                    cur_score_list = []
                    for cur_fold in range(cv_folds): 
                        if (cv_folds > 1) & (score_type == 'org'):
                            cur_score_list.append(all_dict[predictor]['result'][cur_fold]['interp_' + score_point_metric])  
                        else:
                            cur_score_list.append(all_dict[predictor]['result'][cur_fold][score_type + '_' + score_point_metric])
                                              
                    all_dict[predictor]['result'][score_type + '_' + score_point_metric]= np.mean(cur_score_list,axis= 0)
                    all_dict[predictor]['result'][score_type + '_' + score_point_metric +'_se']= np.std(cur_score_list,axis= 0)/np.sqrt(cv_folds)
                    if score_type + '_' + score_point_metric in  ['interp_recalls','interp_fprs','org_recalls','org_fprs']:
                        all_dict[predictor]['result'][score_type + '_' + score_point_metric +'_upper']= np.minimum(all_dict[predictor]['result'][score_type + '_' + score_point_metric] + all_dict[predictor]['result'][score_type + '_' + score_point_metric +'_se'], 1)
                        all_dict[predictor]['result'][score_type + '_' + score_point_metric +'_lower']= np.maximum(all_dict[predictor]['result'][score_type + '_' + score_point_metric] - all_dict[predictor]['result'][score_type + '_' + score_point_metric +'_se'], 0)
                        
                #statistics for each predictor                            
                for score_metric in score_metrics:
                    compare_score_list = []
                    cur_score_list = []    
                    
                    for cur_fold in range(cv_folds):   
                        compare_score_list.append(all_dict[compare_predictor]['result'][cur_fold][score_type + '_' + score_metric])                            
                        cur_score_list.append(all_dict[predictor]['result'][cur_fold][score_type + '_' + score_metric]) 
                                                
                    if run_bootstrapping == 0:                               
                        all_dict[predictor]['result'][score_type + '_' + score_metric + '_df']= cv_folds -1                        
                        all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean']= np.mean(cur_score_list)                    
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_se']= np.std(cur_score_list,ddof= 1)/np.sqrt(cv_folds)
                        if two_sided == 0:                        
                            all_dict[predictor]['result'][score_type + '_' + score_metric +'_pvalue']= stats.ttest_rel(compare_score_list,cur_score_list)[1]/2 # one sided test
                        else:
                            all_dict[predictor]['result'][score_type + '_' + score_metric +'_pvalue']= stats.ttest_rel(compare_score_list,cur_score_list)[1] # one sided test
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_effect_size'] = np.mean(np.array(compare_score_list) - np.array(cur_score_list))
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_effect_size_se'] = np.std(np.array(compare_score_list) - np.array(cur_score_list))/np.sqrt(cv_folds)
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_ci'] = "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric +'_effect_size'] - runtime['t_value']*all_dict[predictor]['result'][score_type + '_' + score_metric +'_effect_size_se']) + ' ~ inf'
                    
                    if run_bootstrapping == 1:                               
                        all_dict[predictor]['result'][score_type + '_' + score_metric + '_df']= cv_folds -1                        
                        all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean']= np.mean(cur_score_list)                    
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_se']= np.std(cur_score_list,ddof= 1)
                        
                        #calculate z score
                        diff = np.array(compare_score_list) - np.array(cur_score_list)
#                         cur_z_statistic = (np.mean(compare_score_list) - np.mean(cur_score_list))/np.sqrt(np.std(compare_score_list,ddof = 1)**2 + np.std(cur_score_list,ddof = 1)**2)
                        cur_z_statistic = np.mean(diff)/np.std(diff,ddof = 1)

                        if two_sided == 0:                        
                            all_dict[predictor]['result'][score_type + '_' + score_metric +'_pvalue']= stats.norm.sf(abs(cur_z_statistic))
                        else:
                            all_dict[predictor]['result'][score_type + '_' + score_metric +'_pvalue']= stats.norm.sf(abs(cur_z_statistic))*2
                            
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_effect_size'] = np.mean(diff)
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_effect_size_se'] = np.std(diff,ddof = 1)                                                
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_ci'] = "{:.3f}".format(np.percentile(diff,5)) + ' ~ inf'

                    if predictor == compare_predictor:
                        all_dict[predictor]['result'][score_type + '_' + score_metric +'_display'] =  "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean']) + '±' + "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_se'])
                    else:                    
#                         all_dict[predictor]['result'][score_type + '_' + score_metric +'_display'] =  "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean']) + '±' + "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_se']) + ' (' + all_dict[predictor]['result'][score_type + '_' + score_metric +'_ci'] +',' + "{:.2e}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_pvalue']) +')'

                        if all_dict[predictor]['result'][score_type + '_' + score_metric + '_pvalue'] <= 0.05:                                                                          
                            all_dict[predictor]['result'][score_type + '_' + score_metric +'_display'] = "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean']) + '±' + "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_se']) + ' [' +  "{:.1e}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_pvalue']) +'*]'
                        else:
                            all_dict[predictor]['result'][score_type + '_' + score_metric +'_display'] =  "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean']) + '±' + "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_se']) + ' [' + "{:.1e}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_pvalue']) +']'

                    if cv_folds == 1:
                         all_dict[predictor]['result'][score_type + '_' + score_metric +'_display'] =  "{:.3f}".format(all_dict[predictor]['result'][score_type + '_' + score_metric + '_mean'])                            
                                                                                        
        #**************************************************************
        # save statistics output to csv file 
        #**************************************************************
        output_cols = ['predictor']
        for score_type in score_types:
            for score_metric in score_metrics:
                for score_statistic in score_statistics:
                    output_cols.append(score_type + '_' + score_metric + '_' + score_statistic)
        score_output = pd.DataFrame(columns = output_cols + ['size','positive_size','negative_size'])      
          
        for predictor in predictors:
            cur_score_list = []     
            for score_type in score_types:
                for score_metric in score_metrics:
                    for score_statistic in score_statistics:    
                        cur_score_list.append(all_dict[predictor]['result'][score_type + '_' + score_metric + '_' + score_statistic])  
                        
            cur_score_list.append(all_dict[predictor]['data']['size'])
            cur_score_list.append(all_dict[predictor]['data']['positive_size'])
            cur_score_list.append(all_dict[predictor]['data']['negative_size'])
                                                                  
            score_output.loc[predictor] = [predictor] + cur_score_list                                                  
        score_output.to_csv(output_file.replace('img','csv') + '.csv')


        #**************************************************************
        # Plot curves for each predictor
        #**************************************************************
        plot_predictors = list(set(predictors) - set(runtime['no_plot_predictors']))
           
        for i in range(len(plot_metrics)):        
            plot_metric = plot_metrics[i]
            plot_metric_order = plot_metric_orders[i]
            plot_output_file = output_file + '_' + plot_metric + '.png'
        
            fig = plt.figure(figsize=(fig_x, fig_y),dpi = dpi)
            plt.clf()
#             plt.rc( 'text', usetex=True ) 
            plt.rcParams["font.family"] = "Helvetica"    
            ax = plt.subplot()
            
            plot_score_metric = plot_metric.split('_')[1]
            plot_score_type = plot_metric.split('_')[0]
    
    
            print('****' + plot_metric + '****')     

            color_index = 0
            for predictor in plot_predictors:
                if plot_score_metric == 'auroc':
                    if plot_score_type == 'interp':
                        ax = alm_fun.plot_cv_roc_ax(all_dict[predictor]['result']['interp_fprs'],all_dict[predictor]['result']['interp_tprs'],all_dict[predictor]['result']['interp_fprs_upper'],all_dict[predictor]['result']['interp_fprs_lower'],ax,color = predictor_dict['colors'][predictor],size_factor = size_factor)
                    if plot_score_type == 'org':
                        ax = alm_fun.plot_cv_roc_ax(all_dict[predictor]['result']['org_fprs'],all_dict[predictor]['result']['org_tprs'],all_dict[predictor]['result']['org_fprs_upper'],all_dict[predictor]['result']['org_fprs_lower'],ax,color = predictor_dict['colors'][predictor],size_factor = size_factor)
                    print(predictor)                        
        
                if plot_score_metric == 'aubprc':
                    if plot_score_type == 'interp':
                        ax = alm_fun.plot_cv_prc_ax(all_dict[predictor]['result']['interp_recalls'],all_dict[predictor]['result']['interp_precisions'],all_dict[predictor]['result']['interp_recalls_upper'],all_dict[predictor]['result']['interp_recalls_lower'],ax,color = predictor_dict['colors'][predictor],size_factor = size_factor)                    
                    if plot_score_type == 'org':
                        ax = alm_fun.plot_cv_prc_ax(all_dict[predictor]['result']['org_recalls'],all_dict[predictor]['result']['org_precisions'],all_dict[predictor]['result']['org_recalls_upper'],all_dict[predictor]['result']['org_recalls_lower'],ax,color = predictor_dict['colors'][predictor],size_factor = size_factor)                    
                    print(predictor)
                color_index += 1  
                   

            #**************************************************************************************************************************                        
            # Sort the predictor with AUBPRC       
            #**************************************************************************************************************************
            sort_metrics = plot_metric_order + '_mean'
            sort_metric_list = []
            for predictor in plot_predictors:
                sort_metric_list.append(all_dict[predictor]['result'][sort_metrics])                                                                                                                            
            sort_indices = list(np.argsort(sort_metric_list))
            sort_indices.reverse()    
            plot_predictors = [plot_predictors[i] for i in sort_indices]  
            
            #**************************************************************************************************************************                        
            # Create Figure Legend   (output AUBUPRC and AUROC)
            #**************************************************************************************************************************        
            legend_data = []
            columns = 0
            if len(plot_predictors) > 10 :
                n_rows = int(np.ceil(len(plot_predictors)/2))
                columns = 2       
            else:
                n_rows = len(plot_predictors)
                columns = 1
         
            # set legend table data             
            for i in range(n_rows):
                cur_legend_info = []
                legend_col_label = []            
                for column in range(columns):   
                    if column > 0:
                        legend_col_label.append(' ')
                        cur_legend_info.append(' ')
                                        
                    if  i+column*n_rows < len(plot_predictors):              
                        if plot_score_metric == 'auroc':
                            legend_col_label.append('Method')
                            cur_legend_info.append(predictor_dict['names'][plot_predictors[i+column*n_rows]])
                            legend_col_label.append('AUROC [P]')
                            cur_legend_info.append(all_dict[plot_predictors[i+ column*n_rows]]['result'][plot_score_type + '_' + plot_score_metric +'_display'])                                                                                                                                                                                                                                                                                 
                        if plot_score_metric == 'aubprc':
                            legend_col_label.append('Method')
                            cur_legend_info.append(predictor_dict['names'][plot_predictors[i+column*n_rows]])
                            legend_col_label.append('AUBPRC [P]')
                            cur_legend_info.append(all_dict[plot_predictors[i+column*n_rows]]['result'][plot_score_type + '_' + plot_score_metric +'_display'])
                            legend_col_label.append('R90BP [P]')
                            cur_legend_info.append(all_dict[plot_predictors[i+column*n_rows]]['result'][plot_score_type + '_brfp_display'])
                        if show_size == 1:
                            legend_col_label.append('Size')
                            cur_legend_info.append('[' +  str(all_dict[plot_predictors[i+column*n_rows]]['data']['size']) + ',' + 'P: ' + str(all_dict[plot_predictors[i+column*n_rows]]['data']['positive_size']) + ' N: ' + str(all_dict[plot_predictors[i+column*n_rows]]['data']['negative_size']) + ']')
                            
                    else:
                        if plot_score_metric == 'auroc':
                            legend_col_label.append('Method')
                            cur_legend_info.append('')
                            legend_col_label.append('AUROC [P]')
                            cur_legend_info.append('')                                                                                                                                                                                                                                                                                 
                        if plot_score_metric == 'aubprc':
                            legend_col_label.append('Method')
                            cur_legend_info.append('')
                            legend_col_label.append('AUBPRC [P]')
                            cur_legend_info.append('')
                            legend_col_label.append('R90BP [P]')
                            cur_legend_info.append('')
                        if show_size == 1:
                            legend_col_label.append('Size')
                            cur_legend_info.append('')
                            
                                     
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
                    if  i+column*n_rows < len(plot_predictors):     
                        
                        if column == 0:                                                                                                                       
                            if plot_score_metric == 'auroc':         
                                legend_table._cells[(i+1,column*(2+show_size))].get_text().set_color(predictor_dict['colors'][plot_predictors[i + column*n_rows]])
                            if plot_score_metric == 'aubprc':
                                legend_table._cells[(i+1,column*(3+show_size))].get_text().set_color(predictor_dict['colors'][plot_predictors[i + column*n_rows]])                            
                        else:
                            if plot_score_metric == 'auroc':         
                                legend_table._cells[(i+1,column*(2+show_size) + 1)].get_text().set_color(predictor_dict['colors'][plot_predictors[i + column*n_rows]])
    #                             legend_table._cells[(i+1,column*(2+show_size))].get_text().set_backgroundcolor('black')
                            if plot_score_metric == 'aubprc':
                                legend_table._cells[(i+1,column*(3+show_size) + 1)].get_text().set_color(predictor_dict['colors'][plot_predictors[i + column*n_rows]])
    #                             legend_table._cells[(i+1,column*(3+show_size))].get_text().set_backgroundcolor('black')
                                                              
                        
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
                      
            ax.set_title(title, size=40*size_factor, pad = 20)
            fig.tight_layout()
            plt.savefig(plot_output_file)  
            
    def plot_target_result(self,runtime):
                
        #### load target data with predictions ( via create_varity_target_data in alm_humandb.py)
        
        varity_target_df = pd.read_csv(runtime['target_predicted_file'])
        
        
        cv_folds = 1                                                                                                
        test_result_dict = {}
        filter_indices_dict = {}
        filter_indices_dict[0] = plot_selected_target_df.index
        clinvar_pvids_performance.loc[cur_pvid,:] = np.nan                                         
        for predictor in predictors:
            cur_test_result_dict = {}
            cur_test_result_dict['test_y_truth_dict'] = {}
            cur_test_result_dict['test_y_predicted_dict'] = {}                                    
            cur_test_result_dict['test_y_truth_dict'][0] = plot_selected_target_df['target_label']
            cur_test_result_dict['test_y_predicted_dict'][0] = plot_selected_target_df[predictor]                      
            test_result_dict[predictor] = cur_test_result_dict                                      
            metrics_dict = alm_fun.classification_metrics(plot_selected_target_df['label'], plot_selected_target_df[predictor] )[1]
            clinvar_pvids_performance.loc[cur_pvid,'size'] = metrics_dict['size']
            clinvar_pvids_performance.loc[cur_pvid,'prior'] = metrics_dict['prior']
            clinvar_pvids_performance.loc[cur_pvid, predictor] = metrics_dict['aubprc']
                         
        output_file = self.project_path + 'output/img/' + cur_pvid + '_cutoff_' + str(maf_cutoff) + '_' + target + '_'  + type + '_' + experimental_score + '_scores_result.png'                               
        self.plot_classification_curve(runtime,type,predictors,compare_predictor, cv_folds, filter_indices_dict,test_result_dict,output_file,size_factor = 1.25, title = title,dpi = 600)

    def plot_mave_result(self,runtime):
        alm_predictor = self.proj.predictor[runtime['predictor']]
        alm_dataset = alm_predictor.data_instance
        cur_test_df = pd.read_csv(runtime['independent_test_file']) 
        cur_test_df = cur_test_df.loc[cur_test_df[runtime['dependent_variable']].notnull(),:]
        cur_test_df = cur_test_df.loc[cur_test_df['mave_label_confidence'] >= runtime['mave_label_confidence_cutoff'],:]
        cur_test_df = cur_test_df.loc[cur_test_df['symbol'].isin(['TPK1','CALM1','PTEN','CBS','BRCA1','VKORC1']),:]
#         cur_test_df = cur_test_df.loc[cur_test_df['symbol'].isin(['CALM1']),:] 
        
#         cur_tmpt_df = cur_test_df.loc[cur_test_df['symbol'] == 'TPMT',:]
#         cur_tmpt_df.to_csv(runtime['independent_test_file'].split('.')[0] + '_TPMT.csv',index = False)
#         cur_test_df = cur_test_df.loc[cur_test_df['VARITY_R_weight'] <= runtime['mave_weight_cutoff'],:]
#         cur_test_df = cur_test_df.loc[cur_test_df['VARITY_ER_weight'] <= runtime['mave_weight_cutoff'],:]
#         cur_test_df = cur_test_df.loc[cur_test_df['symbol'].isin(['UBE2I','TPMT','SUMO1','PTEN','CBS','BRCA1','VKORC1']),:]        
#         cur_test_df = cur_test_df.loc[cur_test_df['symbol'].isin(['GDI1','TECR','MTHFR','TPK1','CALM1','PTEN','CBS','BRCA1','VKORC1']),:]
#         cur_test_df = cur_test_df.loc[cur_test_df['symbol'].isin(['UBE2I','SUMO1','PTEN','CBS','BRCA1','VKORC1']),:]
         
        cur_test_df = cur_test_df.loc[self.varity_obj.filter_test(alm_predictor,cur_test_df,runtime),:]               
        unique_keys = list(cur_test_df[runtime['correlation_key_col']].unique())
                
        correlation_df = pd.DataFrame(columns = [runtime['correlation_key_col'],'predictor','correlation','p_value','mean','ste','star'])
        i = 0                           
        for key in unique_keys:           
            for plot_predictor in runtime['compare_predictors']:
                alm_plot_predictor = self.proj.predictor[plot_predictor]
                if plot_predictor in cur_test_df.columns:
                    score_name = plot_predictor
                else:
                    score_name = alm_plot_predictor.features[0]
                                    
                cur_truth = cur_test_df.loc[cur_test_df[runtime['correlation_key_col']] == key,runtime['dependent_variable']]
                cur_prediction = cur_test_df.loc[cur_test_df[runtime['correlation_key_col']] == key,score_name]
                
                
                
                if runtime['correlation_type'] == 'Pearson':
                    cur_correlation = alm_fun.pcc_cal(cur_truth,cur_prediction)
                if runtime['correlation_type'] == 'Spearman':
                    cur_correlation = alm_fun.spc_cal(cur_truth,cur_prediction)
                                                
                correlation_df.loc[i,:] = [key,plot_predictor,abs(cur_correlation),np.nan,np.nan,np.nan,np.nan]                
                i += 1
        pass 
        correlation_pivot_df =  pd.pivot(correlation_df[[runtime['correlation_key_col'],'predictor','correlation']],index = 'predictor',columns = runtime['correlation_key_col'] ,values = 'correlation')        
        correlation_df = correlation_df.sort_values(['predictor',runtime['correlation_key_col']])                        
        correlation_compare_indices = correlation_df['predictor'] == runtime['compare_to_predictor']
         
        correlation_df['predictor_name'] = correlation_df['predictor'].apply(lambda x: x + '(•)' if x in runtime['nucleotide_predictors'] else x)        
        for predictor_name in runtime['compare_predictors']:
            
            correlation_indices = correlation_df['predictor'] == predictor_name
            if predictor_name != runtime['compare_to_predictor']:
                
                valid_cols = correlation_pivot_df.columns[correlation_pivot_df.loc[predictor_name,:].notnull()]
                cur_correlation_pvalue = stats.ttest_rel(correlation_pivot_df.loc[predictor_name,valid_cols],correlation_pivot_df.loc[runtime['compare_to_predictor'],valid_cols])[1]/2
                cur_correlation_effect_size = np.mean(correlation_pivot_df.loc[runtime['compare_to_predictor'],valid_cols] - correlation_pivot_df.loc[predictor_name,valid_cols])
                cur_correlation_effect_size_se = np.std(correlation_pivot_df.loc[runtime['compare_to_predictor'],valid_cols] - correlation_pivot_df.loc[predictor_name,valid_cols])/np.sqrt(len(unique_keys))
                cur_correlation_ci = "{:.3f}".format(cur_correlation_effect_size - runtime['t_value'] * cur_correlation_effect_size_se) + ' ~ inf'                
                correlation_df.loc[correlation_indices,'star'] = self.get_stars_from_pvalue(cur_correlation_pvalue,pvalue_staronly = True)
                correlation_df.loc[correlation_indices,'p_value']  =  cur_correlation_pvalue
                if cur_correlation_pvalue < 0.05:
                    correlation_df.loc[correlation_indices,'p_value_str']  =  'P=' + "{:.1e}".format(cur_correlation_pvalue) + '*'
                else:
                    correlation_df.loc[correlation_indices,'p_value_str']  =  'P=' + "{:.1e}".format(cur_correlation_pvalue)
                correlation_df.loc[correlation_indices,'effect_size']  =  cur_correlation_effect_size
                correlation_df.loc[correlation_indices,'effect_size_se']  =  cur_correlation_effect_size_se
                correlation_df.loc[correlation_indices,'ci']  =  cur_correlation_ci

            correlation_df.loc[correlation_indices,'mean']  =  np.round(correlation_df.loc[correlation_indices,'correlation'].mean(),3)
            correlation_df.loc[correlation_indices,'ste']  =  np.round(correlation_df.loc[correlation_indices,'correlation'].std()/np.sqrt(len(unique_keys)),3)
            
        correlation_df['mean_str'] = correlation_df['mean'].apply(lambda x: "{:.3f}".format(x))
        correlation_df['ste_str'] = correlation_df['ste'].apply(lambda x: "{:.3f}".format(x)) 
        correlation_df['display'] = correlation_df['predictor_name'] + ':' + correlation_df['mean_str'] + '±' + \
                                    correlation_df['ste_str'] + ' [' + correlation_df['p_value_str'] + ']'
                                    
        compare_index = correlation_df['predictor'] == runtime['compare_to_predictor']
        correlation_df.loc[compare_index,'display'] = correlation_df.loc[compare_index,'predictor'] + ':' + correlation_df.loc[compare_index,'mean_str'] + '±' + correlation_df.loc[compare_index,'ste_str']
          
      
        if predictor_name != runtime['compare_to_predictor']:
            correlation_df.loc[correlation_indices,'display'] = correlation_df.loc[correlation_indices,'predictor'] + ': ' + correlation_df.loc[correlation_indices,'mean'].astype('str') + '±' + correlation_df.loc[correlation_indices,'ste'].astype('str') + ' [' + correlation_df.loc[correlation_indices,'p_value_str'] + ']'                                  
        else:
            correlation_df.loc[correlation_indices,'display'] = correlation_df.loc[correlation_indices,'predictor'] + ': ' + correlation_df.loc[correlation_indices,'mean'].astype('str') + '±' + correlation_df.loc[correlation_indices,'ste'].astype('str')
            
        pass
        correlation_df = correlation_df.sort_values(['mean'],ascending = False)
        
        output_figure_file = self.fun_perfix(runtime, 'img') + '_' + runtime['independent_test_name'] + '_filter' + '_' + str(runtime['filter_test_score'])  +  '_' + runtime['correlation_type'] + '_' + runtime['dependent_variable'] + '.png'
        output_csv_file  =  self.fun_perfix(runtime, 'csv') + '_' + runtime['independent_test_name'] + '_filter' + '_' + str(runtime['filter_test_score'])  +  '_' + runtime['correlation_type'] + '_' + runtime['dependent_variable'] + '.csv'
        
        correlation_output_df = correlation_df.drop(columns = ['symbol','correlation']).drop_duplicates()
        correlation_pivot_df['predictor'] = correlation_pivot_df.index
        correlation_pivot_df = correlation_pivot_df.reset_index(drop = True)
        correlation_output_df = pd.merge(correlation_output_df,correlation_pivot_df,how = 'left')
        
        correlation_output_df.to_csv(output_csv_file.replace('.csv','_output.csv'))
                
        ylabel = runtime['correlation_key_col']
        ylabel = ''
        xlabel = runtime['correlation_type'] + ' Correlation Coefficient'                
        self.plot_correlation_barplot (runtime,runtime['plot_title'],runtime['correlation_key_col'],xlabel,ylabel,correlation_df,output_figure_file,output_csv_file)                
        
        
    def plot_ldlr_result(self,runtime):
        correlation_df = pd.read_csv(runtime['ldlr_correlation_file'])
        correlation_df['predictor'] = np.nan
        for plot_predictor in runtime['compare_predictors']:
            alm_plot_predictor = self.proj.predictor[plot_predictor]
            if 'VARITY' in plot_predictor:
                score_name = plot_predictor
            else:
                score_name = alm_plot_predictor.features[0]
                
            correlation_df.loc[correlation_df['score'] == score_name,'predictor'] = plot_predictor
        pass
        
        correlation_df.columns = ['score','mean','ste','effect_size','ci','p_value','predictor']
        
        correlation_df['mean_str'] = correlation_df['mean'].apply(lambda x: "{:.3f}".format(x))
        correlation_df['ste_str'] = correlation_df['ste'].apply(lambda x: "{:.3f}".format(x))
        correlation_df['effect_size_str'] = correlation_df['effect_size'].apply(lambda x: "{:.3f}".format(x))
        correlation_df['p_value_str']  =  correlation_df['p_value'].apply(lambda x: 'P=' + "{:.1e}".format(x) + '*' if x<0.05 else 'P=' + "{:.1e}".format(x))
# 
#         correlation_df['display'] = correlation_df['predictor'] + ':' + correlation_df['mean_str'] + '±' + \
#                                     correlation_df['ste_str'] + ' [' + correlation_df['p_value_str'] + ',CI=' + correlation_df['ci'] + ']'
        
        correlation_df['predictor_name'] = correlation_df['predictor'].apply(lambda x: x + '(•)' if x in runtime['nucleotide_predictors'] else x )
        correlation_df['display'] = correlation_df['predictor_name'] + ':' + correlation_df['mean_str'] + '±' + \
                                    correlation_df['ste_str'] + ' [' + correlation_df['p_value_str'] + ']'
                                                             
                                    
        compare_index = correlation_df['score'] == runtime['compare_to_predictor']
        correlation_df.loc[compare_index,'display'] = correlation_df.loc[compare_index,'predictor'] + ':' + correlation_df.loc[compare_index,'mean_str'] + '±' + correlation_df.loc[compare_index,'ste_str']                           
        ylabel = ''
        xlabel = runtime['correlation_type'] + ' Correlation Coefficient'
        
        output_figure_file  = runtime['ldlr_correlation_file'].replace('/csv','/img').replace('.csv','.png')
        output_csv_file  = runtime['ldlr_correlation_file'].replace('.csv','_plot.csv')
        
        correlation_df = correlation_df.loc[correlation_df['predictor'].notnull(),:]
        self.plot_correlation_barplot (runtime,runtime['plot_title'],runtime['correlation_key_col'],xlabel,ylabel,correlation_df,output_figure_file,output_csv_file)  
    
    
    def plot_correlation_barplot(self,runtime,plot_title,key_col,xlabel,ylabel,correlation_df,output_figure_file,output_csv_file):                
        fig = plt.figure(figsize=(30, 20))
        plt.rcParams["font.family"] = "Helvetica"                  
        ax = plt.subplot()
        
        correlation_output_df = correlation_df[['predictor_name','predictor','display','mean','ste','effect_size','ci','p_value']]
        correlation_output_df = correlation_output_df.drop_duplicates()  
        correlation_output_df = correlation_output_df.loc[~correlation_output_df['predictor'].isin(runtime['no_plot_predictors']),:]
        
        display_list =  list(correlation_output_df['display'])
#         ax = sns.barplot(x='mean', y="predictor", hue_order = correlation_output_df['display'].unique(),hue="display", data=correlation_output_df,ax = ax)
        ax = sns.barplot(x='mean', y="predictor_name", data=correlation_output_df,ax = ax, linewidth=2.5, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2")
        
        i = 0
        for p in ax.patches:
            ax.annotate(display_list[i].split(':')[1], 
                           (0.055, p.get_y() + p.get_height() / 2), 
                           ha = 'center', va = 'baseline', 
                           size=32,
                           xytext = (0, -10), 
                           textcoords = 'offset points')            
#             ax.annotate(display_list[i], 
#                            (p.get_x() + p.get_width() / 2., p.get_height()), 
#                            ha = 'center', va = 'center', 
#                            size=15,
#                            xytext = (0, -12), 
#                            textcoords = 'offset points')
            i = i+1
        
#         ax.set_ylim(0,1)       
        ax.set_xlabel(xlabel, size=45,labelpad = 10)
        ax.set_ylabel(ylabel, size=45,labelpad = 10)
        ax.tick_params(labelsize=40)
        ax.set_title(plot_title, size=50,pad = 20)
#         ax.legend(loc='upper center', ncol = 2 ,fontsize = '32')
#         plt.setp(ax.get_legend().get_title(), fontsize='35')    
        fig.tight_layout()
        plt.savefig(output_figure_file,dpi = 300)
#         correlation_output_df = correlation_df[['predictor','mean','ste','effect_size','ci','p_value']]
#         correlation_output_df = correlation_output_df.drop_duplicates()  
        correlation_df.to_csv(output_csv_file)

        
        
    def plot_correlation_barplot_old(self,runtime,plot_title,key_col,xlabel,ylabel,correlation_df,output_figure_file,output_csv_file):                
        fig = plt.figure(figsize=(30, 12))
        plt.rcParams["font.family"] = "Helvetica"                  
        ax = plt.subplot()    
        ax = sns.barplot(x=key_col, hue_order = correlation_df['display'].unique(),hue="display", y="correlation", data=correlation_df,ax = ax)
        ax.set_ylim(0,1)       
        ax.set_xlabel(xlabel, size=40,labelpad = 10)
        ax.set_ylabel(ylabel, size=40,labelpad = 10)
        ax.tick_params(labelsize=35)
        ax.set_title(plot_title, size=50,pad = 20)
        ax.legend(loc='upper center', ncol = 2 ,fontsize = '32')
        plt.setp(ax.get_legend().get_title(), fontsize='35')    
        fig.tight_layout()
        plt.savefig(output_figure_file)
        correlation_output_df = correlation_df[['predictor','mean','ste','effect_size','ci','p_value']]
        correlation_output_df = correlation_output_df.drop_duplicates()  
        correlation_output_df.to_csv(output_csv_file)                
                
    def plot_test_result(self,runtime):         
        alm_fun.show_msg(runtime['log'],1,'plot test result started......')
        alm_predictor = self.proj.predictor[runtime['predictor']]
        alm_dataset = alm_predictor.data_instance      
        
        if runtime['cur_test_fold'] == -1:
            test_folds = list(range(alm_predictor.data_instance.test_split_folds))
        else:
            test_folds = [runtime['cur_test_fold']]    
            
        if (runtime['independent_test_file'] != '') | (len(test_folds) == 1):
            run_bootstrapping = 1                             
            #stratified bootstrapping           
            n_folds = runtime['num_bootstrap'] 
        else:
            run_bootstrapping = 0
            n_folds = len(test_folds)            
                  
        
        # Get the filter_indices for each test fold
        remain_indices_dict = {}
        test_result_dict = {}
        
        if runtime['plot_test_with_saved_data'] == 0:                             
            if run_bootstrapping == 1:                                             
                if runtime['independent_test_file'] != '':
                    cur_test_df = pd.read_csv(runtime['independent_test_file'])
                else:                
                    cur_test_indices = alm_dataset.test_splits_df[test_folds[0]]['no_gradient']
                    cur_test_df = alm_dataset.train_data_index_df.loc[cur_test_indices,:] 
                    for plot_predictor in runtime['compare_predictors']: 
                        plot_runtime = runtime.copy()
                        plot_runtime['predictor'] = plot_predictor                    
                        cur_test_result_dict_file = self.fun_perfix(plot_runtime, 'npy',1,target_action = 'test_cv_prediction') + '_test_cv_results.npy'
                        cur_test_result_dict = np.load(cur_test_result_dict_file).item()
                        cur_test_df[plot_predictor] = cur_test_result_dict['test_y_predicted_dict'][test_folds[0]]                   
          
                cur_test_df = cur_test_df.loc[cur_test_df[runtime['dependent_variable']].isin([0,1]),:]            
                remain_indices = self.varity_obj.filter_test(alm_predictor,cur_test_df,runtime)
                cur_test_df = cur_test_df.loc[remain_indices,:].reset_index(drop = True)         
    
                boostrap_indices = {}  
                
                if runtime['num_bootstrap'] == 1:
                    boostrap_indices[0] = list(cur_test_df.index)
                    remain_indices_dict[0] = cur_test_df.index                    
                else: 
                    for i in range(runtime['num_bootstrap']):                
                        labels = np.array(cur_test_df[runtime['dependent_variable']])
                        boostrap_indices[i] = sklearn.utils.resample(list(cur_test_df.index),stratify = labels)
                        remain_indices_dict[i] = cur_test_df.index                    
                for plot_predictor in runtime['compare_predictors']: 
                    plot_runtime = runtime.copy()
                    plot_runtime['predictor'] = plot_predictor  
                    alm_plot_predictor = self.proj.predictor[plot_predictor] 
                    if plot_predictor in cur_test_df.columns:
                        score_name = plot_predictor
                    else:
                        score_name = alm_plot_predictor.features[0]
    
                    cur_test_result_dict = {}
                    cur_test_result_dict['test_y_truth_dict'] = {}
                    cur_test_result_dict['test_y_predicted_dict'] = {}  
                    
                    for i in range(runtime['num_bootstrap']):                     
                        cur_test_bs_df = cur_test_df.loc[boostrap_indices[i],:]
                        cur_test_bs_df = cur_test_bs_df.reset_index(drop = True)                     
                        cur_test_result_dict['test_y_truth_dict'][i] = cur_test_bs_df[runtime['dependent_variable']]
                        cur_test_result_dict['test_y_predicted_dict'][i] = cur_test_bs_df[score_name]
                        
                    test_result_dict[plot_predictor] = cur_test_result_dict            
            else:                               
                for cur_fold in test_folds:     
                    print ('current fold: ' + str(cur_fold))   
                    cur_test_indices = alm_dataset.test_splits_df[cur_fold]['no_gradient']
                    cur_test_df = alm_dataset.train_data_index_df.loc[cur_test_indices,:]                                 
                    remain_indices_dict[cur_fold] = self.varity_obj.filter_test(alm_predictor,cur_test_df,runtime)
    
                for plot_predictor in runtime['compare_predictors']: 
                    plot_runtime = runtime.copy()
                    plot_runtime['predictor'] = plot_predictor                    
                    cur_test_result_dict_file = self.fun_perfix(plot_runtime, 'npy',1,target_action = 'test_cv_prediction') + '_test_cv_results.npy'
                    cur_test_result_dict = np.load(cur_test_result_dict_file).item()
                    test_result_dict[plot_predictor] = cur_test_result_dict                                             
                                       
        # Define the the output file name
        if runtime['independent_test_file'] == '':
            output_file = self.fun_perfix(runtime, 'img') + '_filter' + '_' + str(runtime['filter_test_score'])
        else:
            output_file = self.fun_perfix(runtime, 'img') + '_' + runtime['independent_test_name'] + '_filter' + '_' + str(runtime['filter_test_score'])

        self.plot_classification_curve(runtime,runtime['plot_metric'],runtime['plot_metric_order'],runtime['compare_predictors'],runtime['compare_to_predictor'],n_folds, remain_indices_dict,test_result_dict,output_file,run_bootstrapping = run_bootstrapping, size_factor = runtime['size_factor'],table_scale_factor= runtime['table_scale_factor'], show_size = runtime['plot_show_size'],fig_x = runtime['fig_x'],fig_y = runtime['fig_y'],dpi = runtime['dpi'] )

        alm_fun.show_msg(runtime['log'],1,'plot test result ended......')

    def plot_mv_result(self,runtime):       
         
        alm_predictor = self.proj.predictor[runtime['predictor']]
        extra_data_index = alm_predictor.data_instance.extra_data_index        
        extra_data = alm_predictor.data_instance.extra_train_data_df_lst[extra_data_index].copy()   
        cur_qip_dict = alm_predictor.qip[runtime['mv_qip']]
        cur_extra_data = extra_data.loc[extra_data['set_name'].isin(cur_qip_dict['set_list'])]
        cur_extra_data['index_for_order'] = cur_extra_data.index
            
        if cur_qip_dict['direction'] == 0:
            cur_extra_data  = cur_extra_data.sort_values([cur_qip_dict['qip_col'],'index_for_order'])
        else:
            cur_extra_data  = cur_extra_data.sort_values([cur_qip_dict['qip_col'],'index_for_order'],ascending = False)
                        
        n = int(cur_qip_dict['mv_data_points'])
        r = cur_qip_dict['mv_size_percent']/100
        m = cur_extra_data.shape[0]            
        x= int(m*(1-r)/(n-1))
        mv_size = int(r*m)    
        mv_data_points =  cur_qip_dict['mv_data_points']                 
        cur_direction = cur_qip_dict['direction']            
       
        if cur_direction == 0:
            direction = 'low to high'
        if cur_direction == 1:
            direction = 'high to low'
        
        addon_set_name = str(cur_qip_dict['set_list'])
        property = runtime['mv_qip']
            
        sorted_result_file = self.fun_perfix(runtime, 'csv',1,target_action = 'mv_analysis')  + '_' + runtime['mv_qip'] + '.csv'
        cols = ['window','train_' + alm_predictor.tune_obj,'train_' + alm_predictor.tune_obj + '_ste','test_' + alm_predictor.tune_obj ,'test' + alm_predictor.tune_obj]
        cur_metric = alm_predictor.tune_obj      
        cur_sorted_hp_df = pd.read_csv(sorted_result_file)
        cur_sorted_hp_df['range_start'] = np.nan
        cur_sorted_hp_df['range_end'] = np.nan

                        
        varity_0_score_df = cur_sorted_hp_df.loc[cur_sorted_hp_df.index[0],:]
        varity_1_score_df = cur_sorted_hp_df.loc[cur_sorted_hp_df.index[-1],:]
        varity_0_score = varity_0_score_df[cur_metric]
        varity_1_score = varity_1_score_df[cur_metric]
        cur_sorted_hp_df = cur_sorted_hp_df.loc[cur_sorted_hp_df.index[1:-1],:]
        cur_sorted_hp_df['mv_id'] = cur_sorted_hp_df['mv_id'].astype(int)
                                
        sorted_score_mean = cur_sorted_hp_df[cur_metric].mean()
        sorted_score_se = cur_sorted_hp_df[cur_metric].std()/np.sqrt(cur_sorted_hp_df.shape[0])
        sorted_ci_plus = 1.96*sorted_score_se
        sorted_ci_minus = 0 - 1.96*sorted_score_se                    

        cur_sorted_hp_df[cur_metric + '_diff_base'] = cur_sorted_hp_df[cur_metric] - varity_0_score            
        
        
        fig = plt.figure(figsize=(runtime['fig_x'],runtime['fig_y']))
        plt.rcParams["font.family"] = "Helvetica"  
        plt.clf()        
        ax = plt.subplot()
        ax.plot(cur_sorted_hp_df['mv_id'],cur_sorted_hp_df[cur_metric],linewidth=6,marker='o', markersize=10,color = '#558ED5')
                                
        max_index = cur_sorted_hp_df.loc[cur_sorted_hp_df[cur_metric] == cur_sorted_hp_df[cur_metric].max(),:].index[0]
        sorted_spc = alm_fun.spc_cal (cur_sorted_hp_df['mv_id'],cur_sorted_hp_df[cur_metric])            
        sorted_low_spc = alm_fun.spc_cal (cur_sorted_hp_df.loc[1:max_index,'mv_id'],cur_sorted_hp_df.loc[1:max_index,cur_metric])            
        sorted_high_spc = alm_fun.spc_cal (cur_sorted_hp_df.loc[max_index:mv_data_points,'mv_id'],cur_sorted_hp_df.loc[max_index:mv_data_points,cur_metric])
                                                                
#             ax.set_xticks([1,20,40,60,80,100])
        ax.plot([cur_sorted_hp_df['mv_id'].min(),cur_sorted_hp_df['mv_id'].max()],[sorted_score_mean,sorted_score_mean],color = 'black')
        ax.plot([cur_sorted_hp_df['mv_id'].min(),cur_sorted_hp_df['mv_id'].max()],[sorted_score_mean + sorted_ci_plus,sorted_score_mean + sorted_ci_plus],ls = '-.',color = 'black')
        ax.plot([cur_sorted_hp_df['mv_id'].min(),cur_sorted_hp_df['mv_id'].max()],[sorted_score_mean + sorted_ci_minus,sorted_score_mean + sorted_ci_minus],ls = '-.',color = 'black')     
        ax.set_ylabel('10 Folds AUBPRC',size = 28,labelpad = 10)
        ax.set_xlabel('Moving windows (window size: ' + str('{:,}'.format(mv_size))  +', PCC: ' + str('{:.3f}'.format(sorted_spc)) +  ')', size = 28,labelpad = 10)
        ax.set_title('Add-on set(s): ' + addon_set_name + '\n (Examples ordered by ' + property + ' ' + direction + ')' ,size = 32,pad = 20)
        ax.tick_params(labelsize=20)
        
        fig.tight_layout(pad = 3)
        plt.savefig(self.fun_perfix(runtime, 'img',1)  + '_' + property + '.png')

        print ('OK')
                  
    def plot_data_weight(self,runtime):        
        
        alm_predictor = self.proj.predictor[runtime['predictor']]        
        cur_hp_npy = self.fun_perfix(runtime,'npy',target_action = 'hp_tuning')+ '_hp_dict.npy'       
        cur_hp_dict = np.load(cur_hp_npy).item()
        [alpha,beta] = self.update_sample_weights(cur_hp_dict,runtime)            
        extra_data = alm_predictor.data_instance.extra_train_data_df_lst[0].copy()
        core_data = alm_predictor.data_instance.train_data_index_df.copy()               
        all_data = pd.concat([core_data,extra_data])
        
                
        extra_data_nonzero = extra_data.loc[extra_data['weight'] != 0 ,:]
        extra_data_nonzero['target_index'] = extra_data_nonzero.index
        extra_data_nonzero['target_index'] = extra_data_nonzero['target_index'].apply(lambda x: 'extra-' + str(x))
        core_data_nonzero = core_data.loc[core_data['weight'] != 0 ,:]  
        core_data_nonzero['target_index'] = core_data_nonzero.index 
        core_data_nonzero['target_index'] = core_data_nonzero['target_index'].apply(lambda x: 'core-' + str(x))   
        all_data_nonzero = pd.concat([core_data_nonzero,extra_data_nonzero])                
        all_data_nonzero['target_index'].to_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_' + runtime['predictor'] + '_loo_indices.csv',index = False)
        
        weight_scale = 1/all_data_nonzero['weight'].max()
        
        
        key_cols = ['p_vid','aa_pos','aa_ref','aa_alt']
        annotation_cols = ['clinvar_id','clinvar_source','hgmd_source','gnomad_source','humsavar_source','mave_source',
                           'clinvar_label','hgmd_label','gnomad_label','humsavar_label','mave_label','label',
                           'train_clinvar_source','train_hgmd_source','train_gnomad_source','train_humsavar_source','train_mave_source']                           
        score_cols = ['Polyphen2_selected_HVAR_score','Polyphen2_selected_HDIV_score','PROVEAN_selected_score','SIFT_selected_score',
                      'CADD_raw','PrimateAI_score','Eigen-raw_coding','GenoCanyon_score','integrated_fitCons_score','REVEL_score',
                      'M-CAP_score','LRT_score','MutationTaster_selected_score','MutationAssessor_selected_score',
                      'FATHMM_selected_score','MetaSVM_score','MetaLR_score','DANN_score','GERP++_RS',
                      'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','MPC_selected_score','mistic_score',
                      'mpc_score','mpc_obs_exp','mpc_mis_badness','mpc_fitted_score','DeepSequence_score','mave_input','mave_norm','mave_score']        
        feature_cols = ['PROVEAN_selected_score','SIFT_selected_score','evm_epistatic_score','integrated_fitCons_score','LRT_score','GERP++_RS',
                        'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','blosum100','in_domain','asa_mean','aa_psipred_E',
                        'aa_psipred_H','aa_psipred_C','bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_abs_max',
                        'mw_delta','pka_delta','pkb_delta','pi_delta','hi_delta','pbr_delta','avbr_delta','vadw_delta','asa_delta','cyclic_delta','charge_delta',
                        'positive_delta','negative_delta','hydrophobic_delta','polar_delta','ionizable_delta','aromatic_delta','aliphatic_delta','hbond_delta',
                        'sulfur_delta','essential_delta','size_delta']
        
        qip_cols = ['gnomAD_exomes_AF','gnomAD_exomes_AC','gnomAD_exomes_nhomalt','mave_label_confidence','clinvar_review_star','accessibility']
        
        other_cols = ['set_name','log_af']
        
        
        all_data['weight'] = all_data['weight'] * weight_scale        
        all_data[key_cols + feature_cols + ['weight','label']].to_csv(runtime['project_path'] + 'output/csv/' + runtime['predictor'] + '_training.csv',index = False)
                
        all_input_data = all_data[key_cols + feature_cols + annotation_cols + score_cols + feature_cols + qip_cols + other_cols]
        all_input_data['extra_data'] = 1
        all_input_data.loc[all_input_data['set_name'].str.contains('core'),'extra_data'] = 0
        all_input_data.to_csv(runtime['project_path'] + 'output/csv/' + runtime['predictor'] + '_input_data.csv',index = False)

#         all_data_nonzero.loc[[160230,53630,153634],:].to_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_' + runtime['predictor'] + '_loo_test.csv',index = False)
        
#         all_data.loc[all_data['set_name'] == 'core_clinvar_0','weight']
#         all_data.loc[all_data['weight'] != 0 ,:].shape
#         all_data_nonzero.loc[~all_data_nonzero['set_name'].isin(['core_clinvar_1','core_clinvar_0']),'weight'].shape                
#         all_data['set_name'].value_counts()
#         all_data_nonzero['set_name'].value_counts()         
#         all_data_nonzero['weight'].sum()         

        all_data['weight'] = 1        
        weight_norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
        weight_mapper = cm.ScalarMappable(norm=weight_norm, cmap=cm.summer_r)           
        
        n = len(runtime['plot_qip_cols']) 
        if n > 1 :
            n_subplots = int(2*np.ceil(n/2)) * len(runtime['plot_weight_type'])
        else:
            n_subplots = n * len(runtime['plot_weight_type'])
            
        if n <= 2:
            fig = plt.figure(figsize=(21, 6))
        else:
            fig = plt.figure(figsize=(30, 30))
                           
        plt.rcParams["font.family"] = "Helvetica"

        cur_subplot = 0
        for plot_weight_type in runtime['plot_weight_type']:                  
            for i in range(n):     
                plot_sets = runtime['plot_sets'][i]        
                plot_qip_col = runtime['plot_qip_cols'][i]
                
                cur_data_df = all_data.loc[all_data['set_name'].isin(plot_sets)]
                cur_data_df['org_index'] = cur_data_df.index
                cur_data_df = cur_data_df.sort_values([plot_qip_col,'org_index'])
                cur_data_df = cur_data_df.reset_index(drop = True )        
                #### find all qips associated with the plot sets
                plot_qips = []
                for cur_qip in alm_predictor.qip.keys():
                    cur_qip_dict = alm_predictor.qip[cur_qip]
                    
                    if runtime['plot_final_weight'] == 1:
                        if (alm_predictor.name in cur_qip_dict['predictors']) & (cur_qip_dict['enable'] == 1):                    
                            for cur_set in cur_qip_dict['set_list']:
                                if cur_set in plot_sets:
                                    plot_qips.append(cur_qip)
                                    
                    if runtime['plot_final_weight'] == 0:
                        if (alm_predictor.name in cur_qip_dict['predictors']) & (cur_qip_dict['enable'] == 1) & (cur_qip_dict['qip_col'] == plot_qip_col):                                    
                            for cur_set in cur_qip_dict['set_list']:
                                if cur_set in plot_sets:
                                    plot_qips.append(cur_qip) 
                plot_qips = list(set(plot_qips))
                
                # assign weight for each QIP,  and determine the final weight for each variants using the input QIPs
                for cur_qip in plot_qips:
                    cur_qip_dict = alm_predictor.qip[cur_qip]
                    if cur_qip_dict['weight_function'] == 'logistic':
                        k = cur_hp_dict[cur_qip_dict['hyperparameters'][0]]                        
                        L = cur_hp_dict[cur_qip_dict['hyperparameters'][1]]
                        x0 = cur_hp_dict[cur_qip_dict['hyperparameters'][2]]
                        qip_col = cur_qip_dict['qip_col']
                        qip_set_list = cur_qip_dict['set_list']
                        qip_normalized_col = cur_qip + '_normalized'                
                        cur_qip_data_df = cur_data_df.loc[cur_data_df['set_name'].isin(qip_set_list),:]
                        cur_qip_weight =  L/(1+np.exp(0-k*(cur_qip_data_df[qip_normalized_col]-x0)))
        #                 cur_data_df[cur_qip + '_weight'] = 1
        #                 cur_data_df.loc[cur_data_df['set_name'].isin(qip_set_list),cur_qip + '_weight'] = cur_qip_weight
                        cur_data_df.loc[cur_data_df['set_name'].isin(qip_set_list),'weight'] = cur_data_df.loc[cur_data_df['set_name'].isin(qip_set_list),'weight'] * cur_qip_weight
              
                if n_subplots == 1:
                    ax = plt.subplot()
                else:                       
                    ax = plt.subplot(n_subplots/2,2,cur_subplot+1)     
                    cur_subplot = cur_subplot + 1     

                cur_data_df['weight'] = cur_data_df['weight'] *weight_scale                                                      
                cur_data_df['weight_color'] = cur_data_df['weight'].apply(lambda x: matplotlib.colors.rgb2hex(weight_mapper.to_rgba(x)))
                
                
        
                if plot_weight_type == 1:
                    ax.scatter(range(1,cur_data_df.shape[0]+1),cur_data_df['weight'])
                    ax.set_ylabel('weight',size = 25,labelpad = 15, fontweight='bold') 
                    ax.set_xlabel('variants ordered by ' + runtime['plot_qip_cols_name'][i] ,size = 25,labelpad = 15,fontweight='bold')
                    ax.set_ylim(0,1) 
                    
                if plot_weight_type == 0:  
                    ax.set_ylabel(runtime['plot_qip_cols_name'][i],size = 25,labelpad = 15, fontweight='bold')                 
                    ax.set_xlabel('variants ordered by ' + runtime['plot_qip_cols_name'][i] ,size = 25,labelpad = 15,fontweight='bold') 
                                        
                    if plot_qip_col == 'gnomAD_exomes_nhomalt':
                        ax.bar(range(1,cur_data_df.shape[0]+1),np.log10(cur_data_df[plot_qip_col]) + 0.05, bottom = -0.05, color = cur_data_df['weight_color'],width = 1)
                        ax.set_ylim(-0.05,5)        
                                                    
                    if plot_qip_col == 'log_af':
                        ax.bar(range(1,cur_data_df.shape[0]+1),0 - cur_data_df[plot_qip_col], color = cur_data_df['weight_color'],width = 1)
#                         ax.set_ylim(0,6)
                        ax.set_ylim(-0.05,6.5)
                        ax.set_xlabel('variants ordered by allele frequency' ,size = 25,labelpad = 15,fontweight='bold')
                                    
                    if plot_qip_col == 'clinvar_review_star':
                        ax.bar(range(1,cur_data_df.shape[0]+1), cur_data_df[plot_qip_col] + 0.1,bottom = -0.05, color = cur_data_df['weight_color'],width = 1)
                        
                    if plot_qip_col == 'mave_label_confidence':
                        ax.bar(range(1,cur_data_df.shape[0]+1), cur_data_df[plot_qip_col], color = cur_data_df['weight_color'],width = 1)
#                         ax.set_ylim(0.5,1)
                        ax.set_ylim(0.45,1.05)
                        
                    if plot_qip_col == 'accessibility':
                        ax.bar(range(1,cur_data_df.shape[0]+1), cur_data_df[plot_qip_col]+ 0.05,bottom = -0.05, color = cur_data_df['weight_color'],width = 1)
                        ax.set_ylim(-0.05,0.3)
                                       
                    if plot_qip_col == 'gnomAD_exomes_nhomalt':
                        ax.bar(range(1,cur_data_df.shape[0]+1), cur_data_df[plot_qip_col], color = cur_data_df['weight_color'],width = 1)
                                                                                                        

                    
                ax.set_title(runtime['plot_qip_titles'][i],size = 32,pad = 20,fontweight='bold')                
                ax.tick_params(labelsize=25)    
            
        fig.tight_layout(pad = 3)
        fig.subplots_adjust(right = 0.88)                              
        cbar_ax = fig.add_axes([0.90, 0.2, 0.02,0.6])        
        cb = fig.colorbar(weight_mapper, cax=cbar_ax)     
        cb.set_label('weight',size = 30,fontweight='bold',labelpad = 15)
        cb.ax.tick_params(labelsize=25)        
        plt.savefig(self.fun_perfix(runtime,'img') + '_hp_weight_' + str(runtime['plot_weight_type']) + '.png')           
             
                
#         plot_qip = runtime['plot_qip']
# 
#         plot_qip_col = alm_predictor.qip[plot_qip]['qip_col']

#         
#         qip_vmin = np.nanmin(cur_data_df[plot_qip_col])
#         qip_vmax = np.nanmax(cur_data_df[plot_qip_col])         
#         qip_norm = matplotlib.colors.Normalize(vmin=qip_vmin, vmax=qip_vmax, clip=True)
#         qip_mapper = cm.ScalarMappable(norm=qip_norm, cmap=cm.summer_r)
#         cur_data_df['qip_color'] = cur_data_df[plot_qip_col].apply(lambda x: matplotlib.colors.rgb2hex(qip_mapper.to_rgba(x)))
  
#         #### find all qips associated with the plot sets
#         plot_qips = []
#         for cur_qip in alm_predictor.qip.keys():
#             cur_qip_dict = alm_predictor.qip[cur_qip]
#             if (alm_predictor.name in cur_qip_dict['predictors']) & (cur_qip_dict['enable'] == 1):
#                 for cur_set in cur_qip_dict['set_list']:
#                     if cur_set in plot_sets:
#                         plot_qips.append(cur_qip)
#         plot_qips = list(set(plot_qips))
   



#                 if cur_qip != plot_qip:
#                     print (cur_qip)
# #                     ax.scatter(range(1,cur_data_df.shape[0]+1), cur_data_df[cur_qip + '_weight'],color = 'blue')                    
#                 else:  
#                     print (cur_qip)                  
#                     ax.scatter(range(1,cur_data_df.shape[0]+1), cur_data_df[cur_qip + '_weight'],color = cur_data_df['qip_color'])
#                     ax.bar(range(1,cur_data_df.shape[0]+1),cur_data_df[cur_qip + '_weight'],color = cur_data_df['qip_color'],width = 1)
#                      




#         ### revision
# #         ax.scatter(range(1,cur_data_df.shape[0]+1), cur_data_df['weight'] , color = 'black')        
#         ax.set_ylabel('Weight',size = 25,labelpad = 15, fontweight='bold') 
#         ax.set_xlabel('Rank of each ordered variant',size = 25,labelpad = 15,fontweight='bold') 
#         ax.set_ylim(-0.01,1)                                            
#         ax.set_title('Add-on set: ' + str(plot_sets) + ' order by ' + plot_qip_col,size = 32,pad = 20,fontweight='bold')                
#         ax.tick_params(labelsize=25)    
#         fig.tight_layout(pad = 3)
#         fig.subplots_adjust(right = 0.88)                              
#         cbar_ax = fig.add_axes([0.90, 0.2, 0.02,0.6])        
#         cb = fig.colorbar(qip_mapper, cax=cbar_ax)     
#         cb.set_label(plot_qip_col,size = 30,fontweight='bold',labelpad = 15)
#         cb.ax.tick_params(labelsize=25)
#           
#         plt.savefig(self.fun_perfix(runtime,'img') + '_' + plot_qip + '_hp_weight_new.png')  
         
#         all_nonzero_data = all_data.loc[all_data['weight'] != 0,:]
#         max_weight = all_nonzero_data['weight'].max()
#         all_nonzero_data['weight'] = (1/max_weight) * all_nonzero_data['weight']
#         all_nonzero_data_summary = all_nonzero_data.groupby(['label'])['weight'].agg(['sum','count']).reset_index()
#         print(all_nonzero_data_summary)


#         ### first mauscript
#         weight_norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
#         weight_mapper = cm.ScalarMappable(norm=weight_norm, cmap=cm.summer_r)        
#         cur_data_df['weight_color'] = cur_data_df['weight'].apply(lambda x: matplotlib.colors.rgb2hex(weight_mapper.to_rgba(x)))
#         ax.bar(range(1,cur_data_df.shape[0]+1),cur_data_df[plot_qip_col],color = cur_data_df['weight_color'],width = 1)
#         ax.set_ylabel(plot_qip,size = 25,labelpad = 15, fontweight='bold') 
#         ax.set_xlabel('Rank of each ordered variant',size = 25,labelpad = 15,fontweight='bold') 
#         ax.set_title('Add-on set: ' + str(plot_sets) + ' order by ' + plot_qip_col,size = 32,pad = 20,fontweight='bold')                
#         ax.tick_params(labelsize=25)    
#         fig.tight_layout(pad = 3)
#         fig.subplots_adjust(right = 0.88)                              
#         cbar_ax = fig.add_axes([0.90, 0.2, 0.02,0.6])        
#         cb = fig.colorbar(weight_mapper, cax=cbar_ax)     
#         cb.set_label('weight',size = 30,fontweight='bold',labelpad = 15)
#         cb.ax.tick_params(labelsize=25)        
#         plt.savefig(self.fun_perfix(runtime,'img') + '_' + plot_qip + '_hp_weight.png')
#         
        

#         norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
#         mapper = cm.ScalarMappable(norm=norm, cmap=cm.summer_r)
#          
#         cur_data_df['weight_color'] = np.nan
#         for weight in cur_data_df['weight'].unique():
#             weight_color = mapper.to_rgba(weight)
#             cur_data_df.loc[cur_data_df['weight'] == weight,'weight_color'] =  matplotlib.colors.rgb2hex(weight_color)
#         color_full_weight = matplotlib.colors.rgb2hex(mapper.to_rgba(1))
#       
#         fig = plt.figure(figsize=(runtime['fig_x'], runtime['fig_y']))           
#         ax = plt.subplot()
# 
#         ax.bar(range(1,cur_data_df.shape[0]+1), cur_data_df[cur_qip_col],color = cur_data_df['weight_color'],width = 1)
#         ax.set_ylabel(cur_qip_col,size = 25,labelpad = 15, fontweight='bold') 
#         ax.set_xlabel('Rank of each ordered variant',size = 25,labelpad = 15,fontweight='bold') 

#         ax.bar(range(1,cur_data_df.shape[0]+1),cur_data_df['weight'],width = 1)
#         ax.scatter(cur_data_df[cur_qip_col],cur_data_df['weight'])
#         ax.set_xlabel(cur_qip_col,size = 25,labelpad = 15, fontweight='bold') 
#         ax.set_ylabel('Weight of examples',size = 25,labelpad = 15,fontweight='bold')        
                                             
    def plot_feature_shap_interaction(self, runtime):        
        session_id = runtime['session_id']      
        alm_predictor = self.proj.predictor[runtime['predictor']]   
        alm_dataset = alm_predictor.data_instance   
        key_cols = ['p_vid','aa_pos','aa_ref','aa_alt']
                     
        #load current tuned hyper-parameter dictionary
        runtime['hp_dict_file'] = self.fun_perfix(runtime, 'npy',target_action = 'hp_tuning') + '_hp_dict.npy'
        cur_hp_dict = self.load_cur_hp_dict(runtime['hp_dict_file'])
          
        #update the weight hyper-parameter for each extra training example
        [alpha,beta] = self.update_sample_weights(cur_hp_dict,runtime)
        
        train = alm_dataset.train_data_index_df        
        extra_train = alm_dataset.extra_train_data_df_lst[alm_dataset.extra_data_index]
        final_train = pd.concat([extra_train,train])
        final_train = final_train.loc[(final_train['weight'] != 0),:]
                        
#         train_x = final_train[features]
        train_y = final_train['label']                 
        negative_idx = train_y == 1
        positive_idx = train_y == 0
        negative_weights = final_train['weight'][negative_idx].sum()
        positive_weights = final_train['weight'][positive_idx].sum()
        prior_weight = negative_weights/positive_weights
        final_train['weight'][positive_idx] = final_train['weight'][positive_idx]*prior_weight
        final_train['weight'] = final_train['weight'] /final_train['weight'].max()
   
        features = alm_predictor.features + runtime['additional_features']
        shap_output = np.load(runtime['shap_target_npy_file'])
                        
        if 'single' in runtime['shap_target']:
            p_vid = runtime['shap_target'].split('-')[1].split(':')[0]
            aa_ref = runtime['shap_target'].split('-')[1].split(':')[1]
            aa_pos = int(runtime['shap_target'].split('-')[1].split(':')[2])
            aa_alt = runtime['shap_target'].split('-')[1].split(':')[3]

            shap_prediction = pd.read_csv(runtime['shap_target_csv_file'])            
            single_example_index_org = shap_prediction.loc[(shap_prediction['p_vid'] == p_vid) & (shap_prediction['aa_pos'] == aa_pos)  & (shap_prediction['aa_ref'] == aa_ref)  & (shap_prediction['aa_alt'] == aa_alt),:].index
            single_example_index = list(shap_prediction.index).index(single_example_index_org)
            single_example = p_vid + ':' + aa_ref + str(aa_pos) + aa_alt 
            shap_output = shap_output[single_example_index,:,:][:-1,:-1]    
        

        if runtime['shap_type'] == 'output':
             #average matrix
            if shap_output.ndim > 2:
                shap_matrix = np.abs(shap_output).mean(axis = 0)[:-1,:-1]
            else:
                shap_matrix = shap_output
            
        if runtime['shap_type'] == 'performance':
            shap_performance = copy.deepcopy(shap_output)
            shap_performance[train_y == 0,:,:] = 0 - shap_performance[train_y == 0,:,:]
            shap_performance = np.transpose(np.transpose(shap_performance) * np.array(final_train['weight']))
            #average matrix (weighted average)        
            shap_matrix = shap_performance.sum(axis=0)[:-1,:-1] /final_train['weight'].sum()        


        feature_groups_name = ['Conservation Scores','Delta AA Properties','Accessible Surface Area','PPI','BLOSUM','IN/OUT Pfam Domain','Secondary Structure']
#             feature_groups = [self.nine_escores,self.aa_physical_delta_features, ['asa_mean'],self.pisa_features,['blosum100'],['in_domain'],self.aa_psipred_features]
        feature_groups = [['provean_score','sift_score','evm_epistatic_score','integrated_fitCons_score','LRT_score','GERP++_RS','phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds'],
                          ['mw_delta','pka_delta','pkb_delta','pi_delta','hi_delta','pbr_delta','avbr_delta','vadw_delta','asa_delta','cyclic_delta','charge_delta','positive_delta','negative_delta','hydrophobic_delta','polar_delta','ionizable_delta','aromatic_delta','aliphatic_delta','hbond_delta','sulfur_delta','essential_delta','size_delta'],
                          ['asa_mean'],
                          ['bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_abs_max'],
                          ['blosum100'],
                          ['in_domain'],
                          ['aa_psipred_E','aa_psipred_H','aa_psipred_C']]

        if runtime['shap_feature_group'] == 1:                        
            shap_matrix_group = np.zeros([len(feature_groups),len(feature_groups)])
            
            for i in range(len(feature_groups_name)):
                for j in range(len(feature_groups_name)):
                    cur_row_feature_group = feature_groups[i]
                    cur_row_group_name = feature_groups_name[i]
                    cur_row_feature_index = []
                    for feature in cur_row_feature_group:
                        cur_row_feature_index.append(features.index(feature))
                        
                    cur_col_feature_group = feature_groups[j]
                    cur_col_group_name = feature_groups_name[j]
                    cur_col_feature_index = []
                    for feature in cur_col_feature_group:
                        cur_col_feature_index.append(features.index(feature))                    
                    
                    cur_sum = shap_matrix[np.array(cur_row_feature_index)[:,None],np.array(cur_col_feature_index)].sum()
                    shap_matrix_group[i,j] = cur_sum

            feature_group_order = np.argsort(shap_matrix_group.sum(axis=0))
            cur_plot_matrix = shap_matrix_group
            cur_matrix_name = 'shap_' + runtime['shap_type'] + '_' + runtime['shap_target'] + '_group'

            nan_array_group = np.empty((1,len(feature_groups_name)))
            nan_array_group[:] = np.nan
                        
#             pd.DataFrame(shap_matrix).to_csv(self.project_path + 'output/csv/' + cur_matrix_name + '.csv')
#             pd.DataFrame(shap_matrix_group).to_csv(self.project_path + 'output/csv/' + shap_matrix_group + '.csv')               
                                                   
        else:                        
            shap_matrix_df  = pd.DataFrame(shap_matrix)
            shap_matrix_df.index = features
            shap_matrix_df['shap_sum'] = shap_matrix.sum(axis=0)
            shap_matrix_df['feature_group'] = np.nan
            for i in range(len(feature_groups_name)):
                cur_group_name = feature_groups_name[i]
                for feature in feature_groups[i]:
                    shap_matrix_df.loc[feature,'feature_group'] = len(feature_groups_name) - i
            pass
            shap_matrix_df.reset_index(inplace = True,drop = True)
            feature_order = shap_matrix_df.sort_values(['feature_group','shap_sum']).index            
#             feature_order = np.argsort(shap_matrix.sum(axis=0))
            
            cur_plot_matrix = shap_matrix
            cur_matrix_name = 'shap_' + runtime['shap_type'] + '_' + runtime['shap_target']
            nan_array = np.empty((1,len(features)))
            nan_array[:] = np.nan
                        
        ###**********************************************************
        #start figure plotting
        ###**********************************************************    
                        
        if runtime['shap_target'] != 'training':
            example_tag =  '\n[' + single_example +']'
        else:
            example_tag = '\n[All Training Examples]'
        pass
    
        if runtime['shap_feature_group'] == 1: 
            fig_factor = 1
            size_factor = 1.5
            cur_order = feature_group_order
            matrix_columns = list(np.asarray(feature_groups_name)[cur_order])                          
            cur_plot_matrix = cur_plot_matrix[:,cur_order]
            cur_plot_matrix = cur_plot_matrix[cur_order,:]
            cur_plot_matrix = np.vstack((cur_plot_matrix.sum(axis = 0),nan_array_group,cur_plot_matrix))
        else:
            fig_factor = 1.2
            size_factor = 1
            cur_order = feature_order
            matrix_columns = list(np.asarray(features)[cur_order])                          
            cur_plot_matrix = cur_plot_matrix[:,cur_order]
            cur_plot_matrix = cur_plot_matrix[cur_order,:]
            cur_plot_matrix = np.vstack((cur_plot_matrix.sum(axis = 0),nan_array,cur_plot_matrix))   
        pass

        cur_plot_annotation_matrix = copy.deepcopy(cur_plot_matrix)
        cur_plot_annotation_matrix[0,:] = np.round(cur_plot_annotation_matrix[0,:],3)
        cur_plot_annotation_matrix = np.asanyarray(cur_plot_annotation_matrix,dtype = 'U')
        cur_plot_annotation_matrix[1:,:] = ''

                                                                     
        if runtime['shap_type'] == 'performance':
            cur_type_tag = 'model performance'
            cur_cmap = 'RdYlBu_r'
#                 cur_label = 'SHAP Values'
            cur_label = 'Contribution to model performance'
        else:
            cur_type_tag = 'model output'
            cur_cmap = 'RdYlBu_r'
#             cur_label = 'Log odds contribution to \n' + cur_type_tag
            cur_label = 'Contribution to model output'
        pass
        
        cur_savefig = self.fun_perfix(runtime,'img') + '_' + cur_matrix_name + '.png'
        cur_title = 'Feature contribution and interaction on ' + cur_type_tag + ' ' + example_tag

        cur_plot_matrix[cur_plot_matrix == np.float('inf')] = np.nan
        cur_plot_matrix[cur_plot_matrix == np.float('-inf')] = np.nan
            
            
        fig = plt.figure(figsize=(30*fig_factor,20*fig_factor),dpi = 300)                                     
        plt.clf()
        plt.rcParams["font.family"] = "Helvetica"  
        ax = plt.subplot()                
        vmax = np.nanmax(cur_plot_matrix)
        vmin = np.nanmin(cur_plot_matrix)  
        vcenter =  0 
        
        if runtime['shap_feature_group'] == 1:                         
            ax = sns.heatmap(np.rot90(cur_plot_matrix), fmt = "",annot = np.rot90(cur_plot_annotation_matrix),annot_kws={"size": 25*size_factor},cbar_kws={'label': cur_label},
                             norm=colors.SymLogNorm(linthresh = 1e-4, linscale=1,vmin=vmin, vmax=vmax),vmin = -10,vmax = 10, cmap = cur_cmap,ax = ax)
        else:
            ax = sns.heatmap(np.rot90(cur_plot_matrix), fmt = "",annot = False,annot_kws={"size": 25*size_factor},cbar_kws={'label': cur_label},
                            norm=colors.SymLogNorm(linthresh = 1e-4, linscale=1,vmin=vmin, vmax=vmax),vmin = -10,vmax = 10, cmap = cur_cmap,ax = ax)
        
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-10,-1,-0.1,-0.01,-0.001,0,0.001,0.01, 0.1, 1,10])
        cbar.ax.tick_params(labelsize=20*size_factor,length = 10)
        cbar.ax.yaxis.label.set_size(30*size_factor)
        cbar.ax.set_ylabel(cur_label,labelpad= 20*size_factor)
        ax.set_facecolor('white')
        ax.set_xticklabels(['',''] + matrix_columns,fontsize = 30*size_factor,rotation = 90)
        ax.set_yticklabels(matrix_columns[::-1],fontsize = 30*size_factor,rotation = 0)
        ax.tick_params(pad = 10)         
        fig.tight_layout()
        plt.savefig(cur_savefig)   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#         shap_performance_all_examples = shap_performance_by_examples[:,:-1,:-1].sum(axis = 2)
#         
#         shap_performance_feature_all_examples = pd.DataFrame(shap_performance_all_examples)
#         shap_performance_feature_all_examples.columns = self.varity_dict[predictor]['feature'][cur_test_fold]
#         shap_performance_feature_all_examples['index'] = shap_performance_feature_all_examples.index        
#         shap_performance_feature_all_examples= pd.melt(shap_performance_feature_all_examples, ['index'],self.varity_dict[predictor]['feature'][cur_test_fold],'feature','shap')
#         shap_performance_feature_all_examples['hue'] = np.nan
#         shap_performance_feature_all_examples.loc[shap_performance_feature_all_examples['shap'] > 0 ,'hue'] = 'Positive contribution'
#         shap_performance_feature_all_examples.loc[shap_performance_feature_all_examples['shap'] <= 0 ,'hue'] = 'Negative contribution'
#         
#         shap_performance_feature_all_examples_group = np.zeros([shap_performance_all_examples.shape[0],len(feature_groups_name)])
#         for j in range(len(feature_groups_name)):
#             cur_col_feature_group = feature_groups[j]
#             cur_col_group_name = feature_groups_name[j]
#             cur_col_feature_index = []
#             for feature in cur_col_feature_group:
#                 cur_col_feature_index.append(self.varity_dict[predictor]['feature'][cur_test_fold].index(feature))                                                    
#             shap_performance_feature_all_examples_group[:,j] = shap_performance_all_examples[:,cur_col_feature_index].sum(axis=1)
#         pass
#         shap_performance_feature_all_examples_group = pd.DataFrame(shap_performance_feature_all_examples_group)
#         shap_performance_feature_all_examples_group.columns = feature_groups_name
#         shap_performance_feature_all_examples_group['index'] = shap_performance_feature_all_examples_group.index        
#         shap_performance_feature_all_examples_group= pd.melt(shap_performance_feature_all_examples_group, ['index'],feature_groups_name,'feature','shap')
#         shap_performance_feature_all_examples_group['hue'] = np.nan
#         shap_performance_feature_all_examples_group.loc[shap_performance_feature_all_examples_group['shap'] > 0 ,'hue'] = 'Positive contribution'
#         shap_performance_feature_all_examples_group.loc[shap_performance_feature_all_examples_group['shap'] <= 0 ,'hue'] = 'Negative contribution'
#         
# #             shap_performance_all_examples_group.loc[shap_performance_all_examples_group['shap'] > 0 ,'shap'] = -np.log10(shap_performance_all_examples_group.loc[shap_performance_all_examples_group['shap'] > 0 ,'shap'])
# #             shap_performance_all_examples_group.loc[shap_performance_all_examples_group['shap'] < 0 ,'shap'] = np.log10(0-shap_performance_all_examples_group.loc[shap_performance_all_examples_group['shap'] < 0 ,'shap'])
# #             
# #             cur_matrix = 'shap_performance_avg_matrix'
#         if cur_matrix == 'shap_performance_avg_matrix_group':
#             cur_violin_matrix = shap_performance_feature_all_examples_group                
#             cur_violin_order = np.asarray(feature_groups_name)[feature_group_order][::-1]
#         pass
#         if cur_matrix == 'shap_performance_avg_matrix':
#             cur_violin_matrix = shap_performance_feature_all_examples
#             cur_violin_order = np.asarray(self.varity_dict[predictor]['feature'][cur_test_fold])[feature_order][::-1]   
#         pass
#         if 'avg' in cur_matrix:
#             plt.clf()
#             plt.rcParams["font.family"] = "Helvetica"                  
#             fig = plt.figure(figsize=(30*fig_factor, 15*fig_factor),dpi = 600)        
#             color_palette = sns.color_palette("RdBu", n_colors=2)
#             ax = plt.subplot()             
# #                 ax = sns.violinplot(x= "shap", y= 'feature',data = cur_violin_matrix, order = cur_violin_order,palette=['red','royalblue'],hue = "hue",split = True, scale = 'count',scale_hue=False, vmin = -10,vmax = 10)
#             ax = sns.violinplot(x= "shap", y= 'feature',data = cur_violin_matrix, order = cur_violin_order,palette=['royalblue','red'],hue = "hue",split = True, scale = 'count',scale_hue=False, vmin = -10,vmax = 10)
#             ax.set_xscale('symlog',linthresh = 1e-4, linscale=1, base = 10)                 
#             ax.set_ylabel('')
#             ax.set_xlabel(cur_label, size=30*size_factor,labelpad = 20*size_factor)
#             ax.set_xlim(-10,10)
#             ax.tick_params(labelsize= 30*size_factor,rotation = 0,pad = 10)
#             ax.xaxis.set_ticks([-10,-1,0, 1,10])
# #                 ax.set_title(cur_title, size=40*size_factor,pad = 30*size_factor,loc = 'center')
# #                 ax.legend(fontsize = 25*size_factor,loc = 'right bottom')
#             ax.get_legend().remove()
#             fig.tight_layout()
#             if 'group' in cur_matrix:
#                 cur_violin_savefig = self.fun_perfix(runtime,'img') + '_violin_group.png' 
#             else:
#                 cur_violin_savefig = self.fun_perfix(runtime,'img') + '_violin.png' 
#             pass    
#             plt.savefig(cur_violin_savefig)            
# 

#         
#         
#         
#            
#         
#             
#             
#             
#             
# 
# 
# 
#         
#         
#         shap_performance_avg_matrix_df  = pd.DataFrame(shap_performance_avg_matrix)
#         shap_performance_avg_matrix_df.index = self.varity_dict[predictor]['feature'][cur_test_fold]
#         shap_performance_avg_matrix_df['shap_sum'] = shap_performance_avg_matrix.sum(axis=0)
#         shap_performance_avg_matrix_df['feature_group'] = np.nan
#         for i in range(len(feature_groups_name)):
#             cur_group_name = feature_groups_name[i]
#             for feature in feature_groups[i]:
#                 shap_performance_avg_matrix_df.loc[feature,'feature_group'] = len(feature_groups_name) - i
#         pass
#         shap_performance_avg_matrix_df.reset_index(inplace = True,drop = True)
#         feature_order = shap_performance_avg_matrix_df.sort_values(['feature_group','shap_sum']).index
#         
# #         feature_order = range(len(self.varity_dict[predictor]['feature'][cur_test_fold]))
# 
#        
        print ('OK')
# 
#         #separate into positive and negative matrix
#         shap_interaction_performance_by_examples_p = copy.deepcopy(shap_interaction_performance_by_examples)
#         shap_interaction_performance_by_examples_n = copy.deepcopy(shap_interaction_performance_by_examples)
#         shap_interaction_performance_by_examples_p[shap_interaction_performance_by_examples_p<0] = 0
#         shap_interaction_performance_by_examples_n[shap_interaction_performance_by_examples_n>0] = 0                
#         shap_interaction_performance_matrix_p = shap_interaction_performance_by_examples_p.sum(axis=0)[:-1,:-1] /weights.sum()
#         shap_interaction_performance_matrix_n = shap_interaction_performance_by_examples_n.sum(axis=0)[:-1,:-1] /weights.sum()
#         shap_interaction_performance_matrix_all = np.zeros([shap_interaction_performance_matrix.shape[0]*2,shap_interaction_performance_matrix.shape[1]*2])        
#         for i in range(shap_interaction_performance_matrix.shape[0]):
#             for j in range(shap_interaction_performance_matrix.shape[1]):
#                 shap_interaction_performance_matrix_all[2*i,2*j] = shap_interaction_performance_matrix_p[i,j]
#                 shap_interaction_performance_matrix_all[2*i,2*j+1] = shap_interaction_performance_matrix_p[i,j]
#                 shap_interaction_performance_matrix_all[2*i+1,2*j] = 0 - shap_interaction_performance_matrix_n[i,j]
#                 shap_interaction_performance_matrix_all[2*i+1,2*j+1] = 0 - shap_interaction_performance_matrix_n[i,j]
                                                                                          
    def get_stars_from_pvalue(self,p_value,pvalue_staronly):
        str_p_value = ''
        if p_value is not None:   
            if p_value > 0.05:
                if pvalue_staronly == 0:
                    str_p_value = '%.2E' % Decimal(p_value)
                else:
                    str_p_value = ''
                
            if p_value <= 0.05:
                if pvalue_staronly == 0:
                    str_p_value = '*'+ '%.2E' % Decimal(p_value)
                else:
                    str_p_value = '*'
            if p_value <= 0.01:
                if pvalue_staronly == 0:
                    str_p_value = '**'+ '%.2E' % Decimal(p_value)
                else:
                    str_p_value = '**'
            if p_value <= 0.001:
                if pvalue_staronly == 0:
                    str_p_value = '***'+ '%.2E' % Decimal(p_value)  
                else:
                    str_p_value = '***'  
            str_p_value = '['+str_p_value+']'
        else:
            str_p_value = ''
        return(str_p_value)