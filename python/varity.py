import numpy as np
import sys
import os
import glob
import time
import subprocess
import traceback
import string
import random
from datetime import datetime

import alm_project
import alm_ml
import alm_fun
class varity:

    def __init__(self,sys_argv):   
        print ('Class: [varity] [__init__]...... @' + str(datetime.now()))
        #defaut runtime values
        runtime = {}
        # runtime['action'] = ''
        # runtime['session_id']= ''
        # runtime['predictors']= ''
        runtime['compare_predictors'] = []
        runtime['init_hp_config'] = 0
        runtime['load_from_disk'] = 1
        runtime['save_to_disk'] = 1
        runtime['cur_test_fold']= 0
        # runtime['cur_validation_fold']= 0
        runtime['cur_gradient_key']= 'no_gradient'
        runtime['hp_tune_type'] = 'hyperopt'
        runtime['plot_metric']= ['aubprc_interp','auroc_interp']
        runtime['filter_test_score'] = 1
        runtime['shap_train_interaction'] = 0
        runtime['shap_test_interaction'] = 0
        # runtime['loo'] = 0
        # runtime['mv_hp'] = ''
        runtime['old_system'] = 0
        # runtime['cluster'] = 'local'
        runtime['fig_x'] = 30
        runtime['fig_y'] = 20
        # runtime['project_path'] = '..'
        # runtime['cluster'] = 'local'
        runtime['project_path'] = '..'  
        runtime['load_from_disk'] = 1 
        runtime['filtering_hp'] = ''
        runtime['cluster'] = 0
        runtime['run_on_node'] = 0
        runtime['job_name']=''
        runtime['batch_id']=''
        runtime['key_cols'] = []
        runtime['reinitiate_session'] = 0
        runtime['predictor']=''
            
        #read input runtime parameters    
        if isinstance(sys_argv,list):
            for i in range(1,len(sys_argv)):
                key = sys_argv[i].split('=')[0]
                value = sys_argv[i].split('=')[1]
                if '[' in value:
                    value_lst = []
                    value_lst = value[1:-1].split(',')   
                    runtime[key] = value_lst    
                else:    
                    if value.replace('.','0').isnumeric():                                            
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except:
                            print (str(value) + ' is not numeric!')
                    runtime[key] = value      
        else:
            for key in sys_argv:
                runtime[key] = sys_argv[key]
                
        self.sys_argv = sys_argv
        self.runtime = runtime
                
    def varity_action(self,runtime):                            
        action = runtime['action']        
        predictor = runtime['predictor']
        job_id = '-100'
        job_name = 'NA'        
        if runtime['run_on_node'] == 1:      
            if runtime['job_name'] == '':
                if runtime['filtering_hp'] == '':
                    runtime['job_name'] = runtime['session_id'] + '_' + runtime['action'] + '_' + runtime['predictor'] + '_' + str(runtime['cur_test_fold']) + '_' + str(alm_fun.get_random_id(8))
                else:
                    runtime['job_name'] = runtime['session_id'] + '_' + runtime['action'] + '_' + runtime['predictor'] + '_' + str(runtime['cur_test_fold']) + '_' + runtime['filtering_hp'] + '_' + str(alm_fun.get_random_id(8))
                            
            [job_id,job_name] = self.varity_action_cluster(runtime)
            print('\nCommand [' + action + '] on predictor [' + predictor + '] is running on cluster......' )            
        else:                
            #disable commands for 'Final' session            
            if (runtime['action'] in['init_session','mv_analysis','hp_tuning','save_best_hp','test_cv_predition']) & (runtime['session_id'] == 'Final'):
                print (runtime['action'] + ' command is disabled for  ' + '[' + runtime['session_id'] + '] session, this command may change the existing VARITY models in this session. To create new VARITY models, please initiate a new session.')
                sys.exit()
            
            argvs = self.read_config(runtime)         
            #1) create alm_project instance    
            self.proj = alm_project.alm_project(argvs,self)
            #2) create alm_ml instance                 
            self.ml = alm_ml.alm_ml(argvs['ml'])
            self.ml.varity_obj = self   
            self.ml.proj = self.proj
            
                                      
            if  action == 'mv_analysis':                
                self.ml.weights_opt_sp(predictor,runtime)    
                
            if action == 'validation_cv_prediction_sp':
                self.ml.fun_validation_cv_prediction_sp(predictor,runtime)                   
                
            if action == 'single_validation_fold_prediction':
                self.ml.fun_single_validation_fold_prediction(predictor,runtime)        
                
            if action == 'single_fold_prediction':
                self.ml.fun_single_fold_prediction(predictor,runtime)                        
                
            if action == 'plot_mv_result':  
                self.ml.plot_sp_result(predictor,runtime)                
                                
            if action == 'hp_tuning':                
                self.ml.weights_opt_hyperopt(predictor,runtime)     
            
            if action == 'save_best_hp':
                self.ml.save_best_hp_dict_from_trials(predictor,runtime)
                                        
            if action == 'plot_hp_weight':
                self.ml.plot_extra_data(predictor,runtime)
                                
            if action == 'test_cv_prediction':
                self.ml.fun_test_cv_prediction(predictor,runtime)
                
            if action == 'plot_test_result':
                runtime['cur_test_fold'] = -1 #only plot performance on all outer-loop folds
                self.ml.plot_test_result(predictor,runtime)
                
            if action == 'target_prediction':
                self.ml.fun_target_prediction(predictor,runtime)
                              
            alm_fun.show_msg(self.ml.log, self.ml.verbose, '\nCommand [' + action + '] on predictor [' + predictor + '] is finished.' )                     
        return([job_id,job_name])    

    def read_config(self,runtime):
        #check if the config file is available
        session_config_file = runtime['project_path'] + '/config/' + runtime['session_id'] + '.vsc'        
        if not os.path.isfile(session_config_file): # no config file 
            print ('Configuration file: ' + runtime['project_path'] + '/config/' + runtime['session_id'] + '.vsc is missing, a valid session configuration file is needed.')
            sys.exit()                
        else:
            #check the existence of VARITY objects for this session
            session_objects_exist = 0
            for session_object_file in glob.glob(runtime['project_path'] + '/output/npy/' + runtime['session_id'] + '*.npy'):
                if os.path.isfile(session_object_file):
                    session_objects_exist = 1    
                    
            if session_objects_exist == 1:        
                if runtime['action'] == 'init_session':
                    if runtime['reinitiate_session'] == 1:     
                        print ('Reinitiating existed session ' + '[' + runtime['session_id'] +']......')
                        runtime['load_from_disk'] = 0
                        runtime['init_hp_config'] = 1
                    else:
                        print ('**CAUTION**: Reinitiating existing session ' + '[' + runtime['session_id'] +'] will rebuild all objects. Please set [reinitiate_session] runtime parameter to 1 to force reinitiation.')
                        sys.exit()                            
            else:
                if runtime['action'] == 'init_session':                         
                        print ('initiating session ' + '[' + runtime['session_id'] +']......')
                        runtime['load_from_disk'] = 0
                        runtime['init_hp_config'] = 1        
                else:        
                    print ('Session ' +  runtime['session_id'] + ' has not been initiated yet, please run init_session command first.')
                     
        #load old system and update it
        if runtime['old_system'] == 1:  
            runtime['load_from_disk'] = 1
            runtime['init_hp_config'] = 0

        argvs = {}
        para_names = {}    
        block_comment = 0
        for line in  open(session_config_file,'r'):
            line = line.rstrip()
#             print (line)
            if line == '':
                continue 
            if line[3] == '#\\': #block comment start
                block_comment = 1
                
            if line[3] == '#//': #block comment end
                block_comment = 0
                
            if (line[0] == '#')|(block_comment == 1):
                continue
                
            if line[0] == '*':
                cur_key_lst = line.split(':')[0][1:].split('>')                
                cur_argvs = argvs
                for cur_key in cur_key_lst:
                    if cur_key not in cur_argvs.keys():                        
                        cur_argvs[cur_key] = {}
                    cur_argvs = cur_argvs[cur_key]                    
                para_names = line.split(':')[1].lstrip().split('|')
            else:
                if ':' in line:
                    object_name = line.split(':')[0]
                    para_values = line.split(':')[1].lstrip().split('|')
                    if object_name not in cur_argvs.keys():
                        cur_argvs[object_name] = {}                                        
                else:
                    object_name = None
                    para_values = line.lstrip().split('|')
                    
                for i in range(len(para_names)):
                    if para_values[i].replace('.','0').replace('-','0').isnumeric():                                            
                        try:
                            if '.' in para_values[i]:
                                para_values[i] = float(para_values[i])
                            else:
                                para_values[i] = int(para_values[i])
                        except:
                            print (str(para_values[i]) + ' is not numeric!')
                    else:
                        if '[' in para_values[i]:
                            para_values[i] = para_values[i][1:-1].split(',')    
                    if object_name is None:
                        cur_argvs[para_names[i]] = para_values[i]
                    else:                   
                        cur_argvs[object_name][para_names[i]] = para_values[i]
                            
        #update log,project_path and session_id to each object (ml, predictor, data, es)     
        project_path = runtime['project_path']
        session_id = runtime['session_id']        
        job_name = runtime['job_name']
        log_file = project_path + '/output/log/' + session_id + str(job_name) + '.log'
        
        argvs['project'] = {}
        argvs['ml'] = {}
        
        argvs['project']['project_paht'] = project_path
        argvs['project']['log'] = log_file
        argvs['project']['session_id'] = session_id
        
        for object in ['project','ml']:
            argvs[object]['log'] = log_file
            argvs[object]['verbose'] = 1
            argvs[object]['project_path'] = project_path
            argvs[object]['session_id'] = session_id
        
        for object in ['data','estimator','predictor']:    
            for i in argvs[object].keys():        
                argvs[object][i]['log'] = log_file
                argvs[object][i]['verbose'] = 1
                argvs[object][i]['project_path'] = project_path
                argvs[object][i]['session_id'] = session_id
                
        argvs['runtime'] = runtime
        return (argvs)

    def test_split(self,data_name,train_df,sametest_as_data_name):        
        kf_list_new = []           
        return (kf_list_new)      
    
    def cv_split(self,data_name, train_df):
        kf_list_new = []                       
        return (kf_list_new)       
    
    def data_preprocess(self, data_name, target_df, train_df, test_df, extra_train_df_lst):
        return [target_df, train_df, test_df, extra_train_df_lst]
        
    def filter_test(self,input_df):               
        input_df = input_df.loc[~(input_df['hgmd_source'] == 1), :]
        return(input_df)
    
    def fun_monitor_jobs(self,cur_jobs_dict,cur_log,runtime,max_run_time = np.float('inf')):    
        #*************************************************************************************
        #Monitor a list of job names
        #1) if job doesn't exit, and no result, resubmit the job
        #2) if job is running too long, kill and resubmit the job (max_run_time = 0 indictes unlimitted time, value are in seconds)
        #3) report real time job status (how many are still running, which ones are still runing etc)
        #4) exit the function when all the jobs are done (all job results become availble, the jobs could still be running because we wanted that way)
        #*************************************************************************************
        start_time = datetime.now()
        all_parallel_jobs_done = 0     
        while all_parallel_jobs_done == 0:
            all_parallel_jobs_done = 1
            running_jobs_num = 0
            running_jobs = []
            time.sleep(10)
  
            for cur_job_name in cur_jobs_dict.keys():
                cur_job_result = cur_jobs_dict[cur_job_name][0]    
                cur_job_id =  cur_jobs_dict[cur_job_name][1]
                cur_job_runtime = cur_jobs_dict[cur_job_name][2]
                reschedule = 0
                #check the current job running status 
                squeue_cmd = 'squeue -n ' + cur_job_name + ' -o "%T-%M-%R"' 
                return_process = self.fun_run_subprocess(squeue_cmd,runtime)
                job_status = return_process.stdout                                 
                job_info_list = job_status.split('\n')     
                               
                if ("RUNNING" in job_status) | ("PENDING"  in job_status):  #running or pending
                    all_parallel_jobs_done = 0
                    running_jobs_num += 1
                    running_jobs.append(cur_job_name)
                                       
                    job_info =   job_info_list[1][1:-1]                                      
                    job_state = job_info.split('-')[0]
                    job_runtime = datetime.now() - start_time
                    job_runtime_seconds = int(job_runtime.total_seconds())
                    job_node = job_info.split('-')[2]
                    #running too long 
                    if job_runtime_seconds > max_run_time:
                        alm_fun.show_msg (cur_log,1, 'Job: ' + cur_job_name + ' is  running too long!')
                        reschedule = 1 
                else: #not running, check if result is available                         
                    #there might be slightly delay between the job ending and saving the results, so wait 10 seconds                    
                    if not os.path.isfile(cur_job_result): # no result
                        time.sleep(10)
                        if not os.path.isfile(cur_job_result):
                            alm_fun.show_msg (cur_log,1, 'Job: ' + cur_job_name + '  is not running (no result)!')
                            reschedule = 1
          
                if reschedule == 1:
                    all_parallel_jobs_done = 0
                    scancel_cmd = 'scancel -n ' + cur_job_name
                    return_process = self.fun_run_subprocess(scancel_cmd,runtime)                            
                    #reschedule the job
                    [new_job_id,job_name] = self.varity_action_cluster(cur_job_runtime)
                    cur_jobs_dict[cur_job_name][1] = new_job_id                                                                                                                        
                    alm_fun.show_msg (cur_log,1,'Job: ' + cur_job_name + ' rescheduled to ' + str(new_job_id) + '!')                       
                    start_time = datetime.now()                                                    
                                                                
            alm_fun.show_msg (cur_log,1, str(running_jobs_num) + '/' +  str(len(cur_jobs_dict.keys()))  + ' ' +  str(running_jobs) +  ' jobs are still running......')
        return (all_parallel_jobs_done)
            
    def varity_action_cluster(self,runtime):        
        runtime['run_on_node'] = 0
        job_name = runtime['job_name']                     
        job_command = 'python3 ' + self.sys_argv[0]            
        for key in runtime:
            if type(runtime[key]) == 'list':
                value = '[' + ','.join(runtime[key]) + ']'
                value = value.replace(' ','')
            else:
                value = str(runtime[key])
            job_command = job_command + ' ' + key + '=' + value.replace(' ','')            
          
        exclusion_nodes_list = ''
        exclusion_nodes_log = runtime['project_path']   + '/output/log/exclusion_nodes.log'
        if os.path.isfile(exclusion_nodes_log):
            for line in  open(exclusion_nodes_log,'r'):
                exclusion_nodes_list =  exclusion_nodes_list + line.rstrip()[5:] + ','
            exclusion_nodes_list = exclusion_nodes_list[:-1]        
    
        mem = '10240'
        cpus = '1'
        sh_file = open(runtime['project_path'] + '/output/bat/' + str(job_name)   + '.sh','w')  
        sh_file.write('#!/bin/bash' + '\n')
        sh_file.write('# set the number of nodes' + '\n')
        sh_file.write('#SBATCH --nodes=1' + '\n')
        sh_file.write('# set the number of tasks' + '\n')
        sh_file.write('#SBATCH --ntasks=1' + '\n')
        sh_file.write('# set the number of cpus per task' + '\n')
        sh_file.write('#SBATCH --cpus-per-task=' + cpus + '\n')        
        sh_file.write('# set the memory for each node' + '\n')
        sh_file.write('#SBATCH --mem=' + mem + '\n')    
        sh_file.write('# set name of job' + '\n')
        sh_file.write('#SBATCH --job-name=' + str(job_name)  + '\n')
        sh_file.write("srun " + job_command)
        sh_file.close()
        
        if exclusion_nodes_list == '':
            sbatch_cmd = 'sbatch ' + runtime['project_path'] + '/output/bat/' + str(job_name)  + '.sh'
        else:
            sbatch_cmd = 'sbatch --exclude=galen['  + exclusion_nodes_list + '] ' + runtime['project_path'] + '/output/bat/' + str(job_name) + '.sh'
        
        print(sbatch_cmd)
        #check if number of pending jobs
        chk_pending_cmd = 'squeue -u jwu -t PENDING'  
        return_process =  subprocess.run(chk_pending_cmd.split(" "), cwd = runtime['project_path'] + '/output/log/',capture_output = True,text=True)
        pending_list = return_process.stdout                                 
        pending_num = len(pending_list.split('\n'))
        print ('Current number of pending jobs:' + str(pending_num))            
        job_id = '-1'
        if pending_num < 100:      
            retry_num = 0
            while (job_id == '-1') & (retry_num < 10):
                try:
                    return_process = subprocess.run(sbatch_cmd.split(" "), cwd = runtime['project_path'] + '/output/log/',capture_output = True,text=True)
                    time.sleep(0.1)
                    if return_process.returncode == 0:
                        job_id = return_process.stdout.rstrip().split(' ')[-1]
                    else:
                        job_id = '-1'
                        retry_num = retry_num + 1
                        print  (job_name + ' submit error,rescheduling......'  + str(retry_num))
                except:
                    job_id = '-1'
                    print(traceback.format_exc())
                    print (job_name + ' expected error,rescheduling......')
            print (job_name + ' submitted id: ' + job_id) 
        
#             job_monitor_list = argvs['project_path'] + 'log/jobs.log'    
#             if 'weights_opt_hyperopt' in cur_action:
#                 cur_result_file = runtime['project_path'] +'/output/npy/' +  runtime['predictor'] + '_' +  runtime['data_name'] + '_' +  runtime['tune_obj'] +  '_' + str(runtime['cur_test_fold']) + '_' + runtime['session_id'] + '_best.pkl'            
#                 alm_fun.show_msg(job_monitor_list,1,job_name + ';' + sbatch_cmd + ';' + cur_result_file)
#     
#             if 'weights_opt_sp' in cur_action:
#                 cur_result_file = runtime['project_path'] +'/output/csv/' +  runtime['predictor'] + '_' +  runtime['data_name'] + '_' +  runtime['tune_obj'] +  '_' + str(runtime['cur_test_fold']) + '_' + runtime['session_id'] + '_' +  runtime['add_on_set'] + '_spvalue_results.tab'     
#                 alm_fun.show_msg(job_monitor_list,1,job_name + ';' + sbatch_cmd + ';' + cur_result_file)
                 
        return([job_id,job_name])
    
    def fun_run_subprocess(self,cmd,runtime):
        return_code = subprocess.run(cmd.split(" "), cwd = runtime['project_path'] + '/output/log/',capture_output = True,text=True)        
        return (return_code)                   