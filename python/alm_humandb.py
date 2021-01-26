#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import numpy as np
import pandas as pd
import matplotlib
from seaborn.palettes import color_palette
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import requests
import sys
import csv
import os
import glob
import operator
import itertools
import time
import math
import random
import traceback
import re
import gzip
import urllib
import subprocess
import warnings
from subprocess import call
from ftplib import FTP
from scipy import stats
from operator import itemgetter 
from functools import partial
from datetime import datetime
from numpy import inf
from cgi import log
from posix import lstat
import xml.etree.ElementTree as ET
from io import StringIO
import alm_fun
import varity_run
import sklearn
from scipy import stats
from Bio import pairwise2
from Bio import Align

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)   
warnings.filterwarnings("ignore")

#*****************************************************************************************************************************
#file based human database 
#*****************************************************************************************************************************

class alm_humandb:
        
    def __init__(self,sys_argv):        
        print ('Class: [alm_humandb] [__init__]...... @' + str(datetime.now()))
   
#         self.assembly = argvs['assembly']
#         self.db_path = argvs['humandb_path']
#         self.project_path = argvs['project_path']
#         self.python_path = argvs['python_path']
#         self.log = argvs['humandb_path'] + 'log/humandb.log'        
#         self.verbose = 1
#         self.flanking_k = 0
#         self.cluster = 'galen'
#         self.db_version = 'manuscript'  # or 'uptodate'
#         self.argvs = argvs

        #defaut runtime values             
        runtime = {}
        runtime['assembly'] = 'GRCh37'
        runtime['db_path'] = '..'
        runtime['parallel_id'] = 1
        runtime['parallel_num'] = 1
        runtime['single_id'] = ''
        runtime['verbose'] = 1
        runtime['flanking_k'] = 0
        runtime['cluster'] = 'gelen'
        runtime['db_version'] = 'manuscript'
        runtime['batch_id'] = ''
        runtime['job_id'] = ''
        runtime['job_name'] = ''
        runtime['session_id'] = 'Revision_humandb'
        runtime['parallel_batches'] = 100
        runtime['mem'] = '10240'
        runtime['protein_batch_length_col'] = 'protein_len'
        runtime['pisa_folder'] = 'pisa'

    
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

        runtime['log'] = runtime['db_path'] + 'log/humandb.log'                
        self.sys_argv = sys_argv
        self.runtime = runtime
        
        for key in runtime:
            setattr(self, key, runtime[key])


        ####***************************************************************************************************************************************************************
        # Nucleotide and Amino Acids related
        ####***************************************************************************************************************************************************************
        self.lst_nt = ['A', 'T', 'C', 'G']
        self.lst_aa = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q", "U", "*", '_']
        self.lst_aa_21 = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q", "*"]
        self.lst_aa_20 = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q"]
        self.lst_aa3 = ["Ser", "Ala", "Val", "Arg", "Asp", "Phe", "Thr", "Ile", "Leu", "Lys", "Gly", "Tyr", "Asn", "Cys", "Pro", "Glu", "Met", "Trp", "His", "Gln", "Sec", "Ter", 'Unk']
        self.lst_aa3_20 = ["Ser", "Ala", "Val", "Arg", "Asp", "Phe", "Thr", "Ile", "Leu", "Lys", "Gly", "Tyr", "Asn", "Cys", "Pro", "Glu", "Met", "Trp", "His", "Gln"]
        
        self.lst_aaname = ["Serine", "Alanine", "Valine", "Arginine", "Asparitic Acid", "Phenylalanine", "Threonine", "Isoleucine", "Leucine", "Lysine", "Glycine", "Tyrosine", "Asparagine", "Cysteine", "Proline", "Glutamic Acid", "Methionine", "Tryptophan", "Histidine", "Glutamine", "Selenocysteine", "Stop", "Unknown"]
        self.lst_chr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT']
        
        self.dict_aa3 = {}
        for i in range(len(self.lst_aa3)):
            self.dict_aa3[self.lst_aa3[i]] = self.lst_aa[i]
            
        self.dict_aa3_upper = {}
        for i in range(len(self.lst_aa3)):
            self.dict_aa3_upper[self.lst_aa3[i].upper()] = self.lst_aa[i]
            
        self.dict_aa3_20_upper = {}
        for i in range(len(self.lst_aa3_20)):
            self.dict_aa3_20_upper[self.lst_aa3_20[i].upper()] = self.lst_aa_20[i]            
            
        self.dict_aaname = {}
        for i in range(len(self.lst_aa)):
            self.dict_aaname[self.lst_aa[i]] = self.lst_aaname[i]

        self.dict_aaencode = {}
        for i in range(len(self.lst_aa)):
            self.dict_aaencode[self.lst_aa[i]] = i

    def init_humamdb_object(self,obj_name,sub = 1):        
        if not os.path.isdir(self.db_path + obj_name):
            os.mkdir(self.db_path + obj_name)
        if sub == 1:                 
            if not os.path.isdir(self.db_path + obj_name + '/org'):
                os.mkdir(self.db_path + obj_name + '/org')
            if not os.path.isdir(self.db_path + obj_name + '/all'):            
                os.mkdir(self.db_path + obj_name + '/all')
            if not os.path.isdir(self.db_path + obj_name + '/bygene'):    
                os.mkdir(self.db_path + obj_name + '/bygene')
            if not os.path.isdir(self.db_path + obj_name + '/npy'):    
                os.mkdir(self.db_path + obj_name + '/npy') 
            if not os.path.isdir(self.db_path + obj_name + '/log'):    
                os.mkdir(self.db_path + obj_name + '/log')
            if not os.path.isdir(self.db_path + obj_name + '/csv'):
                os.mkdir(self.db_path + obj_name + '/csv')
            if not os.path.isdir(self.db_path + obj_name + '/bat'):    
                os.mkdir(self.db_path + obj_name + '/bat')
        alm_fun.show_msg(self.log,self.verbose, 'Created folder ' + obj_name + ' and its subfolders.' )

    def set_job_name(self,runtime):        
        job_name = runtime['session_id'] + '_' + runtime['db_action'] + '_' + runtime['batch_id']
        return (job_name)
         
    def humandb_action(self,runtime):
        
        if runtime['batch_id'] == '':
            runtime['batch_id'] = str(alm_fun.get_random_id(10))                            
        #### assign job_name to current job if empty
        if runtime['job_name'] == '':
            runtime['job_name'] = self.set_job_name(runtime)
        #### if run_on_node then submit the current job to cluster                  
        if runtime['run_on_node'] == 1:           
            [runtime['job_id'],runtime['job_name']] = self.humandb_action_cluster(runtime)
            print('\n' + runtime['db_action'] + ' is running on cluster......' )  
      
        else:       
            if runtime['db_action'] == 'debug':
                self.debug()      
            if runtime['db_action'] == 'create_all':        
                self.create_hgnc_data()
                self.create_uniprot_data()
                self.create_ensembl66_data()
                self.create_matched_uniprot_mapping()
                self.create_pisa_data()
                self.create_pfam_data()
                self.create_sift_data()
                self.create_provean_data()
                self.create_clinvar_data()                       
            if runtime['db_action'] == 'varity_check_data':
                self.varity_check_data(argvs['parallel_id'])            
            if runtime['db_action'] == 'varity_check_data_jobs':
                self.varity_check_data_jobs()                       
            if runtime['db_action'] == 'create_aa_properties_data':            
                self.create_aa_properties_data()
            if runtime['db_action'] == 'create_codon_usage_data':            
                self.create_codon_usage_data()            
            if runtime['db_action'] == 'create_blosum_data':
                self.create_blosum_data()
            if runtime['db_action'] == 'create_accsum_data':
                self.create_accsum_data()            
            if runtime['db_action']  == 'create_hgnc_data':
                self.create_hgnc_data()
            if runtime['db_action']  == 'create_uniprot_data':
                self.create_uniprot_data()   
            if runtime['db_action']  == 'create_pfam_data':
                self.create_pfam_data()    
            if runtime['db_action']  == 'create_pisa_data':
                self.create_pisa_data()             
            if runtime['db_action'] == 'retrieve_pisa_data':
                self.retrieve_pisa_data(argvs['parallel_id'], argvs['parallel_num']) 
            if runtime['db_action'] == 'retrieve_pisa_data_jobs':
                self.retrieve_pisa_data_jobs(argvs['parallel_num'])             
            if runtime['db_action'] == 'combine_pisa_data':
                self.combine_pisa_data(runtime)              
            if runtime['db_action']  == 'initiate_psipred_data':
                self.initiate_psipred_data()           
            if runtime['db_action'] == 'check_psipred_data':
                self.check_psipred_data()            
            if runtime['db_action'] == 'retrieve_psipred_data':
                self.retrieve_psipred_data(argvs['parallel_id'], argvs['parallel_num']) 
            if runtime['db_action'] == 'retrieve_psipred_data_jobs':
                self.retrieve_psipred_data_jobs(argvs['parallel_num'])             
            if runtime['db_action'] == 'combine_psipred_data':
                self.combine_psipred_data()              
            if runtime['db_action'] =='create_humsavar_data':
                self.create_humsavar_data()            
            if runtime['db_action'] =='create_hgmd_data':
                self.create_hgmd_data()                     
            if runtime['db_action']  == 'create_clinvar_data':
                self.create_clinvar_data()
            if runtime['db_action'] == 'create_mave_data':
                self.create_mave_data() 
            if runtime['db_action'] == 'create_funsum_data':
                self.create_funsum_data() 
            if runtime['db_action'] == 'create_varity_genelist':
                self.create_varity_genelist()                                                     
            if runtime['db_action']  == 'varity_dbnsfp_jobs':
                self.varity_dbnsfp_jobs()
            if runtime['db_action']  == 'varity_dbnsfp_process':
                self.varity_dbnsfp_process(argvs['parallel_id'])
            if runtime['db_action']  == 'varity_process_jobs': 
                self.varity_process_jobs()
            if runtime['db_action']  == 'varity_merge_data': 
                self.varity_merge_data(argvs['parallel_id'])
            if runtime['db_action']  == 'varity_merge_data_jobs':
                self.varity_merge_data_jobs()
            if runtime['db_action']  == 'varity_all_variant_process': 
                self.varity_all_variant_process(argvs['parallel_id'])
            if runtime['db_action']  == 'varity_all_variant_process_jobs':
                self.varity_all_variant_process_jobs()            
            if runtime['db_action']  == 'varity_combine_train_data':                         
                self.varity_combine_train_data()            
            if runtime['db_action'] == 'varity_combine_all_data':
                self.varity_combine_all_data()            
            if runtime['db_action'] == 'varity_mave_final_data':
                self.varity_mave_final_data()            
            if runtime['db_action'] == 'varity_count_all_data':
                self.varity_count_all_data()            
            if runtime['db_action']  == 'varity_train_variant_process':                         
                self.varity_train_variant_process()                
            if runtime['db_action'] == 'varity_train_data_final_process':
                self.varity_train_data_final_process()                  
            if runtime['db_action'] == 'varity_all_data_final_process':
                self.varity_all_data_final_process(argvs['parallel_id'])                  
            if runtime['db_action'] == 'varity_final_process_jobs':
                self.varity_final_process_jobs()      
            if runtime['db_action']  == 'create_ensembl66_data':
                self.create_ensembl66_data()
            if runtime['db_action']  == 'create_matched_uniprot_mapping':            
                self.create_matched_uniprot_mapping()            
            if runtime['db_action']  == 'create_sift_data':
                self.create_sift_data()
            if runtime['db_action']  == 'create_provean_data':
                self.create_provean_data()  
            if runtime['db_action']  == 'create_gnomad_data':
                self.create_gnomad_data()
            if runtime['db_action']  == 'create_evmutation_data':
                self.create_evmutation_data()               
            if runtime['db_action'] == 'create_varity_data_by_uniprotids':
                self.create_varity_data_by_uniprotids(runtime)
            if runtime['db_action'] == 'create_varity_data_by_uniprotid_jobs':
                self.create_varity_data_by_uniprotid_jobs(runtime)                                            
            if runtime['db_action'] == 'run_psipred_by_uniprotid':
                self.run_psipred_by_uniprotid(argvs['single_id'])
            if runtime['db_action'] == 'create_denovodb_data':
                self.create_denovodb_data(runtime)    
            if runtime['db_action'] == 'create_mpc_data':
                self.create_mpc_data()    
            if runtime['db_action'] == 'create_mistic_data':
                self.create_mistic_data()
            if runtime['db_action'] == 'combine_extra_revision_varity_data':                
                self.combine_extra_revision_varity_data(runtime)
            if runtime['db_action'] == 'check_varity_data_by_uniprotids':                
                self.check_varity_data_by_uniprotids(runtime)
            if runtime['db_action'] == 'create_varity_prediction_data_by_uniprotids':                
                self.create_varity_prediction_data_by_uniprotids(runtime)      
            if runtime['db_action'] == 'create_varity_prediction_data_by_uniprotids_jobs':                
                self.create_varity_prediction_data_by_uniprotids_jobs(runtime)                           
            if runtime['db_action'] == 'create_psipred_data_jobs':                
                self.create_psipred_data_jobs(runtime)        
            if runtime['db_action'] == 'create_psipred_by_uniprotids':                
                self.create_psipred_by_uniprotids(runtime)   
            if runtime['db_action'] == 'create_deepsequence_data':                
                self.create_deepsequence_data(runtime)   
            if runtime['db_action'] == 'run_batch_uniprotid_jobs':                
                self.run_batch_uniprotid_jobs(runtime) 
            if runtime['db_action'] == 'run_batch_id_jobs':                
                self.run_batch_id_jobs(runtime)                                 
            if runtime['db_action'] == 'create_varity_target_data':                     
                self.create_varity_target_data(runtime)
            if runtime['db_action'] == 'update_varity_data':                     
                self.update_varity_data(runtime)
            if runtime['db_action'] == 'create_ukb_data':                     
                self.create_ukb_data(runtime)      
            if runtime['db_action'] == 'retrieve_pisa_by_uniprotids':
                self.retrieve_pisa_by_uniprotids(runtime)
            if runtime['db_action'] == 'retrieve_pisa_by_uniprotid_old':
                self.retrieve_pisa_by_uniprotid_old(runtime)                
            if runtime['db_action'] == 'process_pisa_by_uniprotids':
                self.process_pisa_by_uniprotids(runtime)  
            if runtime['db_action'] == 'retrieve_dbref_by_pdbids':
                self.retrieve_dbref_by_pdbids(runtime)  
            if runtime['db_action'] == 'retrieve_pisa_by_pdbids':
                self.retrieve_pisa_by_pdbids(runtime) 
            if runtime['db_action'] == 'process_dbref_for_all_pdbids':
                self.process_dbref_for_all_pdbids(runtime)   
            if runtime['db_action'] == 'retrieve_uniprot_to_pdb_by_uniprotids':                                
                self.retrieve_uniprot_to_pdb_by_uniprotids(runtime)     
            if runtime['db_action'] == 'create_varity_web_by_uniprotids':                                
                self.create_varity_web_by_uniprotids(runtime)   
            if runtime['db_action'] == 'create_varity_all_predictions':                                
                self.create_varity_all_predictions(runtime)                  

        return([runtime['job_id'],runtime['job_name']])  

    def debug(self):
        
        varity_all_genes = pd.read_csv(self.db_path + '/varity/all/varity_all_genes.csv')
        print (str(len(varity_all_genes['p_vid'].unique())))
        
        x = pd.DataFrame(varity_all_genes.groupby(['p_vid'])['symbol'].agg('count')).reset_index()
        x.columns = ['p_vid','count']
        x.loc[x['count'] > 1,:]
        
        
        
#         mistic_df = pd.read_csv(self.db_path + '/mistic/all/MISTIC_GRCh37_avg_duplicated_scores.csv',dtype = {'chr':'str'}) 
#         mistic_df.loc[mistic_df['chr'] == 'chrX','chr'] = 'X'
        
#         mistic_df = pd.read_csv(self.db_path + '/mistic/org/MISTIC_GRCh37.tsv', sep = '\t',skiprows = 1)
#         mistic_df.columns = ['chr','nt_pos','nt_ref','nt_alt','mistic_score','mistic_pred']  
#         
#         print(mistic_df.loc[mistic_df['nt_pos'] == 11200225,:])
        
        
#         accsum_all = pd.read_csv(self.db_path + 'accsum/csv/accsum.csv')[['aa_ref','aa_alt','accessibility']]
#         accsum_all.columns = ['aa_ref','aa_alt','accessibility_all']        
#         accsum_aa = pd.read_csv(self.db_path + 'accsum/csv/accsum_freq_aa.csv')[['aa_ref','aa_alt','accessibility']] 
#         accsum_aa.columns = ['aa_ref','aa_alt','accessibility_aa']       
#         accsum = pd.merge(accsum_all,accsum_aa,how = 'left')            
#         print(alm_fun.spc_cal(accsum['accessibility_all'],accsum['accessibility_aa']))
#         accsum.to_csv(self.db_path + 'accsum/csv/accsum_merged.csv',index = False)
#         
#             
#         humsavar_snv = pd.read_csv(self.db_path + 'humsavar/all/humsavar_snv.csv')
#         humsavar_group_snv_df = humsavar_snv.groupby(['p_vid','aa_pos','aa_ref','aa_alt']).count().reset_index()
#         humsavar_group_snv_df.loc[humsavar_group_snv_df['humsavar_label'] > 1 ,:]
#         
#         humsavar_snv.loc[(humsavar_snv['p_vid'] == 'O00255') & (humsavar_snv['aa_pos'] == 161) ,:]
#         
#         
#         B39_batch_df = pd.read_csv(self.db_path + 'varity/csv/B39_varity_batch_snv.csv')
#         
#         B39_batch_df.loc[B39_batch_df['asa_mean'].notnull(),:].shape
#         
#         
#         hgmd_snv = pd.read_csv(self.db_path + 'hgmd/all/hgmd_snv.csv')
#         hgmd_missense_snv = hgmd_snv.loc[hgmd_snv['hgmd_id'].str[0:2] == 'CM',:]
#         
#         hgmd_group_snv_df = hgmd_missense_snv.groupby(['chr','nt_pos','nt_ref','nt_alt']).count().reset_index()
#         duplicated_pos = list(hgmd_group_snv_df.loc[hgmd_group_snv_df['hgmd_id'] > 1 ,'nt_pos'])
#         
#         hgmd_missense_snv.loc[hgmd_missense_snv['nt_pos'].isin(duplicated_pos)].sort_values(['nt_pos'])
#         hgmd_snv.loc[hgmd_snv['nt_pos']==19561175,:].sort_values(['nt_pos'])
        print ('OK')
               
    def humandb_action_cluster(self,runtime):
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
        exclusion_nodes_log = runtime['project_path']   + 'output/log/exclusion_nodes.log'
        if os.path.isfile(exclusion_nodes_log):
            for line in  open(exclusion_nodes_log,'r'):
                exclusion_nodes_list =  exclusion_nodes_list + line.rstrip()[5:] + ','
            exclusion_nodes_list = exclusion_nodes_list[:-1]        
    
#         if (runtime['db_action'] == 'create_mpc_data') | (runtime['db_action'] == 'create_mistic_data') | (runtime['db_action'] == 'create_varity_data_by_uniprotids'):
#             mem = '30720'
#         else:
#             mem = '10240'
            
        mem = str(runtime['mem'])
        cpus = '1'
        sh_file = open(runtime['project_path'] + 'output/bat/' + str(job_name)   + '.sh','w')  
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
            sbatch_cmd = 'sbatch ' + runtime['project_path'] + 'output/bat/' + str(job_name)  + '.sh'
        else:
            sbatch_cmd = 'sbatch --exclude=galen['  + exclusion_nodes_list + '] ' + runtime['project_path'] + 'output/bat/' + str(job_name) + '.sh'
        
        print(sbatch_cmd)
        #check if number of pending jobs
        chk_pending_cmd = 'squeue -u jwu -t PENDING'  
        return_process =  subprocess.run(chk_pending_cmd.split(" "), cwd = runtime['project_path'] + 'output/log/',capture_output = True,text=True)
        pending_list = return_process.stdout                                 
        pending_num = len(pending_list.split('\n'))
        print ('Current number of pending jobs:' + str(pending_num))            
        job_id = '-1'
        if pending_num < 100:      
            retry_num = 0
            while (job_id == '-1') & (retry_num < 10):
                try:
                    return_process = subprocess.run(sbatch_cmd.split(" "), cwd = runtime['project_path'] + 'output/log/',capture_output = True,text=True)
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

        return([job_id,job_name])       
    
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
                            alm_fun.show_msg (cur_log,1, cur_job_result + ' does not exist!')
                            reschedule = 1
          
                if reschedule == 1:
                    all_parallel_jobs_done = 0
                    scancel_cmd = 'scancel -n ' + cur_job_name
                    return_process = self.fun_run_subprocess(scancel_cmd,runtime)                            
                    #reschedule the job
                    [new_job_id,job_name] = self.humandb_action_cluster(cur_job_runtime)
                    cur_jobs_dict[cur_job_name][1] = new_job_id                                                                                                                        
                    alm_fun.show_msg (cur_log,1,'Job: ' + cur_job_name + ' rescheduled to ' + str(new_job_id) + '!')                       
                    start_time = datetime.now()                                                    
                                                                
#             alm_fun.show_msg (cur_log,1, str(running_jobs_num) + '/' +  str(len(cur_jobs_dict.keys()))  + ' ' +  str(running_jobs) +  ' jobs are still running......')
            alm_fun.show_msg (cur_log,1, str(running_jobs_num) + '/' +  str(len(cur_jobs_dict.keys()))  +  ' jobs are still running......')
        return (all_parallel_jobs_done)
        
    def fun_run_subprocess(self,cmd,runtime):
        return_code = subprocess.run(cmd.split(" "), cwd = runtime['project_path'] + 'output/log/',capture_output = True,text=True)        
        return (return_code)                           
     
    def fun_id_batches(self,ids,runtime):   
        
        #************************************************************************************************************************************************************************
        ### arrange batch so that can run on cluster (parallel jobs)
        #************************************************************************************************************************************************************************                
        parallel_batches = runtime['parallel_batches']
        varity_batchid_df = pd.DataFrame()
        varity_batchid_df[runtime['batch_id_name']] = ids        
        varity_batchid_df['varity_batch_id'] = np.nan
        
        #### total length of all the uniprot_ids
        total_ids_num = len(ids)
        avg_batch_ids_num = int(total_ids_num/parallel_batches)
        
        varity_batch = str(alm_fun.get_random_id(4))        
        cur_varity_batch_id = 1
        cur_total_batch_num = 0
        for cur_index in varity_batchid_df.index:            
            varity_batchid_df.loc[cur_index,'varity_batch_id'] = varity_batch + '_' + str(cur_varity_batch_id)
            cur_total_batch_num= cur_total_batch_num + 1
            if cur_total_batch_num >= avg_batch_ids_num :
                cur_total_batch_num = 0
                cur_varity_batch_id = cur_varity_batch_id + 1   
                     
        varity_batchid_df[[runtime['batch_id_name'],'varity_batch_id']].to_csv(self.project_path + 'output/log/' + runtime['db_action'] + '_' +  runtime['batch_id'] + '_varity_batch_id.csv',index = False)
        
        return(varity_batchid_df)     
          
    def fun_uniprot_id_batches(self,ids,runtime):   
        
        #************************************************************************************************************************************************************************
        ### arrange batch so that can run on cluster (parallel jobs)
        #************************************************************************************************************************************************************************
                
        parallel_batches = runtime['parallel_batches']
        length_col = runtime['protein_batch_length_col']
        varity_all_genes_df = pd.read_csv(self.db_path + 'varity/all/varity_all_genes.csv')
        varity_run_genes_df = varity_all_genes_df.loc[varity_all_genes_df['p_vid'].isin(ids),:]                        
        varity_run_genes_df = varity_run_genes_df.sort_values(['chr','p_vid'])
        varity_run_genes_df['varity_batch_id'] = np.nan
        
        #### total length of all the uniprot_ids
        total_protein_length = varity_run_genes_df[length_col].sum()
        avg_batch_protein_length = int(total_protein_length/parallel_batches)
        
        varity_genes_batch = str(alm_fun.get_random_id(4))        
        cur_varity_batch_id = 1
        cur_total_batch_length = 0
        for cur_index in varity_run_genes_df.index:            
            varity_run_genes_df.loc[cur_index,'varity_batch_id'] = varity_genes_batch + '_' + str(cur_varity_batch_id)
            cur_total_batch_length = cur_total_batch_length + varity_run_genes_df.loc[cur_index,length_col]
            if cur_total_batch_length >= avg_batch_protein_length :
                cur_total_batch_length = 0
                cur_varity_batch_id = cur_varity_batch_id + 1   
                     
        varity_run_genes_df[['chr','p_vid',length_col,'varity_batch_id']].to_csv(self.project_path + 'output/log/' + runtime['db_action'] + '_' +  runtime['batch_id'] + '_varity_batch_id.csv',index = False)
        
        return(varity_run_genes_df)
    
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
        cur_log = self.project_path + 'output/log/run_batch_id_jobs_' + runtime['batch_id_db_action'] + '_' + runtime['batch_id'] + '.log'
        alm_fun.show_msg (cur_log,1, '# of ids: ' + str(len(run_ids)) + ', # of batches: ' + str(len(varity_batch_ids)))                   
        for varity_batch_id in varity_batch_ids:              
            new_runtime = runtime.copy()            
            new_runtime[runtime['batch_id_name'] + 's'] = str(list(varity_run_ids_df.loc[varity_run_ids_df['varity_batch_id'] == varity_batch_id,runtime['batch_id_name']].unique())).replace("'","")                        
            new_runtime['varity_batch_id'] = varity_batch_id                                    
            new_runtime['run_on_node'] = 1    
            new_runtime['db_action'] = runtime['batch_id_db_action']                        
            new_runtime['job_name'] = runtime['batch_id_db_action']  + '_' + new_runtime['batch_id'] +  '_' + varity_batch_id                
            new_runtime['cur_result'] = self.project_path + 'output/log/' + runtime['batch_id_db_action'] + '_' + varity_batch_id + '_done.log'                        
            if not os.path.isfile(new_runtime['cur_result']):
                alm_fun.show_msg (cur_log,1, 'Processing batch'  + str(varity_batch_id) + '......')                                                                                      
                [job_id,job_name] = self.humandb_action(new_runtime)
                alm_fun.show_msg (cur_log,1, 'Job ID:' + str(job_id) + ', Job Name: ' + job_name)
                cur_jobs[job_name] = []
                cur_jobs[job_name].append(new_runtime['cur_result'])
                cur_jobs[job_name].append(job_id)
                cur_jobs[job_name].append(new_runtime)  
            else:
                alm_fun.show_msg (cur_log,1, 'Varity genes batch ' + str(varity_batch_id) + ' result is available.')
         
        if self.fun_monitor_jobs(cur_jobs,cur_log,runtime) == 1:
            alm_fun.show_msg (cur_log,1, 'Batch: '  +  runtime['batch_id'] + 'all varity genes batch ids are processed.')   
    
    def run_batch_uniprotid_jobs(self,runtime):
        #### Check how many Uniprot IDs need to run        
        exist_ids = []
        all_ids = list(pd.read_csv(runtime['batch_uniprotid_file'])['p_vid'])
        for exist_file in glob.glob(os.path.join(runtime['batch_uniprotid_exist_files'])):
            if os.stat(exist_file).st_size != 0:
                exist_ids.append(exist_file.split('/')[-1].split('.')[0].split('_')[0])
        run_ids = list(set(all_ids) - set(exist_ids))
        
        #### exclude the long protein TTN and MUC16 (no secondary structure yet, Psipred running very slow)
        run_ids = list(set(run_ids) - set(['Q8WZ42','Q8WXI7']))
        

        varity_run_genes_df = self.fun_uniprot_id_batches(run_ids, runtime)
        varity_batch_ids = list(varity_run_genes_df['varity_batch_id'].unique())
        
        ##### Fire parallel jobs 
        cur_jobs= {}                                                  
        cur_log = self.project_path + 'output/log/run_batch_uniprotid_jobs_' + runtime['batch_uniprotid_db_action'] + '_' + runtime['batch_id'] + '.log'
        alm_fun.show_msg (cur_log,1, '# of uniprot ids: ' + str(len(run_ids)) + ', # of batches: ' + str(len(varity_batch_ids)))                   
        for varity_batch_id in varity_batch_ids:              
            new_runtime = runtime.copy()            
            new_runtime['uniprot_ids'] = str(list(varity_run_genes_df.loc[varity_run_genes_df['varity_batch_id'] == varity_batch_id,'p_vid'].unique())).replace("'","")                        
            new_runtime['varity_batch_id'] = varity_batch_id                                    
            new_runtime['run_on_node'] = 1    
            new_runtime['db_action'] = runtime['batch_uniprotid_db_action']                        
            new_runtime['job_name'] = runtime['batch_uniprotid_db_action']  + '_' + new_runtime['batch_id'] +  '_' + varity_batch_id                
            new_runtime['cur_result'] = self.project_path + 'output/log/' + runtime['batch_uniprotid_db_action'] + '_' + varity_batch_id + '_done.log'                        
            if not os.path.isfile(new_runtime['cur_result']):
                alm_fun.show_msg (cur_log,1, 'Processing batch'  + str(varity_batch_id) + '......')                                                                                      
                [job_id,job_name] = self.humandb_action(new_runtime)
                alm_fun.show_msg (cur_log,1, 'Job ID:' + str(job_id) + ', Job Name: ' + job_name)
                cur_jobs[job_name] = []
                cur_jobs[job_name].append(new_runtime['cur_result'])
                cur_jobs[job_name].append(job_id)
                cur_jobs[job_name].append(new_runtime)  
            else:
                alm_fun.show_msg (cur_log,1, 'Varity genes batch ' + str(varity_batch_id) + ' result is available.')
         
        if self.fun_monitor_jobs(cur_jobs,cur_log,runtime) == 1:
            alm_fun.show_msg (cur_log,1, 'Batch: '  +  runtime['batch_id'] + 'all varity genes batch ids are processed.') 

    def run_varity_predictions(self,runtime):
        sys_argv = {}
        sys_argv['action']  = 'target_prediction'
        sys_argv['session_id'] = runtime['varity_session_id']
        sys_argv['predictor'] = runtime['varity_predictor']
        sys_argv['run_on_node'] = 0
        sys_argv['load_exsiting_model'] = 0             
        sys_argv['hp_tune_type'] = 'hyperopt_logistic'     
        sys_argv['project_path'] = runtime['varity_project_path']
        sys_argv['db_path'] = runtime['db_path']
        sys_argv['target_type'] = runtime['target_type']
        sys_argv['target_dataframe'] = runtime['target_dataframe']
        sys_argv['target_dataframe_name'] = runtime['varity_batch_id']
        sys_argv['target_dependent_variable'] = 'random_label'
        sys_argv['predictor'] = runtime['varity_predictor']
        sys_argv['shap_test_interaction'] = 1

        [job_id,job_name,result_dict] = varity_run.varity_run(sys_argv)
        return(result_dict)

    def run_preocess_dbnsfp_output(self,input_data,cur_log):        
        #************************************************************************************************************************************************************************
        ## Define a few functions for processing dbnsfp result
        #************************************************************************************************************************************************************************   
        def retrieve_ensembl_canonical_index(vep_canonical):
            try:
                vep_canonical_list = vep_canonical.split(";")
                canonical_index = vep_canonical_list.index('YES')
            except:
                canonical_index = np.nan
            return canonical_index

        def retrieve_uniprot_canonical_index(uniprot_hgnc_id,uniprot_ids):
            try:
                unprot_ids_list = uniprot_ids.split(";")
                canonical_index = unprot_ids_list.index(uniprot_hgnc_id)
            except:
                canonical_index = np.nan
            return canonical_index        
                         
        def retrieve_value_by_canonical_index(values,canonical_index):
            try:
                if not np.isnan(canonical_index):
                    values_list = str(values).split(";")
                    value = values_list[np.int(canonical_index)]
                else:
                    value = np.nan
            except:
                value = np.nan
            return value
                         
        def retrieve_aapos(uniprot_accs, uniprot_acc, uniprot_aaposs):
            try:        
                uniprot_accs_list = uniprot_accs.split(";")
                uniprot_poss_list = uniprot_aaposs.split(";")
                  
                if len(uniprot_poss_list) == 1:
                    uniprot_aa_pos = uniprot_poss_list[0]
                else:
                    unprot_accs_dict = {uniprot_accs_list[x]:x for x in range(len(uniprot_accs_list))}        
                    uniprot_aa_pos = uniprot_poss_list[unprot_accs_dict.get(uniprot_acc,np.nan)]
                if not chk_int(uniprot_aa_pos):
                    uniprot_aa_pos = np.nan
                else:
                    uniprot_aa_pos = int(uniprot_aa_pos)
                      
            except:
                uniprot_aa_pos = np.nan
            return uniprot_aa_pos
              
        def chk_int(str):
            try:
                x = int(str)        
                return True
            except:
                return False
          
        def chk_float(str):
            try:
                x = float(str)        
                return True
            except:
                return False
              
        def get_value_byfun(values, fun):
            try:
                value_list = values.split(";")
                value_list = [float(x) for x in value_list if chk_float(x)]
                if fun == 'min':
                    value = min(value_list)
                if fun == 'max':
                    value = max(value_list)
            except:
                value = np.nan
            return value
          
        def get_residue_by_pos(seq,pos):
            try:
                residue = seq[pos-1]
            except:
                residue = np.nan
            return residue
        pass        
        #************************************************************************************************************************************************************************
        ## SETP1: Retreive and process selected columns from DBNSFP output
        #************************************************************************************************************************************************************************
#         input_data = pd.read_csv(dbnsfp_output_file, sep = '\t',dtype = {'hg19_chr':'str'}) 
        alm_fun.show_msg(cur_log,self.verbose,'** STEP 1 **: Retrieving DBNSFP records in the output......')                                                
        basic_cols = ['p_vid','#chr','pos(1-based)','hg19_chr','hg19_pos(1-based)', 'ref', 'alt','aapos','aaref','aaalt','genename','Ensembl_geneid','Ensembl_transcriptid','Ensembl_proteinid','Uniprot_acc','VEP_canonical'] + \
                     ['refcodon','codonpos','codon_degeneracy']        
        score_cols = ['MPC_score','SIFT_score','SIFT4G_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','LRT_score','MutationTaster_score','MutationAssessor_score','FATHMM_score','PROVEAN_score','VEST4_score','PrimateAI_score'] + \
                     ['MetaSVM_score','MetaLR_score','M-CAP_score','REVEL_score','MutPred_score','CADD_raw','DANN_score','fathmm-MKL_coding_score','Eigen-raw_coding','GenoCanyon_score'] + \
                     ['integrated_fitCons_score','GERP++_RS','phyloP100way_vertebrate','phyloP30way_mammalian','phastCons100way_vertebrate','phastCons30way_mammalian','SiPhy_29way_logOdds'] 
        gnomad_cols = ['gnomAD_exomes_AC','gnomAD_exomes_AN','gnomAD_exomes_AF','gnomAD_exomes_nhomalt','gnomAD_exomes_controls_AC','gnomAD_exomes_controls_AN','gnomAD_exomes_controls_AF','gnomAD_exomes_controls_nhomalt'] 
        gene_cols = ['Uniprot_acc(HGNC/Uniprot)','CCDS_id','Refseq_id','ucsc_id','MIM_id','MIM_phenotype_id','MIM_disease','gnomAD_pLI','gnomAD_pRec','gnomAD_pNull','HIPred_score'] + \
                    ['RVIS_EVS','RVIS_percentile_EVS','LoF-FDR_ExAC','RVIS_ExAC','RVIS_percentile_ExAC'] 
        all_cols = basic_cols + score_cols + gnomad_cols + gene_cols
        output_snv= input_data[all_cols]                  
        #### Use hgnc uniprot_id as the key to search canonical transcript
        output_snv['Uniprot_acc(HGNC/Uniprot)'] = output_snv['Uniprot_acc(HGNC/Uniprot)'].astype(str)
#         output_snv['p_vid'] = output_snv['Uniprot_acc(HGNC/Uniprot)'].apply(lambda x: x.split(';')[-1])        
        output_snv['uniprot_canonical_index'] = output_snv.apply(lambda x: retrieve_uniprot_canonical_index(x['p_vid'],x['Uniprot_acc']),axis = 1)
        output_snv['aa_pos'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['aapos'],x['uniprot_canonical_index']),axis = 1)
        output_snv = output_snv.rename(columns = {'hg19_chr':'chr','hg19_pos(1-based)':'nt_pos','ref':'nt_ref','alt':'nt_alt','aaref':'aa_ref','aaalt':'aa_alt'})
        
    
        #### upgrade from hg19 to hg38
#         output_snv = output_snv.rename(columns = {'hg19_chr':'hg19_chr','hg19_pos(1-based)': 'hg19_nt_pos', '#chr':'chr', 'pos(1-based)':'nt_pos','ref':'nt_ref','alt':'nt_alt','aaref':'aa_ref','aaalt':'aa_alt'})
#         output_snv['chr'] = output_snv['chr'].apply(lambda x: 'Chr' + str(x))
#         output_snv['hg19_chr'] = output_snv['hg19_chr'].apply(lambda x: 'Chr' + str(x))
                 
        #### get the score for the canonical trascript if there are mutliple scores due to multiple transcripts     
        output_snv['SIFT_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['SIFT_score'],x['uniprot_canonical_index']),axis = 1)        
        output_snv['SIFT4G_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['SIFT4G_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['Polyphen2_selected_HDIV_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['Polyphen2_HDIV_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['Polyphen2_selected_HVAR_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['Polyphen2_HVAR_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['PROVEAN_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['PROVEAN_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['VEST4_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['VEST4_score'],x['uniprot_canonical_index']),axis = 1)       
        output_snv['FATHMM_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['FATHMM_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['MutationAssessor_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['MutationAssessor_score'],x['uniprot_canonical_index']),axis = 1)        
        output_snv['MutationTaster_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['MutationTaster_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['MPC_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['MPC_score'],x['uniprot_canonical_index']),axis = 1)        
#         output_snv['MutationTaster_selected_score'] = output_snv.apply(lambda x: get_value_byfun(x['MutationTaster_score'],'max'),axis = 1)      
        alm_fun.show_msg(cur_log,self.verbose,'Total VARITY records in the output: ' + str(output_snv.shape))
        
        #************************************************************************************************************************************************************************
        ## SETP2: Remove non-missense and irregular records
        #************************************************************************************************************************************************************************
        alm_fun.show_msg(cur_log,self.verbose,'** STEP 2 **: Removing non-missense and irregular variants......')        
                                                                    
        #### replace "." with np.nan        
        output_snv = output_snv.replace('.',np.nan)
        output_snv = output_snv.replace('-',np.nan)
        ##### remove irregular records (only keep valid missense variants)
        output_snv = output_snv.loc[output_snv['chr'].notnull() & output_snv['nt_pos'].notnull() & output_snv['nt_ref'].isin(['A','T','C','G']) & output_snv['nt_alt'].isin(['A','T','C','G'])
                                  & output_snv['aa_pos'].notnull() & output_snv['aa_ref'].isin(self.lst_aa_20) & output_snv['aa_alt'].isin(self.lst_aa_20) 
                                  & output_snv['p_vid'].notnull() ,:]
#         output_snv_less = output_snv[['p_vid','chr','nt_pos','nt_ref','nt_alt','aa_pos','aa_ref','aa_alt']]
        alm_fun.show_msg(cur_log,self.verbose,'Total VARITY records after removing non-missense and irregular variants : ' + str(output_snv.shape))
        
        #************************************************************************************************************************************************************************
        ## SETP3: Set the correct data type for each column
        #************************************************************************************************************************************************************************              
        alm_fun.show_msg(cur_log,self.verbose,'** STEP 3 **: Set the correct data type for each column......')        
        float_cols = [x for x in output_snv.columns if '_score' in x] + ['gnomAD_exomes_controls_AF','gnomAD_exomes_AF']
        float_cols =  set(float_cols) - set(['SIFT_score','SIFT4G_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','MutationTaster_score','MutationAssessor_score','FATHMM_score','PROVEAN_score','VEST4_score','MPC_score'])        
        for float_col in float_cols:
            output_snv[float_col] = output_snv[float_col].astype(float)              
        int_cols = ['nt_pos','aa_pos','gnomAD_exomes_AC','gnomAD_exomes_AN','gnomAD_exomes_nhomalt','gnomAD_exomes_controls_AC','gnomAD_exomes_controls_AN','gnomAD_exomes_controls_nhomalt']
        for int_col in int_cols:
            output_snv.loc[output_snv[int_col].isnull(),int_col] = 0
            output_snv[int_col] = output_snv[int_col].astype(int)        
        output_snv['chr'] = output_snv['chr'].astype(str)
        #make the chr null case to 'Z' ????? 
        output_snv.loc[output_snv['chr'].isnull(),'chr'] = 'Z'       
    
                      
        #************************************************************************************************************************************************************************
        ## SETP4: Add annotation from different databases (gnomAD,ClinVAR,HumsaVAR,HGMD,MAVE)
        #************************************************************************************************************************************************************************
        alm_fun.show_msg(cur_log,self.verbose,'** STEP 4 **: Adding annotation from different databases......')
        #### gnomAD                                            
        output_snv['gnomad_source'] = 0
        output_snv.loc[output_snv['gnomAD_exomes_AC'] > 0, 'gnomad_source'] = 1  
        output_snv['gnomad_label'] = np.nan
        output_snv.loc[(output_snv['gnomad_source'] == 1), 'gnomad_label'] = 0                
        #### ClinVAR
        clinvar_snv = pd.read_csv(self.db_path + 'clinvar/all/clinvar_snv.csv',dtype = {'chr':'str'})
        clinvar_snv['clinvar_source'] = 1     
        clinvar_snv = clinvar_snv.loc[clinvar_snv['clinvar_label'].isin([0,1]),:]
        output_snv = output_snv.merge(clinvar_snv,how = 'left')
        alm_fun.show_msg(cur_log,self.verbose,'Total Varity records after merge CLINVAR : ' + str(output_snv.shape))
        #### HumsaVAR   
        humsavar_snv = pd.read_csv(self.db_path + 'humsavar/all/humsavar_snv.csv')
        humsavar_snv['humsavar_source'] = 1   
        humsavar_snv = humsavar_snv.loc[humsavar_snv['humsavar_label'].isin([0,1]),:]
        output_snv = output_snv.merge(humsavar_snv,how = 'left')
        alm_fun.show_msg(cur_log,self.verbose,'Total Varity records after merge HUMSAVAR : ' + str(output_snv.shape))
        #### HGMD 2015 
        hgmd_snv = pd.read_csv(self.db_path + 'hgmd/all/hgmd_snv.csv')        
        hgmd_snv['hgmd_source'] = 1   
        hgmd_snv = hgmd_snv.loc[hgmd_snv['hgmd_label'].isin([0,1]),:]        
        output_snv = output_snv.merge(hgmd_snv,how = 'left')
        alm_fun.show_msg(cur_log,self.verbose,'Total Varity records after merge HGMD : ' + str(output_snv.shape))
        #### MAVE                          
        mave_missense = pd.read_csv(self.db_path + 'mave/all/all_mave_for_varity.csv')  
        mave_missense['mave_source'] = 1     
        mave_missense = mave_missense.loc[mave_missense['mave_label'].isin([0,1]),:]
        output_snv = output_snv.merge(mave_missense,how = 'left')
        alm_fun.show_msg(cur_log,self.verbose,'Total Varity records after merge MAVE : ' + str(output_snv.shape))             
        
        #************************************************************************************************************************************************************************
        ## SETP5: Add VARITY features
        #************************************************************************************************************************************************************************
        dbnsfp_missense_num = output_snv.shape[0]
        alm_fun.show_msg(cur_log,self.verbose,'** STEP 5 **: Adding VARITY features......')               
        #### create a list of p_vids involved in current process and prepare corresponding feature resource dataframe
        pisa_df = None
        psipred_df = None
        pfam_df = None
        evmutation_df = None
        
        lst_p_vids = output_snv['p_vid'].unique()    
        for p_vid in lst_p_vids:
            p_vid = str(p_vid)
            alm_fun.show_msg(cur_log,self.verbose,"Current Uniprot ID: " +p_vid)   
            
            #### combine pisa
            pisa_gene_file = self.db_path + 'pisa/bygene/' + p_vid + '_pisa.csv'
            if os.path.isfile (pisa_gene_file):            
                cur_pisa_df = pd.read_csv(self.db_path + 'pisa/bygene/' + p_vid + '_pisa.csv')               
                if pisa_df is None:
                    pisa_df = cur_pisa_df
                else:
                    pisa_df = pd.concat([pisa_df,cur_pisa_df])
                                                                 
            #### combined psipred                  
            psipred_gene_file =self.db_path + 'psipred/bygene/' + p_vid + '.ss2' 
            if os.path.isfile(psipred_gene_file):
                cur_psipred_df = pd.read_csv(psipred_gene_file, skiprows=[0, 1], header=None, sep='\s+')
                cur_psipred_df = cur_psipred_df.loc[:, [0, 1, 2]]
                cur_psipred_df.columns = ['aa_pos', 'aa_ref', 'aa_psipred']
                cur_psipred_df = cur_psipred_df.drop_duplicates()    
                cur_psipred_df['p_vid'] = p_vid                      
                psipred_lst = ['E','H','C']
                for ss in psipred_lst:
                    cur_psipred_df['aa_psipred' + '_' + ss] = cur_psipred_df['aa_psipred'].apply(lambda x: int(x == ss))
                
                if psipred_df is None:
                    psipred_df = cur_psipred_df
                else:
                    psipred_df = pd.concat([psipred_df,cur_psipred_df])
                      
     
            #### combined pfam                  
            pfam_gene_file = self.db_path + 'pfam/bygene/' + p_vid + '_pfam.csv'
            if os.path.isfile(pfam_gene_file):
                cur_pfam_df = pd.read_csv(self.db_path + 'pfam/bygene/' + p_vid + '_pfam.csv')
                cur_pfam_df.columns = ['p_vid', 'a_start', 'a_end', 'e_start', 'e_end', 'hmm_id', 'hmm_name', 'type', 'hmm_start', 'hmm_end', 'hmm_length', 'bit_score', 'e_value', 'clan']
                if pfam_df is None:
                    pfam_df = cur_pfam_df
                else:
                    pfam_df = pd.concat([pfam_df,cur_pfam_df])

            #### combined evmutation             
            for evmutation_gene_file in glob.glob(os.path.join(self.db_path + 'evmutation/org/' , p_vid + '*.csv')):
                cur_evmutation_df = pd.read_csv(evmutation_gene_file, sep=';')
                cur_evmutation_df.columns = ['mutation', 'aa_pos', 'aa_ref', 'aa_alt', 'evm_epistatic_score', 'evm_independent_score', 'evm_frequency', 'evm_conservation']
                cur_evmutation_df['p_vid'] = p_vid               
                if evmutation_df is None:
                    evmutation_df = cur_evmutation_df[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt','evm_epistatic_score','evm_independent_score']]
                else:
                    evmutation_df = pd.concat([evmutation_df,cur_evmutation_df])
            
        #### merge pisa   
        pisa_cols = ['asa_mean','asa_std','asa_count','bsa_max','solv_ne_max','bsa_ratio_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min']        
        if pisa_df is not None:  
            pisa_df = pisa_df.reset_index(drop = True)                      
            output_snv = pd.merge(output_snv, pisa_df, how='left')  
            alm_fun.show_msg(cur_log,self.verbose,"merge with pisa: " + str(output_snv.shape))
        else:
            alm_fun.show_msg(cur_log,self.verbose,"pisa info is not available......") 

        #### merge psipred            
        psipred_cols = ['aa_psipred','aa_psipred_E','aa_psipred_H','aa_psipred_C']
        if psipred_df is not None:
            psipred_df = psipred_df.reset_index(drop = True)
            output_snv = pd.merge(output_snv, psipred_df, how='left')         
            alm_fun.show_msg(cur_log,self.verbose,"merge with psipred: " + str(output_snv.shape))
        else:
            alm_fun.show_msg(cur_log,self.verbose,"psipred info is not available......")   
                    
        # merge pfam                      
        pfam_cols = ['pfam_id','in_domain']             
        if pfam_df is not None:
            pfam_df = pfam_df.reset_index(drop = True)                                    
            pfam_df.columns = ['p_vid', 'a_start', 'a_end', 'e_start', 'e_end', 'hmm_id', 'hmm_name', 'type', 'hmm_start', 'hmm_end', 'hmm_length', 'bit_score', 'e_value', 'clan']
            for i in pfam_df.index:
                cur_hmmid = pfam_df.loc[i, "hmm_id"]
                cur_pvid = pfam_df.loc[i, 'p_vid'] 
                cur_aa_start = pfam_df.loc[i, 'a_start']
                cur_aa_end = pfam_df.loc[i, 'a_end']
                output_snv.loc[(output_snv['p_vid'] == cur_pvid) & (output_snv['aa_pos'] >= cur_aa_start) & (output_snv['aa_pos'] <= cur_aa_end), 'pfam_id'] = cur_hmmid
            output_snv['in_domain'] = np.nan
            output_snv.loc[output_snv['pfam_id'].notnull(), 'in_domain'] = 1
            output_snv.loc[output_snv['pfam_id'].isnull(), 'in_domain'] = 0            
            alm_fun.show_msg(cur_log,self.verbose,"merge with pfam: " + str(output_snv.shape))
        else:
            alm_fun.show_msg(cur_log,self.verbose,"pfam info is not available......") 
          
        # merge EVMutation  
        evmutation_cols = ['mutation','evm_epistatic_score','evm_independent_score','evm_frequency','evm_conservation']                                    
        if evmutation_df is not None:
            evmutation_df = evmutation_df.groupby(['p_vid', 'aa_pos', 'aa_ref', 'aa_alt'])['evm_epistatic_score','evm_independent_score'].mean()
            evmutation_df = evmutation_df.reset_index()            
            output_snv = pd.merge(output_snv, evmutation_df, how='left')
            alm_fun.show_msg(cur_log,self.verbose,"merge with evmutation: " + str(output_snv.shape))
        else:
            alm_fun.show_msg(cur_log,self.verbose,"evmutation info is not available......")
                              
        #### aa_ref and aa_alt AA properties
        aa_properties = self.load_aa_properties()
        aa_properties_features = aa_properties.columns                
        aa_properties_ref_features = [x + '_ref' for x in aa_properties_features]
        aa_properties_alt_features = [x + '_alt' for x in aa_properties_features]   
        aa_properties_ref = aa_properties.copy()
        aa_properties_ref.columns = aa_properties_ref_features
        aa_properties_alt = aa_properties.copy()
        aa_properties_alt.columns = aa_properties_alt_features                
        output_snv = pd.merge(output_snv, aa_properties_ref, how='left')
        output_snv = pd.merge(output_snv, aa_properties_alt, how='left')         
        for x in aa_properties_features:
            if x != 'aa':
                output_snv[x+'_delta'] = output_snv[x+'_ref'] - output_snv[x+'_alt']                  
        alm_fun.show_msg(cur_log,self.verbose,"merge with aa properties: " + str(output_snv.shape))          
        #### merge with the blosum properties        
        [df_blosums, dict_blosums]  = self.load_blosums()       
        output_snv = pd.merge(output_snv, df_blosums, how='left')
        alm_fun.show_msg(cur_log,self.verbose,"merge with blosums: " + str(output_snv.shape))
        #### merge accessibility          
        accsum_df = self.load_accsum()
        output_snv = pd.merge(output_snv,accsum_df,how = 'left')
        alm_fun.show_msg(cur_log,self.verbose,"merge with accsums: " + str(output_snv.shape))
                                
        #### add features that were missing 
        possible_missing_features = pisa_cols + psipred_cols + pfam_cols + evmutation_cols 
        missing_features = set(possible_missing_features) - set(output_snv.columns)
        for missing_feature in missing_features:
             output_snv[missing_feature] = np.nan      
             
        #### check the number of records after merge features and annotations
        dbnsfp_process_error_log = self.db_path + 'varity/log/dbnsfp_process_error.log'        
        dbnsfp_missense_num_after_merge = output_snv.shape[0]
        if dbnsfp_missense_num_after_merge- dbnsfp_missense_num > 0: 
            alm_fun.show_msg(dbnsfp_process_error_log,self.verbose,"Duplicated snv detcted after processing dbnsfp output, please check " + cur_log )
                        
                       
        #************************************************************************************************************************************************************************
        ## SETP6: Add extra scores that are not existed in DBNSFP
        #************************************************************************************************************************************************************************
        alm_fun.show_msg(cur_log,self.verbose,'** STEP 6 **: Add extra scores that are not existed in DBNSFP......')
        # merge MISTIC
        mistic_cols = ['mistic_score']
        mistic_df = pd.read_csv(self.db_path + 'mistic/all/MISTIC_GRCh37_avg_duplicated_scores.csv',dtype = {'chr':'str'}) 
        mistic_df.loc[mistic_df['chr'] == 'chrX','chr'] = 'X'         
        output_snv = pd.merge(output_snv, mistic_df, how='left')
        alm_fun.show_msg(cur_log,self.verbose,"merge with MISTIC: " + str(output_snv.shape))
         
        # merge MPC
        mpc_cols = ['mpc_obs_exp','mpc_mis_badness','mpc_fitted_score','mpc_score']
        mpc_df = pd.read_csv(self.db_path + 'mpc/all/mpc_values_v2_avg_duplicated_scores.csv',dtype = {'chr':'str'})
        output_snv = pd.merge(output_snv, mpc_df, how='left')
        alm_fun.show_msg(cur_log,self.verbose,"merge with MPC: " + str(output_snv.shape))
                     
        # merge deepsequence
        deepsequence_df = pd.read_csv(self.db_path + 'deepsequence/all/all_deepsequence_scores.csv')
        output_snv = pd.merge(output_snv, deepsequence_df, how='left')
        alm_fun.show_msg(cur_log,self.verbose,"merge with DeepSeqeunce: " + str(output_snv.shape))
        
        # add more complete sift score        
        output_snv = self.add_sift(output_snv)
        
        # add more complete provean score 
        output_snv = self.add_provean(output_snv)
        
        #************************************************************************************************************************************************************************
        ## SETP7: Determine the label based on quality order of different databases if a variant is labeled differently by different databases
        #************************************************************************************************************************************************************************                
        output_csv = self.update_varity_labels(output_snv,cur_log)                
        return(output_snv)
        
    def create_varity_web_by_uniprotids (self,runtime):         
        uniprot_seq_dict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy').item()   
        id2hgnc_dict = np.load(self.db_path + 'hgnc/npy/id2hgnc_dict.npy').item()
        hgnc2id_dict = np.load(self.db_path + 'hgnc/npy/hgnc2id_dict.npy').item()     
        uniprot_ids = runtime['uniprot_ids']  
        
        cur_log = self.project_path + 'output/log/' + 'create_varity_web_by_uniprotids_'+ runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/' + 'create_varity_web_by_uniprotids_' + runtime['varity_batch_id'] + '_done.log'
        ####***************************************************************************************************************************************************************
        #### Generate files used for web application 
        ####***************************************************************************************************************************************************************
        for cur_pvid in uniprot_ids:    
            
            if os.path.isfile(self.db_path + 'varity/bygene/' + cur_pvid + '_varity_snv.csv'):                              
                p_name = hgnc2id_dict['symbol'].get(id2hgnc_dict['uniprot_ids'].get(cur_pvid,''),'')
                p_seq = uniprot_seq_dict.get(cur_pvid,'')   
                selected_cols = ['aa_alt', 'aa_pos', 'aa_ref', 'p_vid', 'chr','nt_pos','nt_ref','nt_alt','gnomAD_exomes_AF','gnomAD_exomes_AC', 'gnomAD_exomes_nhomalt'] + \
                ['clinvar_source', 'clinvar_review_star', 'clinvar_clinsig_level', 'clinvar_clin_sig', 'clinvar_id', 'hgmd_source', 'mave_source'] + \
                ['humsavar_source', 'gnomad_source', 'clinvar_label', 'hgmd_label', 'humsavar_label', 'provean_score', 'sift_score'] + \
                ['evm_epistatic_score', 'integrated_fitCons_score', 'LRT_score', 'GERP++_RS', 'phyloP30way_mammalian', 'phastCons30way_mammalian'] + \
                ['SiPhy_29way_logOdds', 'blosum100', 'in_domain', 'asa_mean', 'aa_psipred_E', 'aa_psipred_H', 'aa_psipred_C'] + \
                ['bsa_max', 'h_bond_max', 'salt_bridge_max', 'disulfide_bond_max', 'covelent_bond_max', 'solv_ne_min','solv_ne_max','solv_ne_abs_max'] + \
                ['mw_delta', 'pka_delta', 'pkb_delta', 'pi_delta', 'hi_delta', 'pbr_delta', 'avbr_delta', 'vadw_delta', 'asa_delta'] + \
                ['cyclic_delta', 'charge_delta', 'positive_delta', 'negative_delta', 'hydrophobic_delta', 'polar_delta', 'ionizable_delta'] + \
                ['aromatic_delta', 'aliphatic_delta', 'hbond_delta', 'sulfur_delta', 'essential_delta', 'size_delta']
                                    
                cur_varity_snv_df = pd.read_csv(self.db_path + 'varity/bygene/' + cur_pvid + '_varity_snv.csv')[selected_cols]
                snv_cols = ['gnomAD_exomes_AF','gnomAD_exomes_AC', 'gnomAD_exomes_nhomalt','gnomad_source'] + ['nt_pos','nt_ref','nt_alt','clinvar_source','clinvar_review_star','clinvar_clinsig_level','clinvar_clin_sig','clinvar_id','clinvar_label']
                score_cols = ['provean_score', 'sift_score','evm_epistatic_score', 'integrated_fitCons_score', 'LRT_score', 'GERP++_RS', 'phyloP30way_mammalian', 'phastCons30way_mammalian','SiPhy_29way_logOdds']
                cur_varity_missense_df = cur_varity_snv_df.drop(columns = snv_cols + score_cols)
                cur_varity_missense_df = cur_varity_missense_df.drop_duplicates()
                
                cur_snv_list_df  = cur_varity_snv_df.groupby(['aa_alt', 'aa_pos', 'aa_ref', 'p_vid'])[snv_cols].agg(list).reset_index()
                cur_snv_score_mean_df = cur_varity_snv_df.groupby(['aa_alt', 'aa_pos', 'aa_ref', 'p_vid'])[score_cols].agg('mean').reset_index()
                
                cur_snv_gnomad_sum_df = cur_varity_snv_df.groupby(['aa_alt', 'aa_pos', 'aa_ref', 'p_vid'])['gnomAD_exomes_AF'].agg('sum').reset_index()
                cur_snv_gnomad_sum_df.loc[(cur_snv_gnomad_sum_df['gnomAD_exomes_AF'] == 0)|cur_snv_gnomad_sum_df['gnomAD_exomes_AF'].isnull(),'gnomAD_exomes_AF'] = 1e-06
                cur_snv_gnomad_sum_df['gnomad_af_log10'] = 0 - np.log10(cur_snv_gnomad_sum_df['gnomAD_exomes_AF'])
                cur_snv_gnomad_sum_df = cur_snv_gnomad_sum_df.drop(columns = ['gnomAD_exomes_AF'])
    
                cur_varity_missense_df = pd.merge(cur_varity_missense_df,cur_snv_list_df,how = 'left')
                cur_varity_missense_df = pd.merge(cur_varity_missense_df,cur_snv_score_mean_df,how = 'left')
                cur_varity_missense_df = pd.merge(cur_varity_missense_df,cur_snv_gnomad_sum_df,how = 'left')
    
    #             cur_varity_missense_df.to_csv(self.db_path + 'varity/bygene/varity_web_' + p_name + '[' + cur_pvid + ']' + '_missense.csv', index = False)
    #             cur_snv_list_df.to_csv(self.db_path + 'varity/bygene/varity_web_' + p_name + '[' + cur_pvid + ']' + '_list.csv', index = False)
                            
                cur_varity_snv_predicted_df = pd.read_csv(self.db_path + 'varity/bygene/' + cur_pvid + '_varity_snv_predicted.csv')
                cur_varity_snv_predicted_df = cur_varity_snv_predicted_df.drop(columns = ['Unnamed: 0','chr','nt_pos','nt_ref','nt_alt'])
                prediction_socre_cols = list(set(cur_varity_snv_predicted_df.columns) - set(['aa_alt', 'aa_pos', 'aa_ref', 'p_vid']))            
                cur_varity_missense_predicted_df = cur_varity_snv_predicted_df.groupby(['aa_alt', 'aa_pos', 'aa_ref', 'p_vid'])[prediction_socre_cols].agg('mean').reset_index()
                
                ####*************************************************************************************************************************************************************
                # Create full length (all position all possible substitutions) matrix
                ####*************************************************************************************************************************************************************        
                aa_len = len(p_seq)
                print (cur_pvid + ' length: ' + str(aa_len))
                protein_all_matrix = np.full((21, len(p_seq)), np.nan)                                                           
                protein_all_matrix_df = pd.DataFrame(protein_all_matrix, columns=range(1, len(p_seq) + 1), index=self.lst_aa_21)
                protein_all_matrix_df['aa_alt'] = protein_all_matrix_df.index
                protein_all_matrix_df = pd.melt(protein_all_matrix_df, ['aa_alt'])
                protein_all_matrix_df = protein_all_matrix_df.rename(columns={'variable': 'aa_pos', 'value': 'aa_ref'})        
                protein_all_matrix_df['aa_ref'] = protein_all_matrix_df['aa_pos'].apply(lambda x: list(p_seq)[x - 1])
                protein_all_matrix_df['p_vid'] = cur_pvid
                protein_all_matrix_df['aa_pos'] = protein_all_matrix_df['aa_pos'].astype(int)
                varity_protein_df = protein_all_matrix_df.merge(cur_varity_missense_df,how = 'left')
                varity_protein_df = varity_protein_df.merge(cur_varity_missense_predicted_df,how = 'left')
                        
    #             ####*************************************************************************************************************************************************************
    #             #psipred (can be retrieved on the fly if doesn't exist)
    #             ####*************************************************************************************************************************************************************
                psipred_file = self.db_path + 'psipred/bygene/' + cur_pvid + '_psipred.csv'      
                if os.path.isfile(psipred_file):
#                     self.run_psipred_by_uniprotid(cur_pvid)                             
                    psipred = pd.read_csv(psipred_file)
                    psipred.columns = ['aa_psipred','aa_pos','ss_end_pos']
                    psipred['aa_alt'] = '*'
                    varity_protein_df = pd.merge(varity_protein_df,psipred,how = 'left')
                else:
                    varity_protein_df['aa_psipred'] = np.nan
                    varity_protein_df['ss_end_pos'] = np.nan
                                 
    #             ####*************************************************************************************************************************************************************
    #             #pfam 
    #             ####*************************************************************************************************************************************************************
                pfam_file = self.db_path + 'pfam/bygene/' + cur_pvid + '_pfam.csv'  
                if os.path.isfile(pfam_file):
                    pfam = pd.read_csv(pfam_file)[['p_vid','a_start','a_end','hmm_id']]
                    pfam.columns = ['p_vid','aa_pos','pfam_end_pos','hmm_id']
                    pfam['aa_alt'] = '*'
                    varity_protein_df = pd.merge(varity_protein_df,pfam,how = 'left')  
                else:
                    varity_protein_df['pfam_end_pos'] = np.nan
                    varity_protein_df['hmm_id'] = np.nan
    #                 
                ####*************************************************************************************************************************************************************
                # ASA_Mean Normalization 
                ####*************************************************************************************************************************************************************                                    
    
                varity_protein_df['asa_mean_normalized'] = (varity_protein_df['asa_mean'] - np.nanmin(varity_protein_df['asa_mean'])) / (np.nanmax(varity_protein_df['asa_mean']) - np.nanmin(varity_protein_df['asa_mean']))
                
                ####*********************************************************************************************************************************************************
                # Score colors
                ####*********************************************************************************************************************************************************
                v_max_varity = 1
                v_center_varity = 0.5
                v_min_varity = 0
                v_max_varity_color = '#C6172B'
                v_center_varity_color = '#FFFFFF'
                v_min_varity_color = '#3155C6'
                n_gradient_max_varity = 5
                n_gradient_min_varity = 5                
                [lst_max_colors_varity, lst_min_colors_varity] = alm_fun.create_color_gradients(v_max_varity, v_min_varity, v_center_varity, v_max_varity_color, v_min_varity_color, v_center_varity_color, n_gradient_max_varity, n_gradient_min_varity)
                varity_protein_df['VARITY_R_colorcode'] = varity_protein_df['VARITY_R'].apply(lambda x: alm_fun.get_colorcode(x, v_max_varity, v_min_varity, v_center_varity, n_gradient_max_varity, n_gradient_min_varity, lst_max_colors_varity, lst_min_colors_varity))
                varity_protein_df['VARITY_R_LOO_colorcode'] = varity_protein_df['VARITY_R_LOO'].apply(lambda x: alm_fun.get_colorcode(x, v_max_varity, v_min_varity, v_center_varity, n_gradient_max_varity, n_gradient_min_varity, lst_max_colors_varity, lst_min_colors_varity))
                varity_protein_df['VARITY_ER_colorcode'] = varity_protein_df['VARITY_ER'].apply(lambda x: alm_fun.get_colorcode(x,v_max_varity, v_min_varity, v_center_varity, n_gradient_max_varity, n_gradient_min_varity, lst_max_colors_varity, lst_min_colors_varity))
                varity_protein_df['VARITY_ER_LOO_colorcode'] = varity_protein_df['VARITY_ER_LOO'].apply(lambda x: alm_fun.get_colorcode(x,v_max_varity, v_min_varity, v_center_varity, n_gradient_max_varity, n_gradient_min_varity, lst_max_colors_varity, lst_min_colors_varity))                
        
        #         [lst_max_colors_sift,lst_min_colors_sift] = alm_fun.create_color_gradients(1, 0, 0.05,'#C6172B','#FFFFFF','#3155C6',10,10)
                [lst_max_colors_sift, lst_min_colors_sift] = alm_fun.create_color_gradients(1, 0, 0, '#FFFFFF', '#3155C6', '#3155C6', 10, 10)
                varity_protein_df['sift_colorcode'] = varity_protein_df['sift_score'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_sift, lst_min_colors_sift))
               
        #         [lst_max_colors_polyphen, lst_min_colors_polyphen] = alm_fun.create_color_gradients(1, 0, 0, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)
        #         varity_protein_df['polyphen_colorcode'] = varity_protein_df['polyphen_score'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_polyphen, lst_min_colors_polyphen))
        
                [lst_max_colors_gnomad, lst_min_colors_gnomad] = alm_fun.create_color_gradients(10, 0.3, 0.3, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)                            
                varity_protein_df['gnomad_af_log10_colorcode'] = varity_protein_df['gnomad_af_log10'].apply(lambda x: alm_fun.get_colorcode(x, 10, 0.3, 0.3, 10, 10, lst_max_colors_gnomad, lst_min_colors_gnomad))
        
                [lst_max_colors_provean, lst_min_colors_provean] = alm_fun.create_color_gradients(4, -13, -13, '#FFFFFF', '#3155C6', '#3155C6', 10, 10)
                varity_protein_df['provean_colorcode'] = varity_protein_df['provean_score'].apply(lambda x: alm_fun.get_colorcode(x, 4, -13, -13, 10, 10, lst_max_colors_provean, lst_min_colors_provean))
                    
                [lst_max_colors_asa, lst_min_colors_asa] = alm_fun.create_color_gradients(1, 0, 0, '#3155C6', '#FFFFFF', '#FFFFFF', 10, 10)  
                varity_protein_df['asa_colorcode'] = varity_protein_df['asa_mean_normalized'].apply(lambda x: alm_fun.get_colorcode(x, 1, 0, 0, 10, 10, lst_max_colors_asa, lst_min_colors_asa))
         
                ####*********************************************************************************************************************************************************
                #GERP++ name not accepted in javascript
                ####*********************************************************************************************************************************************************
                new_cols = [x.replace('++','') for x in varity_protein_df.columns ]        
                varity_protein_df.columns = new_cols
                
                ####*********************************************************************************************************************************************************
                # Round VARITY values and  SHAP values , also convert SHAP values to percentage
                ####*********************************************************************************************************************************************************
                varity_protein_df = varity_protein_df.round(3)
                for col in new_cols:
                    if 'total_shap' not in col:
                        if 'shap_r' in col:
                            varity_protein_df[col] = 100*varity_protein_df[col]/varity_protein_df['total_shap_r']
                            varity_protein_df[col] = varity_protein_df[col].map('{:,.2f}'.format) + '%'
                            varity_protein_df.loc[varity_protein_df[col]== 'nan%',col] = 'N/A' 
                        if 'shap_er' in col:
    #                         print (col)
                            varity_protein_df[col] = 100*varity_protein_df[col]/varity_protein_df['total_shap_er']
                            varity_protein_df[col] = varity_protein_df[col].map('{:,.2f}'.format) + '%'
                            varity_protein_df.loc[varity_protein_df[col]== 'nan%',col] = 'N/A' 
                                 
                output_cols = ['aa_alt', 'aa_pos', 'aa_ref', 'p_vid', 'chr','nt_pos','nt_ref','nt_alt','gnomAD_exomes_AF','gnomAD_exomes_AC', 'gnomAD_exomes_nhomalt'] + \
                ['clinvar_source', 'clinvar_review_star', 'clinvar_clinsig_level', 'clinvar_clin_sig', 'clinvar_id', 'hgmd_source', 'mave_source'] + \
                ['humsavar_source', 'gnomad_source', 'clinvar_label', 'hgmd_label', 'humsavar_label', 'provean_score', 'sift_score'] + \
                ['evm_epistatic_score', 'integrated_fitCons_score', 'LRT_score', 'GERP_RS', 'phyloP30way_mammalian', 'phastCons30way_mammalian'] + \
                ['SiPhy_29way_logOdds', 'blosum100', 'in_domain', 'asa_mean', 'aa_psipred_E', 'aa_psipred_H', 'aa_psipred_C'] + \
                ['bsa_max', 'h_bond_max', 'salt_bridge_max', 'disulfide_bond_max', 'covelent_bond_max', 'solv_ne_min','solv_ne_max','solv_ne_abs_max'] + \
                ['mw_delta', 'pka_delta', 'pkb_delta', 'pi_delta', 'hi_delta', 'pbr_delta', 'avbr_delta', 'vadw_delta', 'asa_delta'] + \
                ['cyclic_delta', 'charge_delta', 'positive_delta', 'negative_delta', 'hydrophobic_delta', 'polar_delta', 'ionizable_delta'] + \
                ['aromatic_delta', 'aliphatic_delta', 'hbond_delta', 'sulfur_delta', 'essential_delta', 'size_delta'] + \
                ['provean_score_shap_r', 'sift_score_shap_r', 'evm_epistatic_score_shap_r', 'integrated_fitCons_score_shap_r'] + \
                ['LRT_score_shap_r', 'GERP_RS_shap_r', 'phyloP30way_mammalian_shap_r', 'phastCons30way_mammalian_shap_r', 'SiPhy_29way_logOdds_shap_r'] + \
                ['blosum100_shap_r', 'in_domain_shap_r', 'asa_mean_shap_r', 'aa_psipred_E_shap_r', 'aa_psipred_H_shap_r', 'aa_psipred_C_shap_r', 'bsa_max_shap_r'] + \
                ['h_bond_max_shap_r', 'salt_bridge_max_shap_r', 'disulfide_bond_max_shap_r', 'covelent_bond_max_shap_r', 'solv_ne_abs_max_shap_r', 'mw_delta_shap_r'] + \
                ['pka_delta_shap_r', 'pkb_delta_shap_r', 'pi_delta_shap_r', 'hi_delta_shap_r', 'pbr_delta_shap_r', 'avbr_delta_shap_r', 'vadw_delta_shap_r'] + \
                ['asa_delta_shap_r', 'cyclic_delta_shap_r', 'charge_delta_shap_r', 'positive_delta_shap_r', 'negative_delta_shap_r', 'hydrophobic_delta_shap_r'] + \
                ['polar_delta_shap_r', 'ionizable_delta_shap_r', 'aromatic_delta_shap_r', 'aliphatic_delta_shap_r', 'hbond_delta_shap_r', 'sulfur_delta_shap_r'] + \
                ['essential_delta_shap_r', 'size_delta_shap_r', 'base_shap_r', 'total_shap_r', 'provean_score_shap_er', 'sift_score_shap_er'] + \
                ['evm_epistatic_score_shap_er', 'integrated_fitCons_score_shap_er', 'LRT_score_shap_er', 'GERP_RS_shap_er', 'phyloP30way_mammalian_shap_er'] + \
                ['phastCons30way_mammalian_shap_er', 'SiPhy_29way_logOdds_shap_er', 'blosum100_shap_er', 'in_domain_shap_er', 'asa_mean_shap_er', 'aa_psipred_E_shap_er'] + \
                ['aa_psipred_H_shap_er', 'aa_psipred_C_shap_er', 'bsa_max_shap_er', 'h_bond_max_shap_er', 'salt_bridge_max_shap_er', 'disulfide_bond_max_shap_er'] + \
                ['covelent_bond_max_shap_er', 'solv_ne_abs_max_shap_er', 'mw_delta_shap_er', 'pka_delta_shap_er', 'pkb_delta_shap_er', 'pi_delta_shap_er', 'hi_delta_shap_er'] + \
                ['pbr_delta_shap_er', 'avbr_delta_shap_er', 'vadw_delta_shap_er', 'asa_delta_shap_er', 'cyclic_delta_shap_er', 'charge_delta_shap_er'] + \
                ['positive_delta_shap_er', 'negative_delta_shap_er', 'hydrophobic_delta_shap_er', 'polar_delta_shap_er', 'ionizable_delta_shap_er'] + \
                ['aromatic_delta_shap_er', 'aliphatic_delta_shap_er', 'hbond_delta_shap_er', 'sulfur_delta_shap_er', 'essential_delta_shap_er',] + \
                ['size_delta_shap_er', 'base_shap_er', 'total_shap_er', 'VARITY_R', 'VARITY_ER','VARITY_R_LOO', 'VARITY_ER_LOO'] + \
                ['aa_psipred', 'ss_end_pos', 'pfam_end_pos', 'hmm_id', 'gnomad_af_log10', 'asa_mean_normalized', 'VARITY_R_colorcode'] + \
                ['VARITY_R_LOO_colorcode', 'VARITY_ER_colorcode', 'VARITY_ER_LOO_colorcode', 'sift_colorcode', 'gnomad_af_log10_colorcode'] + \
                ['provean_colorcode', 'asa_colorcode']                                                      
                varity_protein_df[output_cols].to_csv(self.db_path + 'varity/bygene/' + cur_pvid + '_' + p_name + '_varity_web.txt', sep = '\t',index = False, float_format='%.3f')
                alm_fun.show_msg(cur_log,self.verbose,cur_pvid + ' is done.')
            else:
                alm_fun.show_msg(cur_log,self.verbose,cur_pvid + ' snv file does not exist.')            
        alm_fun.show_msg(cur_log,self.verbose,runtime['varity_batch_id'] + ' is done.')
        alm_fun.show_msg(cur_done_log,1,runtime['varity_batch_id'] + ' is done.')

    def create_varity_all_predictions(self,runtime):        
        cur_log = self.project_path + 'output/log/create_varity_all_predictions.log'
        id2hgnc_dict = np.load(self.db_path + 'hgnc/npy/id2hgnc_dict.npy').item()  
        hgnc2id_dict = np.load(self.db_path + 'hgnc/npy/hgnc2id_dict.npy').item()                  
        #****************************************************************************************************************************
        #1) Create final output files combining all predictions
        #2) Create Supported Uniprot ID list
        #****************************************************************************************************************************
        key_cols = ['chr','nt_pos','nt_ref','nt_alt','p_vid','aa_pos','aa_ref','aa_alt']
        score_cols = ['VARITY_R','VARITY_ER','VARITY_R_LOO','VARITY_ER_LOO']
        count = 0 
        supproted_ids_file = open(self.db_path + 'varity/all/varity_supported_ids.txt','w')
        supproted_ids_file.write('symbol' + '\t' + 'uniprot_id' + '\n')
        for file in glob.glob(self.db_path + 'varity/bygene/*_varity_snv_predicted.csv'):   
            cur_pvid = file.split('/')[-1].split('.')[0].split('_')[0]  
            p_name = hgnc2id_dict['symbol'].get(id2hgnc_dict['uniprot_ids'].get(cur_pvid,''),'')                           
            cur_predicted_df = pd.read_csv(file)
            if count == 0 :
                cur_predicted_df[key_cols + score_cols].to_csv(self.db_path + 'varity/all/varity_all_predictions.txt', sep = '\t',index = False)                                
            else:
                cur_predicted_df[key_cols + score_cols].to_csv(self.db_path + 'varity/all/varity_all_predictions.txt', sep = '\t',mode = 'a', header = False ,index = False)
            
            supproted_ids_file.write(p_name + '\t' + cur_pvid + '\n')            
            count = count + 1
        supproted_ids_file.close()
        alm_fun.show_msg (cur_log,1, "Total number of supported uniprot ids:  " + str(count))
        
        

    def create_varity_genelist(self):  
        
        def find_canonical_uniprot_id(ids):
            ids_lst = ids.split('|')
            cid = ''
            cid_len = 0
            cid_random = 0
            for id in ids_lst:
                cur_cid_len = len(uniprot_seq_dict.get(id,''))
                if cid == '':
                    cid = id
                    cid_len = cur_cid_len
                else:
                    
                    if cid_len == cur_cid_len:# Mulitiple isoforms have equal length, can not determine by the algorithm, pick the first one   
                        cid_random = 1
                    
                    if cid_len < cur_cid_len: #longest protein as the canonical one
                        cid = id
                        cid_len = cur_cid_len
                        cid_random = 0
                    
            return (pd.Series([cid, cid_len, cid_random]))

        print ("Creating varity data......")
        self.init_humamdb_object('varity') 
        cur_log = self.db_path + 'varity/log/varity.log'
        uniprot_seq_dict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy').item()
        hgnc = pd.read_csv(self.db_path + 'hgnc/org/hgnc_complete_set.txt',sep ='\t',dtype = {'location':'str'})        
        varity_genes_df = hgnc.loc[hgnc['uniprot_ids'].notnull(),:]    
        varity_genes_df['uniprot_id'] = np.nan
        varity_genes_df['protein_len'] = 0
        varity_genes_df['uncertain_canonical'] = 0
        varity_genes_df['manual_canonical'] = 0
        varity_genes_df[['uniprot_id','protein_len','uncertain_canonical']] = varity_genes_df.apply(lambda x: find_canonical_uniprot_id(x['uniprot_ids']),axis = 1) 
                
        # manual canonical uniprot_id assignment for the uncertain_cannonical ones and HLA proteins        
        varity_genes_df.loc[varity_genes_df['symbol'] == 'HLA-A',['uniprot_id','protein_len','manual_canonical']] = ['P04439',371,1]
        varity_genes_df.loc[varity_genes_df['symbol'] == 'HLA-B',['uniprot_id','protein_len','manual_canonical']] = ['P01889',362,1]
        varity_genes_df.loc[varity_genes_df['symbol'] == 'HLA-DRB1',['uniprot_id','protein_len','manual_canonical']] = ['P01911',266,1]
        varity_genes_df.loc[varity_genes_df['symbol'] == 'SIRPB1',['uniprot_id','protein_len','manual_canonical']] = ['O00241',398,1]
        
        # remove uniprot_id that is not reviewed        
        uniprot_human_reviewed_ids = list(np.load(self.db_path + 'uniprot/npy/uniprot_human_reviewed_ids.npy'))        
        varity_genes_df = varity_genes_df.loc[varity_genes_df['uniprot_id'].isin(uniprot_human_reviewed_ids)]
        alm_fun.show_msg(self.log,self.verbose,'Total number of reviewed VARITY genes : ' + str(varity_genes_df.shape[0]))
                
        # check the record where protein length = 0, update manually        
#         varity_genes_df.loc[varity_genes_df['protein_len'] == 0,['hgnc_id','symbol','uniprot_ids','uniprot_id','protein_len']]

        hgnc2id_dict = np.load(self.db_path + 'hgnc/npy/hgnc2id_dict.npy').item()
        id2hgnc_dict = np.load(self.db_path + 'hgnc/npy/id2hgnc_dict.npy').item()
        uniprot_seq_dict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy').item()
        
        #************************************************************************************************************************************************************************
        #Combine ClinVAR, HumsaVAR, MAVE, Denovodb, DeepSequence, disease associated genes 
        #************************************************************************************************************************************************************************              
        clinvar_disease_genes = list(pd.read_csv(self.db_path + 'clinvar/all/clinvar_disease_genes.txt',header = None)[0])         
        alm_fun.show_msg(self.log,self.verbose,'Total number of CLINVAR disease genes : ' + str(len(clinvar_disease_genes)))        
        varity_genes_df['clinvar_disease_gene'] = 0
        varity_genes_df.loc[varity_genes_df['hgnc_id'].isin(clinvar_disease_genes),'clinvar_disease_gene'] = 1
                
        humsavar_disease_genes = list(pd.read_csv(self.db_path + 'humsavar/all/humsavar_disease_genes.txt',header = None)[0])         
        alm_fun.show_msg(self.log,self.verbose,'Total number of HUMSAVAR disease genes : ' + str(len(humsavar_disease_genes)))
        varity_genes_df['humsavar_disease_gene'] = 0
        varity_genes_df.loc[varity_genes_df['hgnc_id'].isin(humsavar_disease_genes),'humsavar_disease_gene'] = 1
  
        mave_genes = list(pd.read_csv(self.db_path + 'mave/all/mave_genes.txt',header = None)[0]) + ['HGNC:23663']  #add VKORC1
        alm_fun.show_msg(self.log,self.verbose,'Total number of MAVE genes : ' + str(len(mave_genes)))
        varity_genes_df['mave_gene'] = 0
        varity_genes_df.loc[varity_genes_df['hgnc_id'].isin(mave_genes),'mave_gene'] = 1
        
        denovodb_enriched_gene_names = list(pd.read_csv(self.db_path + 'denovodb/all/denovodb_enriched_genes.csv')['symbol'])        
        denovodb_enriched_genes = [id2hgnc_dict['symbol'][x] for x in denovodb_enriched_gene_names]    
        alm_fun.show_msg(self.log,self.verbose,'Total number of Denovodb enriched genes (Ceo et al.,2019) : ' + str(len(denovodb_enriched_genes)))
        varity_genes_df['denovodb_enriched_gene'] = 0
        varity_genes_df.loc[varity_genes_df['hgnc_id'].isin(denovodb_enriched_genes),'denovodb_enriched_gene'] = 1
        
        ukb_gene_names = ['LDLR','PCSK9','LPL','ACE2','CHEK2','HMBS','LMNA','SDHB','SOD1','STK11','TECR','TMPRSS2','VHL','APOA1','LRP6','APOB','APOA5',
                          'LDLRAP1','HMGCR','MTHFR','TPK1','BRCA1','PPARG','PTEN','ANGPTL4','CETP','ANGPTL3','PDCD1','CTLA4','LAG3','CD200R1','BTLA',
                          'CBS','SUMO1','CALM1','UBE2I','UBQLN2','UBQLN1']
        ukb_genes = [id2hgnc_dict['symbol'][x] for x in ukb_gene_names]    
        alm_fun.show_msg(self.log,self.verbose,'Total number of UK Biobank genes : ' + str(len(ukb_genes)))
        varity_genes_df['ukb_gene'] = 0
        varity_genes_df.loc[varity_genes_df['hgnc_id'].isin(ukb_genes),'ukb_gene'] = 1
        
        deepsequence_gene_names = ['ADRB2','BRCA1','CALM1','MAPK1','TP53','SUMO1','HRAS','PTEN','UBE2I','TPMT','TPK1']
        deepsequence_genes = [id2hgnc_dict['symbol'][x] for x in deepsequence_gene_names]    
        alm_fun.show_msg(self.log,self.verbose,'Total number of DeepSeqeunce genes : ' + str(len(deepsequence_genes)))
        varity_genes_df['deepsequence_gene'] = 0
        varity_genes_df.loc[varity_genes_df['hgnc_id'].isin(deepsequence_genes),'deepsequence_gene'] = 1
        
        varity_disease_genes_lst = list(set(clinvar_disease_genes + mave_genes + humsavar_disease_genes + denovodb_enriched_genes + ukb_genes + deepsequence_genes))
        alm_fun.show_msg(self.log,self.verbose,'Total number of VARITY disease genes : ' + str(len(varity_disease_genes_lst)))        
        varity_genes_df['varity_disease_gene'] = 0
        varity_genes_df.loc[varity_genes_df['hgnc_id'].isin(varity_disease_genes_lst),'varity_disease_gene'] = 1
        
        varity_revision_genes_lst = list(set(mave_genes + denovodb_enriched_genes + ukb_genes + deepsequence_genes))
        alm_fun.show_msg(self.log,self.verbose,'Total number of VARITY revision genes : ' + str(len(varity_revision_genes_lst)))        
        varity_genes_df['varity_revision_gene'] = 0
        varity_genes_df.loc[varity_genes_df['hgnc_id'].isin(varity_revision_genes_lst),'varity_revision_gene'] = 1
                
        varity_genes_df['p_vid'] = varity_genes_df['uniprot_id']
        varity_genes_df.loc[varity_genes_df['location'].isnull(),'location'] = ''        
        varity_genes_df['chr'] = varity_genes_df.apply(lambda x: x['location'].replace('p','q').split('q')[0],axis = 1)
        varity_genes_df.loc[varity_genes_df['location'] == 'mitochondria','chr'] = 'MT'   
        varity_genes_df.loc[~varity_genes_df['chr'].isin(self.lst_chr),'chr'] = np.nan  
        
        pdb_to_uniprot = pd.read_csv(self.db_path + 'pisa/org/pdb_chain_uniprot_processed.csv',dtype={"PDB": str})[['PDB','SP_PRIMARY']] 
        pdb_to_uniprot.columns = ['pdb','p_vid']    
        pdb_to_uniprot = pdb_to_uniprot.drop_duplicates()
        pdb_to_uniprot_count = pdb_to_uniprot.groupby(['p_vid']).agg('count').reset_index()
        pdb_to_uniprot_count.columns = ['p_vid','pdb_count']
        
        varity_genes_df = varity_genes_df.merge(pdb_to_uniprot_count,how = 'left')
                
        varity_genes_df.to_csv(self.db_path + 'varity/all/varity_all_genes.csv',index = False)
        varity_genes_df.loc[varity_genes_df['varity_disease_gene'] == 1,['p_vid','symbol','hgnc_id','chr']].to_csv(self.db_path + 'varity/all/varity_disease_genes.csv',index = False)
        varity_genes_df.loc[varity_genes_df['varity_revision_gene'] == 1,['p_vid','symbol','hgnc_id','chr']].to_csv(self.db_path + 'varity/all/varity_revision_genes.csv',index = False)
        varity_genes_df.loc[varity_genes_df['ukb_gene'] == 1,['p_vid','symbol','hgnc_id','chr']].to_csv(self.db_path + 'varity/all/varity_ukb_genes.csv',index = False)        
        varity_genes_df.loc[varity_genes_df['denovodb_enriched_gene'] == 1,['p_vid','symbol','hgnc_id','chr']].to_csv(self.db_path + 'varity/all/varity_denovodb_enriched_genes.csv',index = False)        
        varity_genes_df.loc[varity_genes_df['mave_gene'] == 1,['p_vid','symbol','hgnc_id','chr']].to_csv(self.db_path + 'varity/all/varity_mave_genes.csv',index = False)
        varity_genes_df.loc[varity_genes_df['deepsequence_gene'] == 1,['p_vid','symbol','hgnc_id','chr']].to_csv(self.db_path + 'varity/all/varity_deepsequence_genes.csv',index = False)
        
                            
        #### create varity uniprot_id to symbol dict 
        varity_gene_dict = {}
        varity_gene_dict['uniprot2symbol'] = {}
        varity_gene_dict['uniprot2hgnc'] = {}
        varity_gene_dict['symbol2uniprot'] = {}
        varity_gene_dict['hgnc2uniprot'] = {}
        
        for idx in varity_genes_df.index:
            cur_uniprot_id = varity_genes_df.loc[idx,'p_vid']
            cur_symbol = varity_genes_df.loc[idx,'symbol']
            cur_hgnc_id= varity_genes_df.loc[idx,'hgnc_id']
            varity_gene_dict['uniprot2symbol'][cur_uniprot_id] = cur_symbol
            varity_gene_dict['uniprot2symbol'][cur_symbol] = cur_uniprot_id
            varity_gene_dict['symbol2uniprot'][cur_symbol] = cur_hgnc_id
            varity_gene_dict['hgnc2uniprot'][cur_hgnc_id] = cur_uniprot_id
        
        np.save(self.db_path + 'varity/npy/varity_gene_dict.npy', varity_gene_dict)          
        alm_fun.show_msg(cur_log,self.verbose,'Varity data initiated.\n')
        
    def create_psipred_by_uniprotids(self,runtime):
        cur_log =  self.project_path + 'output/log/create_psipred_by_uniprotids_' + runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/create_psipred_by_uniprotids_' + runtime['varity_batch_id'] + '_done.log'
        
        for uniprot_id in runtime['uniprot_ids']:        
            #**********************************************************************************************
            # PISPRED : after installation, remeber to edit the runpsipred script to make it work!!!!!!!
            #**********************************************************************************************               
            if os.path.isfile(self.db_path + 'psipred/bygene/' + uniprot_id + '.ss2') :
                alm_fun.show_msg(cur_log,self.verbose, uniprot_id + ' psipred exists.')
            else:
                alm_fun.show_msg(cur_log,self.verbose, 'strat to retrieve the psipred info of ' + uniprot_id + '......' )
                psipred_cmd = "runpsipred " + self.db_path + "uniprot/bygene/" + uniprot_id + ".fasta"                
                subprocess.run(psipred_cmd.split(" "), cwd = self.db_path + '../../tools/psipred/psipred/' )
                alm_fun.show_msg(cur_log,self.verbose, uniprot_id + ' psipred is retrieved.')
    
            psipred_file = self.db_path + 'psipred/bygene/' + uniprot_id + '.ss2'        
            cur_psipred_df = pd.read_csv(psipred_file, skiprows=[0, 1], header=None, sep='\s+')
            cur_psipred_df = cur_psipred_df.loc[:, [0, 1, 2]]
            cur_psipred_df.columns = ['aa_pos', 'aa_ref', 'aa_psipred']
            
            #create files for secondary structure regions
            if not os.path.isfile(self.db_path + 'psipred/bygene/' + uniprot_id + '_psipred.csv') :                     
                sc_seq = ''.join(list(cur_psipred_df['aa_psipred']))
                psipred_seq_file =  open(self.db_path + 'psipred/bygene/' + uniprot_id + '_psipred.csv','w')
                psipred_seq_str = 'aa_psipred,aa_pos,ss_end_pos\n'
                sc_start = ''
                cur_start_pos = 1
                for i in range(1,len(sc_seq)+1):
                    cur_sc = sc_seq[i-1]
                    if cur_sc != sc_start:
                        pre_start_pos = cur_start_pos
                        if i != 1:
                            pre_end_pos = i-1
                            psipred_seq_str += sc_start + ',' + str(pre_start_pos) + ',' + str(pre_end_pos) + '\n'                                        
                        sc_start = cur_sc
                        cur_start_pos = i
                    if i == cur_psipred_df.shape[0]:
                        psipred_seq_str += sc_start + ',' + str(cur_start_pos) + ',' + str(i) + '\n'
                pass            
                psipred_seq_file.write(psipred_seq_str)      
                psipred_seq_file.close()
            alm_fun.show_msg(cur_log,self.verbose, uniprot_id + ' psipred is done.')
        
        alm_fun.show_msg(cur_done_log,self.verbose, runtime['varity_batch_id'] + ' is done.')
                            
    def create_varity_data_by_uniprotids(self,runtime):        
        uniprot_ids = runtime['uniprot_ids']  
        combined_dbnsfp_output_df = None
        cur_log = self.project_path + 'output/log/' + 'create_varity_data_by_uniprotids_'+ runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/' + 'create_varity_data_by_uniprotids_' + runtime['varity_batch_id'] + '_done.log'
        ####***************************************************************************************************************************************************************
        #### retrieve the DBNSFP output
        ####***************************************************************************************************************************************************************
        for uniprot_id in uniprot_ids:                            
            dbnsfp_output_exist = 0
            dbnsfp_file_exist = 0                   
            dbnsfp_output_file = self.db_path + 'varity/bygene/' + uniprot_id + '_dbnsfp.out'            
            if os.path.isfile(dbnsfp_output_file):
                dbnsfp_file_exist = 1
                if (os.stat(dbnsfp_output_file).st_size != 0):                                    
                    input_data = pd.read_csv(dbnsfp_output_file, sep = '\t',dtype = {'hg19_chr':'str'})
                    if input_data.shape[0] > 0:
                        dbnsfp_output_exist = 1
                        alm_fun.show_msg(cur_log,self.verbose,'dbNSFP result for ' + uniprot_id + ' is available.')
                
            if dbnsfp_file_exist == 0:
                alm_fun.show_msg(cur_log,self.verbose,'Run dbNSFP for  ' + uniprot_id + '......')                  
                varity_genes_file = open(self.db_path + 'varity/bygene/' + uniprot_id + '_dbnsfp.input','w')
                varity_genes_file.write('Uniprot:' + uniprot_id + '\n')
                varity_genes_file.close()   
                dbnsfp_cmd = "java search_dbNSFP40a -v hg19 -i " + self.db_path + "varity/bygene/" + uniprot_id + "_dbnsfp.input -o " + self.db_path + "varity/bygene/" + uniprot_id + "_dbnsfp.out"        
                subprocess.run(dbnsfp_cmd.split(" "), cwd = self.db_path + 'dbnsfp_v4.0_0503/')                
                if os.path.isfile(dbnsfp_output_file):                
                    input_data = pd.read_csv(dbnsfp_output_file, sep = '\t',dtype = {'hg19_chr':'str'})
                    if input_data.shape[0] == 0:                
                        alm_fun.show_msg(cur_log,self.verbose,'There is no dbNSFP result after retrieval for ' + uniprot_id +'.')
                    else:
                        dbnsfp_output_exist = 1
                    
            if dbnsfp_output_exist == 1:                
                #### use uniprot_id that used in searching dbNSFP as p_vid 
                input_data['p_vid'] = uniprot_id
                if combined_dbnsfp_output_df is None:
                    combined_dbnsfp_output_df = input_data
                else:
                    combined_dbnsfp_output_df = pd.concat([combined_dbnsfp_output_df,input_data]) 
                alm_fun.show_msg(cur_log,self.verbose,'# of dbNSFP records for  ' + uniprot_id +' is ' + str(input_data.shape[0]))
        ####***************************************************************************************************************************************************************
        #### Process the combined DBNSFP output and then save processed result for individual UniProt IDs.                         
        ####***************************************************************************************************************************************************************
        if combined_dbnsfp_output_df is not None:   
#             #### add MISTIC scores
#             combined_dbnsfp_output_df = self.add_mistic(combined_dbnsfp_output_df)
            alm_fun.show_msg(cur_log,self.verbose,'Process combined dbNSFP output [#: ' + str(combined_dbnsfp_output_df.shape[0]) + ']......')
            varity_snv_by_uniprotids = self.run_preocess_dbnsfp_output(combined_dbnsfp_output_df,cur_log)                        
            #### save for individual UniProt IDs            
            for uniprot_id in uniprot_ids:
                varity_snv_by_uniprotid = varity_snv_by_uniprotids.loc[varity_snv_by_uniprotids['p_vid'] == uniprot_id,:]
                if varity_snv_by_uniprotid.shape[0] > 0 :
                    varity_snv_by_uniprotid.to_csv(self.db_path + 'varity/bygene/' + uniprot_id + '_varity_snv.csv',index = False)
                
#             varity_snv_by_uniprotids.to_csv(self.db_path + 'varity/csv/' + runtime['varity_batch_id'] + '_varity_batch_snv.csv',index = False) 
                                       
        alm_fun.show_msg(cur_log,self.verbose,runtime['varity_batch_id'] + ' is done.')
        alm_fun.show_msg(cur_done_log,1,runtime['varity_batch_id'] + ' is done.')

    def create_varity_prediction_data_by_uniprotids(self,runtime):
        uniprot_ids = runtime['uniprot_ids']  
        cur_log = self.project_path + 'output/log/' + 'create_varity_prediction_data_by_uniprotids_'+ runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/' + 'create_varity_prediction_data_by_uniprotids_' + runtime['varity_batch_id'] + '_done.log'

        ####combine the varity_snv for all uniprot_ids
        varity_batch_snv_df = None
        for uniprot_id in uniprot_ids:
            cur_snv_file = self.db_path + 'varity/bygene/' + uniprot_id + '_varity_snv.csv'
            if os.path.isfile(cur_snv_file):
                cur_snv_df = pd.read_csv(cur_snv_file,low_memory = False)
                alm_fun.show_msg(cur_log,self.verbose, 'Uniprot ID: ' + uniprot_id) 
    #             alm_fun.show_msg(cur_log,self.verbose, str(cur_snv_df.dtypes))
                alm_fun.show_msg(cur_log,self.verbose, str(cur_snv_df.shape))
                if cur_snv_df.shape[0] > 0 :
                    if varity_batch_snv_df is None:
                        varity_batch_snv_df = cur_snv_df 
                    else:
                        varity_batch_snv_df = pd.concat([varity_batch_snv_df,cur_snv_df])
            else:
                alm_fun.show_msg(cur_log,self.verbose, 'Uniprot ID: ' + uniprot_id + ' has no snv data.')     
            
        if varity_batch_snv_df is not None:
            alm_fun.show_msg(cur_log,self.verbose,'Number of varity batch records: ' + str(varity_batch_snv_df.shape[0]))
    
            ####make predictions for VARITY_R and VARITY_ER
            runtime['target_type'] = 'dataframe'
            runtime['target_dataframe'] = varity_batch_snv_df
            runtime['varity_predictor'] = 'VARITY_ER'
            er_results = self.run_varity_predictions(runtime) 
            if er_results['shap_output_target'] is not None:          
                er_predictions_df = er_results['shap_output_target']            
                rename_dict = {x: x+'_er' for x in er_predictions_df.columns if 'shap' in x}
                er_predictions_df = er_predictions_df.rename(columns = rename_dict)
                
            else:
                er_predictions_df = er_results['target_predictions']
                
            alm_fun.show_msg(cur_log,self.verbose,'VARITY_ER prediction is done.')  
                  
            runtime['varity_predictor'] = 'VARITY_R'
            r_results = self.run_varity_predictions(runtime) 
            if r_results['shap_output_target'] is not None:          
                r_predictions_df = r_results['shap_output_target']            
                rename_dict = {x: x+'_r' for x in r_predictions_df.columns if 'shap' in x}
                r_predictions_df = r_predictions_df.rename(columns = rename_dict)
                
            else:
                r_predictions_df = r_results['target_predictions']
            alm_fun.show_msg(cur_log,self.verbose,'VARITY_R prediction is done.')  
                    
            varity_batch_prediction_df = pd.merge(er_predictions_df,r_predictions_df,how = 'left')
            alm_fun.show_msg(cur_log,self.verbose,'Number of varity batch records after merge prediction R and ER: ' + str(varity_batch_prediction_df.shape[0]))
     
            #### load VARITY LOO predictions  
            target_loo_er_df = pd.read_csv(self.db_path + '/varity/csv/' + runtime['varity_session_id'] + '_VARITY_ER_loo_predictions_with_keycols.csv')                   
            target_loo_r_df = pd.read_csv(self.db_path + '/varity/csv/' + runtime['varity_session_id'] + '_VARITY_R_loo_predictions_with_keycols.csv') 
                                           
            varity_batch_prediction_df = pd.merge(varity_batch_prediction_df,target_loo_er_df,how = 'left')
            varity_batch_prediction_df.loc[varity_batch_prediction_df['VARITY_ER_LOO'].isnull(),'VARITY_ER_LOO'] = varity_batch_prediction_df.loc[varity_batch_prediction_df['VARITY_ER_LOO'].isnull(),'VARITY_ER']
            alm_fun.show_msg(cur_log,self.verbose,'Number of varity batch records after merge varity_loo_er_df: ' + str(varity_batch_prediction_df.shape[0]))
                         
            varity_batch_prediction_df = pd.merge(varity_batch_prediction_df,target_loo_r_df,how = 'left')
            varity_batch_prediction_df.loc[varity_batch_prediction_df['VARITY_R_LOO'].isnull(),'VARITY_R_LOO'] = varity_batch_prediction_df.loc[varity_batch_prediction_df['VARITY_R_LOO'].isnull(),'VARITY_R']
            alm_fun.show_msg(cur_log,self.verbose,'Number of varity batch records after merge varity_loo_r_df: ' + str(varity_batch_prediction_df.shape[0]))
     
            ### save predictions for individual uniprot_id
            for p_vid in varity_batch_prediction_df['p_vid'].unique():                
                varity_pvid_prediction_df = varity_batch_prediction_df.loc[varity_batch_prediction_df['p_vid'] == p_vid,:]
                varity_pvid_prediction_df.to_csv(self.db_path + '/varity/bygene/' + p_vid + '_varity_snv_predicted.csv')
                
        alm_fun.show_msg(cur_log,self.verbose, runtime['varity_batch_id'] + ' is done.')
        alm_fun.show_msg(cur_done_log,self.verbose, runtime['varity_batch_id'] + ' is done.')            
                    

    def create_varity_target_data(self,runtime):
        cur_log = self.project_path + 'output/log/create_varity_target_data_' + runtime['target_data_name'] +'.log'
        key_cols = ['chr','nt_pos','nt_ref','nt_alt','p_vid','aa_pos','aa_ref','aa_alt']
        annotation_cols = ['clinvar_id','clinvar_source','hgmd_source','gnomad_source','humsavar_source','mave_source',
                           'clinvar_label','hgmd_label','gnomad_label','humsavar_label','mave_label','label',
                           'train_clinvar_source','train_hgmd_source','train_gnomad_source','train_humsavar_source','train_mave_source']                           
        score_cols = ['Polyphen2_selected_HVAR_score','Polyphen2_selected_HDIV_score','PROVEAN_selected_score','SIFT_selected_score',
                      'CADD_raw','PrimateAI_score','Eigen-raw_coding','GenoCanyon_score','integrated_fitCons_score','REVEL_score',
                      'M-CAP_score','LRT_score','MutationTaster_selected_score','MutationAssessor_selected_score',
                      'FATHMM_selected_score','MetaSVM_score','MetaLR_score','DANN_score','GERP++_RS',
                      'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','MPC_selected_score','mistic_score',
                      'mpc_score','deepsequence_score','mave_input','mave_norm','mave_score','sift_score','provean_score']        
        feature_cols = ['provean_score','sift_score','evm_epistatic_score','integrated_fitCons_score','LRT_score','GERP++_RS',
                        'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','blosum100','in_domain','asa_mean','aa_psipred_E',
                        'aa_psipred_H','aa_psipred_C','bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_abs_max',
                        'mw_delta','pka_delta','pkb_delta','pi_delta','hi_delta','pbr_delta','avbr_delta','vadw_delta','asa_delta','cyclic_delta','charge_delta',
                        'positive_delta','negative_delta','hydrophobic_delta','polar_delta','ionizable_delta','aromatic_delta','aliphatic_delta','hbond_delta',
                        'sulfur_delta','essential_delta','size_delta']
        
        qip_cols = ['gnomAD_exomes_AF','gnomAD_exomes_AC','gnomAD_exomes_nhomalt','mave_label_confidence','clinvar_review_star','accessibility']
        
        other_cols = ['mpc_obs_exp','mpc_mis_badness','mpc_fitted_score','RVIS_EVS','RVIS_percentile_EVS','LoF-FDR_ExAC','RVIS_ExAC','RVIS_percentile_ExAC']
        
        all_cols = list(key_cols + annotation_cols + list(set(score_cols + feature_cols)) + qip_cols + other_cols)
        
        if runtime['target_data_type'] == 'snv':
            input_snv_df = pd.read_csv(runtime['target_snv_file'],dtype = {'chr':'str'})
            input_genes = list(input_snv_df['p_vid'].unique())
        if runtime['target_data_type'] == 'uniprot_files':
            input_genes_df = pd.read_csv(runtime['target_uniprotid_file'])
            input_genes = list(input_genes_df['p_vid'].unique())
        if runtime['target_data_type'] == 'uniprot_ids':            
            input_genes = runtime['uniprot_ids']  
            
        target_snv_df = None
        target_snv_predicted_df = None
        varity_gene_dict = np.load(self.db_path + 'varity/npy/varity_gene_dict.npy').item()
        for p_vid in input_genes:
            symbol = varity_gene_dict['uniprot2symbol'][p_vid]
            alm_fun.show_msg(cur_log,self.verbose, p_vid + ':[' + symbol + ']')
            cur_snv_file = self.db_path + 'varity/bygene/' + p_vid + '_varity_snv.csv'
            cur_snv_predicted_file = self.db_path + 'varity/bygene/' + p_vid + '_varity_snv_predicted.csv'
            if os.path.isfile(cur_snv_file):            
                cur_snv_df = pd.read_csv(cur_snv_file,dtype = {'chr':'str'})[all_cols]
                cur_snv_df['symbol'] = symbol                
                if runtime['target_data_type'] == 'snv':
                    cur_input_snv_df = input_snv_df.loc[input_snv_df['p_vid'] == p_vid,:]  
#                     print(cur_input_snv_df.dtypes)
#                     print(cur_snv_df.dtypes)              
                    cur_input_snv_df = pd.merge(cur_input_snv_df,cur_snv_df,how = 'left')
                    cur_snv_df = cur_input_snv_df                
                if target_snv_df is None:
                    target_snv_df = cur_snv_df
                else:
                    target_snv_df = pd.concat([target_snv_df,cur_snv_df])
            else:
                alm_fun.show_msg(cur_log,self.verbose, p_vid + ', no snv data.')
            
#             if os.path.isfile(cur_snv_predicted_file):
#                 cur_snv_predicted_df = pd.read_csv(cur_snv_predicted_file,dtype = {'chr':'str'})
#                 if runtime['target_data_type'] == 'snv':
#                     cur_input_snv_df = input_snv_df.loc[input_snv_df['p_vid'] == p_vid,:]                
#                     cur_input_snv_df = pd.merge(cur_input_snv_df,cur_snv_predicted_df,how = 'left')
#                     cur_snv_predicted_df = cur_input_snv_df  
#                                 
#                 if target_snv_predicted_df is None:
#                     target_snv_predicted_df = cur_snv_predicted_df
#                 else:
#                     target_snv_predicted_df = pd.concat([target_snv_predicted_df,cur_snv_predicted_df])
#             else:
#                 alm_fun.show_msg(cur_log,self.verbose, p_vid + ', no snv predicted data.')

#         target_df = pd.merge(target_snv_df,target_snv_predicted_df,how = 'left')
        
        ####take care the LOO columns
        
#         loo_cols = [x for x in target_df.columns if 'LOO' in x]
#         for loo_col in loo_cols:
#             non_loo_col = loo_col[0:-4]
#             target_df[non_loo_col + '_train_flag'] = 0
#             target_df.loc[target_df[loo_col].notnull(),non_loo_col + '_train_flag'] = 1
#             target_df.loc[target_df[loo_col].isnull(),loo_col] = target_df.loc[target_df[loo_col].isnull(),non_loo_col]
        
        
        target_snv_df.to_csv (self.db_path + 'varity/all/' + runtime['target_data_name'] + '.csv', index = False)
         
    def check_varity_data_by_uniprotids(self,runtime):
        cur_log = self.db_path + 'varity/log/check_varity_data_by_uniprotids.log'
        varity_genes = pd.read_csv(runtime['batch_uniprotid_file'])
        varity_genes['pisa_available'] = 0
        varity_genes['psipred_available'] = 0
        varity_genes['pfam_available'] = 0        
        varity_genes['dbnsfp_available'] = 0
        varity_genes['varity_snv_available'] = 0
        varity_genes['varity_snv_prediction_available'] = 0
        
        count = 0
                    
        for p_vid in varity_genes['p_vid']:
            count = count + 1    
            print (p_vid + '-' + str(count))        
            #### check dbNSFP input, dbNSFP output, varity_snv             
            pisa_file = self.db_path + 'pisa/bygene/' + p_vid + '_pisa.csv'
            psipred_file = self.db_path + 'psipred/bygene/' + p_vid + '.ss2'
            pfam_file = self.db_path + 'pfam/bygene/' + p_vid + '_pfam.csv'
            dbnsfp_output_file = self.db_path + 'varity/bygene/' + p_vid + '_dbnsfp.out'
            varity_snv_file = self.db_path + 'varity/bygene/' + p_vid + '_varity_snv.csv'
            varity_snv_prediction_file = self.db_path + 'varity/bygene/' + p_vid + '_varity_snv_predicted.csv'
                        
            if os.path.isfile(pisa_file):
                varity_genes.loc[varity_genes['p_vid'] == p_vid, 'pisa_available'] = 1
            if os.path.isfile(psipred_file):
                varity_genes.loc[varity_genes['p_vid'] == p_vid, 'psipred_available'] = 1
            if os.path.isfile(pfam_file):
                varity_genes.loc[varity_genes['p_vid'] == p_vid, 'pfam_available'] = 1  
            if os.path.isfile(dbnsfp_output_file):
                varity_genes.loc[varity_genes['p_vid'] == p_vid, 'dbnsfp_available'] = 1       
            if os.path.isfile(varity_snv_file):
                varity_genes.loc[varity_genes['p_vid'] == p_vid, 'varity_snv_available'] = 1        
            if os.path.isfile(varity_snv_prediction_file):
                varity_genes.loc[varity_genes['p_vid'] == p_vid, 'varity_snv_prediction_available'] = 1    
                                                                                                                              
        cols = ['p_vid','hgnc_id','symbol','chr','pisa_available','psipred_available','pfam_available','dbnsfp_available','varity_snv_available','varity_snv_prediction_available']            
        varity_genes[cols].to_csv(runtime['batch_uniprotid_file'].split('.')[0] + '_status.csv',index = False)         
            
    def update_varity_data(self,runtime):    
        cur_log = self.project_path + 'output/log/' + 'update_varity_data_'+ runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/' + 'update_varity_data_' + runtime['varity_batch_id'] + '_done.log'
        uniprot_ids = runtime['uniprot_ids']
        for uniprot_id in uniprot_ids:     
            varity_snv_file = self.db_path + 'varity/bygene/' + uniprot_id + '_varity_snv.csv' 
            if os.path.isfile(varity_snv_file):  
                          
                cur_snv_df = pd.read_csv(varity_snv_file)                
                if runtime['update_vairty_data_type'] == 'mave_label_confidence':
                    cur_snv_df['mave_label_confidence'] = np.abs(cur_snv_df['mave_label'] - cur_snv_df['mave_score'])
                    
                if runtime['update_vairty_data_type'] == 'psipred':
                    #### remove psipred columns first
                    psipred_cols = ['aa_psipred','aa_psipred_E','aa_psipred_H','aa_psipred_C']
                    cur_snv_df = cur_snv_df.drop(columns = psipred_cols)
                    
                    psipred_df = None
                    psipred_gene_file =self.db_path + 'psipred/bygene/' + uniprot_id + '.ss2' 
                    if os.path.isfile(psipred_gene_file):
                        cur_psipred_df = pd.read_csv(psipred_gene_file, skiprows=[0, 1], header=None, sep='\s+')
                        cur_psipred_df = cur_psipred_df.loc[:, [0, 1, 2]]
                        cur_psipred_df.columns = ['aa_pos', 'aa_ref', 'aa_psipred']
                        cur_psipred_df = cur_psipred_df.drop_duplicates()    
                        cur_psipred_df['p_vid'] = uniprot_id                      
                        psipred_lst = ['E','H','C']
                        for ss in psipred_lst:
                            cur_psipred_df['aa_psipred' + '_' + ss] = cur_psipred_df['aa_psipred'].apply(lambda x: int(x == ss))
                        
                        if psipred_df is None:
                            psipred_df = cur_psipred_df
                        else:
                            psipred_df = pd.concat([psipred_df,cur_psipred_df])

                        #### merge psipred                                    
                        if psipred_df is not None:
                            psipred_df = psipred_df.reset_index(drop = True)
                            cur_snv_df = pd.merge(cur_snv_df, psipred_df, how='left')         
                            alm_fun.show_msg(cur_log,self.verbose,"merge with psipred: " + str(cur_snv_df.shape))
                        else:
                            alm_fun.show_msg(cur_log,self.verbose,"psipred info is not available......")   

                cur_snv_df.to_csv(varity_snv_file,index = False)
                alm_fun.show_msg(cur_log,self.verbose, uniprot_id + ' updated......')
            else:
                alm_fun.show_msg(cur_log,self.verbose, uniprot_id + ' no snv file found......')
        alm_fun.show_msg(cur_log,self.verbose, 'update_varity_data ' + runtime['varity_batch_id'] + ' is done.')
        alm_fun.show_msg(cur_done_log,self.verbose, 'update_varity_data ' + runtime['varity_batch_id'] + ' is done.')              
            
    def update_varity_labels(self,input_data,cur_log):                 

        ####***************************************************************************************************************************************************************
        # Decide the labels for different type of data (MAVE, ClinVAR, HumsaVAR, gnomAD)
        ####***************************************************************************************************************************************************************            

        
        ####***************************************************************************************************************************************************************
        # Decide the final labels ( if there are overlap in terms of label from different sources, use clinvar label)
        ####***************************************************************************************************************************************************************
        input_data.loc[input_data['clinvar_source'].isnull(),'clinvar_source'] = 0
        input_data.loc[input_data['hgmd_source'].isnull(),'hgmd_source'] = 0
        input_data.loc[input_data['humsavar_source'].isnull(),'humsavar_source'] = 0
        input_data.loc[input_data['gnomad_source'].isnull(),'gnomad_source'] = 0
        input_data.loc[input_data['mave_source'].isnull(),'mave_source'] = 0
     
        input_data['train_clinvar_source'] = input_data['clinvar_source']
        input_data.loc[~input_data['clinvar_label'].isin([0,1]) ,'train_clinvar_source'] = 0
        input_data['train_hgmd_source'] = input_data['hgmd_source']
        input_data.loc[~input_data['hgmd_label'].isin([0,1]) ,'train_hgmd_source'] = 0
        input_data['train_humsavar_source'] = input_data['humsavar_source']
        input_data.loc[~input_data['humsavar_label'].isin([0,1]) ,'train_humsavar_source'] = 0
        input_data['train_gnomad_source'] = input_data['gnomad_source']
        input_data.loc[~input_data['gnomad_label'].isin([0,1]) ,'train_gnomad_source'] = 0
        input_data['train_mave_source'] = input_data['mave_source']
        input_data.loc[~input_data['mave_label'].isin([0,1]) ,'train_mave_source'] = 0
         
        # when there is a ClinVAR label use it as the final label       
        input_data.loc[input_data['train_clinvar_source'] == 1 ,'train_hgmd_source'] = 0  
        input_data.loc[input_data['train_clinvar_source'] == 1 ,'train_humsavar_source'] = 0       
        input_data.loc[input_data['train_clinvar_source'] == 1 ,'train_mave_source'] = 0
        input_data.loc[input_data['train_clinvar_source'] == 1 ,'train_gnomad_source'] = 0
         
        # after Clinvar when there is a HGMD label use it as the final label                
        input_data.loc[input_data['train_hgmd_source'] == 1 ,'train_humsavar_source'] = 0  
        input_data.loc[input_data['train_hgmd_source'] == 1 ,'train_mave_source'] = 0    
        input_data.loc[input_data['train_hgmd_source'] == 1 ,'train_gnomad_source'] = 0  
 
        # after HGMD, when there is a HumsaVAR label use it as final label
        input_data.loc[input_data['train_humsavar_source'] == 1 ,'train_mave_source'] = 0
        input_data.loc[input_data['train_humsavar_source'] == 1 ,'train_gnomad_source'] = 0    
 
        # after humsaVAR, when there is a MAVE label use it as final label
        input_data.loc[input_data['train_mave_source'] == 1 ,'train_gnomad_source'] = 0
     
        # decide the final label
        input_data['label'] = -1
        input_data.loc[input_data['train_clinvar_source'] == 1 ,'label'] = input_data.loc[input_data['train_clinvar_source'] == 1 ,'clinvar_label']
        input_data.loc[input_data['train_hgmd_source'] == 1 ,'label'] = input_data.loc[input_data['train_hgmd_source'] == 1 ,'hgmd_label']
        input_data.loc[input_data['train_humsavar_source'] == 1 ,'label'] = input_data.loc[input_data['train_humsavar_source'] == 1 ,'humsavar_label']
        input_data.loc[input_data['train_mave_source'] == 1 ,'label'] = input_data.loc[input_data['train_mave_source'] == 1 ,'mave_label']
        input_data.loc[input_data['train_gnomad_source'] == 1 ,'label'] = input_data.loc[input_data['train_gnomad_source'] == 1 ,'gnomad_label']

        return(input_data)
            
    def create_denovodb_data(self,runtime):   
        self.init_humamdb_object("denovodb")
          
        cur_log = self.db_path + 'denovodb/log/create_denovodb_data.log'
                     
        hgnc2id_dict = np.load(self.db_path + 'hgnc/npy/hgnc2id_dict.npy').item()
        id2hgnc_dict = np.load(self.db_path + 'hgnc/npy/id2hgnc_dict.npy').item()
        
        varity_gene_dict = np.load(self.db_path + 'varity/npy/varity_gene_dict.npy').item()
        
        
        ####coe et al enriched genes
        coe_enriched_df = pd.read_csv(runtime['db_path'] + 'denovodb/org/denovodb_coe_genes_org.csv')            
        coe_enriched_df['ch_lgd_valid_q'] = 1
        coe_enriched_df.loc[coe_enriched_df['ch_lgd_q'].isnull(),'ch_lgd_q'] = 1
        coe_enriched_df.loc[coe_enriched_df['lgd_count'] > 1,'ch_lgd_valid_q'] = coe_enriched_df.loc[coe_enriched_df['lgd_count'] > 1,'ch_lgd_q']
        coe_enriched_df['dr_lgd_valid_q'] = 1
        coe_enriched_df.loc[coe_enriched_df['dr_lgd_q'].isnull(),'dr_lgd_q'] = 1
        coe_enriched_df.loc[coe_enriched_df['lgd_count'] > 1,'dr_lgd_valid_q'] = coe_enriched_df.loc[coe_enriched_df['lgd_count'] > 1,'dr_lgd_q']
        coe_enriched_df['ch_missense_valid_q'] = 1
        coe_enriched_df.loc[coe_enriched_df['ch_missense_q'].isnull(),'ch_missense_q'] = 1
        coe_enriched_df.loc[coe_enriched_df['missense_count'] > 1,'ch_missense_valid_q'] = coe_enriched_df.loc[coe_enriched_df['missense_count'] > 1,'ch_missense_q']
        coe_enriched_df['dr_missense_valid_q'] = 1
        coe_enriched_df.loc[coe_enriched_df['dr_missense_q'].isnull(),'dr_missense_q'] = 1        
        coe_enriched_df.loc[coe_enriched_df['missense_count'] > 1,'dr_missense_valid_q'] = coe_enriched_df.loc[coe_enriched_df['missense_count'] > 1,'dr_missense_q']        
        coe_enriched_df['ch_cadd_valid_q'] = 1
        coe_enriched_df.loc[coe_enriched_df['ch_cadd_q'].isnull(),'ch_cadd_q'] = 1           
        coe_enriched_df.loc[coe_enriched_df['missense_count'] > 1,'ch_cadd_valid_q'] = coe_enriched_df.loc[coe_enriched_df['missense_count'] > 1,'ch_cadd_q']
                 
        coe_enriched_df['fdr_cutoff'] = coe_enriched_df.apply(lambda x: np.min([x['ch_lgd_valid_q'],x['dr_lgd_valid_q'],x['ch_missense_valid_q'],x['dr_missense_valid_q'],x['ch_cadd_valid_q']]),axis = 1)            
        coe_enriched_df.to_csv(runtime['db_path'] + 'denovodb/all/denovodb_coe_genes.csv', index = False)
        

        #####denovodb SCC and non-SCC
        denovodb_scc_df = pd.read_csv(self.db_path + 'denovodb/org/denovo-db.ssc-samples.variants.tsv',sep = '\t',skiprows = 1 )
        denovodb_non_scc_df = pd.read_csv(self.db_path + 'denovodb/org/denovo-db.non-ssc-samples.variants.tsv',sep = '\t',skiprows = 1)
        denovodb_scc_df['scc'] = 1
        denovodb_non_scc_df['scc'] = 0         
        denovodb_df = pd.concat([denovodb_scc_df,denovodb_non_scc_df]) 
        alm_fun.show_msg(cur_log,self.verbose,  '# of denovodb missense variants: ' +  str(denovodb_df.shape[0]) )
        
        #####denovodb missesne        
        denovodb_missense_df =denovodb_df.loc[(denovodb_df['FunctionClass'] == 'missense') & denovodb_df['cDnaVariant'].notnull(),:]
        alm_fun.show_msg(cur_log,self.verbose,  '# of denovodb missense variants: ' +  str(denovodb_missense_df.shape[0]) )                
          
        denovodb_missense_df['hgnc_id'] = denovodb_missense_df.apply(lambda x: id2hgnc_dict['symbol'].get(x['Gene'],np.nan),axis = 1)
        denovodb_missense_df = denovodb_missense_df.loc[denovodb_missense_df['hgnc_id'].notnull(),:]
        alm_fun.show_msg(cur_log,self.verbose,  '# of denovodb missense variants with valid hgnc_id: ' +  str(denovodb_missense_df.shape[0]) )
        
        denovodb_missense_df['p_vid'] = denovodb_missense_df.apply(lambda x: varity_gene_dict['hgnc2uniprot'].get(x['hgnc_id'],np.nan),axis = 1)
        denovodb_missense_df = denovodb_missense_df.loc[denovodb_missense_df['p_vid'].notnull(),:] 
        alm_fun.show_msg(cur_log,self.verbose,  '# of denovodb missense variants with valid reviewed uniprot_id: ' +  str(denovodb_missense_df.shape[0]) )
        
        denovodb_missense_df['denovo_label'] = 1
        denovodb_missense_df.loc[denovodb_missense_df['PrimaryPhenotype'] == 'control','denovo_label'] = 0
        denovodb_missense_df['chr'] = denovodb_missense_df['Chr'].astype(str)
        denovodb_missense_df['nt_pos'] = denovodb_missense_df['Position'].astype(int)
        denovodb_missense_df['nt_ref'] = denovodb_missense_df['Variant'].str[0]
        denovodb_missense_df['nt_alt'] = denovodb_missense_df['Variant'].str[-1]
        denovodb_missense_df = denovodb_missense_df[['StudyName','PubmedID','PrimaryPhenotype','rsID','scc','Validation','Gene','chr','nt_pos','nt_ref','nt_alt','p_vid','hgnc_id','denovo_label']]
        denovodb_missense_df['hg19_coord'] = denovodb_missense_df.apply(lambda x: 'chr' + x['chr'] + ':' + str(x['nt_pos']),axis = 1)
        denovodb_missense_df = denovodb_missense_df.drop_duplicates()
        
    
        denovodb_missense_group_df = denovodb_missense_df.groupby(['chr','nt_pos','nt_ref','nt_alt'])['StudyName','PubmedID','PrimaryPhenotype','rsID','scc','Validation','Gene','p_vid','hgnc_id','denovo_label'].agg(set).reset_index()        
        #remove inconsistent labels
        denovodb_missense_group_df = denovodb_missense_group_df.loc[(denovodb_missense_group_df['denovo_label']== {1}) | (denovodb_missense_group_df['denovo_label']== {0}),:]
        #remove same variant that has multiple p_vids 
        denovodb_missense_group_df['p_vid_count'] = denovodb_missense_group_df['p_vid'].apply(lambda x: len(x))
        denovodb_missense_group_df = denovodb_missense_group_df.loc[denovodb_missense_group_df['p_vid_count']==1 ,:]
        
        denovodb_missense_group_df['denovo_label'] = denovodb_missense_group_df['denovo_label'].apply(lambda x: list(x)[0])
        denovodb_missense_group_df['p_vid'] = denovodb_missense_group_df['p_vid'].apply(lambda x: list(x)[0])
        denovodb_missense_group_df['symbol'] = denovodb_missense_group_df['Gene'].apply(lambda x: list(x)[0])
        denovodb_missense_group_df['hgnc_id'] = denovodb_missense_group_df['hgnc_id'].apply(lambda x: list(x)[0])
        

#         #### denovodb enriched snvs (genes that enriched in missense variants, coe et al 2019)        
#         denovodb_enriched_gene_df = pd.read_csv(self.db_path + 'denovodb/all/denovodb_enriched_genes.csv')
#         denovodb_enriched_gene_df = denovodb_enriched_gene_df.drop(columns = ['PubMed'])
#         denovodb_enriched_genes = list(denovodb_enriched_gene_df['Gene'].unique())
#         denovodb_enriched_df = denovodb_missense_group_df.loc[denovodb_missense_group_df['Gene'].isin(denovodb_enriched_genes),:]        
#         denovodb_enriched_df = pd.merge(denovodb_enriched_df,denovodb_enriched_gene_df)
#         
        #### denovodb enriched snvs (genes that enriched in missense variants, coe et al 2019)  
        fdr_cutoff = 0.5   
        denovodb_coe_gene_df = pd.read_csv(self.db_path + 'denovodb/all/denovodb_coe_genes.csv')
        denovodb_coe_gene_df = denovodb_coe_gene_df.loc[denovodb_coe_gene_df['fdr_cutoff'] <= fdr_cutoff, ['symbol','fdr_cutoff']]        
        denovodb_coe_genes = list(denovodb_coe_gene_df['symbol'].unique())
        denovodb_enriched_df = denovodb_missense_group_df.loc[denovodb_missense_group_df['symbol'].isin(denovodb_coe_genes),:]        
        denovodb_enriched_df = pd.merge(denovodb_enriched_df,denovodb_coe_gene_df)
        
        #P62805 has two symbol associated with it, causing duplicate recrods 
        denovodb_enriched_df = denovodb_enriched_df.loc[denovodb_enriched_df['p_vid'] != 'P62805',:]
        
        denovodb_enriched_genes_df = denovodb_enriched_df[['p_vid','symbol','hgnc_id','chr','fdr_cutoff']]
        denovodb_enriched_genes_df = denovodb_enriched_genes_df.drop_duplicates()
        denovodb_enriched_genes_df.columns = ['p_vid','symbol','hgnc_id','chr','fdr_cutoff']
        print (str(denovodb_enriched_genes_df.shape))
        denovodb_enriched_genes_df.to_csv(self.db_path + 'denovodb/all/denovodb_enriched_genes.csv',index = False)
         
        denovodb_enriched_df.to_csv(self.db_path + 'denovodb/all/denovodb_enriched_snv.csv',index = False)
        alm_fun.show_msg(cur_log,self.verbose,  '# of denovodb missense variants in genes that enriched in missense variants: ' +  str(denovodb_enriched_df.shape[0]) ) 
        
#         #### denovodb snvs in VARITY disease genes 
#         varity_disease_genes_df = pd.read_csv(self.db_path + 'varity/all/varity_disease_genes.csv')
#         varity_disease_genes = list(varity_disease_genes_df['p_vid'].unique())
#         denovodb_disease_df = denovodb_missense_group_df.loc[denovodb_missense_group_df['p_vid'].isin(varity_disease_genes),:]         
#         denovodb_disease_df.to_csv(self.db_path + 'denovodb/csv/denovodb_disease_snv.csv',index = False)
#         alm_fun.show_msg(cur_log,self.verbose,  '# of denovodb missense variants in varity disease genes: ' +  str(denovodb_disease_df.shape[0]) )                


        alm_fun.show_msg(self.log,self.verbose,'denovodb data created.')


        #### denovodb pg snvs (genes that have both negative and positive variants) 
#         positivie_genes = set(denovodb_missense_group_df.loc[denovodb_missense_group_df['denovo_label'] == 1,'p_vid'].unique())
#         negative_genes = set(denovodb_missense_group_df.loc[denovodb_missense_group_df['denovo_label'] == 0,'p_vid'].unique())        
#         pn_genes = positivie_genes.intersection(negative_genes)
# 
#         denovodb_pg_genes_df = denovodb_missense_group_df.loc[denovodb_missense_group_df['p_vid'].isin(pn_genes),['p_vid','Gene','hgnc_id','chr']]
#         denovodb_pg_genes_df = denovodb_pg_genes_df.drop_duplicates()
#         denovodb_pg_genes_df.columns = ['p_vid','symbol','hgnc_id','chr']
#         denovodb_pg_genes_df.to_csv(self.db_path + 'denovodb/all/denovodb_pn_genes.csv')
#         
#         denovodb_missense_pg_df = denovodb_missense_group_df.loc[denovodb_missense_group_df['p_vid'].isin(pn_genes),:]
#         denovodb_missense_pg_df.to_csv(self.db_path + 'denovodb/csv/denovodb_pn_snv.csv')
#         alm_fun.show_msg(cur_log,self.verbose,  '# of denovodb missense variants in genes that have both negative and positive variants: ' +  str(denovodb_missense_pg_df.shape[0]) ) 
#         
    def create_mpc_data(self):
        #ftp://ftp.broadinstitute.org/pub/ExAC_release/release1/regional_missense_constraint/
        self.init_humamdb_object('mpc')
        mpc_df =  pd.read_csv(self.db_path + 'mpc/org/fordist_constraint_official_mpc_values_v2.txt',sep = '\t')
        mpc_df = mpc_df[['chrom','pos','ref','alt','obs_exp','mis_badness','fitted_score','MPC']]
        mpc_df.columns = ['chr','nt_pos','nt_ref','nt_alt','mpc_obs_exp','mpc_mis_badness','mpc_fitted_score','mpc_score']
        new_mpc_df = mpc_df.groupby(['chr','nt_pos','nt_ref','nt_alt'])['mpc_obs_exp','mpc_mis_badness','mpc_fitted_score','mpc_score'].agg('mean').reset_index()
        print ("Before and after avg the duplicated scores : " + str(mpc_df.shape[0]) + ',' + str(new_mpc_df.shape[0])) 
        new_mpc_df.to_csv(self.db_path + 'mpc/all/mpc_values_v2_avg_duplicated_scores.csv',index = False)
                              
    def create_deepsequence_data(self,runtime):  
        self.init_humamdb_object("deepsequence")
        cur_log = self.db_path + 'deepsequence/log/deepsequence.log'
        alm_fun.show_msg(cur_log,self.verbose,'deepsequence data created.')  
        
        #### data from paper (PMID: 32627955) Livesey Y Marsh 2020 "Using deep mutational scanning to benchmark variant effect predictors and identify disease mutations"        
        data_file = self.db_path + 'deepsequence/org/dms_with_deepsequence.xls'
        varity_all_genes = pd.read_csv(self.db_path + 'varity/all/varity_all_genes.csv')
        human_genes = ['ADRB2','BRCA1','HRAS','MAPK1','TP53','PTEN','SUMO1','TPK1','TPMT','UBE2I']
        
        all_deepsequence_df = None
        for cur_gene in human_genes:                    
            cur_gene_df = pd.read_csv(self.db_path + 'deepsequence/org/' + cur_gene + '.csv')
            cur_uniprot_id = varity_all_genes.loc[varity_all_genes['symbol'] == cur_gene,'uniprot_id'].values[0]            
            cur_gene_df['p_vid'] = cur_uniprot_id
            cur_gene_df['aa_ref'] = cur_gene_df['variant'].str[0]
            cur_gene_df['aa_pos'] = cur_gene_df['variant'].str[1:-1]
            cur_gene_df['aa_pos'] = cur_gene_df['aa_pos'].astype(int)
            cur_gene_df['aa_alt'] = cur_gene_df['variant'].str[-1]            
            #### save to deepsequence bygene
            cur_deepsequence_df = cur_gene_df[['p_vid','aa_pos','aa_ref','aa_alt','DeepSequence']]
            cur_deepsequence_df.columns = ['p_vid','aa_pos','aa_ref','aa_alt','deepsequence_score']
            cur_deepsequence_df.to_csv(self.db_path + 'deepsequence/bygene/' + cur_uniprot_id + '_deepsequence.csv',index = False)
            #### save to MAVE org fold
            if cur_gene in ['ADRB2','HRAS','MAPK1','TP53']:
                cur_gene_df[['p_vid','aa_pos','aa_ref','aa_alt'] + [x for x in cur_gene_df.columns if 'DMS' in x]].to_csv(self.db_path + 'mave/org/' + cur_uniprot_id + '_' + cur_gene + '_mave_org.csv', index = False)
                
            if all_deepsequence_df is None:
                all_deepsequence_df = cur_deepsequence_df
            else:
                all_deepsequence_df = pd.concat([all_deepsequence_df,cur_deepsequence_df])

        all_deepsequence_df.to_csv(self.db_path + 'deepsequence/all/all_deepsequence_scores.csv',index = False)
        
    def create_varity_training_data(self):

        if merge_mave == 1:
            #************************************************************************************        
            # Adding MAVE data (part of the MAVE data are lost after dbnsfp filtering
            #************************************************************************************
            mave_missense = pd.read_csv(self.db_path + 'mave/all/mave_missense.csv')  
            input_data = input_data.merge(mave_missense,'outer')        
            alm_fun.show_msg(cur_log,self.verbose,'Varity records after merging with MAVE : ' + str(input_data.shape[0]))
            
            input_data.loc[input_data['SIFT_selected_score'].isnull(),'SIFT_selected_score'] = input_data.loc[input_data['SIFT_selected_score'].isnull(),'sift_score']
            input_data.loc[input_data['PROVEAN_selected_score'].isnull(),'PROVEAN_selected_score'] = input_data.loc[input_data['PROVEAN_selected_score'].isnull(),'provean_score']
        else:
            #************************************************************************************        
            # Adding MAVE data (part of the MAVE data are lost after dbnsfp filtering)
            #************************************************************************************
            mave_missense = pd.read_csv(self.db_path + 'mave/all/mave_missense.csv')  
            input_data = pd.merge(input_data,mave_missense,how = 'left')       
        ####***************************************************************************************************************************************************************
        # Check the label conflicts
        ####***************************************************************************************************************************************************************
#         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] == input_data_final['clinvar_label']),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
#         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] == input_data_final['clinvar_label']) & (input_data_final['humsavar_label'] != -1) * (input_data_final['clinvar_label'] != -1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
#         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] != input_data_final['clinvar_label']),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
#         
#         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  0 ) & (input_data_final['clinvar_label'] == 1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']]
#         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  1 ) & (input_data_final['clinvar_label'] == 0),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
#         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  -1 ) & (input_data_final['clinvar_label'] == 0),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
#         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  -1 ) & (input_data_final['clinvar_label'] == 1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
#         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  0 ) & (input_data_final['clinvar_label'] == -1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
#         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  1 ) & (input_data_final['clinvar_label'] == -1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
     
#         input_data_final.loc[(input_data_final['mave_label'] == 1) & (input_data_final['clinvar_label'] == 0),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level']].shape
#         input_data_final.loc[(input_data_final['mave_label'] == 0) & (input_data_final['clinvar_label'] == 1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level']].shape
#         input_data_final.loc[(input_data_final['mave_label'] == 1) & (input_data_final['gnomad_label'] == 0),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level']].shape
#         input_data_final.loc[(input_data_final['clinvar_label'] == 1) & (input_data_final['gnomad_label'] == 0),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level']].shape
         
           
        print ("OK")
        
    def combine_extra_revision_varity_data(self,runtime):
        cur_log = self.db_path + 'varity/log/' + 'combine_extra_revision_varity_data.log'
        varity_disease_genes_df = pd.read_csv(self.db_path + 'varity/all/varity_disease_genes.csv')
        varity_revision_genes = list(varity_disease_genes_df.loc[varity_disease_genes_df['revision'] == 1,'uniprot_id'])   
        runtime['uniprot_ids'] = varity_revision_genes
        combined_varity_df = None
        for uniprot_id in runtime['uniprot_ids']:
            alm_fun.show_msg(cur_log,self.verbose,'Combining ' + uniprot_id +'.......')
            cur_csv_file  = self.db_path + "varity/bygene/varity_snv_" + uniprot_id + ".csv"
            if os.path.isfile(cur_csv_file):
                cur_csv_df = pd.read_csv(self.db_path + "varity/bygene/varity_snv_" + uniprot_id + ".csv")
                if combined_varity_df is None:
                    combined_varity_df = cur_csv_df
                else:
                    combined_varity_df = pd.concat([combined_varity_df,cur_csv_df])                
        combined_varity_df.to_csv(self.db_path + "varity/all/varity_revision_data.csv")
    
    def create_aa_properties_data(self):
        self.init_humamdb_object("aa_properties")
        cur_log = self.db_path + 'aa_properties/log/aa_properties.log'        
        if not os.path.isfile(self.db_path + 'aa_properties/org/aa.txt'):       
            wget_cmd = "wget https://www.dropbox.com/s/wcgb9ztue97miwz/aa.txt"
            subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'aa_properties/org/')
            alm_fun.show_msg(cur_log,self.verbose,'aa properties data downloaded.\n')
        else:
            alm_fun.show_msg(cur_log,self.verbose,'aa_properties data exists already.')
            
    def load_aa_properties(self):
        aa_properties = pd.read_csv(self.db_path + 'aa_properties/org/aa.txt', sep='\t')
        aa_properties.drop_duplicates(inplace=True)
        aa_properties.drop(['aa_name'], axis=1, inplace=True)
        # aa_properties_features = ['aa','mw','pka','pkb','pi', 'cyclic','charged','charge','hydropathy_index','hydrophobic','polar','ionizable','aromatic','aliphatic','hbond','sulfur','pbr','avbr','vadw','asa']
        aa_properties_features = ['aa', 'mw', 'pka', 'pkb', 'pi', 'hi', 'pbr', 'avbr', 'vadw', 'asa', 'pbr_10', 'avbr_100', 'vadw_100', 'asa_100', 'cyclic', 'charge', 'positive', 'negative', 'hydrophobic', 'polar', 'ionizable', 'aromatic', 'aliphatic', 'hbond', 'sulfur', 'essential', 'size']
        aa_properties.columns = aa_properties_features    
        return (aa_properties)
    
    def create_codon_usage_data(self):
        self.init_humamdb_object("codon_usage")
        cur_log = self.db_path + 'codon_usage/log/codon_usage.log'
        alm_fun.show_msg(cur_log,self.verbose,'codon_usage data created.')
                
    def load_condon_dicts(self):
        codon_usage_df = pd.read_csv(self.db_path + 'codon_usage/org/codon_usage_human.txt', sep='\t', header=None)
        codon_usage_df.columns = ['codon_r', 'codon', 'aa', 'freq_aa', 'freq_all', 'count']
        codon_usage_df.loc[codon_usage_df['aa'] == '_', 'aa'] = '*'
        
        dict_aa_codon = {}
        dict_codon = {}
        dict_codon_freq_aa = {}
        dict_codon_freq_all = {}
        for i in range(codon_usage_df.shape[0]):
            cur_aa = codon_usage_df.loc[i, 'aa']
            cur_codon = codon_usage_df.loc[i, 'codon']
            
            if dict_aa_codon.get(cur_aa, '') == '':
                dict_aa_codon[cur_aa] = [cur_codon]
            else:
                dict_aa_codon[cur_aa] = dict_aa_codon[cur_aa] + [cur_codon]
            dict_codon[cur_codon] = cur_aa
            dict_codon_freq_aa[cur_codon] = codon_usage_df.loc[i, 'freq_aa']
            dict_codon_freq_all[cur_codon] = codon_usage_df.loc[i, 'freq_all']
            
        return [dict_aa_codon,dict_codon,dict_codon_freq_aa,dict_codon_freq_all]
    
    def create_blosum_data(self):
        self.init_humamdb_object("blosum")
        alm_fun.show_msg(self.log,self.verbose,'blosum data created.')  
    
    def create_dbnsfp_data(self):
        self.init_humamdb_object('dbnsfp',sub = 0) 
        cur_log = self.db_path + 'dbnsfp/dbnsfp.log'
        
        if not os.path.isfile(self.db_path + 'dbnsfp/dbnsfp.zip'):        
            wget_cmd = "git clone https://github.com/chentinghao/download_google_drive.git"
            subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'dbnsfp/')
            alm_fun.show_msg(cur_log,self.verbose,'download_google_drive downloaded.\n')
            
            download_cmd = "python download_gdrive.py 1XQI2m_403yq-TLxJ1QHtkzKE7_c_9Gal " + self.db_path + 'dbnsfp/dbnsfp.zip'
            subprocess.run(download_cmd.split(" "), cwd = self.db_path + 'dbnsfp/')
            
            unzip_cmd = "unzip " + self.db_path + "dbnsfp/dbnsfp.zip"
            subprocess.run(unzip_cmd.split(" "), cwd = self.db_path + 'dbnsfp/')
        
        alm_fun.show_msg(cur_log,self.verbose,'dbnsfp data created.')
            
    def create_accsum_data(self ):  
        self.init_humamdb_object("accsum")
        [dict_aa_codon,dict_codon,dict_codon_freq_aa,dict_codon_freq_all] = self.load_condon_dicts()     
        accsum_df = pd.DataFrame(columns=['aa_ref', 'aa_alt', 'accessibility'])        
        titv_ratio = 2
        for aa_ref in self.lst_aa_21:
            for aa_alt in self.lst_aa_21:
                cur_df = pd.DataFrame.from_records([(aa_ref, aa_alt, self.get_aa_accessibility(aa_ref, aa_alt,dict_aa_codon,dict_codon_freq_all, titv_ratio))], columns=['aa_ref', 'aa_alt', 'accessibility'])
                accsum_df = accsum_df.append(cur_df)
        accsum_df['accessibility_ste'] = 0
        accsum_df['accessibility'] = accsum_df['accessibility']
        output_path = self.db_path + 'accsum/csv/'
        accsum_df.to_csv(output_path + 'accsum.csv', index=False)
        alm_fun.show_msg(self.log,self.verbose,'accsum data created.')  
        
    def get_aa_accessibility(self, aa_ref, aa_alt, dict_aa_codon,dict_codon_freq_all,titv_ratio=1):       
        aa_ref_codons = dict_aa_codon[aa_ref]
        aa_alt_codons = dict_aa_codon[aa_alt]
        access = 0
        
        for ref_codon in aa_ref_codons:
            for alt_codon in aa_alt_codons:
                if alm_fun.hamming_distance(ref_codon, alt_codon) == 1:
                    if alm_fun.get_codon_titv(ref_codon, alt_codon) == 'ti':
#                         access += dict_codon_freq_all[ref_codon] * dict_codon_freq_all[alt_codon]
                        access += dict_codon_freq_all[ref_codon]/9
                    if alm_fun.get_codon_titv(ref_codon, alt_codon) == 'tv':
#                         access += dict_codon_freq_all[ref_codon] * dict_codon_freq_all[alt_codon] / titv_ratio
                        access += dict_codon_freq_all[ref_codon]/9
#         if access == 0 :
#             access = np.nan
        return (access)
                      
    def make_aasum(self, sum_name, centralities, properties, value_df, value_score_names, aasum_prefix, aasum_folder, quality_cutoff=None, weightedby_columns=[''], weightedby_columns_inverse=[0]):
        
        def cal_weighted_average(value_df, groupby, value_score, weighted_by, inverse):

            def single_group_by(x):
                groupby_cols = list(x[0])
                groupby_df = x[1]
                groupby_df = groupby_df.loc[groupby_df[value_score].notnull() & groupby_df[weighted_by].notnull()]
                
                weighted_by_cols = groupby_df[weighted_by]
                value_cols = groupby_df[value_score]
                if inverse == 1:
                    weighted_by_cols = 1 / weighted_by_cols
                weighted_by_cols = weighted_by_cols / np.sum(weighted_by_cols)
                weighted_mean = np.dot(value_cols, weighted_by_cols)
                cur_row = groupby_cols + [weighted_mean]
                return cur_row            
            
            if groupby != None:
                group_obj = value_df.groupby(groupby)   
                group_list = list(group_obj)     
                return_list = [single_group_by(x) for x in group_list]       
                return_obj = pd.DataFrame(return_list, columns=groupby + [value_score])
            else:
                weighted_by_cols = value_df[weighted_by]
                value_cols = value_df[value_score]
                if inverse == 1:
                    weighted_by_cols = 1 / weighted_by_cols
                weighted_by_cols = weighted_by_cols / np.sum(weighted_by_cols)
                weighted_mean = np.dot(value_cols, weighted_by_cols)
                return_obj = weighted_mean
            return (return_obj)
        
        def make_aasum_sub(centrality, value_df, value_score, aasum_groupby, aasum_name, aasum_folder, weighted_by, weighted_by_inverse):                         
            if centrality == 'mean':    
                if weighted_by == '':
                    aasum_value = value_df.groupby(aasum_groupby)[value_score].mean().reset_index()
                else:
                    aasum_value = cal_weighted_average(value_df, aasum_groupby, value_score, weighted_by, weighted_by_inverse)           
            if centrality == 'median':
                aasum_value = value_df.groupby(aasum_groupby)[value_score].median().reset_index()
            if centrality == 'normalization':                    
                aasum_value = value_df.groupby(aasum_groupby)[value_score].sum().reset_index()
                total_value = aasum_value[value_score].sum()                                        
                aasum_value[value_score] = aasum_value[value_score] / total_value
            if centrality == 'logodds':            
                if weighted_by == '':
                    aasum_value = value_df.groupby(aasum_groupby)[value_score].mean().reset_index()
                    total_mean_value = aasum_value[value_score].mean()                                        
                    aasum_value[value_score] = -np.log2((aasum_value[value_score] / total_mean_value))
                    # aasum_value[value_score] = aasum_value[value_score] / total_mean_value
                else:
                    aasum_value = cal_weighted_average(value_df, aasum_groupby, value_score, weighted_by, weighted_by_inverse)
                    total_mean_value = cal_weighted_average[value_df, None, value_score, weighted_by, weighted_by_inverse]                                        
                    aasum_value[value_score] = -np.log2((aasum_value[value_score] / total_mean_value))                    
                    # aasum_value[value_score] = aasum_value[value_score] / total_mean_value
                
            aasum_value.rename(columns={value_score: aasum_name}, inplace=True)
            aasum_std = value_df.groupby(aasum_groupby)[value_score].std().reset_index()
            aasum_std.rename(columns={value_score: aasum_name + '_std'}, inplace=True)
            aasum_count = value_df.groupby(aasum_groupby)[value_score].count().reset_index()
            aasum_count.rename(columns={value_score: aasum_name + '_count'}, inplace=True)
            
            aasum = pd.merge(aasum_value, aasum_std, how='left')
            aasum = pd.merge(aasum, aasum_count, how='left')
            aasum[aasum_name + '_ste'] = aasum[aasum_name + '_std'] / np.sqrt(aasum[aasum_name + '_count'])            
            return(aasum)
        
        pass
        aasum_groupby = ['aa_ref', 'aa_alt']
        for value_score in value_score_names:
            cur_value_df = value_df.copy()
            if quality_cutoff != None:
                cur_value_df = cur_value_df.loc[cur_value_df[value_score].notnull() & (cur_value_df['quality_score'] > quality_cutoff), :]
            else:
                cur_value_df = cur_value_df.loc[cur_value_df[value_score].notnull() , :]
            for centrality in centralities:
                for i in range(len(weightedby_columns)):
                    weighted_by = weightedby_columns[i]
                    weighted_by_inverse = weightedby_columns_inverse[i]
                    if weighted_by == '':  
                        aasum_name = aasum_prefix + value_score + '_' + centrality
                    else:
                        aasum_name = aasum_prefix + value_score + '_' + centrality + '_' + weighted_by 
                    aasum = make_aasum_sub(centrality, cur_value_df, value_score, aasum_groupby, aasum_name, aasum_folder, weighted_by, weighted_by_inverse)
                    aasum.to_csv(aasum_folder + '/' + aasum_name + '.csv', index=False)
                    for property in properties:
                        lst_property = property.split(',')
                        property_aasum_groupby = aasum_groupby + lst_property
                        property_aasum_name = aasum_name + '_' + property.replace(',', '_')                         
                        # create funsum for each property value
                        k = 0;
                        for property_groupby in list(cur_value_df.groupby(lst_property)):
                            k += 1;
                            cur_property_value = str(property_groupby[0])
                            cur_property_value_df = property_groupby[1]
                            cur_property_aasum_name = property_aasum_name + '_' + str(cur_property_value)   
                            cur_property_aasum_df = make_aasum_sub(centrality, cur_property_value_df, value_score, property_aasum_groupby, cur_property_aasum_name, aasum_folder, weighted_by, weighted_by_inverse)                            
                            cur_property_aasum_df.columns = aasum_groupby + lst_property + [property_aasum_name, property_aasum_name + '_std', property_aasum_name + '_count', property_aasum_name + '_ste']
                            
                            if k == 1:
                                all_property_aasum_df = cur_property_aasum_df
                            else:
                                all_property_aasum_df = pd.concat([all_property_aasum_df, cur_property_aasum_df], axis=0)
                        all_property_aasum_df.to_csv(aasum_folder + '/' + property_aasum_name + '.csv', index=False)        
    
    def make_funsums(self, funsum_genes, funsum_scores,funsum_centralities,funsum_properties,funsum_dmsfiles,funsum_weightedby_columns,funsum_weightedby_columns_inverse,quality_cutoff, pos_importance_type, pos_importance_value = 1, output_path = None):
        cur_log = self.db_path + 'funsum/log/funsum.log'
        if output_path is None:
            output_path = self.db_path + 'funsum/csv/'
        sum_name = 'funsum'
        aasum_prefix = 'funsum_'
        aasum_folder = output_path
        value_df = pd.read_csv(funsum_dmsfiles)
        alm_fun.show_msg(cur_log,self.verbose,"# of total variant effect records: " + str(value_df.shape[0]))        
        value_df = value_df.loc[(value_df['p_vid'].isin(funsum_genes)) & (value_df[pos_importance_type] <= pos_importance_value) & (value_df['aa_ref'] != value_df['aa_alt']), :]
        alm_fun.show_msg(cur_log,self.verbose,"# of total variant effect records used for FunSUM: " + str(value_df.shape[0]))            
        self.make_aasum(sum_name, funsum_centralities, funsum_properties, value_df, funsum_scores, aasum_prefix, aasum_folder, quality_cutoff, funsum_weightedby_columns, funsum_weightedby_columns_inverse)
        
    def create_funsum_data(self):
        self.init_humamdb_object('funsum') 
        cur_log = self.db_path + 'funsum/log/funsum.log'
        funsum_genes = ['P63279','P63165','P62166','Q9H3S4','P0DP23','Q9NZ01','P31150','P42898','P35520']     
        funsum_scores = ['fitness']
        funsum_centralities = ['mean']
        funsum_properties = ['in_domain','aa_psipred']
        funsum_dmsfiles = self.db_path + 'mave/all/mave_missense_forfunsum.csv'
        funsum_weightedby_columns = ['']
        funsum_weightedby_columns_inverse = [0]        
        quality_cutoff = 0 
        pos_importance_type = 'pos_importance_median'
        pos_importance_value = 0.8
        self.make_funsums(funsum_genes, funsum_scores,funsum_centralities,funsum_properties,funsum_dmsfiles,funsum_weightedby_columns,funsum_weightedby_columns_inverse,quality_cutoff, pos_importance_type, pos_importance_value)
        alm_fun.show_msg(cur_log,self.verbose,'funsum data created.')
    
    def create_sublocation_data(self):
        self.init_humamdb_object("sublocation")
        cur_log = self.db_path + 'sublocation/log/sublocation.log'        
        if not os.path.isfile(self.db_path + 'sublocation/org/uniprot_sublocation.txt'):       
#             wget_cmd = "wget https://www.dropbox.com/s/wcgb9ztue97miwz/aa.txt"
            subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'sublocation/org/')
            alm_fun.show_msg(cur_log,self.verbose,'aa properties data downloaded.\n')
        else:
            alm_fun.show_msg(cur_log,self.verbose,'sublocation data exists already.')
            
#         sublocation_df = pd.read_csv(self.db_path + 'sublocation/org/uniprot_sublocation.txt',sep = '\t')   
        print("OK")
        sublocation_org = open(self.db_path + 'sublocation/org/uniprot_sublocation.txt', 'r')
        sublocation_processed =  open(self.db_path + 'sublocation/all/uniprot_sublocation_processed.txt', 'w')
        i = 0
        for line in sublocation_org:
            if i > 0:
#                 line = line.rstrip()
                lst_line = line.split('\t')
                p_vid = lst_line[0]
                topo_domain = lst_line[1]
                intramembrane = lst_line[2]
                transmembrane = lst_line[3].rstrip()
                
#                 if topo_domain != '':
#                     for x in topo_domain.split('; '):
#                         if x.split(' ')[0] == 'TOPO_DOM':
# #                             print(x)
#                             d_start = x.split(' ')[1]
#                             d_end = x.split(' ')[2]
# #                             d_type = x.split(' ')[3]
#                             d_type = '0'          
#                             sublocation_processed.write(p_vid + '\t' + d_start + '\t' + d_end + '\t' + d_type + '\n')
#                             

                if intramembrane != '':
                    for x in intramembrane.split('; '):
                        if x.split(' ')[0] == 'INTRAMEM':
                            print(x)
                            d_start = x.split(' ')[1]
                            d_end = x.split(' ')[2]
#                             d_type = x.split(' ')[0]
                            d_type = '1'                                         
                            for aa_pos in range(int(d_start),int(d_end)+1):
                                sublocation_processed.write(p_vid + '\t' + str(aa_pos) + '\t' + d_type + '\n')
                            
                if transmembrane != '':
#                     print(transmembrane)
                    for x in transmembrane.split('; '):
                        if x.split(' ')[0] == 'TRANSMEM':
#                             print(x)
                            d_start = x.split(' ')[1]
                            d_end = x.split(' ')[2]
#                             d_type = x.split(' ')[0]
                            d_type = '1'                
                            for aa_pos in range(int(d_start),int(d_end)+1):
                                sublocation_processed.write(p_vid + '\t' + str(aa_pos) + '\t' + d_type + '\n')                                       

            i += 1
#             print (str(i))
        pass
        sublocation_org.close()
        sublocation_processed.close()
        
#         sublocation_df = pd.read_csv(self.db_path + 'sublocation/all/uniprot_sublocation_processed.txt',sep = '\t')
#         sublocation_df.columns = ['p_vid','s_start','s_end','s_type']
        
    def create_hgnc_data(self):
        def fill_dict(hngc_id,input,in_dict):
            lst_input = input.split('|')
            in_dict.update({x:hngc_id for x in lst_input})                        
        ####***************************************************************************************************************************************************************    
        # EBI FTP
        ####***************************************************************************************************************************************************************
        # HGNC compplete set 
        # ftp://ftp.ebi.ac.uk/pub/databases/genenames/new/tsv/hgnc_complete_set.txt          
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('hgnc')        
#         self.ebi_ftp_obj = alm_fun.create_ftp_object('ftp.ebi.ac.uk')
#         return_info = alm_fun.download_ftp(self.ebi_ftp_obj, '/pub/databases/genenames/new/tsv/', 'hgnc_complete_set.txt', self.db_path + 'hgnc/org/hgnc_complete_set.txt')        
        hgnc = pd.read_csv(self.db_path + 'hgnc/org/hgnc_complete_set.txt',sep ='\t',dtype = {'location':'str'}) 
        hgnc.loc[hgnc['location'].isnull(),'location'] = ''
        hgnc['location'] = hgnc.apply(lambda x: x['location'].replace('p','q'),axis = 1)
        hgnc['chr'] = hgnc.apply(lambda x: x['location'].split('q')[0],axis = 1)
        hgnc.loc[hgnc['location'] == 'mitochondria','chr'] = 'MT'   
        hgnc.loc[~hgnc['chr'].isin(self.lst_chr),'chr'] = np.nan          
        #******************************************************
        # IDs to hgnc
        #******************************************************
        id2hgnc_dict = {}
        id_lst = ['prev_symbol','symbol','ensembl_gene_id','refseq_accession','uniprot_ids','ucsc_id']
        
        for id in id_lst:
            cur_hgnc_ids = hgnc.loc[hgnc[id].notnull(),['hgnc_id',id]] 
            cur_id_dict = {}
            cur_hgnc_ids.apply(lambda x: fill_dict(x['hgnc_id'],x[id],cur_id_dict),axis = 1)
            id2hgnc_dict[id] = cur_id_dict
        pass
        
        #combine previous symbol and symbol 
        symbol_dict = id2hgnc_dict['prev_symbol']
        symbol_dict.update(id2hgnc_dict['symbol'])
        id2hgnc_dict['symbol'] = symbol_dict
    
        np.save(self.db_path + 'hgnc/npy/id2hgnc_dict.npy', id2hgnc_dict)        
        #******************************************************
        # hgnc to IDS
        #******************************************************
        hgnc2id_dict = {}
        id_lst = ['symbol','ensembl_gene_id','refseq_accession','uniprot_ids','ucsc_id','chr']        
        for id in id_lst:
            cur_id_dict = {hgnc.loc[i,'hgnc_id']:hgnc.loc[i,id] for i in hgnc.index }             
            hgnc2id_dict[id] = cur_id_dict
        pass
        np.save(self.db_path + 'hgnc/npy/hgnc2id_dict.npy', hgnc2id_dict)      
        alm_fun.show_msg(self.log,self.verbose,'Created hgnc data.')

    def create_uniprot_data(self):
        def fill_dict(uniprot_id,input,in_dict):
            lst_input = input.split(';')
            in_dict.update({x:uniprot_id for x in lst_input})
        ####***************************************************************************************************************************************************************    
        # UniProt FTP
        ####***************************************************************************************************************************************************************
        # Reviewed uniprot fasta file
        # ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz      
        # Reviewd uniprot isoforms fasta fil e
        # ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot_varsplic.fasta.gz
        # Uniprot ID mapping
        # ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz
        ####***************************************************************************************************************************************************************        
        self.init_humamdb_object('uniprot')        
        cur_log = self.db_path + 'uniprot/log/uniprot.log'
        self.uniprot_ftp_obj = alm_fun.create_ftp_object('ftp.uniprot.org')
        
        
#         ******************************************************
#          All IDs maps to uniprot
#         ******************************************************
#         id_maps = pd.read_csv(self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab', sep='\t', header=None) 
#         id_maps.columns = ['UniProtKB-AC', 'UniProtKB-ID', 'GeneID', 'RefSeq', 'GI', 'PDB', 'GO', 'UniRef100', 'UniRef90', 'UniRef50', 'UniParc', 'PIR', 'NCBI-taxon', 'MIM', 'UniGene', 'PubMed', 'EMBL', 'EMBL-CDS', 'Ensembl', 'Ensembl_TRS', 'Ensembl_PRO', 'Additional PubMed']
#         id2uniprot_dict = {}
#         id_lst = ['RefSeq','Ensembl_PRO']        
#         for id in id_lst:
#             cur_id_maps = id_maps.loc[id_maps['UniProtKB-AC'].isin(uniprot_human_reviewed_ids) & id_maps[id].notnull(), ['UniProtKB-AC',id]] 
#             cur_id_dict = {}
#             cur_id_maps.apply(lambda x: fill_dict(x['UniProtKB-AC'],x[id],cur_id_dict),axis = 1)
#             id2uniprot_dict[id] = cur_id_dict
#         pass
#         np.save(self.db_path + 'uniprot/npy/id2uniprot_dict.npy', id2uniprot_dict)

        
        
        if self.db_version == 'uptodate':      
            return_info = alm_fun.download_ftp(self.uniprot_ftp_obj, '/pub/databases/uniprot/current_release/knowledgebase/complete/', 'uniprot_sprot.fasta.gz', self.db_path + 'uniprot/org/uniprot_sprot.fasta.gz')
            if (return_info == 'updated') | (return_info == 'downloaded'):
                alm_fun.gzip_decompress(self.db_path + 'uniprot/org/uniprot_sprot.fasta.gz', self.db_path + 'uniprot/org/uniprot_sprot.fasta')
#         if self.db_version == 'manuscript':
#             return_info = alm_fun.download_ftp(self.uniprot_ftp_obj, '/pub/databases/uniprot/previous_releases/release-2019_02/knowledgebase/', 'uniprot_sprot-only2019_02.tar.gz', self.db_path + 'uniprot/org/uniprot_sprot-only2019_02.tar.gz')
#             if (return_info == 'updated') | (return_info == 'downloaded'):
#                 alm_fun.gzip_decompress(self.db_path + 'uniprot/org/uniprot_sprot-only2019_02.tar.gz', self.db_path + 'uniprot/org/uniprot_sprot-only2019_02.tar')            
#                 gzip_cmd = "tar -xvf " + self.db_path + "uniprot/org/uniprot_sprot-only2019_02.tar"
#                 subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'uniprot/org/')
            
        
#         return_info = alm_fun.download_ftp(self.uniprot_ftp_obj, '/pub/databases/uniprot/current_release/knowledgebase/complete/', 'uniprot_sprot_varsplic.fasta.gz', self.db_path + 'uniprot/org/uniprot_sprot_varsplic.fasta.gz')
#         if (return_info == 'updated') | (return_info == 'downloaded'):
#             alm_fun.gzip_decompress(self.db_path + 'uniprot/org/uniprot_sprot_varsplic.fasta.gz', self.db_path + 'uniprot/org/uniprot_sprot_varsplic.fasta')  
# 
#         return_info = alm_fun.download_ftp(self.uniprot_ftp_obj, '/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/', 'HUMAN_9606_idmapping_selected.tab.gz', self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab.gz')
#         if (return_info == 'updated') | (return_info == 'downloaded'):
#             alm_fun.gzip_decompress(self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab.gz', self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab')  

        #******************************************************
        # Uniprot sequence dictionary and reviewed ids
        #******************************************************
        p_fa_dict = {}
#         uniprot_human_reviewed_isoform_ids = []
        uniprot_human_reviewed_ids = []
        p_fa = open(self.db_path + 'uniprot/org/uniprot_sprot.fasta', 'r') 
        for line in p_fa:
            if line[0] == ">" :
                cur_key = line.split('|')[1]
                if 'OS=Homo sapiens' in line :                    
                    uniprot_human_reviewed_ids.append(cur_key)
                p_fa_dict[cur_key] = ''
            else:
                p_fa_dict[cur_key] += line.strip()
                
#         p_fa_isoform = open(self.db_path + 'uniprot/org/uniprot_sprot_varsplic.fasta', 'r') 
#         for line in p_fa_isoform:
#             if line[0] == ">" :
#                 cur_key = line.split('|')[1]
#                 if 'OS=Homo sapiens' in line : 
#                     uniprot_human_reviewed_isoform_ids.append(cur_key)
#                 p_fa_dict[cur_key] = ''
#             else:
#                 p_fa_dict[cur_key] += line.strip()  
                
        #******************************************************
        # Make individual fasta file for all reviewed uniprot ids
        #******************************************************   
        for key in p_fa_dict.keys():
            if key in uniprot_human_reviewed_ids:
                cur_fasta = open(self.db_path + 'uniprot/bygene/' + key + '.fasta','w')
                cur_fasta.write(">" + key + '\n')
                cur_fasta.write(p_fa_dict[key])
                cur_fasta.close()
                  
        #******************************************************
        # Save the results
        #******************************************************         
        np.save(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy', p_fa_dict)
        np.save(self.db_path + 'uniprot/npy/uniprot_human_reviewed_ids.npy', uniprot_human_reviewed_ids)
#         np.save(self.db_path + 'uniprot/npy/uniprot_human_reviewed_isoform_ids.npy', uniprot_human_reviewed_isoform_ids)


        alm_fun.show_msg(cur_log,self.verbose,'Created uniprot data.')
    
    def create_mistic_data(self):
        self.init_humamdb_object("mistic")
#         cur_log = self.db_path + 'mistic/log/mistic.log'        
#         if not os.path.isfile(self.db_path + 'mistic/org/MISTIC_GRCh37.tsv.gz'):       
#             wget_cmd = "wget http://lbgi.fr/mistic/static/data/MISTIC_GRCh37.tsv.gz"
#             subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'mistic/org/')                    
#             alm_fun.show_msg(cur_log,self.verbose,'mistic data downloaded.\n')
#             alm_fun.gzip_decompress(self.db_path + 'mistic/org/MISTIC_GRCh37.tsv.gz', self.db_path + 'clinvar/org/MISTIC_GRCh37.tsv')            
#         else:
#             alm_fun.show_msg(cur_log,self.verbose,'mistic data exists already.')
        mistic_df = pd.read_csv(self.db_path + '/mistic/org/MISTIC_GRCh37.tsv', sep = '\t',skiprows = 1)
        mistic_df.columns = ['chr','nt_pos','nt_ref','nt_alt','mistic_score','mistic_pred']        
        new_mistic_df = mistic_df.groupby(['chr','nt_pos','nt_ref','nt_alt'])['mistic_score'].agg('mean').reset_index() 
        print ("Before and after avg the duplicated scores : " + str(mistic_df.shape[0]) + ',' + str(new_mistic_df.shape[0]))       
        new_mistic_df.to_csv(self.db_path + '/mistic/all/MISTIC_GRCh37_avg_duplicated_scores.csv',index = False) 
            
    def create_hgmd_data(self):
        self.init_humamdb_object("hgmd")
        cur_log = self.db_path + 'hgmd/log/hgmd.log'        
        if not os.path.isfile(self.db_path + 'hgmd/org/hgmd_2015.txt'):       
            wget_cmd = "wget https://www.dropbox.com/s/5p5ng2qjk7uuqjs/hgmd_2015_hg19.txt"
            subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'hgmd/org/')
            wget_cmd = "wget https://www.dropbox.com/s/0d0i6bsg8lkvy1x/hgmd_2015.txt"
            subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'hgmd/org/')            
            alm_fun.show_msg(cur_log,self.verbose,'hgmd data downloaded.\n')
        else:
            alm_fun.show_msg(cur_log,self.verbose,'hgmd data exists already.')
        hgmd_snv = pd.read_csv(self.db_path + 'hgmd/org/hgmd_2015.txt',sep ='\t')    
        hgmd_snv = hgmd_snv.loc[hgmd_snv['amino'].notnull(),['acc_num','tag']]
        hgmd_snv.columns = ['hgmd_id','hgmd_tag']

        hgmd_hg19 = pd.read_csv(self.db_path + 'hgmd/org/hgmd_2015_hg19.txt',sep ='\t',dtype = {'chrom':'str'})
        hgmd_hg19 = hgmd_hg19[['chrom','id','pos','ref','alt']]
        hgmd_hg19.columns = ['chr','hgmd_id','nt_pos','nt_ref','nt_alt']
        
        hgmd_snv = hgmd_snv.merge(hgmd_hg19,how = 'left')
        hgmd_snv = hgmd_snv.loc[hgmd_snv['chr'].notnull() & hgmd_snv['nt_pos'].notnull(),:] 
        
        #### same nt coordinate has multiple annotations (different studies, splicing/missense etc)        
        alm_fun.show_msg(cur_log,self.verbose,'Total number of records: ' + str(hgmd_snv.shape[0]))                
        hgmd_snv = hgmd_snv.groupby(['chr','nt_pos','nt_ref','nt_alt']).agg(','.join).reset_index()        
        hgmd_snv['hgmd_source'] = 1
        hgmd_snv['hgmd_label'] = 1            
        alm_fun.show_msg(cur_log,self.verbose,'Total number of records after combining annotations with identical snv: ' + str(hgmd_snv.shape[0]))
        
        hgmd_snv.to_csv(self.db_path +'hgmd/all/hgmd_snv.csv',index = False)
    
    def create_humsavar_data(self):
        def get_aa_ref_humsavar(x):
            x = x[33:47].strip()
            if 'p.' not in x:
                return np.nan
            else:
                y = x[2:5]
                if y in self.dict_aa3.keys():
                    return self.dict_aa3[y]
                else:
                    return '?'            
    
        def get_aa_pos_humsavar(x):
            x = x[33:47].strip()        
            if 'p.' not in x:
                return -1
            else:
                y = x[5:-3]
                if y.isdigit():
                    return y
                else:
                    return -1    
            
        def get_aa_alt_humsavar(x):
            x = x[33:47].strip()
            if 'p.' not in x:
                return np.nan
            else:
                if '=' in x:
                    return '*'
                else:  
                    y = x[-3:]
                    if y in self.dict_aa3.keys():
                        return self.dict_aa3[y]
                    else:
                        return '?' 
                    
        self.init_humamdb_object ('humsavar')  
        cur_log = self.db_path + 'humsavar/log/humsavar.log'

        if not os.path.isfile(self.db_path + 'humsavar/org/humsavar.txt'):       
            wget_cmd = "wget https://www.uniprot.org/docs/humsavar.txt"
            subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'humsavar/org/')
            alm_fun.show_msg(cur_log,self.verbose,'humsavar data downloaded.\n')
        else:
            alm_fun.show_msg(cur_log,self.verbose,'humsavar data exists already.')

        humsavar_snv = pd.read_csv(self.db_path + 'humsavar/org/humsavar.txt',skiprows = 50,sep = '\t',header = None)        
        humsavar_snv.columns = ['humsavar_info']
        humsavar_snv['p_vid'] = humsavar_snv['humsavar_info'].apply(lambda x: x[10:16])
        humsavar_snv['aa_ref'] = humsavar_snv['humsavar_info'].apply(lambda x: get_aa_ref_humsavar(x))
        humsavar_snv['aa_pos'] = humsavar_snv['humsavar_info'].apply(lambda x: get_aa_pos_humsavar(x))
        humsavar_snv['aa_pos'] = humsavar_snv['aa_pos'].astype(int)
        humsavar_snv['aa_alt'] = humsavar_snv['humsavar_info'].apply(lambda x: get_aa_alt_humsavar(x))
        humsavar_snv['humsavar_clin_sig'] = humsavar_snv['humsavar_info'].apply(lambda x: x[48:61].strip())
        
        humsavar_snv = humsavar_snv[['p_vid','aa_ref','aa_alt','aa_pos','humsavar_clin_sig']]
        humsavar_snv = humsavar_snv.drop_duplicates()

        humsavar_snv['humsavar_label'] = -1
        humsavar_snv.loc[(humsavar_snv['humsavar_clin_sig'] == 'Disease'), 'humsavar_label'] = 1
        humsavar_snv.loc[(humsavar_snv['humsavar_clin_sig'] == 'Polymorphism'), 'humsavar_label'] = 0
        humsavar_snv.to_csv(self.db_path + 'humsavar/all/humsavar_snv.csv',index = False)
        
        humsavar_disease_pvids = humsavar_snv.loc[humsavar_snv['humsavar_label'] == 1,'p_vid'].unique()
        id2hgnc_dict = np.load(self.db_path + 'hgnc/npy/id2hgnc_dict.npy').item()
        humsavar_disease_genes = [id2hgnc_dict['uniprot_ids'].get(x,'') for x in humsavar_disease_pvids]        
        humsavar_disease_genes_file = open(self.db_path + 'humsavar/all/humsavar_disease_genes.txt', 'w')
        for gene in humsavar_disease_genes:
            if gene != '':
                humsavar_disease_genes_file.write(gene + '\n') 
                                
        alm_fun.show_msg(cur_log,self.verbose,'HUMSAVAR data created.\n')

    def create_clinvar_data(self): 
        def chk_method(input):
            unique_methods = np.unique(input)
            if (len(unique_methods) == 1) & (unique_methods[0] == 'literature only'):
                return(1)
            else:
                return(0)           
        ####***************************************************************************************************************************************************************    
        # NCBI FTP
        ####***************************************************************************************************************************************************************
        # Clinvar 
        # ftp://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
        ####***************************************************************************************************************************************************************        
        self.init_humamdb_object ('clinvar')  
        cur_log = self.db_path + 'clinvar/log/clinvar.log'
        self.ncbi_ftp_obj = alm_fun.create_ftp_object('ftp.ncbi.nlm.nih.gov')  
            
        return_info = alm_fun.download_ftp(self.ncbi_ftp_obj, '/pub/clinvar/tab_delimited/', 'variant_summary.txt.gz', self.db_path + 'clinvar/org/variant_summary.txt.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'clinvar/org/variant_summary.txt.gz', self.db_path + 'clinvar/org/variant_summary.txt')      
            
            
        return_info = alm_fun.download_ftp(self.ncbi_ftp_obj, '/pub/clinvar/tab_delimited/', 'submission_summary.txt.gz', self.db_path + 'clinvar/org/variant_summary.txt.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'clinvar/org/variant_summary.txt.gz', self.db_path + 'clinvar/org/submission_summary.txt')             
                       
    
        #load clinvar submission
        self.clinvar_submission_file = 'clinvar/org/submission_summary.txt'        
        clinvar_submission_raw = pd.read_csv(self.db_path + self.clinvar_submission_file, sep='\t',skiprows = 15  )
        alm_fun.show_msg(cur_log,self.verbose,'Total number of CLINVAR submission records : ' + str(clinvar_submission_raw.shape[0]))
        
        clinvar_submission = clinvar_submission_raw[['#VariationID','CollectionMethod']]
        clinvar_submission.columns = ['clinvar_id','clinvar_collection_method']
        clinvar_submission_group = clinvar_submission.groupby(['clinvar_id']) ['clinvar_collection_method'].agg(list).reset_index()
        clinvar_submission_group.columns = ['clinvar_id','clinvar_collection_methods']
        clinvar_submission_group['clinvar_literature_only'] = clinvar_submission_group['clinvar_collection_methods'].apply(lambda x: chk_method(x))

        #load clinvar rawdata
        self.clinvar_raw_file = 'clinvar/org/variant_summary.txt'        
        clinvar_raw = pd.read_csv(self.db_path + self.clinvar_raw_file, sep='\t',dtype={'Chromosome':'str'})
        alm_fun.show_msg(cur_log,self.verbose,'Total number of CLINVAR records : ' + str(clinvar_raw.shape[0]))

        clinvar_snv = clinvar_raw[['Assembly','Type','Chromosome', 'Start', 'ReferenceAllele', 'AlternateAllele', 'ReviewStatus','LastEvaluated', 'ClinicalSignificance', 'NumberSubmitters', 'PhenotypeIDS', 'PhenotypeList','VariationID','HGNC_ID']]
        clinvar_snv.columns = ['assembly','type','chr', 'nt_pos', 'nt_ref', 'nt_alt', 'clinvar_review_status','clinvar_evaluate_time', 'clinvar_clin_sig', 'clinvar_ev_num', 'clinvar_phenotype_id', 'clinvar_phenotype_name','clinvar_id','clinvar_hgnc_id']        

        #filter irregular chromosome
        clinvar_snv = clinvar_snv.loc[clinvar_snv['chr'].isin(self.lst_chr),:]
        alm_fun.show_msg(cur_log,self.verbose,'clinVAR records after filter irregular chromosome records : ' + str(clinvar_snv.shape[0]))         
 
        # Get list of disease genes
        clinvar_disease_clin_sigs = ['Pathogenic','Pathogenic/Likely pathogenic','Likely pathogenic']        
        clinvar_disease_genes = list(clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'].isin(clinvar_disease_clin_sigs)) & (clinvar_snv['clinvar_hgnc_id'].str.contains('HGNC')),'clinvar_hgnc_id'].unique())        
        alm_fun.show_msg(cur_log,self.verbose,'Number of disease genes : ' + str(len(clinvar_disease_genes)))
         
        #filter non GRCh37 records
        clinvar_snv = clinvar_snv.loc[(clinvar_snv['assembly'] == self.assembly),:]
        alm_fun.show_msg(cur_log,self.verbose,'clinVAR records after filter non GRCh37 records : ' + str(clinvar_snv.shape[0]))                        
 

        # filter non-ACMG term 
        clinvar_clin_sig_lst = ['Pathogenic','Pathogenic/Likely pathogenic','Likely pathogenic','Uncertain significance','Benign','Benign/Likely benign','Likely benign']        
        clinvar_snv = clinvar_snv.loc[clinvar_snv['clinvar_clin_sig'].isin(clinvar_clin_sig_lst),:]
        alm_fun.show_msg(cur_log,self.verbose,'clinVAR records after filter nan-ACMG records : ' + str(clinvar_snv.shape[0]))
         
        #filter non snv
        clinvar_snv = clinvar_snv.loc[(clinvar_snv['type'] == 'single nucleotide variant'),:]  
        alm_fun.show_msg(cur_log,self.verbose,'clinVAR records after filter non-SNV records : ' + str(clinvar_snv.shape[0]))
  
        #filter non-disease gene 
        clinvar_snv = clinvar_snv.loc[clinvar_snv['clinvar_hgnc_id'].isin(clinvar_disease_genes),:]
        alm_fun.show_msg(cur_log,self.verbose,'clinVAR records after filter non-disease gene records : ' + str(clinvar_snv.shape[0]))
         
        clinvar_snv['clinvar_review_star'] = -1
        clinvar_snv.loc[clinvar_snv['clinvar_review_status'] == 'practice guideline','clinvar_review_star'] = 4
        clinvar_snv.loc[clinvar_snv['clinvar_review_status'] == 'reviewed by expert panel','clinvar_review_star'] = 3
        clinvar_snv.loc[clinvar_snv['clinvar_review_status'] == 'criteria provided, multiple submitters, no conflicts','clinvar_review_star'] = 2
        clinvar_snv.loc[clinvar_snv['clinvar_review_status'] == 'criteria provided, single submitter','clinvar_review_star'] = 1    
        clinvar_snv.loc[clinvar_snv['clinvar_review_status'] == 'criteria provided, conflicting interpretations','clinvar_review_star'] = 1
        clinvar_snv.loc[clinvar_snv['clinvar_review_status'] == 'no assertion criteria provided','clinvar_review_star'] = 0
        clinvar_snv.loc[clinvar_snv['clinvar_review_status'] == 'no interpretation for the single variant','clinvar_review_star'] = 0
                        
        clinvar_snv['clinvar_clinsig_level'] = -1
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Pathogenic'), 'clinvar_clinsig_level'] = 3
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Benign'), 'clinvar_clinsig_level'] = 3                        
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Pathogenic/Likely pathogenic'), 'clinvar_clinsig_level'] = 2
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Benign/Likely benign'), 'clinvar_clinsig_level'] = 2
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Likely pathogenic'), 'clinvar_clinsig_level'] = 1
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Likely benign'), 'clinvar_clinsig_level'] = 1 
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Uncertain significance'), 'clinvar_clinsig_level'] = 0
          
        clinvar_snv['clinvar_label'] = -1
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Pathogenic'), 'clinvar_label'] = 1
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Benign'), 'clinvar_label'] = 0                        
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Pathogenic/Likely pathogenic'), 'clinvar_label'] = 1
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Benign/Likely benign'), 'clinvar_label'] = 0
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Likely pathogenic'), 'clinvar_label'] = 1
        clinvar_snv.loc[(clinvar_snv['clinvar_clin_sig'] == 'Likely benign'), 'clinvar_label'] = 0 
                        
        clinvar_disease_genes_file = open(self.db_path + 'clinvar/all/clinvar_disease_genes.txt', 'w')
        for disease_gene in clinvar_disease_genes:
            clinvar_disease_genes_file.write(disease_gene + '\n')
        
        clinvar_disease_genes_file.close()  
        clinvar_snv = clinvar_snv[['chr', 'nt_pos', 'nt_ref', 'nt_alt', 'clinvar_review_status','clinvar_review_star','clinvar_evaluate_time', 'clinvar_clin_sig', 'clinvar_clinsig_level','clinvar_ev_num','clinvar_label','clinvar_phenotype_id', 'clinvar_phenotype_name','clinvar_id','clinvar_hgnc_id']]
        clinvar_snv_new = pd.merge(clinvar_snv,clinvar_submission_group,how = 'left')
#         clinvar_snv_new.loc[(clinvar_snv_new['clinvar_clinsig_level'].isin([1,2,3])) & (clinvar_snv_new['clinvar_literature_only'] == 1),:]                               
        clinvar_snv_new.to_csv(self.db_path + 'clinvar/all/clinvar_snv.csv',index = False)        
        alm_fun.show_msg(cur_log,self.verbose,'CLINVAR data created.\n')
         
    def run_mave_normalization(self,mave_input_file,normalized,flip_flag,floor_flag):        
        dms_gene_df = pd.read_csv(mave_input_file, sep = '\t')
        syn_keep_index = dms_gene_df.loc[dms_gene_df['aa_ref'] == dms_gene_df['aa_alt'],:].index
        stop_keep_index = dms_gene_df.loc[dms_gene_df['aa_alt'] == '*',:].index        
        dms_gene_df['fitness_input_se'] = dms_gene_df['fitness_input_sd'] / np.sqrt(dms_gene_df['num_replicates'])
        dms_gene_df['fitness_input_filtered'] = dms_gene_df['fitness_input']
        dms_gene_df['fitness_input_filtered_sd'] = dms_gene_df['fitness_input_sd']
        dms_gene_df['fitness_input_filtered_se'] = dms_gene_df['fitness_input_se']

        ####*************************************************************************************************************************************************************
        # step2: fitness normalization
        ####*************************************************************************************************************************************************************
        if int(normalized) == 0:            
            syn_median = np.median(dms_gene_df.loc[syn_keep_index,'fitness_input_filtered'])
            stop_median = np.median(dms_gene_df.loc[stop_keep_index,'fitness_input_filtered'])            
            dms_gene_df['fitness_org'] = (dms_gene_df['fitness_input_filtered'] - stop_median) / (syn_median - stop_median)
            dms_gene_df['fitness_sd_org'] = dms_gene_df['fitness_input_filtered_sd'] / (syn_median - stop_median) 
            dms_gene_df['fitness_se_org'] = dms_gene_df['fitness_input_filtered_se'] / (syn_median - stop_median)        
        else:
            dms_gene_df['fitness_org'] = dms_gene_df['fitness_input_filtered']
            dms_gene_df['fitness_sd_org'] = dms_gene_df['fitness_input_filtered_sd']
            dms_gene_df['fitness_se_org'] = dms_gene_df['fitness_input_filtered_se']
            
            dms_gene_df['syn_filtered'] = 1
            syn_keep_index = (dms_gene_df['annotation'] == 'SYN') & dms_gene_df['fitness_input_filtered'].notnull()
            dms_gene_df.loc[syn_keep_index,'syn_filtered'] = 0
            syn_median = np.median(dms_gene_df.loc[syn_keep_index,'fitness_input_filtered'])
                                
            dms_gene_df['stop_filtered'] = 1
            stop_keep_index = (dms_gene_df['annotation'] == 'STOP') & dms_gene_df['fitness_input_filtered'].notnull()
            dms_gene_df.loc[stop_keep_index,'stop_filtered'] = 0
                   
        ####*************************************************************************************************************************************************************
        # step3: fitness reverse and floor
        ####*************************************************************************************************************************************************************
        dms_gene_df['fitness_reverse'] = dms_gene_df['fitness_org']
        dms_gene_df['fitness_sd_reverse'] = dms_gene_df['fitness_sd_org']
        dms_gene_df['fitness_se_reverse'] = dms_gene_df['fitness_se_org']
        
        dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_sd_reverse'] = dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_sd_reverse'] / np.power(dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_reverse'],2)
        dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_se_reverse'] = dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_se_reverse'] / np.power(dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_reverse'],2)        
        dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_reverse'] = 1 / dms_gene_df.loc[dms_gene_df['fitness_reverse'] > 1 , 'fitness_reverse']

        if flip_flag == 1:            
            dms_gene_df['fitness'] = dms_gene_df['fitness_reverse']
            dms_gene_df['fitness_sd'] = dms_gene_df['fitness_sd_reverse'] 
            dms_gene_df['fitness_se'] = dms_gene_df['fitness_se_reverse']
        else:
            dms_gene_df['fitness'] = dms_gene_df['fitness_org']
            dms_gene_df['fitness_sd'] = dms_gene_df['fitness_sd_org']
            dms_gene_df['fitness_se'] = dms_gene_df['fitness_se_org']
            
        # floor
        if floor_flag == 1:
            dms_gene_df.loc[dms_gene_df['fitness'] < 0, 'fitness'] = 0
        return(dms_gene_df)
            
    def create_mave_data_old(self):               
        self.init_humamdb_object ('mave')  
        cur_log = self.db_path + 'mave/log/mave.log'
                
        if not os.path.isfile(self.db_path + 'mave/all/mave_missense.csv'):                    
            if not os.path.isfile(self.db_path + 'mave/org/funregressor_training_from_imputation.csv') :                 
                wget_cmd = "wget https://www.dropbox.com/s/m8i41a5l3mk1h5i/funregressor_training_from_imputation.csv"
                subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'mave/org/')
                alm_fun.show_msg(cur_log,self.verbose,'mave data downloaded.\n')
            
            #load Roth MAVE data from imputation pipeline
            mave_data = pd.read_csv(self.db_path + 'mave/org/funregressor_training_from_imputation.csv')        
            mave_data = mave_data[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt','fitness_input','fitness_org','fitness_sd_org','fitness','fitness_prob','fitness_se_reg','fitness_refine','fitness_se_refine','quality_score']]
    
            alm_fun.show_msg(cur_log,self.verbose,'Total number of MAVE records : ' + str(mave_data.shape[0]))
            
            mave_data = mave_data.loc[mave_data['fitness_input'].notnull(),:]
            alm_fun.show_msg(cur_log,self.verbose,'MAVE records after removing no experimental score : ' + str(mave_data.shape[0]))
            
            mave_missense = mave_data.loc[(mave_data['aa_ref'] != mave_data['aa_alt']) & (mave_data['aa_alt'] != '*'),:]
            alm_fun.show_msg(cur_log,self.verbose,'MAVE records after taking only missense variants : ' + str(mave_missense.shape[0]))
            
    #         mave_missense = self.add_pisa(mave_missense)
    #         mave_missense = self.add_psipred(mave_missense)
    #         mave_missense = self.add_pfam(mave_missense)
            mave_missense = self.add_sift(mave_missense)
            mave_missense = self.add_provean(mave_missense)
                    
            #Add position importance
            pos_importance_groupby = mave_missense.groupby(['p_vid','aa_pos'])['fitness'].agg(['min', 'max','median','mean','count','std']).reset_index()
            pos_importance_groupby.columns = ['p_vid','aa_pos','pos_importance_min','pos_importance_max','pos_importance_median','pos_importance_mean','pos_importance_count','pos_importance_std']        
            mave_missense = pd.merge(mave_missense,pos_importance_groupby,how = 'left')
            mave_missense['mave_source'] = 1
            
            #Add label and label confidence
            mave_missense['mave_label'] = -1
            mave_missense.loc[mave_missense['fitness'] >= 0.5, 'mave_label'] = 0
            mave_missense.loc[mave_missense['fitness'] < 0.5, 'mave_label'] = 1
            mave_missense['mave_label_confidence'] = np.abs(mave_missense['mave_label'] - mave_missense['fitness']) 
         
            mave_missense_for_funsum = mave_missense.copy()       
            mave_pvids = list(mave_data['p_vid'].unique())
            id2hgnc_dict = np.load(self.db_path + 'hgnc/npy/id2hgnc_dict.npy').item()
            mave_genes = [id2hgnc_dict['uniprot_ids'][x] for x in mave_pvids]        
            alm_fun.show_msg(cur_log,self.verbose,'Number of MAVE genes : ' + str(len(mave_genes)))
            
            mave_genes_file = open(self.db_path + 'mave/all/mave_genes.txt', 'w')
            for gene in mave_genes:
                mave_genes_file.write(gene + '\n')
    
            mave_missense_for_funsum.to_csv(self.db_path + 'mave/all/mave_missense_forfunsum.csv',index = False)                    
            mave_missense.to_csv(self.db_path + 'mave/all/mave_missense.csv',index = False)
            alm_fun.show_msg(cur_log,self.verbose,'MAVE data created.\n')   
        else:
            alm_fun.show_msg(cur_log,self.verbose,'MAVE data exists already.\n')            
    
    def create_mave_data(self):            
#         #**********************************************************************************************************
#         # Create MAVE Imputation input data for each MAVE gene, imputation pipeline format (from VARITY manuscript)
#         #**********************************************************************************************************
        varity_mave_df = pd.read_csv(self.db_path + 'mave/org/funregressor_training_from_imputation.csv')             
        for p_vid in varity_mave_df['p_vid'].unique():            
            cur_mave_df = varity_mave_df.loc[varity_mave_df['p_vid'] == p_vid,['aa_pos','aa_ref','aa_alt','quality_score','num_replicates','fitness_input','fitness_input_sd']]
            cur_mave_df = cur_mave_df.loc[cur_mave_df['fitness_input'].notnull(),:]
            cur_mave_df['num_replicates'] = 2
            cur_mave_df.to_csv(self.db_path + 'mave/bygene/' + p_vid + '_mave_input.txt',sep = '\t', index = False)
         
#         #**********************************************************************************************************
#         # VARITY revision, add a few more genes that have MAVE data from Livesey et al 2020 Molecular system biology 
#         # ['ADRB2','HRAS','MAPK1','TP53'], need more investigation on each of them to determine the fitness
#         #**********************************************************************************************************
#         
#         #**********************************************************************************************************
#         # VKORC1 Chiasson et al. eLife 2020;9:e58026. DOI: https://doi.org/10.7554/eLife.58026
#         #**********************************************************************************************************        
        vkor_mave_df = pd.read_csv(self.db_path + 'mave/org/VKOR.csv')[['position','start','end','abundance_se','abundance_expts','abundance_score','abundance_sd']]        
        vkor_mave_df.columns = ['aa_pos','aa_ref','aa_alt','quality_score','num_replicates','fitness_input','fitness_input_sd']
        vkor_mave_df.loc[vkor_mave_df['aa_alt'] == 'X','aa_alt'] = '*'
        vkor_mave_df = vkor_mave_df.loc[vkor_mave_df['aa_ref']!= 'Z',:]
        vkor_mave_df = vkor_mave_df.loc[vkor_mave_df['num_replicates'].notnull(),:]
        vkor_mave_df['aa_pos'] = vkor_mave_df['aa_pos'].astype(int)
        vkor_mave_df['num_replicates'] = vkor_mave_df['num_replicates'].astype(int)
        vkor_mave_df.to_csv(self.db_path + 'mave/bygene/' + 'Q9BQB6' + '_mave_input.txt',sep = '\t', index = False)

#         all_mave_df = self.add_pisa(all_mave_df)
#         all_mave_df = self.add_psipred(all_mave_df)
#         all_mave_df = self.add_pfam(all_mave_df)
#         all_mave_df = self.add_sift(all_mave_df)
#         all_mave_df = self.add_provean(all_mave_df)

        self.init_humamdb_object ('mave')  
        cur_log = self.db_path + 'mave/log/mave.log'
        #******************************************************************************
        #Roth Lab MAVE data
        #******************************************************************************
        all_mave_df = None
        for mave_input_file in glob.glob(os.path.join(self.db_path + 'mave/bygene/' , '*_mave_input.txt')):
            cur_pvid = mave_input_file.split('/')[-1].split('_')[0]
            mave_normalized_df = self.run_mave_normalization(mave_input_file, 0, 1, 1)
            mave_normalized_df['p_vid'] = cur_pvid
            if all_mave_df is None:
                all_mave_df = mave_normalized_df
            else:
                all_mave_df = pd.concat([all_mave_df,mave_normalized_df])
                
        alm_fun.show_msg(cur_log,self.verbose,'Total number of MAVE records : ' + str(all_mave_df.shape[0]))        
        all_mave_df = all_mave_df.loc[all_mave_df['fitness_input'].notnull(),:]
        alm_fun.show_msg(cur_log,self.verbose,'MAVE records after removing no experimental score : ' + str(all_mave_df.shape[0]))        
        all_mave_df = all_mave_df.loc[(all_mave_df['aa_ref'] != all_mave_df['aa_alt']) & (all_mave_df['aa_alt'] != '*'),:]
        alm_fun.show_msg(cur_log,self.verbose,'MAVE records after taking only missense variants : ' + str(all_mave_df.shape[0]))
                
        #Add position importance
        pos_importance_groupby = all_mave_df.groupby(['p_vid','aa_pos'])['fitness'].agg(['min', 'max','median','mean','count','std']).reset_index()
        pos_importance_groupby.columns = ['p_vid','aa_pos','pos_importance_min','pos_importance_max','pos_importance_median','pos_importance_mean','pos_importance_count','pos_importance_std']        
        all_mave_df = pd.merge(all_mave_df,pos_importance_groupby,how = 'left')
        all_mave_df['mave_source'] = 1
        
        #Add label and label confidence
        all_mave_df['mave_label'] = np.nan
        all_mave_df.loc[all_mave_df['fitness'] >= 0.5, 'mave_label'] = 0
        all_mave_df.loc[all_mave_df['fitness'] < 0.5, 'mave_label'] = 1
        all_mave_df['mave_label_confidence'] = np.abs(all_mave_df['mave_label'] - all_mave_df['fitness']) 
       
        mave_pvids = list(all_mave_df['p_vid'].unique())
        id2hgnc_dict = np.load(self.db_path + 'hgnc/npy/id2hgnc_dict.npy').item()
        mave_genes = [id2hgnc_dict['uniprot_ids'][x] for x in mave_pvids]        
        alm_fun.show_msg(cur_log,self.verbose,'Number of MAVE genes : ' + str(len(mave_genes)))
        
        mave_genes_file = open(self.db_path + 'mave/all/mave_genes.txt', 'w')
        for gene in mave_genes:
            mave_genes_file.write(gene + '\n')

        all_mave_df.to_csv(self.db_path + 'mave/all/all_mave_for_funsum.csv',index = False)   
        all_mave_for_varity = all_mave_df[['p_vid','aa_pos','aa_ref','aa_alt','fitness_input','fitness_org','fitness','mave_source','mave_label','mave_label_confidence']]
        all_mave_for_varity.columns = ['p_vid','aa_pos','aa_ref','aa_alt','mave_input','mave_norm','mave_score','mave_source','mave_label','mave_label_confidence']
                         
        all_mave_for_varity.to_csv(self.db_path + 'mave/all/all_mave_for_varity.csv',index = False)
        alm_fun.show_msg(cur_log,self.verbose,'MAVE data created.\n')                  

    def create_gnomad_data(self):
        ####***************************************************************************************************************************************************************    
        # Gnomad (using google clound not FTP) 
        ####***************************************************************************************************************************************************************
        # wget https://storage.googleapis.com/gnomad-public/release/2.1.1/vcf/exomes/gnomad.exomes.r2.1.1.sites.vcf.bgz          
        ####***************************************************************************************************************************************************************        
        self.init_humamdb_object('gnomad')  
        stime = time.time()
        gnomadLogFile = self.db_path + 'gnomad/gnomad_process_log'
        gnomadFile = self.db_path + 'gnomad/gnomad.exomes.r2.1.1.sites.vcf'
        gnomFile_output = self.db_path + 'gnomad/gnomad_output_snp.txt'
        gnomFile_output_af = self.db_path + 'gnomad/gnomad_output_snp_af.txt'
        vcfheaderFile_output = self.db_path + 'gnomad/vcfheader_output_snp.txt'
        count = 0
        newline = ''
        newline_af = ''
        vcfheader = ''
         
        with open(gnomadFile) as infile:
            for line in infile:                                 
                if not re.match('#', line):
                    count += 1  
                    if (count % 10000 == 0):
                        alm_fun.show_msg(gnomadLogFile, 1, str(count) + ' records have been processed.\n')
                        with open(gnomFile_output, "a") as f:
                            f.write(newline)
                            newline = ''
                        with open(gnomFile_output_af, "a") as f:
                            f.write(newline_af)
                            newline_af = ''                                                        
                    line_list = line.split('\t')             
                    info = line_list[-1]
                    line_list.pop(-1)
                    if re.match('AC', info): 
                        # print(info)
                        try:                
                            info_dict = {x.split("=")[0]:x.split("=")[1] for x in info.split(';')}
                        except:
                            info_dict = {x.split("=")[0]:x.split("=")[1] if "=" in x else x + '=' for x in info.split(';')}

                        try:
                            line_list.append(info_dict.get('AC', ''))  # add AC
                            line_list.append(info_dict.get('AN', ''))  # add AN
                            line_list.append(info_dict.get('AF', ''))  # add AF
                            line_list.append(info_dict.get('AF_EAS', ''))  # add AF_ESA
                            line_list.append(info_dict.get('GC', ''))  # add GC
     
                            vep_list = info_dict.get('CSQ', '').split(',')                            
                            vep_allele_dict = {}    
                            for vep in vep_list:
                                vep_sub_list = vep.split('|')                      
                                if (vep_sub_list[26] == 'YES') & (vep_sub_list[1] == 'missense_variant') :  # 'CANONICAL'      
                                    if vep_sub_list[0] not in vep_allele_dict.keys():  
                                        vep_allele_dict[vep_sub_list[0]] = []                                                          
                                    vep_allele_dict[vep_sub_list[0]].append([vep_sub_list[15].split('/')[0], vep_sub_list[14], vep_sub_list[15].split('/')[1], vep_sub_list[1], vep_sub_list[3], vep_sub_list[4], vep_sub_list[6], vep_sub_list[10], vep_sub_list[11], vep_sub_list[17], vep_sub_list[30], vep_sub_list[31], vep_sub_list[20]])
#                                     vep_allele_dict[vep_sub_list[0]] = [vep_sub_list[15].split('/')[0],vep_sub_list[14],vep_sub_list[15].split('/')[1],vep_sub_list[1],vep_sub_list[3],vep_sub_list[4],vep_sub_list[6],vep_sub_list[10],vep_sub_list[11],vep_sub_list[17],vep_sub_list[30],vep_sub_list[31],vep_sub_list[20]]
                                if (vep_sub_list[26] == 'YES') & (vep_sub_list[1] == 'synonymous_variant') :  # 'CANONICAL'
                                    if vep_sub_list[0] not in vep_allele_dict.keys():  
                                        vep_allele_dict[vep_sub_list[0]] = []   
                                    vep_allele_dict[vep_sub_list[0]].append([vep_sub_list[15], vep_sub_list[14], vep_sub_list[15], vep_sub_list[1], vep_sub_list[3], vep_sub_list[4], vep_sub_list[6], vep_sub_list[10], vep_sub_list[11], vep_sub_list[17], vep_sub_list[30], vep_sub_list[31], vep_sub_list[20]])                                    
#                                     vep_allele_dict[vep_sub_list[0]] = [vep_sub_list[15],vep_sub_list[14],vep_sub_list[15],vep_sub_list[1],vep_sub_list[3],vep_sub_list[4],vep_sub_list[6],vep_sub_list[10],vep_sub_list[11],vep_sub_list[17],vep_sub_list[30],vep_sub_list[31],vep_sub_list[20]]                                                                                                                                                                           
                            # multiple allele for same position and ref
                            allele_list = line_list[4].split(',')                            
                            if len(allele_list) > 1:
                                ac_list = line_list[7].split(',')
                                an_list = line_list[8].split(',')
                                af_list = line_list[9].split(',')
                                af_esa_list = line_list[10].split(',')
                                if line_list[11] != '':
                                    gc_list_1 = line_list[11].split(',')
                                    gc_list = []
                                    for i in range(int(len(gc_list_1) / 3)):
                                        gc_list.append(gc_list_1[3 * i:3 * i + 3])
                                else:
                                    gc_list = [['-1', '-1', '-1']] * len(allele_list)                             
                                
                                for i in range(len(allele_list)):
                                    newline_af += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [line_list[8]] + [af_list[i]] + [af_esa_list[i]] + [str(gc_list[i])] + [gc_list[i][0]] + [gc_list[i][1]] + [gc_list[i][2]]) + '\n'
                                    if allele_list[i] in list(vep_allele_dict.keys()):   
                                        for j in range(len(vep_allele_dict[allele_list[i]])):                                                
                                            newline += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [line_list[8]] + [af_list[i]] + [af_esa_list[i]] + [str(gc_list[i])] + [gc_list[i][0]] + [gc_list[i][1]] + [gc_list[i][2]] + vep_allele_dict[allele_list[i]][j]) + '\n'
                                    else:
                                        newline += '\t'.join (line_list[0:4] + [allele_list[i]] + line_list[5:7] + [ac_list[i]] + [line_list[8]] + [af_list[i]] + [af_esa_list[i]] + [str(gc_list[i])] + [gc_list[i][0]] + [gc_list[i][1]] + [gc_list[i][2]] + ['N/A'] * 13) + '\n'
                            else:                                
                                if line_list[11] != '':
                                    gc_list = line_list[11].split(',')
                                else:
                                    gc_list = ['-1', '-1', '-1']
                                
                                newline_af += '\t'.join (line_list[0:11] + [str(gc_list)] + [gc_list[0]] + [gc_list[1]] + [gc_list[2]]) + '\n'                                
                                if allele_list[0] in list(vep_allele_dict.keys()):    
                                    for j in range(len(vep_allele_dict[allele_list[0]])):                                                
                                        newline += '\t'.join (line_list[0:11] + [str(gc_list)] + [gc_list[0]] + [gc_list[1]] + [gc_list[2]] + vep_allele_dict[allele_list[0]][j]) + '\n'
                                else:
                                    newline += '\t'.join (line_list[0:11] + [str(gc_list)] + [gc_list[0]] + [gc_list[1]] + [gc_list[2]] + ['N/A'] * 13) + '\n'
                        except:
                            alm_fun.show_msg(gnomadLogFile, 1, 'Records ' + str(count) + ' raise error:\n' + traceback.format_exc() + '\n' + str(line_list))
                            newline += '\t'.join (line_list + ['N/A'] * 16) + '\n'
                            newline_af += '\t'.join (line_list + ['N/A'] * 16) + '\n'
                else:
                    vcfheader += line + '\n'                
        with open(gnomFile_output, "a") as f:
            f.write(newline)
        with open(gnomFile_output_af, "a") as f:
            f.write(newline_af)
        with open(vcfheaderFile_output, "a") as f:
            f.write(vcfheader)  
        etime = time.time() 
        print("Elapse time was %g seconds" % (etime - stime))
    
    def initiate_psipred_data(self):
        ####***************************************************************************************************************************************************************    
        # EBI FTP
        ####***************************************************************************************************************************************************************
        # Uniprot ID to PDB ids 
        # (ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz)       
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('psipred') 
        cur_log = self.db_path + 'psipred/log/psipred.log'    
        alm_fun.show_msg(cur_log,self.verbose,'Initiated psipred data.')
        
    def create_pfam_data(self):
        ####***************************************************************************************************************************************************************    
        # EBI FTP
        ####***************************************************************************************************************************************************************
        # Pfam
        # ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/proteomes/9606.tsv.gz       
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('pfam') 
        cur_log = self.db_path + 'pfam/log/pfam.log'
        self.ebi_ftp_obj = alm_fun.create_ftp_object('ftp.ebi.ac.uk')
        return_info = alm_fun.download_ftp(self.ebi_ftp_obj, '/pub/databases/Pfam/current_release/proteomes/', '9606.tsv.gz', self.db_path + 'pfam/org/9606.tsv.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):        
            alm_fun.gzip_decompress(self.db_path + 'pfam/org/9606.tsv.gz', self.db_path + 'pfam/org/9606.tsv')  
        
        pfam = pd.read_csv(self.db_path + 'pfam/org/9606.tsv', header=None, skiprows=3, sep='\t')
        pfam.columns = ['p_vid', 'a_start', 'a_end', 'e_start', 'e_end', 'hmm_id', 'hmm_name', 'type', 'hmm_start', 'hmm_end', 'hmm_length', 'bit_score', 'e_value', 'clan']
        p_vids = pfam.p_vid.unique()        
        for p_vid in p_vids:
            pfam.loc[(pfam.p_vid == p_vid), :].to_csv(self.db_path + 'pfam/bygene/' + p_vid + '_pfam.csv', index=None)
        alm_fun.show_msg(cur_log,self.verbose,'pFAM data created.\n')
    
    def create_ensembl66_data(self):
        ####***************************************************************************************************************************************************************    
        # ensembl FTP
        ####***************************************************************************************************************************************************************
        # ensembl protein sequences 66 , because provean and sift was using ensembl release 66
        # ftp://ftp.ensembl.org/pub/release-66/fasta/homo_sapiens/pep/Homo_sapiens.GRCh37.66.pep.all.fa.gz     
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('ensembl66')  
        cur_log = self.dbpath = 'ensembl66/log/ensembl66.log'
        self.ensembl_ftp_obj = alm_fun.create_ftp_object('ftp.ensembl.org')      
        return_info = alm_fun.download_ftp(self.ensembl_ftp_obj, '/pub/release-66/fasta/homo_sapiens/pep/', 'Homo_sapiens.GRCh37.66.pep.all.fa.gz', self.db_path + 'ensembl66/org/Homo_sapiens.GRCh37.66.pep.all.fa.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'ensembl66/org/Homo_sapiens.GRCh37.66.pep.all.fa.gz', self.db_path + 'ensembl66/org/Homo_sapiens.GRCh37.66.pep.all.fa')  
                
        ensg2ensp66_dict = {}
        ensp2ensg66_dict = {}
        p_fa_dict = {}        
        cur_file = self.db_path + 'ensembl66/org/Homo_sapiens.GRCh37.66.pep.all.fa'
        if os.path.isfile(cur_file):
            p_fa = open(cur_file, 'r') 
            for line in p_fa:
                if line[0] == ">" :
                    cur_ensp = line.split(' ')[0][1:]
                    cur_ensg = line.split(' ')[3].split(':')[1]
                    ensp2ensg66_dict[cur_ensp] = cur_ensg
                    if cur_ensg in ensg2ensp66_dict.keys():
                        ensg2ensp66_dict[cur_ensg].append(cur_ensp)
                    else:
                        ensg2ensp66_dict[cur_ensg] = [cur_ensp]                    
                    p_fa_dict[cur_ensp] = ''
                else:
                    p_fa_dict[cur_ensp] += line.strip() 
        np.save(self.db_path + 'ensembl66/npy/ensembl66_seq_dict.npy', p_fa_dict) 
        np.save(self.db_path + 'ensembl66/npy/ensg2ensp66_dict.npy', ensg2ensp66_dict)
        np.save(self.db_path + 'ensembl66/npy/ensp2ensg66_dict.npy', ensp2ensg66_dict)

        alm_fun.show_msg(cur_log,self.verbose,'ensembl66 data created.\n')
        
    
    
    def create_sift_data(self):
        ####***************************************************************************************************************************************************************    
        # jcvi FTP
        ####***************************************************************************************************************************************************************
        # SIFT scores
        # ftp://ftp.jcvi.org/pub/data/provean/precomputed_scores/SIFT_scores_and_info_ensembl66_human.tsv.gz    
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('sift')  
        cur_log = self.db_path + 'sift/log/sift.log'
        self.jcvi_ftp_obj = alm_fun.create_ftp_object('ftp.jcvi.org')
        return_info = alm_fun.download_ftp(self.jcvi_ftp_obj, '/pub/data/provean/precomputed_scores/', 'SIFT_scores_and_info_ensembl66_human.tsv.gz', self.db_path + 'sift/org/SIFT_scores_and_info_ensembl66_human.tsv.gz') 
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'sift/org/SIFT_scores_and_info_ensembl66_human.tsv.gz', self.db_path + 'sift/org/SIFT_scores_and_info_ensembl66_human.tsv')  
   
        id2uniprot_matched_dict = np.load(self.db_path + 'uniprot/npy/id2uniprot_matched_dict.npy').item()       
        sift_scores = open(self.db_path + 'sift/org/SIFT_scores_and_info_ensembl66_human.tsv', 'r')
        cur_enspid = None 
        for line in sift_scores:
            if not re.match('#',line):
                lst_line = line.split('\t')
                if cur_enspid != lst_line[0]:
                    if cur_enspid is not None: # close the old file 
                        cur_ensp_file.close()
                        uniprot_id = id2uniprot_matched_dict['ensembl66'].get(cur_enspid,np.nan)
                        if str(uniprot_id) != 'nan':
                            sift = pd.read_csv(self.db_path + 'sift/bygene/' + cur_enspid + '.tsv',header = None,sep = '\t')    
                            sift.columns = ['ensp_id','aa_pos','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','median','n_seq'] 
                            sift['p_vid'] = uniprot_id
                            sift.drop(['median','n_seq'],axis = 1,inplace = True)
                            sift_melt = pd.melt(sift,id_vars=['p_vid','aa_pos'],value_vars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
                            sift_melt.columns = ['p_vid','aa_pos','aa_alt','sift_score']
                            sift_melt.to_csv(self.db_path + 'sift/bygene/' + uniprot_id + '_sift.csv',index = None)                        
                    cur_enspid = lst_line[0]
                    cur_ensp_file = open(self.db_path + 'sift/bygene/' + cur_enspid + '.tsv','w')
                    cur_ensp_file.write(line)
                else:
                    cur_ensp_file.write(line)
        alm_fun.show_msg(cur_log,self.verbose,'sift data created.\n')
        
    def create_provean_data(self):
        ####***************************************************************************************************************************************************************    
        # jcvi FTP
        ####***************************************************************************************************************************************************************
        # provean scores
        # ftp://ftp.jcvi.org/pub/data/provean/precomputed_scores/PROVEAN_scores_ensembl66_human.tsv.gz   
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('provean')  
        self.jcvi_ftp_obj = alm_fun.create_ftp_object('ftp.jcvi.org')
        return_info = alm_fun.download_ftp(self.jcvi_ftp_obj, '/pub/data/provean/precomputed_scores/', 'PROVEAN_scores_ensembl66_human.tsv.gz', self.db_path + 'provean/org/PROVEAN_scores_ensembl66_human.tsv.gz') 
        if (return_info == 'updated') | (return_info == 'downloaded'):
            alm_fun.gzip_decompress(self.db_path + 'provean/org/PROVEAN_scores_ensembl66_human.tsv.gz', self.db_path + 'provean/org/PROVEAN_scores_ensembl66_human.tsv')  
                              
        id2uniprot_matched_dict = np.load(self.db_path + 'uniprot/npy/id2uniprot_matched_dict.npy').item()       
        provean_scores = open(self.db_path + 'provean/org/PROVEAN_scores_ensembl66_human.tsv', 'r')
        cur_enspid = None 
        for line in provean_scores:
            if not re.match('#',line):
                lst_line = line.split('\t')
                if cur_enspid != lst_line[0]:
                    if cur_enspid is not None: # close the old file 
                        cur_ensp_file.close()
                        uniprot_id = id2uniprot_matched_dict['ensembl66'].get(cur_enspid,np.nan)
                        if str(uniprot_id) != 'nan':
                            provean = pd.read_csv(self.db_path + 'provean/bygene/' + cur_enspid + '.tsv',header = None,sep = '\t')    
                            provean.columns = ['ensp_id','aa_pos','A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Del'] 
                            provean['p_vid'] = uniprot_id
                            provean_melt = pd.melt(provean,id_vars=['p_vid','aa_pos'],value_vars = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','Del'])
                            provean_melt.columns = ['p_vid','aa_pos','aa_alt','provean_score']
                            provean_melt.loc[provean_melt['aa_alt'] == 'Del','aa_alt'] = '*'
                            provean_melt.to_csv(self.db_path + 'provean/bygene/' + uniprot_id + '_provean.csv',index = None)                        
                    cur_enspid = lst_line[0]
                    cur_ensp_file = open(self.db_path + 'provean/bygene/' + cur_enspid + '.tsv','w')
                    cur_ensp_file.write(line)
                else:
                    cur_ensp_file.write(line)  
            
    def create_polyphen_data(self):
        alm_fun.show_msg(self.log,self.verbose,'OK')   
            
    def create_evmutation_data(self):
        
        ####***************************************************************************************************************************************************************
        # EVMutation
        # https://marks.hms.harvard.edu/evmutation/data/effects.tar.gz     
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('evmutation') 
        cur_log = self.db_path + 'evmutation/log/evmutation.log'
        
        if not os.path.isfile(self.db_path + 'evmutation/org/effects.tar.gz'):       
            wget_cmd = "wget https://marks.hms.harvard.edu/evmutation/data/effects.tar.gz"
            subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'evmutation/org/')
            alm_fun.show_msg(cur_log,self.verbose,'evmutation data downloaded.\n')
            
            alm_fun.gzip_decompress(self.db_path + "evmutation/org/effects.tar.gz", self.db_path + "evmutation/org/effects.tar")
            
            gzip_cmd = "tar -xvf " + self.db_path + "evmutation/org/effects.tar"
            subprocess.run(wget_cmd.split(" "), cwd = self.db_path + 'evmutation/org/')

            first = 1
            for evmutation_file in glob.glob(os.path.join(self.db_path + 'evmutation/org/' , '*.csv')):
                cur_evmutation_df = pd.read_csv(evmutation_file, sep=';')
                cur_evmutation_df.columns = ['mutation', 'aa_pos', 'aa_ref', 'aa_alt', 'evm_epistatic_score', 'evm_independent_score', 'evm_frequency', 'evm_conservation']
                 
                # convert to uniprot id 
                filename_splits = evmutation_file.split('/')[-1].split('_')
                cur_uniprot_id = filename_splits[0]
                cur_pfam_id = filename_splits[1]
                cur_evmutation_df['p_vid'] = cur_uniprot_id
                cur_evmutation_df['evm_pfam_id'] = cur_pfam_id
     
                if first == 1:
                    cur_evmutation_df.to_csv(self.db_path + 'evmutation/all/evmutation_processed_df.csv', mode='w', index=None)
                    first = 0
                else:
                    cur_evmutation_df.to_csv(self.db_path + 'evmutation/all/evmutation_processed_df.csv', mode='a', index=None, header=False)
                    
            evmutation_processed_df = pd.read_csv(self.db_path + 'evmutation/all/evmutation_processed_df.csv')[['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'evm_epistatic_score', 'evm_independent_score']]
            evmutation_df = evmutation_processed_df.groupby(['p_vid', 'aa_pos', 'aa_ref', 'aa_alt'])['evm_epistatic_score','evm_independent_score'].mean()
            evmutation_df = evmutation_df.reset_index()
            evmutation_df.to_csv(self.db_path + 'evmutation/all/evmutation_df.csv', index=None)
            alm_fun.show_msg(cur_log,self.verbose,'evmutation data created.\n')
        else:
            alm_fun.show_msg(cur_log,self.verbose,'evmutation data exists already.\n')

    def retrieve_pisa_by_uniprotid_old(self,runtime):
        uniprot_id = runtime['uniprot_id']
        cur_log = cur_log = self.db_path + 'pisa/log/retrieve_pisa_by_uniprotid_old_' + uniprot_id + '.log'
        self.pdb_to_uniprot = pd.read_csv(self.db_path + 'pisa/org/pdb_chain_uniprot.csv', skiprows = 1 ,dtype={"PDB": str})
        
        try: 
#             if os.path.isfile(self.db_path + 'pisa/bygene/' + uniprot_id + '_pisa.csv'):
#                 return(0)
#             if os.path.isfile(self.db_path + 'pisa/org/' + uniprot_id + '_interface.xml'):
#                 return(0)                                 
            cur_gene_pdb = self.pdb_to_uniprot.loc[self.pdb_to_uniprot['SP_PRIMARY'] == uniprot_id, :]
            if cur_gene_pdb.shape[0] > 0 :                   
                #******************************************************
                # 1) Download the pisa xml file 
                #******************************************************
                interface_url = 'http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?'                
                interface_xml_file_path  = self.db_path + 'pisa/org/' + uniprot_id + '_interface.xml'            
                interface_xml_file = open(interface_xml_file_path,'w')                
#                 multimers_ls url = 'http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/multimers.pisa?'
#                 multimers_xml_file = open(self.db_path + 'pdb/org/' + uniprot_id + '_multimers.xml', 'w')
#                 multimers_asa_file = open(self.db_path + 'pdb/org/' + uniprot_id + '_multimers.txt', 'w')    
                                   
                pdb_lst = ','.join(cur_gene_pdb['PDB'].unique())                                                                                
                cur_interface_url = interface_url + pdb_lst
                response = urllib.request.urlopen(cur_interface_url) 
                interface_r = response.read().decode('utf-8')
                interface_xml_file.write(interface_r)
                interface_xml_file.close()
                alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa xml file is downloaded.')
                
                #******************************************************
                # 2) Generate molecule and bond file from xml 
                #******************************************************
                interface_molecule_file_path = self.db_path + 'pisa/bygene/' + uniprot_id + '_molecule.txt'
                interface_bond_file_path = self.db_path + 'pisa/bygene/' + uniprot_id + '_bond.txt'                               
                if (not os.path.isfile(interface_molecule_file_path)) | (not os.path.isfile(interface_bond_file_path)):            
                    interface_molecule_file = open(interface_molecule_file_path, 'w')
                    interface_bond_file = open(interface_bond_file_path, 'w')
                    with open(interface_xml_file_path) as infile:
                        infile_str = infile.read()
                        if len(infile_str) == 0:
                            return(0)
                        interface_tree = ET.fromstring(infile_str)
                        for pdb_entry in interface_tree.iter('pdb_entry'):
                            # print (pdb_entry[0].tag + ':' + pdb_entry[0].text)   #pdb_code
                            for interface in pdb_entry.iter("interface"):
                                # print (interface[0].tag + ':' + interface[0].text) #interface id 
                                for h_bonds in interface.iter("h-bonds"):
                                    for bond in h_bonds.iter("bond"):
                                        interface_bond_file.write('H\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n') 
                                        interface_bond_file.write('H\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n') 
                                for salt_bridges in interface.iter("salt-bridges"):
                                    for bond in salt_bridges.iter("bond"):
                                        interface_bond_file.write('S\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')
                                        interface_bond_file.write('S\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')                                   
                                for ss_bonds in interface.iter("ss-bonds"):
                                    for bond in ss_bonds.iter("bond"):
                                        interface_bond_file.write('D\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')
                                        interface_bond_file.write('D\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')                                                  
                                for cov_bonds in interface.iter("cov-bonds"):
                                    for bond in cov_bonds.iter("bond"):
                                        interface_bond_file.write('C\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')
                                        interface_bond_file.write('C\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')                                    
                                for molecule in interface.iter("molecule"):
                                    # print(molecule[1].tag +':' + molecule[1].text) #chain_id
                                    for residue in molecule.iter("residue"):
                                        # print (residue[0].tag + ':' + residue[0].text + '|' + residue[1].tag + ':' + residue[1].text +'|' + residue[5].tag + ':' + residue[5].text)
                                        interface_molecule_file.write(str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + str(molecule[1].text) + '\t' + str(residue[0].text) + '\t' + str(residue[1].text) + '\t' + str(residue[2].text) + '\t' + str(residue[3].text) + '\t' + str(residue[4].text) + '\t' + str(residue[5].text) + '\t' + str(residue[6].text) + '\t' + str(residue[7].text) + '\n')
                    interface_bond_file.close()                    
                    interface_molecule_file.close()
                    alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa xml file is processed.')
                #******************************************************
                # 3) Generate pisa file for each gene  
                #******************************************************                        
                if os.path.isfile(interface_molecule_file_path):
                    if  (os.stat(interface_molecule_file_path).st_size != 0):                                             
                        cur_molecule_df = pd.read_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_molecule.txt', sep='\t', header=None, dtype = object)
                        cur_molecule_df.columns = ['PDB', 'interface', 'CHAIN', 'ser_no', 'residue', 'aa_pos', 'ins_code','bonds','asa','bsa','solv_ne']
                        cur_molecule_df['asa'] = cur_molecule_df['asa'].astype(float)
                        cur_molecule_df['bsa'] = cur_molecule_df['bsa'].astype(float)
                        cur_molecule_df['solv_ne'] = cur_molecule_df['solv_ne'].astype(float)
                        cur_molecule_df['aa_pos'] = cur_molecule_df['aa_pos'].astype(int)
                        
                        if (os.stat(interface_bond_file_path).st_size != 0): 
                            cur_bond_df = pd.read_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_bond.txt', sep='\t', header=None, dtype = object)
                            cur_bond_df.columns = ['bond','CHAIN', 'residue', 'aa_pos', 'ins_code','atom','PDB','interface']  
                            cur_bond_df['aa_pos'] = cur_bond_df['aa_pos'].astype(int)
                            
                            cur_bond_df = cur_bond_df.groupby(['bond','CHAIN','residue','aa_pos','ins_code','PDB','interface'])['atom'].agg('count').reset_index()                                                        
                            cur_bond_df['h_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'H' else 0, axis = 1)
                            cur_bond_df['salt_bridge']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'S' else 0, axis = 1)
                            cur_bond_df['disulfide_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'D' else 0, axis = 1)
                            cur_bond_df['covelent_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'C' else 0, axis = 1)                                        
                            cur_bond_df.drop(columns = ['atom','bond'],inplace = True)
                            cur_bond_df = cur_bond_df.groupby(['CHAIN','residue','aa_pos','ins_code','PDB','interface'])['h_bond','salt_bridge','disulfide_bond','covelent_bond'].agg('sum').reset_index()
                            cur_molecule_df = cur_molecule_df.merge(cur_bond_df,how = 'left')
                        else:
                            cur_molecule_df['h_bond'] = 0
                            cur_molecule_df['salt_bridge'] = 0
                            cur_molecule_df['disulfide_bond'] = 0 
                            cur_molecule_df['covelent_bond'] = 0
                
                        cur_molecule_df['bsa_ratio'] = 0
                        cur_molecule_df.loc[cur_molecule_df['asa'] !=0, 'bsa_ratio'] = cur_molecule_df.loc[cur_molecule_df['asa'] !=0, 'bsa'] / cur_molecule_df.loc[cur_molecule_df['asa'] != 0, 'asa']
                                                  
                        cur_molecule_df_groupby = cur_molecule_df.groupby(['residue', 'aa_pos'])
                        cur_pisa_value_df1 = cur_molecule_df_groupby['asa'].agg(['mean', 'std', 'count']).reset_index().sort_values(['aa_pos'])
                        cur_pisa_value_df2 = cur_molecule_df_groupby['bsa','bsa_ratio','solv_ne','h_bond','salt_bridge','disulfide_bond','covelent_bond'].agg('max').reset_index().sort_values(['aa_pos'])
                        cur_pisa_value_df3 = cur_molecule_df_groupby['solv_ne'].agg('min').reset_index().sort_values(['aa_pos'])
                            
                        cur_pisa_value_df1.columns = ['residue', 'aa_pos', 'asa_mean', 'asa_std', 'asa_count']   
                        cur_pisa_value_df2.columns = ['residue', 'aa_pos', 'bsa_max', 'solv_ne_max','bsa_ratio_max', 'h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max']
                        cur_pisa_value_df3.columns = ['residue', 'aa_pos', 'solv_ne_min']
                        
                        cur_pisa_df = cur_pisa_value_df1.merge(cur_pisa_value_df2,how = 'left')
                        cur_pisa_df = cur_pisa_df.merge(cur_pisa_value_df3,how = 'left')
                         
                        cur_pisa_df['aa_ref'] = cur_pisa_df['residue'].apply(lambda x: self.dict_aa3_upper.get(x, np.nan))
                        cur_pisa_df = cur_pisa_df.loc[cur_pisa_df['aa_ref'].notnull(), ]
                        cur_pisa_df['p_vid'] = uniprot_id
                        cur_pisa_df.drop(['residue'],axis = 1,inplace = True)
                        cur_pisa_df = cur_pisa_df.fillna(0)
                
                        cur_pisa_df.columns = ['aa_pos','asa_mean','asa_std','asa_count','bsa_max','solv_ne_max','bsa_ratio_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min','aa_ref','p_vid']    
                        cur_pisa_df.to_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_pisa.csv', index=False)
                        alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa csv file is generated.')                
            else:
                return(0)    
        except:
            alm_fun.show_msg(cur_log, 1, uniprot_id + traceback.format_exc() + '\n')            
            return(0)             
        
    def retrieve_uniprot_to_pdb_by_uniprotids(self,runtime):
        cur_log = self.project_path + 'output/log/' + 'retrieve_uniprot_to_pdb_by_uniprotids_'+ runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/' + 'retrieve_uniprot_to_pdb_by_uniprotids_' + runtime['varity_batch_id'] + '_done.log'
        uniprot_ids = runtime['uniprot_ids'] 
        for uniprot_id in uniprot_ids:
#            ******************************************************
#            * Download uniprot to pdb xml 
#            ******************************************************            
            gql_url = 'https://1d-coordinates.rcsb.org/graphql?query=%7B%0A%20%20alignment(%0A%20%20%20%20from:UNIPROT,%0A%20%20%20%20to:' + \
            'PDB_INSTANCE,%0A%20%20%20%20queryId:%22' + \
            uniprot_id + \
            '%22%0A%20%20)%7B%0A%20%20%20%20query_sequence%0A%20%20%20%20' +\
            'target_alignment%20%7B%0A%20%20%20%20%20%20target_id%0A%20%20%20%20%20%20target_sequence%0A%20%20%20%20%20%20' + \
            'coverage%7B%0A%20%20%20%20%20%20%20%20query_coverage%0A%20%20%20%20%20%20%20%20' + \
            'query_length%0A%20%20%20%20%20%20%20%20target_coverage%0A%20%20%20%20%20%20%20%20' + \
            'target_length%0A%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20aligned_regions%20%7B%0A%20%20%20%20%20%20%20%20' + \
            'query_begin%0A%20%20%20%20%20%20%20%20query_end%0A%20%20%20%20%20%20%20%20target_begin%0A%20%20%20%20%20%20%20%20' + \
            'target_end%0A%20%20%20%20%20%20%7D%0A%20%20%20%20%7D%0A%20%20%7D%0A%7D%0A'
                                                    
            try:
                response = urllib.request.urlopen(gql_url)
                gql_r = response.read().decode('utf-8') 
                json_file_path  = self.db_path + 'pisa/bygene/' + uniprot_id + '_pdb.json'
                json_file = open(json_file_path,'w')                           
                json_file.write(gql_r)
                json_file.close()
                alm_fun.show_msg(cur_log, 1, uniprot_id + ' gql json file is downloaded.')                        
                break 
            except Exception as e:
                alm_fun.show_msg(cur_log,self.verbose, str(e))
                time.sleep(5)              

            else:
                alm_fun.show_msg(cur_log, 1, uniprot_id + ' gql json file exists.')
              
        alm_fun.show_msg(cur_log,self.verbose, 'retrieve_uniprot_to_pdb_by_uniprotids_' + runtime['varity_batch_id'] + ' is done.')
        alm_fun.show_msg(cur_done_log,self.verbose, 'retrieve_uniprot_to_pdb_by_uniprotids_' + runtime['varity_batch_id'] + ' is done.')
                    
    def retrieve_pisa_by_uniprotids(self,runtime):        
        cur_log = self.project_path + 'output/log/' + 'retrieve_pisa_by_uniprotids_'+ runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/' + 'retrieve_pisa_by_uniprotids_' + runtime['varity_batch_id'] + '_done.log'
        uniprot_ids = runtime['uniprot_ids']
        pdb_to_uniprot = pd.read_csv(self.db_path + 'pisa/org/pdb_chain_uniprot_processed.csv',dtype={"PDB": str})                                       
                    
        for uniprot_id in uniprot_ids:
#            ******************************************************
#            * Download the pisa xml file 
#            ******************************************************            
            cur_gene_pdb = pdb_to_uniprot.loc[pdb_to_uniprot['SP_PRIMARY'] == uniprot_id, :]                            
            interface_url = 'http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?'                
            interface_xml_file_path  = self.db_path + 'pisa/bygene/' + uniprot_id + '_interface.xml'
            run_xml = 1 
            if os.path.isfile(interface_xml_file_path):
                if os.stat(interface_xml_file_path).st_size != 0:
                    run_xml = 0
            if run_xml == 1:                                               
                pdb_lst = ','.join(cur_gene_pdb['PDB'].unique())                                                                                
                cur_interface_url = interface_url + pdb_lst                
                while True:
                    try:
                        response = urllib.request.urlopen(cur_interface_url)
                        interface_r = response.read().decode('utf-8') 
                        interface_xml_file = open(interface_xml_file_path,'w')                           
                        interface_xml_file.write(interface_r)
                        interface_xml_file.close()
                        alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa xml file is downloaded.')                        
                        break 
                    except:
                        time.sleep(5)              

            else:
                alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa xml file exists.')
              
        alm_fun.show_msg(cur_log,self.verbose, 'retrieve_pisa_by_uniprotids_' + runtime['varity_batch_id'] + ' is done.')
        alm_fun.show_msg(cur_done_log,self.verbose, 'retrieve_pisa_by_uniprotids_' + runtime['varity_batch_id'] + ' is done.')
                
          
    def create_pisa_data(self):
        ####***************************************************************************************************************************************************************    
        # EBI FTP
        ####***************************************************************************************************************************************************************
        # Uniprot ID to PDB ids 
        # (ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz)       
        ####***************************************************************************************************************************************************************
#         self.init_humamdb_object('pisa') 
#         cur_log = self.db_path + 'pisa/log/pisa.log' 
#         self.ebi_ftp_obj = alm_fun.create_ftp_object('ftp.ebi.ac.uk')
#         return_info = alm_fun.download_ftp(self.ebi_ftp_obj, '/pub/databases/msd/sifts/flatfiles/csv/', 'pdb_chain_uniprot.csv.gz', self.db_path + 'pisa/org/pdb_chain_uniprot.csv.gz')
#         if (return_info == 'updated') | (return_info == 'downloaded'):        
#             alm_fun.gzip_decompress(self.db_path + 'pisa/org/pdb_chain_uniprot.csv.gz', self.db_path + 'pisa/org/pdb_chain_uniprot.csv')  
#         alm_fun.show_msg(self.log,self.verbose,'Initiated pisa data.')
#                 
        ####**************************************************************************************************************************************************************                
        uniprot_human_reviewed_ids = list(np.load(self.db_path + 'uniprot/npy/uniprot_human_reviewed_ids.npy'))
        
        #### get uniprot to pdb mapping from HUMAN_9606_idmapping_selected.tab
        
#         uniprot_id_maps = pd.read_csv(self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab', sep='\t', header=None) 
#         uniprot_id_maps.columns = ['UniProtKB-AC', 'UniProtKB-ID', 'GeneID', 'RefSeq', 'GI', 'PDB', 'GO', 'UniRef100', 'UniRef90', 'UniRef50', 'UniParc', 'PIR', 'NCBI-taxon', 'MIM', 'UniGene', 'PubMed', 'EMBL', 'EMBL-CDS', 'Ensembl', 'Ensembl_TRS', 'Ensembl_PRO', 'Additional PubMed']
#         uniprot_id_maps = uniprot_id_maps.loc[uniprot_id_maps['UniProtKB-AC'].isin(uniprot_human_reviewed_ids),:] 
#         uniprot_to_pdb = pd.DataFrame(columns = ['p_vid','p_name','pdb_id','chain_id'])
#         new_index = 0
#         for i in uniprot_id_maps.index:
#             cur_p_vid  = uniprot_id_maps.loc[i,'UniProtKB-AC']
#             cur_p_name  = uniprot_id_maps.loc[i,'UniProtKB-ID']
#             cur_pdbs = str(uniprot_id_maps.loc[i,'PDB']).split(';')
#             for cur_pdb in cur_pdbs:
#                 if (cur_pdb != '') & (':' in cur_pdb):
#                     cur_pdb_id = cur_pdb.split(':')[0].strip().lower()
#                     cur_chain_id = cur_pdb.split(':')[1]
#                     uniprot_to_pdb.loc[new_index,:] = [cur_p_vid,cur_p_name,cur_pdb_id,cur_chain_id]
#                     new_index += 1                                
#         uniprot_to_pdb.to_csv(self.db_path + 'pisa/all/uniprot_pdb_chain.csv', index = False)
#         uniprot_to_pdb = pd.read_csv(self.db_path + 'pisa/all/uniprot_pdb_chain.csv')  

        #### save list of pdb_ids and list of uniprot_ids
#         pisa_pdb_ids = list(uniprot_to_pdb['pdb_id'].unique())
#         pisa_pdb_ids_df = pd.DataFrame(columns = ['pdb_id'])
#         pisa_pdb_ids_df['pdb_id'] = pisa_pdb_ids   
#         pisa_pdb_ids_df.to_csv(self.db_path + 'pisa/all/pisa_pdb_ids.csv',index = False)
#         pisa_uniprot_ids = list(uniprot_to_pdb['p_vid'].unique())
#         pisa_uniprot_ids_df = pd.DataFrame(columns = ['p_vid'])
#         pisa_uniprot_ids_df['p_vid'] = pisa_uniprot_ids        
#         pisa_uniprot_ids_df.to_csv(self.db_path + 'pisa/all/pisa_uniprot_ids.csv',index = False)
# 
#         
        #### get uniprot to pdb mapping from pdb_chain_uniprot.csv.gz
        pdb_chain_uniprot = pd.read_csv(self.db_path + 'pisa/org/pdb_chain_uniprot.csv',skiprows = 1)
        pdb_chain_uniprot.columns = ['pdb_id','chain_id','p_vid','res_beg','res_end','pdb_beg','pdb_end','uniprot_beg','uniprot_end']        
        pdb_chain_uniprot = pdb_chain_uniprot.loc[pdb_chain_uniprot['p_vid'].isin(uniprot_human_reviewed_ids),:]
        
        pdb_chain_uniprot.loc[~ pdb_chain_uniprot['pdb_end'].str.lstrip('-').str.replace('.','0').str.isnumeric(),'pdb_end'] = np.nan        
        pdb_chain_uniprot.loc[~ pdb_chain_uniprot['pdb_beg'].str.lstrip('-').str.isnumeric(),'pdb_beg'] = np.nan
        pdb_chain_uniprot['pdb_beg'] = pdb_chain_uniprot['pdb_beg'].astype(float)
        pdb_chain_uniprot['pdb_end'] = pdb_chain_uniprot['pdb_end'].astype(float)
        pdb_chain_uniprot['uniprot_beg'] = pdb_chain_uniprot['uniprot_beg'].astype(float)
        pdb_chain_uniprot['uniprot_end'] = pdb_chain_uniprot['uniprot_end'].astype(float)   
        
        print ('pdb_chain_uniprot genes: '+ str(len(pdb_chain_uniprot['p_vid'].unique())))
                        
        #### only when pdb_beg and pdb_end are both not None
        pdb_chain_uniprot_1 =  pdb_chain_uniprot.loc[pdb_chain_uniprot['pdb_beg'].notnull() & (pdb_chain_uniprot['pdb_end'].notnull()),:]
        print ('pdb_chain_uniprot_1 genes: '+ str(len(pdb_chain_uniprot_1['p_vid'].unique())))
        #### remove pdb_beg and pdb_end both None
        pdb_chain_uniprot_2 =  pdb_chain_uniprot.loc[~(pdb_chain_uniprot['pdb_beg'].isnull() & pdb_chain_uniprot['pdb_end'].isnull()),:]        
        pdb_chain_uniprot_2.loc[pdb_chain_uniprot_2['pdb_beg'].isnull(),'pdb_beg'] = pdb_chain_uniprot_2.loc[pdb_chain_uniprot_2['pdb_beg'].isnull(),'pdb_end'] - pdb_chain_uniprot_2.loc[pdb_chain_uniprot_2['pdb_beg'].isnull(),'uniprot_end'] + pdb_chain_uniprot_2.loc[pdb_chain_uniprot_2['pdb_beg'].isnull(),'uniprot_beg']
        pdb_chain_uniprot_2.loc[pdb_chain_uniprot_2['pdb_end'].isnull(),'pdb_end'] = pdb_chain_uniprot_2.loc[pdb_chain_uniprot_2['pdb_end'].isnull(),'pdb_beg'] + pdb_chain_uniprot_2.loc[pdb_chain_uniprot_2['pdb_end'].isnull(),'uniprot_end'] - pdb_chain_uniprot_2.loc[pdb_chain_uniprot_2['pdb_end'].isnull(),'uniprot_beg']
        print ('pdb_chain_uniprot_2 genes: '+ str(len(pdb_chain_uniprot_2['p_vid'].unique())))
        
        
        pisa_pdb_ids = list(pdb_chain_uniprot['pdb_id'].unique())
        pisa_pdb_ids_df = pd.DataFrame(columns = ['pdb_id'])
        pisa_pdb_ids_df['pdb_id'] = pisa_pdb_ids   
        pisa_pdb_ids_df.to_csv(self.db_path + 'pisa/all/pisa_pdb_ids.csv',index = False)
        pisa_uniprot_ids = list(pdb_chain_uniprot['p_vid'].unique())
        pisa_uniprot_ids_df = pd.DataFrame(columns = ['p_vid'])
        pisa_uniprot_ids_df['p_vid'] = pisa_uniprot_ids        
        pisa_uniprot_ids_df.to_csv(self.db_path + 'pisa/all/pisa_uniprot_ids.csv',index = False)
        pisa_uniprot_ids_df.to_csv(self.db_path + 'pisa/all/pisa3_uniprot_ids.csv',index = False)       
        
        pisa_pdb_ids = list(pdb_chain_uniprot_1['pdb_id'].unique())
        pisa_pdb_ids_df = pd.DataFrame(columns = ['pdb_id'])
        pisa_pdb_ids_df['pdb_id'] = pisa_pdb_ids   
        pisa_pdb_ids_df.to_csv(self.db_path + 'pisa/all/pisa1_pdb_ids.csv',index = False)
        pisa_uniprot_ids = list(pdb_chain_uniprot_1['p_vid'].unique())
        pisa_uniprot_ids_df = pd.DataFrame(columns = ['p_vid'])
        pisa_uniprot_ids_df['p_vid'] = pisa_uniprot_ids        
        pisa_uniprot_ids_df.to_csv(self.db_path + 'pisa/all/pisa1_uniprot_ids.csv',index = False)       
        
        pisa_pdb_ids = list(pdb_chain_uniprot_2['pdb_id'].unique())
        pisa_pdb_ids_df = pd.DataFrame(columns = ['pdb_id'])
        pisa_pdb_ids_df['pdb_id'] = pisa_pdb_ids   
        pisa_pdb_ids_df.to_csv(self.db_path + 'pisa/all/pisa2_pdb_ids.csv',index = False)
        pisa_uniprot_ids = list(pdb_chain_uniprot['p_vid'].unique())
        pisa_uniprot_ids_df = pd.DataFrame(columns = ['p_vid'])
        pisa_uniprot_ids_df['p_vid'] = pisa_uniprot_ids        
        pisa_uniprot_ids_df.to_csv(self.db_path + 'pisa/all/pisa2_uniprot_ids.csv',index = False)       
                
        pdb_chain_uniprot.to_csv(self.db_path + 'pisa/all/pdb_chain_uniprot_reviewed_pisa.csv')
        pdb_chain_uniprot.to_csv(self.db_path + 'pisa/all/pdb_chain_uniprot_reviewed_pisa3.csv')        
        pdb_chain_uniprot_1.to_csv(self.db_path + 'pisa/all/pdb_chain_uniprot_reviewed_pisa1.csv')
        pdb_chain_uniprot_2.to_csv(self.db_path + 'pisa/all/pdb_chain_uniprot_reviewed_pisa2.csv')


        
    def retrieve_dbref_by_pdbids(self,runtime):
        cur_log = self.project_path + 'output/log/' + 'retrieve_dbref_by_pdbids_'+ runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/' + 'retrieve_dbref_by_pdbids_' + runtime['varity_batch_id'] + '_done.log'
        pdb_ids = runtime['pdb_ids']
                                                                
        for pdb_id in pdb_ids:
#            ******************************************************
#            * Download pdb and get dbref
#            ******************************************************                                     
            cur_pdb_url = 'https://files.rcsb.org/view/' + pdb_id + '.pdb'  
            pdb_file_path = self.db_path + 'pisa/bypdb/' + pdb_id + '_dbref.txt'                             
            while True:
                try:
                    response = urllib.request.urlopen(cur_pdb_url)
                    pdb_r = response.read().decode('utf-8') 
                    pdb_file = open(pdb_file_path,'w')                                                                                      
                    pdb_file.write(pdb_r)
                    pdb_file.close()
                    alm_fun.show_msg(cur_log, 1, pdb_id + ' pdb file is downloaded.')                        
                    break 
                except Exception as e:
                    if str(e) == 'HTTP Error 404: Not Found':
                        alm_fun.show_msg(cur_log, 1, pdb_id + ' pdb file not found.')                           
                        break
                    else:
                        alm_fun.show_msg(cur_log, 1, pdb_id + ' error: [' + str(e) + '], try after 10 secondes......')
                        time.sleep(10)              

            else:
                alm_fun.show_msg(cur_log, 1, pdb_id + ' pdb file exists.')
              
        alm_fun.show_msg(cur_log,self.verbose, 'retrieve_dbref_by_pdbids_' + runtime['varity_batch_id'] + ' is done.')
        alm_fun.show_msg(cur_done_log,self.verbose, 'retrieve_dbref_by_pdbids_' + runtime['varity_batch_id'] + ' is done.')
    
    def process_dbref_for_all_pdbids(self,runtime):
        cur_log = self.project_path + 'output/log/' + 'process_dbref_for_all_pdbids.log'
#         dbref_all_file = open(self.db_path + 'pisa/all/all_dbref.txt','w')
#         for pdb_file in glob.glob(self.db_path + 'pisa/bypdb/*_dbref.txt'): 
#             cur_pdb_id = pdb_file.split('/')[-1].split('_')[0]                    
#             alm_fun.show_msg(cur_log,self.verbose, cur_pdb_id)
#             dbref_start = 0            
#             for line in  open(pdb_file,'r'):
#                 if re.match('^DBREF',line):
#                     dbref_start = 1                    
#                     dbref_all_file.write(re.sub("\s+", ",", line.strip()) + ',' + cur_pdb_id + '\n')
#                 else:
#                     if dbref_start == 1:
#                         break
#         dbref_all_file.close()
#         alm_fun.show_msg(cur_log,self.verbose, 'process_dbref_for_all_pdbids is done.')  
         
        dbref_df = pd.read_csv(self.db_path + 'pisa/all/all_dbref.txt',header = None)        
        dbref_df.columns = ['DBREF','pdb_id','chain_id','pdb_beg','pdb_end','deref_type','p_vid_dbref','p_name_dbref','uniprot_beg','uniprot_end','input_pdb_id']
        dbref_df = dbref_df[['pdb_id','chain_id','p_vid_dbref','deref_type','p_name_dbref','pdb_beg','pdb_end','uniprot_beg','uniprot_end','input_pdb_id']]
        dbref_df.loc[~ dbref_df['pdb_end'].str.lstrip('-').str.replace('.','0').str.isnumeric(),'pdb_end'] = np.nan        
        dbref_df.loc[~ dbref_df['pdb_beg'].str.lstrip('-').str.isnumeric(),'pdb_beg'] = np.nan
        dbref_df['pdb_beg'] = dbref_df['pdb_beg'].astype(float)
        dbref_df['pdb_end'] = dbref_df['pdb_end'].astype(float)
        dbref_df['pdb_id'] = dbref_df['pdb_id'].apply(lambda x: x.lower())            
        dbref_df = dbref_df.drop_duplicates()     
                            
        uniprot_pdb_chain = pd.read_csv(self.db_path + 'pisa/all/uniprot_pdb_chain.csv')
        alm_fun.show_msg(cur_log,self.verbose, 'total uniprot_pdb_chain records is ' + str(uniprot_pdb_chain.shape[0]))      
        uniprot_pdb_chain = pd.merge(uniprot_pdb_chain,dbref_df,how = 'left' )
        alm_fun.show_msg(cur_log,self.verbose, 'total records after merging DBREF records is ' + str(uniprot_pdb_chain.shape[0]))
        uniprot_human_reviewed_ids = list(np.load(self.db_path + 'uniprot/npy/uniprot_human_reviewed_ids.npy'))
        uniprot_pdb_chain['p_vid_reviewed'] = uniprot_pdb_chain['p_vid'].isin(uniprot_human_reviewed_ids)
        uniprot_pdb_chain['p_vid_dbref_reviewed'] = uniprot_pdb_chain['p_vid_dbref'].isin(uniprot_human_reviewed_ids)
        uniprot_pdb_chain.to_csv(self.db_path + 'pisa/all/uniprot_pdb_chain_dbref_before_filter.csv',index = False)           
        

        uniprot_pdb_chain = uniprot_pdb_chain.loc[uniprot_pdb_chain['pdb_beg'].notnull() & uniprot_pdb_chain['pdb_end'].notnull() & uniprot_pdb_chain['p_name_dbref'].notnull(),:]
        alm_fun.show_msg(cur_log,self.verbose, 'total records after removing invalid PDB coordinates  is ' + str(uniprot_pdb_chain.shape[0])) 
       
        uniprot_pdb_chain = uniprot_pdb_chain.loc[uniprot_pdb_chain['p_name_dbref'].str.contains("HUMAN",na = False),:]
        uniprot_pdb_chain = uniprot_pdb_chain.loc[uniprot_pdb_chain['deref_type'] == 'UNP',:]        
        alm_fun.show_msg(cur_log,self.verbose, 'total records after removing non-human mapping is ' + str(uniprot_pdb_chain.shape[0]))

        
        uniprot_pdb_chain = uniprot_pdb_chain.loc[~(uniprot_pdb_chain['p_vid_reviewed'] & uniprot_pdb_chain['p_vid_dbref_reviewed'] & (uniprot_pdb_chain['p_vid'] != uniprot_pdb_chain['p_vid_dbref'])),:]
        alm_fun.show_msg(cur_log,self.verbose, 'total records after removing cases (one chain mutiple poteins) is ' + str(uniprot_pdb_chain.shape[0]))
                           
        uniprot_pdb_chain_group = uniprot_pdb_chain.groupby(['p_vid','pdb_id','chain_id'])['pdb_beg'].agg('count').reset_index()                   
        uniprot_pdb_chain_group.columns = ['p_vid','pdb_id','chain_id','count']                     
        uniprot_pdb_chain = pd.merge(uniprot_pdb_chain,uniprot_pdb_chain_group)

        print(str(uniprot_pdb_chain.loc[uniprot_pdb_chain['count'] > 1,:].shape[0]))
        
        uniprot_pdb_chain.to_csv(self.db_path + 'pisa/all/uniprot_pdb_chain_dbref.csv',index = False)           
        
        print('OK')
             
    def retrieve_pisa_by_pdbids(self,runtime):
        cur_log = self.project_path + 'output/log/' + 'retrieve_pisa_by_pdbids_'+ runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/' + 'retrieve_pisa_by_pdbids_' + runtime['varity_batch_id'] + '_done.log'
        pdb_ids = runtime['pdb_ids']
                                                                
        for pdb_id in pdb_ids:
#            ******************************************************
#            * Download pdb and get dbref
#            ******************************************************                                     
            cur_pisa_url = 'http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/interfaces.pisa?' + pdb_id
            pisa_file_path = self.db_path + 'pisa/bypdb/' + pdb_id + '_interface.xml'                             
            while True:
                try:
                    response = urllib.request.urlopen(cur_pisa_url)
                    pisa_r = response.read().decode('utf-8') 
                    pisa_file = open(pisa_file_path,'w')                                                                                      
                    pisa_file.write(pisa_r)
                    pisa_file.close()
                    alm_fun.show_msg(cur_log, 1, pdb_id + ' pisa file is downloaded.')                        
                    break 
                except Exception as e:
                    if str(e) == 'HTTP Error 404: Not Found':
                        alm_fun.show_msg(cur_log, 1, pdb_id + ' pisa file not found.')                            
                        break
                    else:
                        alm_fun.show_msg(cur_log, 1, pdb_id + ' error: [' + str(e) + '], try after 10 secondes......')
                        time.sleep(10)                

            else:
                alm_fun.show_msg(cur_log, 1, pdb_id + ' pisa file exists.')
              
        alm_fun.show_msg(cur_log,self.verbose, 'retrieve_pisa_by_pdbids_' + runtime['varity_batch_id'] + ' is done.')
        alm_fun.show_msg(cur_done_log,self.verbose, 'retrieve_pisa_by_pdbids_' + runtime['varity_batch_id'] + ' is done.')
              
    def process_pisa_by_uniprotids(self,runtime):        
        cur_log = self.project_path + 'output/log/' + 'process_pisa_by_uniprotids_'+ runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/' + 'process_pisa_by_uniprotids_' + runtime['varity_batch_id'] + '_done.log'
        uniprot_ids = runtime['uniprot_ids']
        pdb_to_uniprot = pd.read_csv(self.db_path + 'pisa/all/pdb_chain_uniprot_reviewed_' + runtime['pisa_folder'] + '.csv')   
        uniprot_seq_dict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy').item()
        aligner = Align.PairwiseAligner()  
        aligner.mode = 'local'  
        aligner.open_gap_score = -0.5
        aligner.extend_gap_score = -0.1
        aligner.target_end_gap_score = 0.0
        aligner.query_end_gap_score = 0.0   
        aligner.mismatch_score = -10                                       
        for uniprot_id in uniprot_ids:            
            if os.path.isfile(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_pisa.csv'):
                continue            
                            
            cur_gene_pdb = pdb_to_uniprot.loc[pdb_to_uniprot['p_vid'] == uniprot_id, :]
            if cur_gene_pdb.shape[0] == 0:
                continue
            
            cur_gene_pdb['enable'] = 1          
            cur_gene_pdb['pdb_id'] = cur_gene_pdb['pdb_id'].apply(lambda x: x.upper())   
            #******************************************************
            #* Generate molecule and bond file from xml 
            #******************************************************
            interface_molecule_file_path = self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_molecule.txt'
            interface_bond_file_path = self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_bond.txt'
            interface_molecule_file = open(interface_molecule_file_path, 'w')
            interface_bond_file = open(interface_bond_file_path, 'w')            
            
            for pdb_id in list(cur_gene_pdb['pdb_id'].unique()):
                interface_xml_file_path = self.db_path + 'pisa/bypdb/' + pdb_id.lower() + '_interface.xml'                                           
                xml_file_exist = 0
                if os.path.isfile(interface_xml_file_path):
                    if os.stat(interface_xml_file_path).st_size != 0:       
                        xml_file_exist = 1
                if xml_file_exist == 1:
                    alm_fun.show_msg(cur_log, 1, "Processing " + pdb_id + ' pisa xml file...... ')
                    with open(interface_xml_file_path) as infile:
                        infile_str = infile.read()
                        if len(infile_str) == 0:
                            return(0)
                        interface_tree = ET.fromstring(infile_str)
                        for pdb_entry in interface_tree.iter('pdb_entry'):
                            # print (pdb_entry[0].tag + ':' + pdb_entry[0].text)   #pdb_code
                            for interface in pdb_entry.iter("interface"):
                                # print (interface[0].tag + ':' + interface[0].text) #interface id 
                                for h_bonds in interface.iter("h-bonds"):
                                    for bond in h_bonds.iter("bond"):
                                        interface_bond_file.write('H\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + '1' + '\n') 
                                        interface_bond_file.write('H\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + '2' + '\n') 
                                for salt_bridges in interface.iter("salt-bridges"):
                                    for bond in salt_bridges.iter("bond"):
                                        interface_bond_file.write('S\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + '1' + '\n') 
                                        interface_bond_file.write('S\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + '2' + '\n')                                    
                                for ss_bonds in interface.iter("ss-bonds"):
                                    for bond in ss_bonds.iter("bond"):
                                        interface_bond_file.write('D\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + '1' + '\n') 
                                        interface_bond_file.write('D\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + '2' + '\n')                                                   
                                for cov_bonds in interface.iter("cov-bonds"):
                                    for bond in cov_bonds.iter("bond"):
                                        interface_bond_file.write('C\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + '1' + '\n') 
                                        interface_bond_file.write('C\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + '2' + '\n')                                     
                                for molecule in interface.iter("molecule"):
                                    # print(molecule[1].tag +':' + molecule[1].text) #chain_id                                    
                                        for residue in molecule.iter("residue"):
                                            # print (residue[0].tag + ':' + residue[0].text + '|' + residue[1].tag + ':' + residue[1].text +'|' + residue[5].tag + ':' + residue[5].text)
                                            interface_molecule_file.write(str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' +  str(molecule[0].text) + '\t' + str(molecule[1].text) + '\t' + str(residue[0].text) + '\t' + str(residue[1].text) + '\t' + str(residue[2].text) + '\t' + str(residue[3].text) + '\t' + str(residue[4].text) + '\t' + str(residue[5].text) + '\t' + str(residue[6].text) + '\t' + str(residue[7].text) + '\n')
            interface_bond_file.close()                    
            interface_molecule_file.close()
            alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa xml file is processed.')
            #******************************************************
            # Generate pisa file for each gene  
            #******************************************************   
            cur_molecule_df = pd.DataFrame()
            cur_bond_df = pd.DataFrame()   
            
            ##Read molecule file            
            if os.path.isfile(interface_molecule_file_path):
                if  (os.stat(interface_molecule_file_path).st_size != 0):                                             
                    cur_molecule_df = pd.read_csv(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_molecule.txt', sep='\t', header=None, dtype = object)
                    cur_molecule_df.columns = ['pdb_id', 'interface', 'molecule', 'chain_id', 'ser_no', 'residue', 'pdb_aa_pos', 'ins_code','bonds','asa','bsa','solv_ne']
                    
            alm_fun.show_msg(cur_log, 1, 'Total number of records: ' + str(cur_molecule_df.shape[0]))                                        
            ##Read bond file and combine with molecule file 
            if os.path.isfile(interface_bond_file_path):                        
                if (os.stat(interface_bond_file_path).st_size != 0): 
                    cur_bond_df = pd.read_csv(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_bond.txt', sep='\t', header=None, dtype = object)
                    cur_bond_df.columns = ['bond','chain_id', 'residue', 'pdb_aa_pos', 'ins_code','atom','pdb_id','interface','molecule']
                    cur_bond_df = cur_bond_df.groupby(['bond','chain_id','residue','pdb_aa_pos','ins_code','pdb_id','interface','molecule'])['atom'].agg('count').reset_index()                                                        
                    cur_bond_df['h_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'H' else 0, axis = 1)
                    cur_bond_df['salt_bridge']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'S' else 0, axis = 1)
                    cur_bond_df['disulfide_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'D' else 0, axis = 1)
                    cur_bond_df['covelent_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'C' else 0, axis = 1)                                        
                    cur_bond_df.drop(columns = ['atom','bond'],inplace = True)
                    cur_bond_df = cur_bond_df.groupby(['chain_id','residue','pdb_aa_pos','ins_code','pdb_id','interface','molecule'])['h_bond','salt_bridge','disulfide_bond','covelent_bond'].agg('sum').reset_index()
                    cur_molecule_df = cur_molecule_df.merge(cur_bond_df,how = 'left')
                else:
                    cur_molecule_df['h_bond'] = 0
                    cur_molecule_df['salt_bridge'] = 0
                    cur_molecule_df['disulfide_bond'] = 0 
                    cur_molecule_df['covelent_bond'] = 0
                    
            alm_fun.show_msg(cur_log, 1, 'Total number of records after merging bond info: ' + str(cur_molecule_df.shape[0]))                    
                    
            if cur_molecule_df.shape[0] > 0 :
                ###Only keep the records that in the chain associated with the input uniprot ID, and its corrdiate is in the right range.
                cur_molecule_df = pd.merge(cur_molecule_df,cur_gene_pdb,how = 'left')
                cur_molecule_df = cur_molecule_df.loc[cur_molecule_df['enable'] == 1,:]
#                 cur_molecule_df.to_csv(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_molecule.csv',index = False)
#                 cur_gene_pdb.to_csv(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_to_pdb.csv',index = False)#                 
                alm_fun.show_msg(cur_log, 1, 'Total number of records after only keeping the records for relevant uniprot chain: ' + str(cur_molecule_df.shape[0]))
                
                ##make sure the corrdinates are numeric
                cur_molecule_df['ser_no'] = cur_molecule_df['ser_no'].astype(int)
                cur_molecule_df['pdb_aa_pos'] = cur_molecule_df['pdb_aa_pos'].astype(str)
                cur_molecule_df.loc[~ cur_molecule_df['pdb_aa_pos'].str.lstrip('-').str.replace('.','0').str.isnumeric(),'pdb_aa_pos'] = np.nan                                        
                cur_molecule_df['pdb_aa_pos'] = cur_molecule_df['pdb_aa_pos'].astype(float)                                   
                alm_fun.show_msg(cur_log, 1, 'Total number of records after removing pdb_aa_pos that is not numeric: ' + str(cur_molecule_df.shape[0]))


                ####Determine the coordinates by PDB chain sequence
                if (runtime['pisa_folder'] == 'pisa') | (runtime['pisa_folder'] == 'pisa3'):
                    def keep_chunk(size,t):
                        t0 = t[0]
                        t1 = t[1]             
                        n = len(t0)           
                        
                        cor = [(t1[i][0]+1,t1[i][1],t0[i][0]+1,t0[i][1],t0[i][0]- t1[i][0]) for i in range(n) if t0[i][1] - t0[i][0] >= size]
#                         t0_filtered = [x[0]  for x in t0 if (x[1] - x[0]) >= size]
#                         t1_filtered = [x[0]  for x in t1 if (x[1] - x[0]) >= size]                        
                        return (cor)
                    
                    def get_aa_pos(coordinates, ser_no):
                        aa_pos = np.nan
                        for coordinate in coordinates:
                            if (ser_no >= coordinate[0]) & (ser_no <= coordinate[1]):
                                aa_pos  = ser_no + coordinate[4]
                                
                        return (aa_pos)
                         
                    if cur_molecule_df.shape[0] > 0 :
                        cur_molecule_df['aa_ref'] = cur_molecule_df['residue'].apply(lambda x: self.dict_aa3_20_upper.get(x, 'L')) # L is the most abudent 
                        cur_molecule_df = cur_molecule_df.sort_values(['pdb_id','chain_id','interface','molecule','ser_no'])
                        x = cur_molecule_df.groupby(['pdb_id','chain_id','interface','molecule'])['aa_ref'].agg(list).reset_index()
                        x['pdb_seq'] = x['aa_ref'].apply(lambda x: ''.join(x))
                        x = x.drop(columns = ['aa_ref'])
                        x = x.drop_duplicates()
                        y = cur_molecule_df.groupby(['pdb_id','chain_id','interface','molecule'])['ser_no','pdb_aa_pos'].agg(min).reset_index()
                        y = y.rename(columns = {'ser_no':'ser_no_beg','pdb_aa_pos':'chain_beg'})
                        y = y.drop_duplicates()                
                        z = pd.merge(x,y,how = 'left')
                        z = z.drop(columns = ['interface','molecule'])
                        z = z.drop_duplicates() 
    #                     z.to_csv(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_z.csv',index = False)       
                        uniprot_seq = uniprot_seq_dict[uniprot_id]                      
                        z['coordinates'] = z['pdb_seq'].apply(lambda x: keep_chunk(3,aligner.align(uniprot_seq,x)[0].aligned))                                                        
                        z.to_csv(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_pdb_seq.csv',index = False)
                
                        ##Determine aa_pos
                        cur_molecule_df = cur_molecule_df.merge(z[['pdb_id','chain_id','ser_no_beg','chain_beg','coordinates']],how = 'left')
    #                     cur_molecule_df.to_csv(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_molecule.csv',index = False)
                        cur_molecule_df['aa_pos'] = cur_molecule_df.apply(lambda x: get_aa_pos(x['coordinates'],x['ser_no']),axis = 1) 
                                        
    #                     s = 'SVTCKSGDFSCGGNRCIPQFWRCDGQVDCNGSDEQGCPPKTCSQDEFRCHDGRQFVCDSDRDCLDGSDEASCPVLTCGPASFQCNSSTCIPQLWACDNDPDCEDGSDEWPQRCRGLYVFQGDSSPCSAFEFHCLSGECIHSSWRCDGGPDCKDKSDEENCAVATCRPDEFQCSDGNCIHGSRQCDREYDCKDMSDEVGCVNVTLCEGPNKFKCHSGECITLDKVCNMARDCRDWSDEPIKECGTNECLDNNGGCSHVCNDLKIGYECLCPDGFQLVAQRRCEDIDECQDPDTCSQLCVNLEGGYKCQCEEGFQLDPHTKACKAVGSIAYLFFTNRHERKMTLDRSEYTSLIPNLRNVVALDTEVASNRIYWSDLSQRMICSTQLDRAHGSSYDTVISRDIPDGLAVDWIHSNIYWTDSVLGTVSVADTKGVKRKTLFREQGSKPRAIVVDPVHGFMYWTDWGTPAKIKKGGLNGVDIYSLVTENIQWPNGITLDLLSGRLYWVDSKLHSISSIDVNGGNRKTILEDEKRLAHPFSLAVFEDKVFWTDIINEAIFSANRLTGSDVNLLAENLLSPEDMVLFHQLTQPRGVNWCERTTLSNGGCQYLCLPAPQINPHSPKFTCACPDGMLLARDMRSCLTE'
    # # #                     s = 'GTNECLDNNGGCSHVCNDLKIGYECL'     
    # #                     s = 'SINFDNPVYQKTT'
    #                     alignment = aligner.align(uniprot_seq,s)[0]
    #                     print(alignment) 
    #                     alignment.aligned[0]
    #                     alignment.aligned[1]
    #                     len(s) 
    
                else: # pisa1 , pisa2                   
                    cur_molecule_df = cur_molecule_df.loc[(cur_molecule_df['pdb_aa_pos'] >= cur_molecule_df['pdb_beg']) & (cur_molecule_df['pdb_aa_pos'] <= cur_molecule_df['pdb_end']),:]                
                    alm_fun.show_msg(cur_log, 1, 'Total number of records after only keeping valid coordinates: ' + str(cur_molecule_df.shape[0]))
                    if cur_molecule_df.shape[0] > 0 :            
                        ##Determine aa_pos                                
                        cur_molecule_df['aa_pos'] = cur_molecule_df.apply(lambda x: x['pdb_aa_pos'] + x['uniprot_beg']- x['pdb_beg'] ,axis = 1)
                        
            if cur_molecule_df.shape[0] > 0 :      
                cur_molecule_df = cur_molecule_df.loc[cur_molecule_df['aa_pos'].notnull(),:] # only keep valid aa_pos
                alm_fun.show_msg(cur_log, 1, 'Total number of records after assigning aa_pos: ' + str(cur_molecule_df.shape[0]))
    
            ## create pisa csv
            if cur_molecule_df.shape[0] > 0 :            
      
                cur_molecule_df['asa'] = cur_molecule_df['asa'].astype(float)
                cur_molecule_df['bsa'] = cur_molecule_df['bsa'].astype(float)
                cur_molecule_df['solv_ne'] = cur_molecule_df['solv_ne'].astype(float)
                
                
                #set 0 to nan
                cur_molecule_df.loc[cur_molecule_df['bsa'] == 0,'bsa'] = np.nan
                cur_molecule_df.loc[cur_molecule_df['solv_ne'] == 0,'solv_ne'] = np.nan
                cur_molecule_df.loc[cur_molecule_df['h_bond'] == 0,'h_bond'] = np.nan
                cur_molecule_df.loc[cur_molecule_df['salt_bridge'] == 0,'salt_bridge'] = np.nan
                cur_molecule_df.loc[cur_molecule_df['disulfide_bond'] == 0,'disulfide_bond'] = np.nan
                cur_molecule_df.loc[cur_molecule_df['covelent_bond'] == 0,'covelent_bond'] = np.nan
                
                #absolute value of solv_ne
                cur_molecule_df['solv_ne_abs'] = cur_molecule_df['solv_ne'].apply(lambda x: np.abs(x))
                                    
                cur_molecule_df['ser_no'] = cur_molecule_df['ser_no'].astype(int)                 
                cur_molecule_df['bsa_ratio'] = 0
                cur_molecule_df.loc[cur_molecule_df['asa'] !=0, 'bsa_ratio'] = cur_molecule_df.loc[cur_molecule_df['asa'] !=0, 'bsa'] / cur_molecule_df.loc[cur_molecule_df['asa'] != 0, 'asa']
                                  
                #Create PISA csv                                           
                cur_pisa_value_df1_1 = cur_molecule_df.groupby(['pdb_id','chain_id','residue', 'aa_pos'])['asa'].agg(['mean']).reset_index().sort_values(['aa_pos'])
                cur_pisa_value_df1 = cur_pisa_value_df1_1.groupby(['residue','aa_pos'])['mean'].agg(['mean', 'std', 'count']).reset_index().sort_values(['aa_pos'])
                
                cur_pisa_value_df2 = cur_molecule_df.groupby(['residue', 'aa_pos'])['bsa','bsa_ratio','solv_ne','solv_ne_abs','h_bond','salt_bridge','disulfide_bond','covelent_bond'].agg('max').reset_index().sort_values(['aa_pos'])
                cur_pisa_value_df3 = cur_molecule_df.groupby(['residue', 'aa_pos'])['solv_ne'].agg('min').reset_index().sort_values(['aa_pos'])
                    
                cur_pisa_value_df1.columns = ['residue', 'aa_pos', 'asa_mean', 'asa_std', 'asa_count']   
                cur_pisa_value_df2.columns = ['residue', 'aa_pos', 'bsa_max','bsa_ratio_max', 'solv_ne_max','solv_ne_abs_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max']
                cur_pisa_value_df3.columns = ['residue', 'aa_pos', 'solv_ne_min']
                
                cur_pisa_df = cur_pisa_value_df1.merge(cur_pisa_value_df2,how = 'left')
                cur_pisa_df = cur_pisa_df.merge(cur_pisa_value_df3,how = 'left')
                 
                cur_pisa_df['aa_ref'] = cur_pisa_df['residue'].apply(lambda x: self.dict_aa3_upper.get(x, np.nan))
                cur_pisa_df = cur_pisa_df.loc[cur_pisa_df['aa_ref'].notnull(), ]
                cur_pisa_df['p_vid'] = uniprot_id
                cur_pisa_df.drop(['residue'],axis = 1,inplace = True)
#                 cur_pisa_df = cur_pisa_df.fillna(0)
        
                cur_pisa_df.columns = ['aa_pos','asa_mean','asa_std','asa_count','bsa_max','bsa_ratio_max','solv_ne_max','solv_ne_abs_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min','aa_ref','p_vid']    
                cur_pisa_df.to_csv(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_pisa.csv', index=False)
                cur_molecule_df.to_csv(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_molecule.csv',index = False)
#                 cur_bond_df.to_csv(self.db_path + runtime['pisa_folder'] + '/bygene/' + uniprot_id + '_bond.csv',index = False)
                alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa csv file is generated.')  
                              
        alm_fun.show_msg(cur_log,self.verbose, 'process_pisa_by_uniprotids_' + runtime['varity_batch_id'] + ' is done.')
        alm_fun.show_msg(cur_done_log,self.verbose, 'process_pisa_by_uniprotids_' + runtime['varity_batch_id'] + ' is done.')    
                      
                                  
    def process_pisa_by_uniprotids_old(self,runtime):        
        cur_log = self.project_path + 'output/log/' + 'process_pisa_by_uniprotids_'+ runtime['varity_batch_id'] + '.log'
        cur_done_log = self.project_path + 'output/log/' + 'process_pisa_by_uniprotids_' + runtime['varity_batch_id'] + '_done.log'
        uniprot_ids = runtime['uniprot_ids']

        pdb_to_uniprot = pd.read_csv(self.db_path + 'pisa/all/uniprot_pdb_chain_dbref.csv')               
        for uniprot_id in uniprot_ids:
            
            if os.path.isfile(self.db_path + 'pisa/bygene/' + uniprot_id + '_pisa.csv'):
                continue
            
#             uniprot_seq = uniprot_seq_dict.get(uniprot_id,'')    
            cur_gene_pdb = pdb_to_uniprot.loc[pdb_to_uniprot['p_vid'] == uniprot_id, :]
            cur_gene_pdb['enable'] = 1          
            cur_gene_pdb['pdb_id'] = cur_gene_pdb['pdb_id'].apply(lambda x: x.upper())   
            #******************************************************
            #* Generate molecule and bond file from xml 
            #******************************************************
            interface_molecule_file_path = self.db_path + 'pisa/bygene/' + uniprot_id + '_molecule.txt'
            interface_bond_file_path = self.db_path + 'pisa/bygene/' + uniprot_id + '_bond.txt'
            interface_molecule_file = open(interface_molecule_file_path, 'w')
            interface_bond_file = open(interface_bond_file_path, 'w')            
            
            for pdb_id in list(cur_gene_pdb['pdb_id'].unique()):
                interface_xml_file_path = self.db_path + 'pisa/bypdb/' + pdb_id.lower() + '_interface.xml'                                           
                xml_file_exist = 0
                if os.path.isfile(interface_xml_file_path):
                    if os.stat(interface_xml_file_path).st_size != 0:       
                        xml_file_exist = 1
                if xml_file_exist == 1:
                    alm_fun.show_msg(cur_log, 1, "Processing " + pdb_id + ' pisa xml file...... ')
                    with open(interface_xml_file_path) as infile:
                        infile_str = infile.read()
                        if len(infile_str) == 0:
                            return(0)
                        interface_tree = ET.fromstring(infile_str)
                        for pdb_entry in interface_tree.iter('pdb_entry'):
                            # print (pdb_entry[0].tag + ':' + pdb_entry[0].text)   #pdb_code
                            for interface in pdb_entry.iter("interface"):
                                # print (interface[0].tag + ':' + interface[0].text) #interface id 
                                for h_bonds in interface.iter("h-bonds"):
                                    for bond in h_bonds.iter("bond"):
                                        interface_bond_file.write('H\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n') 
                                        interface_bond_file.write('H\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n') 
                                for salt_bridges in interface.iter("salt-bridges"):
                                    for bond in salt_bridges.iter("bond"):
                                        interface_bond_file.write('S\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')
                                        interface_bond_file.write('S\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')                                   
                                for ss_bonds in interface.iter("ss-bonds"):
                                    for bond in ss_bonds.iter("bond"):
                                        interface_bond_file.write('D\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')
                                        interface_bond_file.write('D\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')                                                  
                                for cov_bonds in interface.iter("cov-bonds"):
                                    for bond in cov_bonds.iter("bond"):
                                        interface_bond_file.write('C\t' + str(bond[0].text) + '\t' + str(bond[1].text) + '\t' + \
                                                                          str(bond[2].text) + '\t' + str(bond[3].text) + '\t' + \
                                                                          str(bond[4].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')
                                        interface_bond_file.write('C\t' + str(bond[5].text) + '\t' + str(bond[6].text) + '\t' + \
                                                                          str(bond[7].text) + '\t' + str(bond[8].text) + '\t' + \
                                                                          str(bond[9].text) + '\t' + str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\n')                                    
                                for molecule in interface.iter("molecule"):
                                    # print(molecule[1].tag +':' + molecule[1].text) #chain_id
                                    for residue in molecule.iter("residue"):
                                        # print (residue[0].tag + ':' + residue[0].text + '|' + residue[1].tag + ':' + residue[1].text +'|' + residue[5].tag + ':' + residue[5].text)
                                        interface_molecule_file.write(str(pdb_entry[0].text) + '\t' + str(interface[0].text) + '\t' + str(molecule[1].text) + '\t' + str(residue[0].text) + '\t' + str(residue[1].text) + '\t' + str(residue[2].text) + '\t' + str(residue[3].text) + '\t' + str(residue[4].text) + '\t' + str(residue[5].text) + '\t' + str(residue[6].text) + '\t' + str(residue[7].text) + '\n')
            interface_bond_file.close()                    
            interface_molecule_file.close()
            alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa xml file is processed.')
            #******************************************************
            # Generate pisa file for each gene  
            #******************************************************   
            cur_molecule_df = pd.DataFrame()
            cur_bond_df = pd.DataFrame()   
            
            ##Read molecule file            
            if os.path.isfile(interface_molecule_file_path):
                if  (os.stat(interface_molecule_file_path).st_size != 0):                                             
                    cur_molecule_df = pd.read_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_molecule.txt', sep='\t', header=None, dtype = object)
                    cur_molecule_df.columns = ['pdb_id', 'interface', 'chain_id', 'ser_no', 'residue', 'pdb_aa_pos', 'ins_code','bonds','asa','bsa','solv_ne']
                    
            alm_fun.show_msg(cur_log, 1, 'Total number of records: ' + str(cur_molecule_df.shape[0]))                                        
            ##Read bond file and combine with molecule file 
            if os.path.isfile(interface_bond_file_path):                        
                if (os.stat(interface_bond_file_path).st_size != 0): 
                    cur_bond_df = pd.read_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_bond.txt', sep='\t', header=None, dtype = object)
                    cur_bond_df.columns = ['bond','chain_id', 'residue', 'pdb_aa_pos', 'ins_code','atom','pdb_id','interface']
                    cur_bond_df = cur_bond_df.groupby(['bond','chain_id','residue','pdb_aa_pos','ins_code','pdb_id','interface'])['atom'].agg('count').reset_index()                                                        
                    cur_bond_df['h_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'H' else 0, axis = 1)
                    cur_bond_df['salt_bridge']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'S' else 0, axis = 1)
                    cur_bond_df['disulfide_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'D' else 0, axis = 1)
                    cur_bond_df['covelent_bond']  = cur_bond_df.apply(lambda x: x['atom'] if x['bond'] == 'C' else 0, axis = 1)                                        
                    cur_bond_df.drop(columns = ['atom','bond'],inplace = True)
                    cur_bond_df = cur_bond_df.groupby(['chain_id','residue','pdb_aa_pos','ins_code','pdb_id','interface'])['h_bond','salt_bridge','disulfide_bond','covelent_bond'].agg('sum').reset_index()
                    cur_molecule_df = cur_molecule_df.merge(cur_bond_df,how = 'left')
                else:
                    cur_molecule_df['h_bond'] = 0
                    cur_molecule_df['salt_bridge'] = 0
                    cur_molecule_df['disulfide_bond'] = 0 
                    cur_molecule_df['covelent_bond'] = 0
                    
            alm_fun.show_msg(cur_log, 1, 'Total number of records after merging bond info: ' + str(cur_molecule_df.shape[0]))                    
                    
            if cur_molecule_df.shape[0] > 0 :
                ###Only keep the records that in the chain associated with the input uniprot ID, and its corrdiate is in the right range.
                cur_molecule_df = pd.merge(cur_molecule_df,cur_gene_pdb,how = 'left')
                cur_molecule_df = cur_molecule_df.loc[cur_molecule_df['enable'] == 1,:]
#                 cur_molecule_df.to_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_molecule.csv',index = False)
#                 cur_gene_pdb.to_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_to_pdb.csv',index = False)#                 
                alm_fun.show_msg(cur_log, 1, 'Total number of records after only keeping the records for relevant uniprot chain: ' + str(cur_molecule_df.shape[0]))
            
                ##make sure the corrdinates are numeric
                cur_molecule_df['pdb_aa_pos'] = cur_molecule_df['pdb_aa_pos'].astype(str)
                cur_molecule_df.loc[~ cur_molecule_df['pdb_aa_pos'].str.lstrip('-').str.replace('.','0').str.isnumeric(),'pdb_aa_pos'] = np.nan                                        
                cur_molecule_df['pdb_aa_pos'] = cur_molecule_df['pdb_aa_pos'].astype(float)                     
                cur_molecule_df['pdb_beg'] = cur_molecule_df['pdb_beg'].astype(str)
                cur_molecule_df.loc[~ cur_molecule_df['pdb_beg'].str.lstrip('-').str.replace('.','0').str.isnumeric(),'pdb_beg'] = np.nan                                        
                cur_molecule_df['pdb_beg'] = cur_molecule_df['pdb_beg'].astype(float)
                cur_molecule_df['pdb_end'] = cur_molecule_df['pdb_end'].astype(str)
                cur_molecule_df.loc[~ cur_molecule_df['pdb_end'].str.lstrip('-').str.replace('.','0').str.isnumeric(),'pdb_end'] = np.nan                                        
                cur_molecule_df['pdb_end'] = cur_molecule_df['pdb_end'].astype(float)                             
                cur_molecule_df = cur_molecule_df.loc[(cur_molecule_df['pdb_aa_pos'] >= cur_molecule_df['pdb_beg']) & (cur_molecule_df['pdb_aa_pos'] <= cur_molecule_df['pdb_end']),:]                
                alm_fun.show_msg(cur_log, 1, 'Total number of records after only keeping valid coordinates: ' + str(cur_molecule_df.shape[0]))
                
            if cur_molecule_df.shape[0] > 0 :            
                ##Determine aa_pos
                cur_molecule_df['aa_pos'] = cur_molecule_df.apply(lambda x: x['pdb_aa_pos'] + x['uniprot_beg']- x['pdb_beg'] ,axis = 1)
                cur_molecule_df = cur_molecule_df.loc[cur_molecule_df['aa_pos'].notnull(),:] # only keep valid aa_pos
                
                alm_fun.show_msg(cur_log, 1, 'Total number of records after assigning aa_pos: ' + str(cur_molecule_df.shape[0]))
    
                cur_molecule_df['asa'] = cur_molecule_df['asa'].astype(float)
                cur_molecule_df['bsa'] = cur_molecule_df['bsa'].astype(float)
                cur_molecule_df['solv_ne'] = cur_molecule_df['solv_ne'].astype(float)                    
                cur_molecule_df['ser_no'] = cur_molecule_df['ser_no'].astype(int)                 
                cur_molecule_df['bsa_ratio'] = 0
                cur_molecule_df.loc[cur_molecule_df['asa'] !=0, 'bsa_ratio'] = cur_molecule_df.loc[cur_molecule_df['asa'] !=0, 'bsa'] / cur_molecule_df.loc[cur_molecule_df['asa'] != 0, 'asa']
                                  
                #Create PISA csv                                   
                cur_molecule_df_groupby = cur_molecule_df.groupby(['residue', 'aa_pos'])
                cur_pisa_value_df1 = cur_molecule_df_groupby['asa'].agg(['mean', 'std', 'count']).reset_index().sort_values(['aa_pos'])
                cur_pisa_value_df2 = cur_molecule_df_groupby['bsa','bsa_ratio','solv_ne','h_bond','salt_bridge','disulfide_bond','covelent_bond'].agg('max').reset_index().sort_values(['aa_pos'])
                cur_pisa_value_df3 = cur_molecule_df_groupby['solv_ne'].agg('min').reset_index().sort_values(['aa_pos'])
                    
                cur_pisa_value_df1.columns = ['residue', 'aa_pos', 'asa_mean', 'asa_std', 'asa_count']   
                cur_pisa_value_df2.columns = ['residue', 'aa_pos', 'bsa_max', 'solv_ne_max','bsa_ratio_max', 'h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max']
                cur_pisa_value_df3.columns = ['residue', 'aa_pos', 'solv_ne_min']
                
                cur_pisa_df = cur_pisa_value_df1.merge(cur_pisa_value_df2,how = 'left')
                cur_pisa_df = cur_pisa_df.merge(cur_pisa_value_df3,how = 'left')
                 
                cur_pisa_df['aa_ref'] = cur_pisa_df['residue'].apply(lambda x: self.dict_aa3_upper.get(x, np.nan))
                cur_pisa_df = cur_pisa_df.loc[cur_pisa_df['aa_ref'].notnull(), ]
                cur_pisa_df['p_vid'] = uniprot_id
                cur_pisa_df.drop(['residue'],axis = 1,inplace = True)
#                 cur_pisa_df = cur_pisa_df.fillna(0)
        
                cur_pisa_df.columns = ['aa_pos','asa_mean','asa_std','asa_count','bsa_max','solv_ne_max','bsa_ratio_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min','aa_ref','p_vid']    
                cur_pisa_df.to_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_pisa.csv', index=False)
                cur_molecule_df.to_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_molecule.csv',index = False)
#                 cur_bond_df.to_csv(self.db_path + 'pisa/bygene/' + uniprot_id + '_bond.csv',index = False)
                alm_fun.show_msg(cur_log, 1, uniprot_id + ' pisa csv file is generated.')  
                              
        alm_fun.show_msg(cur_log,self.verbose, 'process_pisa_by_uniprotids_' + runtime['varity_batch_id'] + ' is done.')
        alm_fun.show_msg(cur_done_log,self.verbose, 'process_pisa_by_uniprotids_' + runtime['varity_batch_id'] + ' is done.')    
            
    def create_ukb_data(self,runtime):
        
        self.init_humamdb_object('ukb')  

    def combine_pisa_data(self,runtime):
        pisa_csv_df = None
        for pisa_csv_file in glob.glob(self.db_path + runtime['pisa_folder'] + '/bygene/*_pisa.csv'):
            cur_pisa_csv_df = pd.read_csv(pisa_csv_file)
            if pisa_csv_df is None:
                pisa_csv_df = cur_pisa_csv_df
            else:
                pisa_csv_df = pd.concat([pisa_csv_df,cur_pisa_csv_df])
            
        pisa_csv_df.to_csv(self.db_path + 'pisa/all/all_' + runtime['pisa_folder'] + '.csv',index = False)
        
    def combine_psipred_data(self):
        psipred_df = None
        for psipred_file in glob.glob(self.db_path + 'psipred/bygene/*.ss2' ):
#         for psipred_file in glob.glob('/Users/joewu/Dropbox/database/humandb/psipred/uniprot_psipred/*.ss2' ):           
            cur_psipred_df = pd.read_csv(psipred_file, skiprows=[0, 1], header=None, sep='\s+')
            cur_psipred_df = cur_psipred_df.loc[:, [0, 1, 2]]
            cur_psipred_df.columns = ['aa_pos', 'aa_ref', 'aa_psipred']                       
            uniprot_id = psipred_file.split('/')[-1][:-4]                              
            if uniprot_id != None:
                cur_psipred_df['p_vid'] = uniprot_id                             
                if psipred_df is None:
                    psipred_df = cur_psipred_df
                else:
                    psipred_df = pd.concat([psipred_df, cur_psipred_df])                                       
        psipred_df.to_csv(self.db_path + 'psipred/all/psipred_df.csv', index=None)
    
    def check_psipred_data(self):
        cur_log = self.db_path + 'psipred/log/check_psipred_data.log'
        varity_genes_df = pd.read_csv(self.db_path + 'varity/all/varity_disease_genes.csv')         
        varity_ids = varity_genes_df.loc[varity_genes_df['uniprot_id'].notnull(),'uniprot_id']
        exist_ids = []
        for file in glob.glob(self.db_path + 'psipred/bygene/*.ss2'):
            cur_id = file.split('/')[-1].split('.')[0]
            exist_ids.append(cur_id)
        long_protein_ids = ['Q8WZ42'] #too long to do the blast 
        run_varity_ids = set(varity_ids) - set(exist_ids)  
        run_varity_ids = list(run_varity_ids - set(long_protein_ids))   
        run_varity_ids.sort()
                              
        total_gene_num = len(run_varity_ids)
        alm_fun.show_msg(cur_log,self.verbose,'total # number of genes need to run psipred is ' + str(total_gene_num)) 
                         
    def retrieve_psipred_data(self,parallel_id,parallel_num):
        cur_log = self.db_path + 'psipred/log/psipred.log'
        parallel_id = int(parallel_id)                 
        
        varity_genes_df = pd.read_csv(self.db_path + 'varity/all/varity_disease_genes.csv')         
        varity_ids = varity_genes_df.loc[varity_genes_df['p_vid'].notnull(),'p_vid']
        alm_fun.show_msg(cur_log,self.verbose,'total # number of vairty genes need  ' + str(len(varity_ids)))
        exist_ids = []
        for file in glob.glob(self.db_path + 'psipred/bygene/*.ss2'):
            cur_id = file.split('/')[-1].split('.')[0]
            exist_ids.append(cur_id)
            
        alm_fun.show_msg(cur_log,self.verbose,'total # number of vairty genes have psipred  ' + str(len(exist_ids)))
        long_protein_ids = ['Q8WZ42'] #too long to do the blast 
        run_varity_ids = set(varity_ids) - set(exist_ids)  
        run_varity_ids = list(run_varity_ids - set(long_protein_ids))   
        run_varity_ids.sort()
                                  
        total_gene_num = len(run_varity_ids)
        alm_fun.show_msg(cur_log,self.verbose,'total # number of genes need to run psipred is ' + str(total_gene_num))  
        alm_fun.show_msg(cur_log,self.verbose,str(run_varity_ids))   
     
        gene_index_array = np.linspace(0,total_gene_num,parallel_num+1 , dtype = int)        
        cur_parallel_indices = list(range(gene_index_array[parallel_id],gene_index_array[parallel_id+1]))
        cur_gene_ids = [run_varity_ids[i] for i in cur_parallel_indices]
        cur_parallel_log = self.db_path + 'psipred/log/psipred_data_parallel_' + str(parallel_id) +  '_' + str(parallel_num) + '.log'          
        for uniprot_id in cur_gene_ids:
            self.retrieve_psipred_by_uniprotid(uniprot_id,cur_parallel_log)
        alm_fun.show_msg(cur_log,self.verbose,'psipred_job_' + str(parallel_id) + '_' + str(parallel_num) + ' is done.')   
     
    def retrieve_psipred_by_uniprotid(self,uniprot_id,cur_parallel_log, overwrite = 1):
        #**********************************************************************************************
        # PISPRED : after installation, remeber to edit the runpsipred script to make it work!!!!!!!
        #**********************************************************************************************                
        if os.path.isfile(self.db_path + 'psipred/bygene/' + uniprot_id + '.ss2') & (overwrite == 0):
            alm_fun.show_msg(cur_parallel_log,self.verbose, uniprot_id + ' psipred exists.')
            return(0)
        else:
            alm_fun.show_msg(cur_parallel_log,self.verbose, 'strat to retrieve the psipred info of ' + uniprot_id + '......' )
            psipred_cmd = "runpsipred " + self.db_path + "uniprot/bygene/" + uniprot_id + ".fasta"                
            subprocess.run(psipred_cmd.split(" "), cwd = self.db_path + '../../tools/psipred/psipred/' )
            alm_fun.show_msg(cur_parallel_log,self.verbose, uniprot_id + ' psipred is retrieved.')
                        
    def create_matched_uniprot_mapping(self):
        
        #*************************************************************************************
        # Create mapping from other protein ids to uniprot ids
        # 1) Other ID - > HGNC id   (1 to 1 relationship) 
        # 2) HGNC id  - > Uniprot Id ( 1 to many relationship)
        # 3) Compare the protein sequence of other id and each uniprot id 
        # 4) Other ID  - > Last Uniprot ID that has matched sequence
        #*************************************************************************************
        
        id2uniprot_matched_dict = {}        
        hgnc2id_dict = np.load(self.db_path + 'hgnc/npy/hgnc2id_dict.npy').item()
        id2hgnc_dict = np.load(self.db_path + 'hgnc/npy/id2hgnc_dict.npy').item()    
        uniprot_seq_dict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy').item()
        ensembl66_seq_dict = np.load(self.db_path + 'ensembl66/npy/ensembl66_seq_dict.npy').item()
        ensp2ensg66_dict = np.load(self.db_path + 'ensembl66/npy/ensp2ensg66_dict.npy').item()
                
        #matched ensembl66 to uniprot
        id2uniprot_matched_dict['ensembl66']={}
        for cur_ensp in ensp2ensg66_dict.keys():
            cur_uniprot_ids = hgnc2id_dict['uniprot_ids'].get(id2hgnc_dict['ensembl_gene_id'].get(ensp2ensg66_dict[cur_ensp],np.nan),np.nan)
            if str(cur_uniprot_ids) != 'nan':
                for cur_uniprot_id in cur_uniprot_ids.split('|'):
                    if ensembl66_seq_dict.get(cur_ensp,np.nan) == uniprot_seq_dict.get(cur_uniprot_id,np.nan):            
                        id2uniprot_matched_dict['ensembl66'][cur_ensp] = cur_uniprot_id    
        np.save(self.db_path + 'uniprot/npy/id2uniprot_matched_dict.npy', id2uniprot_matched_dict) 
    
    def retrieve_vep_record(self,variant_corrdinate):
        server = "https://grch37.rest.ensembl.org"
        ext = "/vep/human/hgvs/" + variant_corrdinate + "?protein=1;uniprot=1;canonical=1"
         
        r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
         
        if not r.ok:
          r.raise_for_status()
          sys.exit()
         
        decoded = r.json()
        print(repr(decoded))
                
    def varity_dbnsfp_jobs(self,single_id = '',chr = ''):        
        if single_id == '':
            varity_genes_df = pd.read_csv(self.db_path + 'varity/all/varity_disease_genes.csv')
        else:
            varity_genes_df = pd.DataFrame(columns = ['p_vid','chr'])
            varity_genes_df.loc[0,'p_vid'] = single_id
            varity_genes_df.loc[0,'chr'] = chr
            
        for chr in varity_genes_df['chr'].unique():
            if chr == 'MT':
                chr = 'M'                
            cur_gene_symbols =  list(varity_genes_df.loc[varity_genes_df['chr'] == chr ,'p_vid'])
            cur_gene_symbols.sort()    
            if len(cur_gene_symbols)!= 0:            
                x = list(np.arange(0,len(cur_gene_symbols),100))
                if len(x) == 1:
                    x.append(len(cur_gene_symbols))
                else:
                    x[-1] = len(cur_gene_symbols)
                    
                for i in range(len(x)-1):
                    if single_id == '':
                        varity_genes_file = open(self.db_path + 'varity/csv/varity_genes_' + chr + '_' + str(i) + '.txt','w')                        
                    else:
                        varity_genes_file = open(self.db_path + 'varity/csv/varity_genes_' + single_id + '_' + str(i) + '.txt','w')
                        
                    for s in cur_gene_symbols[x[i]:x[i+1]]:
                        varity_genes_file.write('Uniprot:' + s + '\n')
                    varity_genes_file.close()   
                    if single_id == '':              
                        alm_fun.show_msg(self.log,self.verbose,'Running dbNSFP for VARITY genes on ' + chr + '_' + str(i) + ' chromosome.')
                    else:
                        alm_fun.show_msg(self.log,self.verbose,'Running dbNSFP for VARITY genes on ' + single_id + '_' + str(i) + ' chromosome.')
                    
                    #*************************************************************************************************
                    #run one by one using one workstation 
                    #*************************************************************************************************
    #                 dbnsfp_cmd = "java search_dbNSFP40b2a -v hg19 -c " + chr + " -i " + self.db_path + "varity/csv/varity_genes_" + chr + ".txt -o " + self.db_path + "varity/csv/varity_dbnsfp_snv_" + chr + ".out"
        #             subprocess.run(dbnsfp_cmd.split(" "), cwd = self.db_path + 'dbnsfp/')
                    
                    #*************************************************************************************************
                    #run parallel jobs (but you need figure out the way to determine when all jobs are finished 
                    #*************************************************************************************************
#                     dbnsfp_cmd = "java search_dbNSFP40b2a -v hg19 -c " + chr + " -i " + self.db_path + "varity/csv/varity_genes_" + chr + '_' + str(i) + ".txt -o " + self.db_path + "varity/csv/varity_dbnsfp_snv_" + chr + '_' + str(i) + ".out"
                   #  Error happened if use -c option search sepecific chromosome  
                    if single_id == '':
                        dbnsfp_cmd = "java search_dbNSFP40a -v hg19 -i " + self.db_path + "varity/csv/varity_genes_" + chr + '_' + str(i) + ".txt -o " + self.db_path + "varity/csv/varity_dbnsfp_snv_" + chr + '_' + str(i) + ".out"
                    else:
                        dbnsfp_cmd = "java search_dbNSFP40a -v hg19 -i " + self.db_path + "varity/csv/varity_genes_" + single_id + '_' + str(i) + ".txt -o " + self.db_path + "varity/csv/varity_dbnsfp_snv_" + single_id + '_' + str(i) + ".out"
           
                    if self.cluster == 'galen':            
                        #*************************************************************************************************
                        #run single job on cluster for current action 
                        #*************************************************************************************************
                        if single_id == '':                     
                            sh_file = open(self.db_path + 'varity/bat/varity_dbnsfp_job_' + chr + '_' + str(i) + '.sh','w')
                        else:
                            sh_file = open(self.db_path + 'varity/bat/varity_dbnsfp_job_' + single_id + '_' + str(i) + '.sh','w')  
                        sh_file.write('#!/bin/bash' + '\n')
                        sh_file.write('# set the number of nodes' + '\n')
                        sh_file.write('#SBATCH --nodes=1' + '\n')
                        sh_file.write('# set max wallclock time' + '\n')
                        sh_file.write('#SBATCH --time=100:00:00' + '\n')
                        sh_file.write('# set name of job' + '\n')
                        if single_id == '':
                            sh_file.write('#SBATCH --job-name=varity_dbnsfp_job_' + chr + '_' + str(i) + '\n')
                        else:
                            sh_file.write('#SBATCH --job-name=varity_dbnsfp_job_' + single_id + '_' + str(i) + '\n')
                        sh_file.write('# mail alert at start, end and abortion of execution' + '\n')
                        sh_file.write('#SBATCH --mail-type=ALL' + '\n')
                        sh_file.write('# send mail to this address' + '\n')
                        sh_file.write('#SBATCH --mail-user=joe.wu.ca@gmail.com' + '\n')      
                        sh_file.write("srun " + dbnsfp_cmd)
                        sh_file.close()
                        
                        if single_id == '':
                            sbatch_cmd = "sbatch " + self.db_path + 'varity/bat/varity_dbnsfp_job_' + chr + '_' + str(i) + '.sh'
                        else:
                            sbatch_cmd = "sbatch " + self.db_path + 'varity/bat/varity_dbnsfp_job_' + single_id + '_' + str(i) + '.sh'
#                         print (sbatch_cmd)
                        subprocess.run(sbatch_cmd.split(" "), cwd = self.db_path + 'dbnsfp/')                    
    
    def varity_dbnsfp_process_jobs(self):    
         
        for file in glob.glob(self.db_path + 'varity/csv/varity_dbnsfp_snv_*.out'):
            parallel_id = file.split(".")[0].split('_')[-2] + '_' + file.split(".")[0].split('_')[-1] 
            alm_fun.show_msg(self.log,self.verbose,'Processing VARITY out file on ' + parallel_id + ' chromosome.')         
            #*************************************************************************************************
            #run parallel jobs 
            #*************************************************************************************************
            varity_process_cmd = "python3 " + "'" + self.project_path +  "python/humandb_debug.py" + "'" + " '" + self.python_path + "'"  + " '"  + self.project_path + "' " + "'" +  self.db_path + "'" + " 'run_humandb_action' " + "'varity_dbnsfp_process' " + "'" +  parallel_id + "' " + "'0' " + "\n"
            
            if self.cluster == 'galen':                              
                sh_file = open(self.db_path + 'varity/bat/varity_dbnsfp_process_' + parallel_id + '.sh','w')  
                sh_file.write('#!/bin/bash' + '\n')
                sh_file.write('# set the number of nodes' + '\n')
                sh_file.write('#SBATCH --nodes=1' + '\n')
                sh_file.write('# set max wallclock time' + '\n')
                sh_file.write('#SBATCH --time=100:00:00' + '\n')
                sh_file.write('# set name of job' + '\n')
                sh_file.write('#SBATCH --job-name=varity_process_' + parallel_id + '\n')
                sh_file.write('# mail alert at start, end and abortion of execution' + '\n')
                sh_file.write('#SBATCH --mail-type=ALL' + '\n')
                sh_file.write('# send mail to this address' + '\n')
                sh_file.write('#SBATCH --mail-user=joe.wu.ca@gmail.com' + '\n')      
                sh_file.write("srun " + varity_process_cmd)
                sh_file.close()
                                
                sbatch_cmd = "sbatch " + self.db_path + 'varity/bat/varity_dbnsfp_process_' + parallel_id + '.sh'
                subprocess.run(sbatch_cmd.split(" "), cwd = self.db_path + 'varity/log/')   
    
    def varity_dbnsfp_process(self,parallel_id,single_id = ''):
        #************************************************************************************************************************************************************************
        ## Process dbnsfp output result
        #************************************************************************************************************************************************************************
        if single_id == '':        
            cur_log = self.db_path + 'varity/log/varity_process_' + str(parallel_id) + '.log'
            alm_fun.show_msg(cur_log,self.verbose,'Varity' + ' chr:' + str(parallel_id) + ' start processing...... ' )
            varity_dbnsfp_data = pd.read_csv(self.db_path +'varity/csv/varity_dbnsfp_snv_' + str(parallel_id) + '.out', sep = '\t',dtype = {'hg19_chr':'str'})                                                
            alm_fun.show_msg(cur_log,self.verbose,'Varity' + ' chr:' + str(parallel_id) + ' records before processing : ' + str(varity_dbnsfp_data.shape[0]))
            varity_dbnsfp_processed_data = self.process_dbnsfp_out(varity_dbnsfp_data,cur_log)                
            varity_dbnsfp_processed_data.to_csv(self.db_path + 'varity/csv/varity_dbnsfp_processed_' + str(parallel_id) + '.csv',index = False)
            alm_fun.show_msg(cur_log,self.verbose,'Varity' + ' chr:' + str(parallel_id) + ' records after processing : ' + str(varity_dbnsfp_processed_data.shape[0]))
        else:
            cur_log = self.db_path + 'varity/log/varity_process_' + str(single_id) + '.log'
            alm_fun.show_msg(cur_log,self.verbose,str(single_id) + ' start processing...... ' )
            varity_dbnsfp_data = pd.read_csv(self.db_path +'varity/csv/varity_dbnsfp_snv_' + str(single_id) + '_0.out', sep = '\t',dtype = {'hg19_chr':'str'})                                                
            alm_fun.show_msg(cur_log,self.verbose, str(single_id) + ' records before processing : ' + str(varity_dbnsfp_data.shape[0]))
            varity_dbnsfp_processed_data = self.process_dbnsfp_out(varity_dbnsfp_data,cur_log)                
            varity_dbnsfp_processed_data.to_csv(self.db_path + 'varity/csv/varity_dbnsfp_processed_' + str(single_id) + '.csv',index = False)
            alm_fun.show_msg(cur_log,self.verbose, str(single_id) + ' records after processing : ' + str(varity_dbnsfp_processed_data.shape[0]))
        return(varity_dbnsfp_processed_data)
                                                                  
    def varity_merge_data(self,parallel_id,single_id = ''):
        
        if single_id == '':
            cur_log = self.db_path + 'varity/log/varity_merge_data_' + str(parallel_id) + '.log'
            varity_dbnsfp_processed_data = pd.read_csv(self.db_path + 'varity/csv/varity_dbnsfp_processed_' + str(parallel_id) + '.csv',dtype = {'chr':'str'})        
            alm_fun.show_msg(cur_log,self.verbose,'Varity records before merging : ' + str(varity_dbnsfp_processed_data.shape[0]))
        else:
            cur_log = self.db_path + 'varity/log/varity_merge_data_' + str(single_id) + '.log'
            varity_dbnsfp_processed_data = pd.read_csv(self.db_path + 'varity/csv/varity_dbnsfp_processed_' + str(single_id) + '.csv',dtype = {'chr':'str'})        
            alm_fun.show_msg(cur_log,self.verbose,'Varity records before merging : ' + str(varity_dbnsfp_processed_data.shape[0]))
        #************************************************************************************
        # Load CLINVAR, HUMSAVAR, HGMD and  MAVE data
        #************************************************************************************
        clinvar_snv = pd.read_csv(self.db_path + 'clinvar/all/clinvar_snv.csv',dtype = {'chr':'str'})
        clinvar_snv['clinvar_source'] = 1
        
        humsavar_snv = pd.read_csv(self.db_path + 'humsavar/all/humsavar_snv.csv')
        humsavar_snv['humsavar_source'] = 1
        
        hgmd_snv = pd.read_csv(self.db_path + 'hgmd/all/hgmd_snv.csv')
        hgmd_snv['hgmd_source'] = 1
        
        mave_missense = pd.read_csv(self.db_path + 'mave/all/mave_missense.csv')  
        mave_pvids = list(mave_missense['p_vid'].unique())
        
        #************************************************************************************
        # Merge varity data with CLINVAR and HUMSAVAR
        #************************************************************************************
        varity_merged_data = varity_dbnsfp_processed_data.merge(clinvar_snv,how = 'left')
        alm_fun.show_msg(cur_log,self.verbose,'Varity records after merge CLINVAR : ' + str(varity_merged_data.shape[0]))
        varity_merged_data = varity_merged_data.merge(humsavar_snv,how = 'left')
        alm_fun.show_msg(cur_log,self.verbose,'Varity records after merge HUMSAVAR : ' + str(varity_merged_data.shape[0]))
        varity_merged_data = varity_merged_data.merge(hgmd_snv,how = 'left')
        alm_fun.show_msg(cur_log,self.verbose,'Varity records after merge HGMD : ' + str(varity_merged_data.shape[0]))

        varity_merged_data['mave_gene_source'] = 0
        varity_merged_data.loc[varity_merged_data['p_vid'].isin(mave_pvids),'mave_gene_source'] = 1
        varity_merged_data['gnomad_source'] = 0
        varity_merged_data.loc[varity_merged_data['gnomAD_exomes_AC'] > 0, 'gnomad_source'] = 1

        if single_id == '':                              
            varity_merged_data.to_csv(self.db_path + 'varity/csv/varity_merged_' + str(parallel_id) + '.csv',index = False)        
            varity_merged_train_data = varity_merged_data.loc[(varity_merged_data['mave_gene_source'] == 1) | (varity_merged_data['clinvar_source'] == 1) | (varity_merged_data['gnomad_source'] == 1) | (varity_merged_data['humsavar_source'] == 1) | (varity_merged_data['hgmd_source'] == 1),: ]                
            alm_fun.show_msg(cur_log,self.verbose,'Varity records after merging - selected for training (CLINVAR variants, HUMSAVAR variants, HGMD variants and MAVE genes, GNOMAD AC >0) : ' + str(varity_merged_train_data.shape[0]))        
            varity_merged_train_data.to_csv(self.db_path + 'varity/csv/varity_merged_train_' + str(parallel_id) + '.csv',index = False)                            
        else:  
            varity_merged_data.to_csv(self.db_path + 'varity/csv/varity_merged_' + str(single_id) + '.csv',index = False)
            
    def varity_merge_data_jobs(self):        
#         for file in glob.glob(self.db_path + 'varity/csv/varity_dbnsfp_processed_*.csv'):
        for file in glob.glob(self.db_path + 'varity/csv/varity_processed_*.csv'):
            parallel_id = file.split(".")[0].split('_')[-2] + '_' + file.split(".")[0].split('_')[-1] 
            alm_fun.show_msg(self.log,self.verbose,'Merge CLINVAR and MAVE with Varity processed file on  ' + parallel_id + ' chromosome.')         
            #*************************************************************************************************
            #run parallel jobs 
            #*************************************************************************************************
            varity_merge_data_cmd = "python3 " + "'" + self.project_path +  "python/humandb_debug.py" + "'" + " '" + self.python_path + "'"  + " '"  + self.project_path + "' " + "'" +  self.db_path + "'" + " 'run_humandb_action' " + "'varity_merge_data' " + "'" +  parallel_id + "' " + "'0' " + "\n"
            
            if self.cluster == 'ccbr':  
                db_sh_file = open(self.db_path + 'varity/bat/varity_merge_data_' + parallel_id + '.sh','w')  
                db_obj = 'varity_merge_data'                
                db_sh_file.write('#!/bin/bash' + '\n')
                db_sh_file.write('$ -l h_vmem=6G' + '\n')
                db_sh_file.write(varity_merge_data_cmd + '\n')
                db_sh_file.close() 
                print(process_cmd)            
                qsub_cmd = "submitjob -m 6 " + self.db_path + 'varity/bat/varity_merge_data_' + chr + '.sh'
                subprocess.run(qsub_cmd.split(" "), cwd = self.db_path + 'varity/bat/')
            
            if self.cluster == 'galen':                              
                sh_file = open(self.db_path + 'varity/bat/varity_merge_data_' + parallel_id + '.sh','w')  
                sh_file.write('#!/bin/bash' + '\n')
                sh_file.write('# set the number of nodes' + '\n')
                sh_file.write('#SBATCH --nodes=1' + '\n')
                sh_file.write('# set max wallclock time' + '\n')
                sh_file.write('#SBATCH --time=100:00:00' + '\n')
                sh_file.write('# set name of job' + '\n')
                sh_file.write('#SBATCH --job-name=varity_merge_data_' + parallel_id + '\n')
                sh_file.write('# mail alert at start, end and abortion of execution' + '\n')
                sh_file.write('#SBATCH --mail-type=ALL' + '\n')
                sh_file.write('# send mail to this address' + '\n')
                sh_file.write('#SBATCH --mail-user=joe.wu.ca@gmail.com' + '\n')      
                sh_file.write("srun " + varity_merge_data_cmd)
                sh_file.close()
                                
                sbatch_cmd = "sbatch " + self.db_path + 'varity/bat/varity_merge_data_' + parallel_id + '.sh'
                subprocess.run(sbatch_cmd.split(" "), cwd = self.db_path + 'varity/log/')   
    
    def varity_combine_train_data(self):
        cur_log = self.db_path + 'varity/log/varity_combine_train_data.log'
        varity_train_data = None
        for file in glob.glob(self.db_path + 'varity/csv/varity_merged_train_*.csv'):
            parallel_id = file.split(".")[0].split('_')[-2] + '_' + file.split(".")[0].split('_')[-1] 
            alm_fun.show_msg(cur_log,self.verbose,'Combine varity train data ' + parallel_id + '......')
            cur_varity_train_data = pd.read_csv(self.db_path + 'varity/csv/varity_merged_train_' + parallel_id + '.csv')
            if varity_train_data is None:
                varity_train_data = cur_varity_train_data
            else:
                varity_train_data = pd.concat([varity_train_data,cur_varity_train_data])
        varity_train_data.to_csv(self.db_path + 'varity/csv/varity_train_data.csv',index = False)
        alm_fun.show_msg(cur_log,self.verbose,'Varity training data merged, total number: ' + str(varity_train_data.shape[0]))
        
        train_p_vids = varity_train_data['p_vid'].unique()
        train_p_vids_file = open(self.db_path + 'varity/all/varity_train_uniprot_genes.txt','w')
        for p_vid in train_p_vids:
            train_p_vids_file.write(p_vid + '\n')
        pass
        train_p_vids_file.close()
    
    def varity_count_all_data(self):
        cur_log = self.db_path + 'varity/log/varity_count_all_data.log'
        varity_total_counts = 0 
        for file in glob.glob(self.db_path + 'varity/csv/varity_merged_*.csv'):
            if not ('train' in file):
                parallel_id = file.split(".")[0].split('_')[-2] + '_' + file.split(".")[0].split('_')[-1] 
                cur_varity_all_data = pd.read_csv(self.db_path + 'varity/csv/varity_merged_' + parallel_id + '.csv')
                alm_fun.show_msg(cur_log,self.verbose,'# of varity data ' + parallel_id + ':' + str(cur_varity_all_data.shape[0]))
                varity_total_counts += cur_varity_all_data.shape[0]
        alm_fun.show_msg(cur_log,self.verbose,'total # of varity data :' + str(varity_total_counts))
                 
    def varity_combine_all_data(self):
        cur_log = self.db_path + 'varity/log/varity_combine_all_data.log'
        varity_all_data = None
        alm_fun.show_msg(cur_log,self.verbose,'Varity all data merge starts......')
        for file in glob.glob(self.db_path + 'varity/csv/varity_all_*_final.csv'):            
            parallel_id = file.split(".")[0].split('_')[-3] + '_' + file.split(".")[0].split('_')[-2] 
            alm_fun.show_msg(cur_log,self.verbose,'Combine varity all data ' + parallel_id + '......')
            cur_varity_all_data = pd.read_csv(self.db_path + 'varity/csv/varity_all_' + parallel_id + '_final.csv')
            key_cols = ['chr','p_vid','aa_pos','aa_ref','aa_alt','fitness_input','fitness_org','fitness','clinvar_source','hgmd_source','mave_source','humsavar_source','gnomad_source','clinvar_label','hgmd_label','humsavar_label','label']   
            score_cols = []             
            score_cols = ['gnomAD_exomes_AF','gnomAD_exomes_AC','gnomAD_exomes_nhomalt','evm_epistatic_score','Polyphen2_selected_HDIV_score','Polyphen2_selected_HVAR_score','PROVEAN_selected_score','SIFT_selected_score','CADD_raw','PrimateAI_score','Eigen-raw_coding','integrated_fitCons_score','REVEL_score','M-CAP_score','LRT_score','MutationTaster_selected_score','MutationAssessor_selected_score','FATHMM_selected_score','VEST4_selected_score','MetaSVM_score','MetaLR_score','DANN_score','GenoCanyon_score','GERP++_RS','phyloP100way_vertebrate','phyloP30way_mammalian','phastCons100way_vertebrate','phastCons30way_mammalian','SiPhy_29way_logOdds']
            features = ['blosum100', 'in_domain', 'asa_mean', 'aa_psipred_E', 'aa_psipred_H', 'aa_psipred_C', 'bsa_max', 'h_bond_max', 'salt_bridge_max', 'disulfide_bond_max', 'covelent_bond_max', 'solv_ne_min', 'mw_delta', 'pka_delta', 'pkb_delta', 'pi_delta', 'hi_delta', 'pbr_delta', 'avbr_delta', 'vadw_delta', 'asa_delta', 'cyclic_delta', 'charge_delta', 'positive_delta', 'negative_delta', 'hydrophobic_delta', 'polar_delta', 'ionizable_delta', 'aromatic_delta', 'aliphatic_delta', 'hbond_delta', 'sulfur_delta', 'essential_delta', 'size_delta', 'PROVEAN_selected_score', 'SIFT_selected_score', 'evm_epistatic_score', 'Eigen-raw_coding', 'integrated_fitCons_score', 'LRT_score', 'GERP++_RS', 'phyloP30way_mammalian', 'phastCons30way_mammalian', 'SiPhy_29way_logOdds']
                                    
            cur_varity_all_data = cur_varity_all_data[list(set(key_cols+ score_cols + features))]    
            if varity_all_data is None:
                varity_all_data = cur_varity_all_data
            else:
                varity_all_data = pd.concat([varity_all_data,cur_varity_all_data])
        varity_all_data.to_csv(self.db_path + 'varity/csv/varity_all_scores_data.csv',index = False)
        alm_fun.show_msg(cur_log,self.verbose,'Varity all data merged, total number: ' + str(varity_all_data.shape[0]))
    
    def varity_mave_final_data(self):
        cur_log = self.db_path + 'varity/log/varity_mave_final_data.log'
        varity_mave_final_data = None
        alm_fun.show_msg(cur_log,self.verbose,'Varity all mave_data merge starts......')
        p_chr_dict = {}
        p_chr_dict['P42898'] = '1_1'
        p_chr_dict['P35520'] = '21_0'
        p_chr_dict['P38398'] = '17_0'
        p_chr_dict['P60484'] = '10_0'
        p_chr_dict['P63279'] = '16_0'
        p_chr_dict['P63165'] = '2_1'
        p_chr_dict['P51580'] = '6_0'                
        for p_vid in p_chr_dict.keys():
            alm_fun.show_msg(cur_log,self.verbose,'Combine mave all data ' + p_vid + '......')
            cur_varity_mave_final_data = pd.read_csv(self.db_path + 'varity/csv/varity_all_' + p_chr_dict[p_vid] + '_final.csv')
            cur_varity_mave_final_data = cur_varity_mave_final_data.loc[cur_varity_mave_final_data['p_vid'] == p_vid,:]
            varity_mave_final_data = pd.concat([varity_mave_final_data,cur_varity_mave_final_data])
        varity_mave_final_data.to_csv(self.db_path + 'varity/csv/varity_mave_final_data.csv',index = False)
        alm_fun.show_msg(cur_log,self.verbose,'Varity mvae final data merged, total number: ' + str(varity_mave_final_data.shape[0]))
     
    def varity_check_data(self,parallel_id):
        cur_log = self.db_path + 'varity/log/varity_check_data.log'
        varity_final_data = pd.read_csv(self.db_path + 'varity/csv/varity_all_' + str(parallel_id) + '_final.csv',dtype = {'chr':'str'},usecols = ['chr','p_vid','aa_pos','aa_ref','aa_alt'])        
        varity_final_data_group = varity_final_data.groupby(['p_vid','aa_pos','aa_ref','aa_alt'])['chr'].agg('count')
        
        if varity_final_data.shape[0] == varity_final_data_group.shape[0]:
            alm_fun.show_msg(cur_log,self.verbose,parallel_id + ' Varity final data: ' + str(varity_final_data.shape[0]))
        else:
            alm_fun.show_msg(cur_log,self.verbose,parallel_id + ' Varity final group data: ' + str(varity_final_data_group.shape[0]) +  ' Varity final data: ' + str(varity_final_data.shape[0]))
        
    def varity_check_data_jobs(self):
        cur_log = self.db_path + 'varity/log/varity_check_data.log'
        alm_fun.show_msg(cur_log,self.verbose,'**********************************************************')
        alm_fun.show_msg(cur_log,self.verbose,'varity_check_data_jobs')
        alm_fun.show_msg(cur_log,self.verbose,'**********************************************************')
        
        for file in glob.glob(self.db_path + 'varity/csv/varity_all_*_final.csv'):
            parallel_id = file.split(".")[0].split('_')[-3] + '_' + file.split(".")[0].split('_')[-2]                        
             #*************************************************************************************************
            #run parallel jobs 
            #*************************************************************************************************
            varity_check_data_cmd = "python3 " +  self.project_path +  "python/humandb_debug.py" + " '" + self.python_path + "'"  + " '"  + self.project_path + "' " + "'" +  self.db_path + "'" + " 'run_humandb_action' " + "'varity_check_data' " + "'" +  parallel_id + "' " + "0" 
            job_name = 'varity_all_check_data_' + parallel_id 
            
            if self.cluster == 'galen':                              
                sh_file = open(self.db_path + 'varity/bat/varity_check_data_' + parallel_id + '.sh','w')  
                sh_file.write('#!/bin/bash' + '\n')
                sh_file.write('# set the number of nodes' + '\n')
                sh_file.write('#SBATCH --nodes=1' + '\n')
                sh_file.write('# set the memory for each node' + '\n')
                sh_file.write('#SBATCH --mem=' + '10240' + '\n')                  
                sh_file.write('# set name of job' + '\n')
                sh_file.write('#SBATCH --job-name=' + job_name + '\n')
                sh_file.write('# mail alert at start, end and abortion of execution' + '\n')
                sh_file.write('#SBATCH --mail-type=ALL' + '\n')
                sh_file.write('# send mail to this address' + '\n')
                sh_file.write('#SBATCH --mail-user=joe.wu.ca@gmail.com' + '\n')      
                sh_file.write("srun " + varity_check_data_cmd)
                sh_file.close()
                                 
                sbatch_cmd = "sbatch " + self.db_path + 'varity/bat/varity_check_data_' + parallel_id + '.sh'
                print (sbatch_cmd)
                job_id = '-1'
                while  job_id == '-1':
                    return_process = subprocess.run(sbatch_cmd.split(" "), cwd = self.db_path + 'varity/log/',capture_output = True,text=True)
                    time.sleep(0.1)
                    if return_process.returncode == 0:
                        job_id = return_process.stdout.rstrip().split(' ')[-1]
                    else:
                        job_id = '-1'
                        print  (job_name + ' submit error,rescheduling......' )
                print (job_name + ' submitted id: ' + job_id)        

    def varity_all_variant_process(self,parallel_id):            
        varity_merged_data = pd.read_csv(self.db_path + 'varity/csv/varity_merged_' + str(parallel_id) + '.csv',dtype = {'chr':'str'})       
        varity_merged_data_variant_processed = self.varity_variant_process(varity_merged_data,0,'varity_all_' + parallel_id)            
        varity_merged_data_variant_processed.to_csv(self.db_path + 'varity/csv/varity_all_variant_processed_' +  str(parallel_id) + '.csv', index = False)

    def varity_all_variant_process_jobs(self):
        print ("runing varaity_all_variant_process_jobs!")
        parallel_ids = []
        for file in glob.glob(self.db_path + 'varity/csv/varity_merged_*.csv'):            
            parallel_id = file.split(".")[0].split('_')[-2] + '_' + file.split(".")[0].split('_')[-1] 
            parallel_ids.append(parallel_id)
            
        parallel_ids = np.unique(parallel_ids)
        for parallel_id in parallel_ids:
            alm_fun.show_msg(self.log,self.verbose,'Run variant process  on  ' + parallel_id + ' chromosome.')         
            #*************************************************************************************************
            #run parallel jobs 
            #*************************************************************************************************
            varity_variant_process_cmd = "python3 " +  self.project_path +  "python/humandb_debug.py" + " '" + self.python_path + "'"  + " '"  + self.project_path + "' " + "'" +  self.db_path + "'" + " 'run_humandb_action' " + "'varity_all_variant_process' " + "'" +  parallel_id + "' " + "0" 
            job_name = 'varity_all_variant_process_' + parallel_id 
            
            if self.cluster == 'galen':                              
                sh_file = open(self.db_path + 'varity/bat/varity_variant_process_' + parallel_id + '.sh','w')  
                sh_file.write('#!/bin/bash' + '\n')
                sh_file.write('# set the number of nodes' + '\n')
                sh_file.write('#SBATCH --nodes=1' + '\n')
                sh_file.write('# set the memory for each node' + '\n')
                sh_file.write('#SBATCH --mem=' + '10240' + '\n')              
                sh_file.write('# set name of job' + '\n')
                sh_file.write('#SBATCH --job-name=' + job_name + '\n')
                sh_file.write('# mail alert at start, end and abortion of execution' + '\n')
                sh_file.write('#SBATCH --mail-type=ALL' + '\n')
                sh_file.write('# send mail to this address' + '\n')
                sh_file.write('#SBATCH --mail-user=joe.wu.ca@gmail.com' + '\n')      
                sh_file.write("srun " + varity_variant_process_cmd)
                sh_file.close()
                                 
                sbatch_cmd = "sbatch " + self.db_path + 'varity/bat/varity_variant_process_' + parallel_id + '.sh'
                print (sbatch_cmd)
                job_id = '-1'
                while  job_id == '-1':
                    return_process = subprocess.run(sbatch_cmd.split(" "), cwd = self.db_path + 'varity/log/',capture_output = True,text=True)
                    time.sleep(0.1)
                    if return_process.returncode == 0:
                        job_id = return_process.stdout.rstrip().split(' ')[-1]
                    else:
                        job_id = '-1'
                        print  (job_name + ' submit error,rescheduling......' )
                print (job_name + ' submitted id: ' + job_id)                 
                        
    def varity_variant_process(self, input_data, merge_mave,data_name):    
        def duplicate_aa_process(input):             
            if input.dtypes == 'float64':            
                print(str(input))    
        cur_log = self.db_path + 'varity/log/' + data_name  + '_variant_process.log'       
        alm_fun.show_msg(cur_log,self.verbose,data_name + ' before variant process : ' + str(input_data.shape[0]))
        input_data = input_data.loc[(input_data['aa_ref'].isin(self.lst_aa_20)) & (input_data['aa_alt'].isin(self.lst_aa_20)) ,:]
        alm_fun.show_msg(cur_log,self.verbose,data_name + ' after removing non-misense variants : ' + str(input_data.shape[0]))
        
        #make the chr null case to 'Z'
        input_data.loc[input_data['chr'].isnull(),'chr'] = 'Z'
#         alm_fun.show_msg(cur_log,self.verbose,data_name + ' after removing variants that are not valid for hg19 : ' + str(input_data.shape[0]))
        
        #************************************************************************************
        # Process the duplicated AA change caused by different NT change at the same position
        # 0) Pick a random one
        # 1) Pick a variant with rules (most common by allele frequency)
        # 2) Combine information of all nt variant
        # 3) Combine information of all nt variant
        # 2019-04-10 1)
        #************************************************************************************
        input_data_group = input_data.groupby(['p_vid','aa_pos','aa_ref','aa_alt'])['chr'].agg('count').reset_index()
        input_data_group.columns = ['p_vid','aa_pos','aa_ref','aa_alt','count']
        duplicate_aa = input_data_group.loc[input_data_group['count'] > 1,:]
                
        input_data = pd.merge(input_data, duplicate_aa, how = 'left')                
        varity_no_duplicate_aa = input_data.loc[input_data['count'].isnull(),:]
        
        alm_fun.show_msg(cur_log,self.verbose,data_name + ' without duplication : ' + str(varity_no_duplicate_aa.shape[0]))
        varity_duplicate_aa = input_data.loc[input_data['count'].notnull(),:]                
        alm_fun.show_msg(cur_log,self.verbose,data_name + ' with duplication : ' + str(varity_duplicate_aa.shape[0]))
        varity_duplicate_aa.to_csv(self.db_path  + 'varity/csv/varity_duplicate_' + data_name + '.csv')   
     
        # the source of each NT change record apply to the AA change record (clinvar and hgmd)
#         varity_duplicate_aa_source = varity_duplicate_aa.groupby(['p_vid','aa_pos','aa_ref','aa_alt'])['hgmd_source','clinvar_source'].agg(duplicate_aa_process).reset_index()
        
        varity_duplicate_aa_source = varity_duplicate_aa.groupby(['p_vid','aa_pos','aa_ref','aa_alt'])['hgmd_source','clinvar_source'].agg(np.nanmax).reset_index()
        varity_duplicate_aa = varity_duplicate_aa.drop(columns = {'hgmd_source','clinvar_source'})
        varity_duplicate_aa = varity_duplicate_aa.merge(varity_duplicate_aa_source,how = 'left')
        
        # Pick a variant with rules (most common by allele frequency)
        varity_duplicate_aa['ac'] = varity_duplicate_aa['gnomAD_exomes_AC']        
        varity_duplicate_aa.loc[varity_duplicate_aa['ac'].isnull(),'ac'] = 0        
        varity_duplicate_aa['index'] = varity_duplicate_aa.index
        
        varity_duplicate_aa_ac_lst = varity_duplicate_aa.groupby(['p_vid','aa_pos','aa_ref','aa_alt'])['ac','index'].agg(list).reset_index()
        varity_duplicate_aa_ac_lst['selected_index'] = varity_duplicate_aa_ac_lst.apply(lambda x: x['index'][x['ac'].index(np.nanmax(x['ac']))],axis = 1)        
        
        varity_duplicate_aa_selected = varity_duplicate_aa.loc[varity_duplicate_aa_ac_lst['selected_index'],:]        
        varity_duplicate_aa_selected = varity_duplicate_aa_selected.drop(columns = {'ac'})     
        
        input_data = pd.concat([varity_duplicate_aa_selected,varity_no_duplicate_aa])
        alm_fun.show_msg(cur_log,self.verbose,'Varity records after removing duplicated AA changes : ' + str(input_data.shape[0]))
        
        if merge_mave == 1:
            #************************************************************************************        
            # Adding MAVE data (part of the MAVE data are lost after dbnsfp filtering
            #************************************************************************************
            mave_missense = pd.read_csv(self.db_path + 'mave/all/mave_missense.csv')  
            input_data = input_data.merge(mave_missense,'outer')        
            alm_fun.show_msg(cur_log,self.verbose,'Varity records after merging with MAVE : ' + str(input_data.shape[0]))
            
            input_data.loc[input_data['SIFT_selected_score'].isnull(),'SIFT_selected_score'] = input_data.loc[input_data['SIFT_selected_score'].isnull(),'sift_score']
            input_data.loc[input_data['PROVEAN_selected_score'].isnull(),'PROVEAN_selected_score'] = input_data.loc[input_data['PROVEAN_selected_score'].isnull(),'provean_score']
        else:
            #************************************************************************************        
            # Adding MAVE data (part of the MAVE data are lost after dbnsfp filtering)
            #************************************************************************************
            mave_missense = pd.read_csv(self.db_path + 'mave/all/mave_missense.csv')  
            input_data = pd.merge(input_data,mave_missense,how = 'left')        
                
        uniprot_seqdict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy',allow_pickle = True).item()
        input_data_variant_processed = self.variants_process(data_name, input_data, uniprot_seqdict, self.flanking_k, nt_input = 1, gnomad_available = 1, gnomad_merge_id = 'uniprot_id',cur_log = cur_log)
        
        return(input_data_variant_processed)
        
    def varity_train_variant_process(self):                    
        varity_train_data = pd.read_csv(self.db_path + 'varity/csv/varity_train_data.csv',dtype = {'chr':'str'})       
        varity_train_variant_processed = self.varity_variant_process(varity_train_data,1,'varity_train_data')    
        varity_train_variant_processed.to_csv(self.db_path + 'varity/csv/varity_train_variant_processed.csv', index = False)
    
    def varity_single_protein_variant_process(self,single_id):
        varity_train_data = pd.read_csv(self.db_path + 'varity/csv/varity_merged_' + single_id + '.csv',dtype = {'chr':'str'})       
        varity_train_variant_processed = self.varity_variant_process(varity_train_data,0,'varity_single_protein_' + single_id)    
        varity_train_variant_processed.to_csv(self.db_path + 'varity/csv/varity_' + single_id + '_variant_processed.csv', index = False)
    
    def varity_single_protein_data_final_process(self,single_id):                            
        varity_single_protein_data = pd.read_csv(self.db_path + 'varity/csv/varity_' + single_id + '_variant_processed.csv') 
        self.varity_final_process(varity_single_protein_data,'varity_all_' +  single_id)        
            
    def varity_train_data_final_process(self):                            
        varity_train_data = pd.read_csv(self.db_path + 'varity/csv/varity_train_variant_processed.csv') 
        self.varity_final_process(varity_train_data,'varity_train_data')        
        
    def varity_all_data_final_process(self,parallel_id):        
        varity_variant_processed_data = pd.read_csv(self.db_path + 'varity/csv/varity_all_variant_processed_' + str(parallel_id) + '.csv',dtype = {'chr':'str'})
        self.varity_final_process(varity_variant_processed_data,'varity_all_' + parallel_id)    
            
    def varity_final_process(self,input_data,data_name):
        cur_log = self.db_path + 'varity/log/' + data_name + '_final_process.log'
        def define_label(mave_label,clinvar_label,gnomad_label):            
            label_lst = [mave_label,clinvar_label,gnomad_label]            
            has_zero = 0 in label_lst
            has_one = 1 in label_lst
                
            if has_zero and has_one:
                return(-1)
            if has_zero and not has_one:
                return(0)
            if not has_zero and has_one:
                return(1)
            if not has_zero and not has_one:
                return(-1) 
                            
        alm_fun.show_msg(cur_log,self.verbose,data_name + ' before final process : ' + str(input_data.shape[0]))
        
        
         ####***************************************************************************************************************************************************************
        # remove P84243 from chromosome 17 which is in chromosome 1 as well (in fact two different genes but share same uniprot id)
        ####***************************************************************************************************************************************************************
        if   data_name ==  'varity_all_17_1':
            input_data = input_data.loc[~((input_data['p_vid'] == 'P84243') & (input_data['chr'] == '1')) ,:]
            alm_fun.show_msg(cur_log,self.verbose,data_name + ' after removing P84243 : ' + str(input_data.shape[0]))

        ####***************************************************************************************************************************************************************
        # check if the aa_ref is correct for aa_pos and p_vid based on the uniprot seq
        ####***************************************************************************************************************************************************************
        uniprot_seqdict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy',allow_pickle = True).item()
        input_data['aa_ref_chk'] = input_data.apply(lambda x: uniprot_seqdict.get(x['p_vid'],'')[x['aa_pos']-1],axis = 1)        
        input_data.loc[input_data['aa_ref_chk'] != input_data['aa_ref'],['p_vid','aa_pos','aa_ref_chk','aa_ref','aa_alt']]            
        input_data = input_data.loc[input_data['aa_ref_chk'] == input_data['aa_ref'],:]
        alm_fun.show_msg(cur_log,self.verbose,data_name + ' after removing aa_ref not the same to aa_ref_chk: ' + str(input_data.shape[0])) 

        if data_name == 'varity_train_data':               
            ####***************************************************************************************************************************************************************
            # exclude 'Q8WZ42' the long protein
            ####***************************************************************************************************************************************************************
            input_data_final = input_data.loc[input_data['p_vid']!= 'Q8WZ42',:]     
            alm_fun.show_msg(cur_log,self.verbose,data_name + ' after removing Q8WZ42: ' + str(input_data_final.shape[0]))

            ####***************************************************************************************************************************************************************
            #some of the psipred info was run previously and the uniprot seq has changed since then, so need to be rerun
            ####***************************************************************************************************************************************************************
    #         input_data_final = input_data_final.loc[input_data_final['aa_psipred'].notnull(),:]      
    #         varity_train_psipred_rerun_ids = input_data_final.loc[input_data_final['p_vid'] == 'O75445',['aapos','Ensembl_proteinid','Uniprot_acc','Uniprot_acc(HGNC/Uniprot)']]
    #         input_data_final.loc[input_data_final['p_vid'] == 'O75445',['p_vid','aa_pos','aa_ref','aa_alt','aa_psipred']]
                            
            ####***************************************************************************************************************************************************************
            # Decide the lables for differnt type of data (MAVE, ClinVAR, Humsavar, Gnomad)
            ####***************************************************************************************************************************************************************            
            input_data_final['mave_label'] = -1
            input_data_final.loc[(input_data_final['mave_source'] == 1) & (input_data_final['fitness'] >= 0.5), 'mave_label'] = 0
            input_data_final.loc[(input_data_final['mave_source'] == 1) & (input_data_final['fitness'] < 0.5), 'mave_label'] = 1
            input_data_final['mave_label_confidence'] = np.nan
            input_data_final.loc[(input_data_final['mave_source'] == 1),'mave_label_confidence'] = np.abs(input_data_final.loc[(input_data_final['mave_source'] == 1),'mave_label'] - input_data_final.loc[(input_data_final['mave_source'] == 1),'fitness']) 
                 
            input_data_final['gnomad_label'] = -1
    #         input_data_final.loc[(input_data_final['gnomad_source'] == 1) & (input_data_final['gnomAD_exomes_nhomalt'] > 0), 'gnomad_label'] = 0
            input_data_final.loc[(input_data_final['gnomad_source'] == 1), 'gnomad_label'] = 0
              
            ####***************************************************************************************************************************************************************
            # Check the label conflicts
            ####***************************************************************************************************************************************************************
    #         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] == input_data_final['clinvar_label']),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
    #         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] == input_data_final['clinvar_label']) & (input_data_final['humsavar_label'] != -1) * (input_data_final['clinvar_label'] != -1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
    #         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] != input_data_final['clinvar_label']),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
    #         
    #         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  0 ) & (input_data_final['clinvar_label'] == 1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']]
    #         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  1 ) & (input_data_final['clinvar_label'] == 0),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
    #         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  -1 ) & (input_data_final['clinvar_label'] == 0),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
    #         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  -1 ) & (input_data_final['clinvar_label'] == 1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
    #         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  0 ) & (input_data_final['clinvar_label'] == -1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
    #         input_data_final.loc[(input_data_final['humsavar_source'] == 1) & (input_data_final['clinvar_source'] == 1) & (input_data_final['humsavar_label'] ==  1 ) & (input_data_final['clinvar_label'] == -1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','humsavar_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level','clinvar_id']].shape
         
    #         input_data_final.loc[(input_data_final['mave_label'] == 1) & (input_data_final['clinvar_label'] == 0),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level']].shape
    #         input_data_final.loc[(input_data_final['mave_label'] == 0) & (input_data_final['clinvar_label'] == 1),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level']].shape
    #         input_data_final.loc[(input_data_final['mave_label'] == 1) & (input_data_final['gnomad_label'] == 0),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level']].shape
    #         input_data_final.loc[(input_data_final['clinvar_label'] == 1) & (input_data_final['gnomad_label'] == 0),['p_vid','aa_pos','aa_ref','aa_alt','fitness','gnomad_label','mave_label','clinvar_label','clinvar_review_star','clinvar_clinsig_level']].shape
     
            ####***************************************************************************************************************************************************************
            # Decide the final labels ( if there are overlap in terms of label from different sources, use clinvar label)
            ####***************************************************************************************************************************************************************
            input_data_final.loc[input_data_final['clinvar_source'].isnull(),'clinvar_source'] = 0
            input_data_final.loc[input_data_final['hgmd_source'].isnull(),'hgmd_source'] = 0
            input_data_final.loc[input_data_final['humsavar_source'].isnull(),'humsavar_source'] = 0
            input_data_final.loc[input_data_final['gnomad_source'].isnull(),'gnomad_source'] = 0
            input_data_final.loc[input_data_final['mave_source'].isnull(),'mave_source'] = 0
         
            input_data_final['train_clinvar_source'] = input_data_final['clinvar_source']
            input_data_final.loc[~input_data_final['clinvar_label'].isin([0,1]) ,'train_clinvar_source'] = 0
            input_data_final['train_hgmd_source'] = input_data_final['hgmd_source']
            input_data_final.loc[~input_data_final['hgmd_label'].isin([0,1]) ,'train_hgmd_source'] = 0
            input_data_final['train_humsavar_source'] = input_data_final['humsavar_source']
            input_data_final.loc[~input_data_final['humsavar_label'].isin([0,1]) ,'train_humsavar_source'] = 0
            input_data_final['train_gnomad_source'] = input_data_final['gnomad_source']
            input_data_final.loc[~input_data_final['gnomad_label'].isin([0,1]) ,'train_gnomad_source'] = 0
            input_data_final['train_mave_source'] = input_data_final['mave_source']
            input_data_final.loc[~input_data_final['mave_label'].isin([0,1]) ,'train_mave_source'] = 0
             
            # when there is a ClinVAR label use it as the final label       
            input_data_final.loc[input_data_final['train_clinvar_source'] == 1 ,'train_hgmd_source'] = 0  
            input_data_final.loc[input_data_final['train_clinvar_source'] == 1 ,'train_humsavar_source'] = 0       
            input_data_final.loc[input_data_final['train_clinvar_source'] == 1 ,'train_mave_source'] = 0
            input_data_final.loc[input_data_final['train_clinvar_source'] == 1 ,'train_gnomad_source'] = 0
             
            # after Clinvar when there is a HGMD label use it as the final label                
            input_data_final.loc[input_data_final['train_hgmd_source'] == 1 ,'train_humsavar_source'] = 0  
            input_data_final.loc[input_data_final['train_hgmd_source'] == 1 ,'train_mave_source'] = 0    
            input_data_final.loc[input_data_final['train_hgmd_source'] == 1 ,'train_gnomad_source'] = 0  
     
            # after HGMD, when there is a HumsaVAR label use it as final label
            input_data_final.loc[input_data_final['train_humsavar_source'] == 1 ,'train_mave_source'] = 0
            input_data_final.loc[input_data_final['train_humsavar_source'] == 1 ,'train_gnomad_source'] = 0    
     
            # after humsaVAR, when there is a MAVE label use it as final label
            input_data_final.loc[input_data_final['train_mave_source'] == 1 ,'train_gnomad_source'] = 0
         
            # decide the final label
            input_data_final['label'] = -1
            input_data_final.loc[input_data_final['train_clinvar_source'] == 1 ,'label'] = input_data_final.loc[input_data_final['train_clinvar_source'] == 1 ,'clinvar_label']
            input_data_final.loc[input_data_final['train_hgmd_source'] == 1 ,'label'] = input_data_final.loc[input_data_final['train_hgmd_source'] == 1 ,'hgmd_label']
            input_data_final.loc[input_data_final['train_humsavar_source'] == 1 ,'label'] = input_data_final.loc[input_data_final['train_humsavar_source'] == 1 ,'humsavar_label']
            input_data_final.loc[input_data_final['train_mave_source'] == 1 ,'label'] = input_data_final.loc[input_data_final['train_mave_source'] == 1 ,'mave_label']
            input_data_final.loc[input_data_final['train_gnomad_source'] == 1 ,'label'] = input_data_final.loc[input_data_final['train_gnomad_source'] == 1 ,'gnomad_label']
                      
            #remove the record with label -1 (unclassfied by clinvar or humsavar)
    #         input_data_final = input_data_final.loc[input_data_final['label'].isin([0,1]),:]
    #         alm_fun.show_msg(cur_log,self.verbose,data_name + ' after removing not 0 or 1 label: ' + str(input_data_final.shape[0])) 
      
        else:
            input_data_final = input_data 
            input_data_final['label'] = np.random.randint(2,size = input_data_final.shape[0])
        
        ####***************************************************************************************************************************************************************
        # Miscellaneous
        # 1) Add weight column and set default value to 1
        # 2) Add contamination column and set default value to 0
        # 3) Set Gnomad columns to 0 if nan  
        # 4) data source labelling
        ####***************************************************************************************************************************************************************
  
        input_data_final['weight'] = 1
        input_data_final['contamination'] = 0
          
        input_data_final.loc[input_data_final['gnomAD_exomes_AC'].isnull(), 'gnomAD_exomes_AC'] = 0
        input_data_final.loc[input_data_final['gnomAD_exomes_AF'].isnull(), 'gnomAD_exomes_AF'] = 0
        input_data_final.loc[input_data_final['gnomAD_exomes_AN'].isnull(), 'gnomAD_exomes_AN'] = 0
        input_data_final.loc[input_data_final['gnomAD_exomes_nhomalt'].isnull(), 'gnomAD_exomes_nhomalt'] = 0
          
        input_data_final.loc[input_data_final['gnomAD_exomes_controls_AC'].isnull(), 'gnomAD_exomes_controls_AC'] = 0
        input_data_final.loc[input_data_final['gnomAD_exomes_controls_AF'].isnull(), 'gnomAD_exomes_controls_AF'] = 0
        input_data_final.loc[input_data_final['gnomAD_exomes_controls_AN'].isnull(), 'gnomAD_exomes_controls_AN'] = 0
        input_data_final.loc[input_data_final['gnomAD_exomes_controls_nhomalt'].isnull(), 'gnomAD_exomes_controls_nhomalt'] = 0
          
        input_data_final.loc[input_data_final['clinvar_review_star'].isnull(),'clinvar_review_star'] = -1
        input_data_final.loc[input_data_final['clinvar_clinsig_level'].isnull(),'clinvar_clinsig_level'] = -1
  
        if data_name == 'varity_train_data':
            input_data_final['fitness'] = 1 - input_data_final['fitness']
            input_data_final_less_gnomad = input_data_final.loc[ ~((input_data_final['gnomAD_exomes_nhomalt'] == 0) & (input_data_final['train_gnomad_source'] == 1)),:]
            input_data_final_less_gnomad.to_csv(self.db_path + 'varity/csv/'  + data_name + '_final_less_gnomad.csv',index = False) 
        
        
        input_data_final.to_csv(self.db_path + 'varity/csv/'  + data_name + '_final.csv',index = False)
            
    def varity_final_process_jobs(self):
        print ("runing varaity_final_process_jobs!")
        parallel_ids = []
        for file in glob.glob(self.db_path + 'varity/csv/varity_all_variant_processed*.csv'):
            print (file)            
            parallel_id = file.split(".")[0].split('_')[-2] + '_' + file.split(".")[0].split('_')[-1]
            parallel_ids.append(parallel_id)
        parallel_ids = np.unique(parallel_ids)
        
        for parallel_id in parallel_ids: 
            alm_fun.show_msg(self.log,self.verbose,'Run varity final process  on  ' + parallel_id + ' chromosome.')         
            #*************************************************************************************************
            #run parallel jobs 
            #*************************************************************************************************
            varity_final_process_cmd = "python3 " +  self.project_path +  "python/humandb_debug.py" + " '" + self.python_path + "'"  + " '"  + self.project_path + "' " + "'" +  self.db_path + "'" + " 'run_humandb_action' " + "'varity_all_data_final_process' " + "'" +  parallel_id + "' " + "0" 
            job_name = 'varity_all_final_process_' + parallel_id 
            
            if self.cluster == 'galen':                              
                sh_file = open(self.db_path + 'varity/bat/varity_final_process_' + parallel_id + '.sh','w')  
                sh_file.write('#!/bin/bash' + '\n')
                sh_file.write('# set the number of nodes' + '\n')
                sh_file.write('#SBATCH --nodes=1' + '\n')
                sh_file.write('# set the memory for each node' + '\n')
                sh_file.write('#SBATCH --mem=' + '10240' + '\n')                  
                sh_file.write('# set name of job' + '\n')
                sh_file.write('#SBATCH --job-name=' + job_name + '\n')
                sh_file.write('# mail alert at start, end and abortion of execution' + '\n')
                sh_file.write('#SBATCH --mail-type=ALL' + '\n')
                sh_file.write('# send mail to this address' + '\n')
                sh_file.write('#SBATCH --mail-user=joe.wu.ca@gmail.com' + '\n')      
                sh_file.write("srun " + varity_final_process_cmd)
                sh_file.close()
                                 
                sbatch_cmd = "sbatch " + self.db_path + 'varity/bat/varity_final_process_' + parallel_id + '.sh'
                print (sbatch_cmd)
                job_id = '-1'
                while  job_id == '-1':
                    return_process = subprocess.run(sbatch_cmd.split(" "), cwd = self.db_path + 'varity/log/',capture_output = True,text=True)
                    time.sleep(0.1)
                    if return_process.returncode == 0:
                        job_id = return_process.stdout.rstrip().split(' ')[-1]
                    else:
                        job_id = '-1'
                        print  (job_name + ' submit error,rescheduling......' )
                print (job_name + ' submitted id: ' + job_id)        
                
    def varity_train_data_analysis(self):

        varity_train_data_final = pd.read_csv(self.db_path + 'varity/csv/varity_train_data_final.csv')
        

        varity_train_data_final_no_gnomad = varity_train_data_final.loc[(varity_train_data_final['mave_source'] == 1) | (varity_train_data_final['clinvar_source'] == 1),:]
        varity_train_data_final_no_gnomad.to_csv(self.db_path + 'varity/csv/varity_train_data_final_no_gnomad.csv',index = False)
        #Check if the nan in each column make sense 
        column_counts_df = varity_train_data.count(axis = 0)
        column_counts_df.to_csv(self.db_path  + 'varity/csv/varity_train_column_counts.csv')
        varity_train_data['p_vid'].isnull().sum()
        
        #check how many duplicated amino acid changes
        

        varity_train_data_duplicate_aa_change = varity_train_data_duplicate_aa_change.loc[varity_train_data_duplicate_aa_change['count'] > 1,:]
        varity_train_data_duplicate_aa_change.to_csv(self.db_path  + 'varity/csv/duplicate_aa_change.csv')
        
        print ("OK")
                                                  
    def process_dbnsfp_out(self,input_out,cur_log):
        #************************************************************************************************************************************************************************
        ## Define a few functions for processing dbnsfp result
        #************************************************************************************************************************************************************************   
        def retrieve_ensembl_canonical_index(vep_canonical):
            try:
                vep_canonical_list = vep_canonical.split(";")
                canonical_index = vep_canonical_list.index('YES')
            except:
                canonical_index = np.nan
            return canonical_index

        def retrieve_uniprot_canonical_index(uniprot_hgnc_id,uniprot_ids):
            try:
                unprot_ids_list = uniprot_ids.split(";")
                canonical_index = unprot_ids_list.index(uniprot_hgnc_id)
            except:
                canonical_index = np.nan
            return canonical_index        
                         
        def retrieve_value_by_canonical_index(values,canonical_index):
            try:
                if not np.isnan(canonical_index):
                    values_list = values.split(";")
                    value = values_list[np.int(canonical_index)]
                else:
                    value = np.nan
            except:
                value = np.nan
            return value
                         
        def retrieve_aapos(uniprot_accs, uniprot_acc, uniprot_aaposs):
            try:        
                uniprot_accs_list = uniprot_accs.split(";")
                uniprot_poss_list = uniprot_aaposs.split(";")
                  
                if len(uniprot_poss_list) == 1:
                    uniprot_aa_pos = uniprot_poss_list[0]
                else:
                    unprot_accs_dict = {uniprot_accs_list[x]:x for x in range(len(uniprot_accs_list))}        
                    uniprot_aa_pos = uniprot_poss_list[unprot_accs_dict.get(uniprot_acc,np.nan)]
                if not chk_int(uniprot_aa_pos):
                    uniprot_aa_pos = np.nan
                else:
                    uniprot_aa_pos = int(uniprot_aa_pos)
                      
            except:
                uniprot_aa_pos = np.nan
            return uniprot_aa_pos
              
        def chk_int(str):
            try:
                x = int(str)        
                return True
            except:
                return False
          
        def chk_float(str):
            try:
                x = float(str)        
                return True
            except:
                return False
              
        def get_value_byfun(values, fun):
            try:
                value_list = values.split(";")
                value_list = [float(x) for x in value_list if chk_float(x)]
                if fun == 'min':
                    value = min(value_list)
                if fun == 'max':
                    value = max(value_list)
            except:
                value = np.nan
            return value
          
        def get_residue_by_pos(seq,pos):
            try:
                residue = seq[pos-1]
            except:
                residue = np.nan
            return residue
        pass
                                            
        basic_cols = ['hg19_chr','hg19_pos(1-based)', 'ref', 'alt','aapos','aaref','aaalt','genename','Ensembl_geneid','Ensembl_transcriptid','Ensembl_proteinid','Uniprot_acc','VEP_canonical'] + \
                     ['refcodon','codonpos','codon_degeneracy']
        
        score_cols = ['SIFT_score','SIFT4G_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','LRT_score','MutationTaster_score','MutationAssessor_score','FATHMM_score','PROVEAN_score','VEST4_score','PrimateAI_score'] + \
                     ['MetaSVM_score','MetaLR_score','M-CAP_score','REVEL_score','MutPred_score','CADD_raw','DANN_score','fathmm-MKL_coding_score','Eigen-raw_coding','GenoCanyon_score'] + \
                     ['integrated_fitCons_score','GERP++_RS','phyloP100way_vertebrate','phyloP30way_mammalian','phastCons100way_vertebrate','phastCons30way_mammalian','SiPhy_29way_logOdds']
 
        gnomad_cols = ['gnomAD_exomes_AC','gnomAD_exomes_AN','gnomAD_exomes_AF','gnomAD_exomes_nhomalt','gnomAD_exomes_controls_AC','gnomAD_exomes_controls_AN','gnomAD_exomes_controls_AF','gnomAD_exomes_controls_nhomalt']
 
        gene_cols = ['Uniprot_acc(HGNC/Uniprot)','CCDS_id','Refseq_id','ucsc_id','MIM_id','MIM_phenotype_id','MIM_disease','gnomAD_pLI','gnomAD_pRec','gnomAD_pNull','HIPred_score']
 
        all_cols = basic_cols + score_cols + gnomad_cols + gene_cols
        output_snv= input_out[all_cols]

        #******************************************************************************************************          
        #Use hgnc uniprot_id as the key to search cnonical transcript
        #******************************************************************************************************
#         output_snv.loc[output_snv['Uniprot_acc(HGNC/Uniprot)'].str.contains(';'),['Uniprot_acc','Uniprot_acc(HGNC/Uniprot)']]
#         output_snv.loc[output_snv['p_vid'] == 'O96033',:].to_csv(self.db_path + 'varity/csv/O96033.csv',index = False)

#         output_snv['ensembl_canonical_index'] = output_snv.apply(lambda x: retrieve_ensembl_canonical_index(x['VEP_canonical']),axis = 1)
        
        
#         output_snv['p_vid'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['Uniprot_acc(HGNC/Uniprot)'],x['Uniprot_acc']),axis = 1)
#         output_snv['p_vid'] = output_snv['Uniprot_acc(HGNC/Uniprot)']

        output_snv['p_vid'] = output_snv['Uniprot_acc(HGNC/Uniprot)'].apply(lambda x: x.split(';')[-1])
        output_snv['uniprot_canonical_index'] = output_snv.apply(lambda x: retrieve_uniprot_canonical_index(x['p_vid'],x['Uniprot_acc']),axis = 1)
        output_snv['aa_pos'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['aapos'],x['uniprot_canonical_index']),axis = 1)
        output_snv = output_snv.rename(columns = {'hg19_chr':'chr','hg19_pos(1-based)': 'nt_pos', 'ref':'nt_ref','alt':'nt_alt','aaref':'aa_ref','aaalt':'aa_alt'})

        #******************************************************************************************************          
        #get the most delterious score for scores have multiple values due to different transcripts
        #******************************************************************************************************
        output_snv['SIFT_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['SIFT_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['SIFT4G_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['SIFT4G_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['Polyphen2_selected_HDIV_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['Polyphen2_HDIV_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['Polyphen2_selected_HVAR_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['Polyphen2_HVAR_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['PROVEAN_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['PROVEAN_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['VEST4_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['VEST4_score'],x['uniprot_canonical_index']),axis = 1)       
        output_snv['FATHMM_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['FATHMM_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['MutationAssessor_selected_score'] = output_snv.apply(lambda x: retrieve_value_by_canonical_index(x['MutationAssessor_score'],x['uniprot_canonical_index']),axis = 1)
        output_snv['MutationTaster_selected_score'] = output_snv.apply(lambda x: get_value_byfun(x['MutationTaster_score'],'max'),axis = 1)
                 
        #************************************************************************************
        # replace "." with np.nan
        #************************************************************************************
        output_snv = output_snv.replace('.',np.nan)
        output_snv = output_snv.replace('-',np.nan)
        score_cols = [x for x in output_snv.columns if '_score' in x]
        score_cols =  set(score_cols) - set(['SIFT_score','SIFT4G_score','Polyphen2_HDIV_score','Polyphen2_HVAR_score','MutationTaster_score','MutationAssessor_score','FATHMM_score','PROVEAN_score','VEST4_score'])
        for score_col in score_cols:
            output_snv[score_col] = output_snv[score_col].astype(float)
            
        #************************************************************************************
        # remove irregular records
        #************************************************************************************
        output_snv = output_snv.loc[output_snv['aa_pos'].notnull() & output_snv['aa_ref'].notnull() & output_snv['aa_alt'].notnull() & output_snv['p_vid'].notnull() ,:]        
        #************************************************************************************
        
        #************************************************************************************
        # remove nonsense and read through records
        #************************************************************************************
#         output_snv = output_snv.loc[(output_snv['aa_ref'] != 'X') & (output_snv['aa_alt'] !='X'),:]        
        #************************************************************************************
        output_snv['aa_pos'] = output_snv['aa_pos'].astype(int)        
        output_snv_processed = output_snv
        
        # run variant processing
        #************************************************************************************    
#         uniprot_seqdict = np.load(self.db_path + 'uniprot/npy/uniprot_seq_dict.npy').item()
#         output_snv_processed = self.variants_process('varity_dbnsfp', output_snv, uniprot_seqdict, self.flanking_k, nt_input = 1, gnomad_available = 1, gnomad_merge_id = 'uniprot_id')

        #************************************************************************************
        #remove records with duplicated coordinatees
        #************************************************************************************
#         output_snv_groupby = output_snv.groupby(['p_vid','aa_pos','aa_ref','aa_alt']).size().reset_index()
#         output_snv_groupby = output_snv_groupby.rename(columns = {0:'aa_counts'})
#         output_snv = pd.merge(output_snv,output_snv_groupby,how = 'left')  
#         output_snv = output_snv.loc[output_snv['aa_counts'] == 1,:]    
#         alm_fun.show_msg(self.log,self.verbose,'Varity records after removing AA duplicated records : ' + str(output_snv.shape[0]))
            
        #************************************************************************************
        # mark polyphen training data
        #************************************************************************************
#         self.polyphen_train = self.get_polyphen_train_data()
#         output_snv = pd.merge(output_snv, self.polyphen_train, how='left')
        return(output_snv_processed)

    def variants_process(self, data_name, variants_df, seq_dict, k, nt_input, gnomad_available, gnomad_merge_id,cur_log):        
        if 'varity_single_protein' in data_name:
            single_id = data_name.split('_')[-1]            
        else:
            single_id = ''
        
        total_stime = time.time()  
        alm_fun.show_msg(cur_log,self.verbose,"variants processing for : [" + str(data_name) + "]")
        variants_df['aa_pos'] = variants_df['aa_pos'].astype(int)
        variants_df['chr'] = variants_df['chr'].astype(str)   
#         variants_df = variants_df.loc[variants_df['p_vid'].isin(seq_dict.keys()), :]
#         variants_df['aa_len'] = variants_df['p_vid'].apply(lambda x: len(seq_dict[x]))
     
#         varaints_df = variants_df.drop_duplicates()
                   
#         ####***************************************************************************************************************************************************************
#         #### merge with the asa (accessible solvent area)
#         ####***************************************************************************************************************************************************************  
#         stime = time.time() 
#         self.pdb_asa = pd.read_csv(self.db_path + 'pdb/csv/asa_df.csv', header=None)
#         self.pdb_asa.columns = ['aa_pos', 'aa_ref', 'asa_mean', 'asa_std', 'asa_count', 'p_vid']                    
#         variants_df = pd.merge(variants_df, self.pdb_asa, how='left') 
#         etime = time.time() 
#         print ("merge with pdb asa: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
#     
        ####***************************************************************************************************************************************************************
        #### merge with the pisa (including accessible solvent area)
        ####***************************************************************************************************************************************************************
        stime = time.time() 
        variants_df = self.add_pisa(variants_df)
        etime = time.time() 
        alm_fun.show_msg(cur_log,self.verbose,"merge with pdb pisa: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
     
        ####***************************************************************************************************************************************************************
        #### merge with the pdb secondary structure
        ####***************************************************************************************************************************************************************  
#         stime = time.time() 
#         self.pdbss = pd.read_csv(self.db_path + 'pdbss/pdbss_final.csv')
#         variants_df = pd.merge(variants_df, self.pdbss, how='left') 
#         # encode      
#         pdbss1_dict = {'E':1, 'H':2, 'C':3, 'T':4}
#         pdbss_dict = {'H':1, 'G':2, 'I':3, 'B':4, 'E':5, 'T':6, 'S':7, 'C':8}
#         variants_df['aa_ss_encode'] = variants_df['aa_ss'].apply(lambda x:pdbss_dict.get(x, np.nan))
#         variants_df['aa_ss1_encode'] = variants_df['aa_ss1'].apply(lambda x:pdbss1_dict.get(x, np.nan)) 
#         etime = time.time()        
#         print ("merge with pdbss: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
#           
        ####***************************************************************************************************************************************************************
        #### merge with the psipred secondary structure
        ####***************************************************************************************************************************************************************
        stime = time.time() 
        variants_df = self.add_psipred(variants_df,single_id)
        etime = time.time()
        alm_fun.show_msg(cur_log,self.verbose,"merge with psipred: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
        
             
        ####***************************************************************************************************************************************************************
        #### merge with the pfam domain information
        ####****************************************************************************************************************************************
        stime = time.time() 
        variants_df = self.add_pfam(variants_df)
        etime = time.time()
        alm_fun.show_msg(cur_log,self.verbose,"merge with pfam: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime)) 
             
        ####***************************************************************************************************************************************************************
        #### merge with the subcelluar localization information
        ####****************************************************************************************************************************************
        stime = time.time() 
        variants_df = self.add_sublocation(variants_df)
        etime = time.time()
        alm_fun.show_msg(cur_log,self.verbose,"merge with sublocation: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime)) 

        ####***************************************************************************************************************************************************************
        #### merge with the allel frequency properties
        ####***************************************************************************************************************************************************************
#         if gnomad_available == 0:
#             stime = time.time() 
#             if nt_input == 1:    
#                 gnomad_nt = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_nt.txt',sep = '\t',dtype={"chr": str})
#                 variants_df = pd.merge(variants_df, self.gnomad_nt, how='left')
#             else: 
# #                 gnomad_aa = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_aa.txt',sep = '\t')
#                 gnomad_aa = pd.read_csv(self.db_path+'gnomad/gnomad_output_snp_aa_uniprot.txt',sep = '\t')
#                 gnomad_aa.rename(columns={gnomad_merge_id: 'p_vid'}, inplace=True)
#                 gnomad_aa['aa_pos'] = gnomad_aa['aa_pos'].astype(int)
#                 variants_df = pd.merge(variants_df, gnomad_aa, how='left')             
#             etime = time.time()         
#             print ("merge with gnomad: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))     
#                 
             
        ####***************************************************************************************************************************************************************   
        # Polyphen (http://genetics.bwh.harvard.edu/pph2/bgi.shtml)
        ####***************************************************************************************************************************************************************
#         stime = time.time()
#         polyphen_new_score = pd.read_csv(self.db_path + 'polyphen/org/' + data_name + '_pph2-full.txt', sep='\t')
#         polyphen_new_score.columns = np.char.strip(polyphen_new_score.columns.get_values().astype(str))
#         polyphen_new_score = polyphen_new_score[['#o_acc', 'o_pos', 'o_aa1', 'o_aa2', 'pph2_prob']]
#         polyphen_new_score.columns = ['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'polyphen_new_score']
#         polyphen_new_score['p_vid'] = polyphen_new_score['p_vid'].str.strip()
#         polyphen_new_score['aa_ref'] = polyphen_new_score['aa_ref'].str.strip()
#         polyphen_new_score['aa_alt'] = polyphen_new_score['aa_alt'].str.strip()
#           
#         polyphen_new_score['polyphen_new_score'] = polyphen_new_score['polyphen_new_score'].astype(str)
#         polyphen_new_score['polyphen_new_score'] = polyphen_new_score['polyphen_new_score'].str.strip()
#         polyphen_new_score = polyphen_new_score.loc[polyphen_new_score['polyphen_new_score'] != '?', :]
#         polyphen_new_score['polyphen_new_score'] = polyphen_new_score['polyphen_new_score'].astype(float)
#         polyphen_new_score = polyphen_new_score.loc[polyphen_new_score['aa_pos'].notnull(), :]
#         polyphen_new_score['aa_pos'] = polyphen_new_score['aa_pos'].astype(int)
#     
#         polyphen_new_score.drop_duplicates(inplace = True)
#         variants_df = pd.merge(variants_df, polyphen_new_score, how='left')
#         etime = time.time()  
#         print ("merge with polyphen: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
         
        ####***************************************************************************************************************************************************************   
        # SIFT and PROVEAN (http://provean.jcvi.org/protein_batch_submit.php?species=human)
        ####***************************************************************************************************************************************************************
#         stime = time.time()
#         sift_new_score = pd.read_csv(self.db_path + 'provean/org/' + data_name + '_provean.tsv', sep='\t')[['PROTEIN_ID', 'POSITION', 'RESIDUE_REF', 'RESIDUE_ALT', 'SCORE', 'SCORE.1']]
#         sift_new_score.columns = ['p_vid', 'aa_pos', 'aa_ref', 'aa_alt', 'provean_new_score', 'sift_new_score']
#         sift_new_score = sift_new_score.drop_duplicates()
#         variants_df = pd.merge(variants_df, sift_new_score, how='left')
#         etime = time.time()
#         print ("merge with sift and provean: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
#         
        ####***************************************************************************************************************************************************************
        #### EV-mutation score (https://marks.hms.harvard.edu/evmutation/human_proteins.html)
        ####***************************************************************************************************************************************************************                
        stime = time.time()
        self.evm_score = pd.read_csv(self.db_path + 'evmutation/all/evmutation_df.csv')
   
        # ['mutation','aa_pos','aa_ref','aa_alt','evm_epistatic_score','evm_independent_score','evm_frequency','evm_conservation']
        variants_df = pd.merge(variants_df, self.evm_score, how='left')
        etime = time.time()
        alm_fun.show_msg(cur_log,self.verbose,"merge with evmutation: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
           
        ####***************************************************************************************************************************************************************
        #### envision
        ####***************************************************************************************************************************************************************
#         stime = time.time()
#         self.envision_score = pd.read_csv(self.db_path + 'envision/csv/envision_score_for_extrapolation_processed.csv')
#         variants_df = pd.merge(variants_df, self.envision_score, how='left')
#         etime = time.time()
#         print ("merge with envision: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime))
#           
        ###***************************************************************************************************************************************************************
        ### primateAI
        ###***************************************************************************************************************************************************************
#         if nt_input == 1:
#             stime = time.time()
#             self.primateai_score = pd.read_csv(self.db_path + 'primateai/PrimateAI_scores_v0.2.tsv', skiprows = 10, sep = '\t')
#             self.primateai_score.columns = ['chr_org','nt_pos','nt_ref','nt_alt','aa_ref','aa_alt','strand','codon','ucsc_id','exac_af','primateai_score']
#             self.primateai_score['chr'] = self.primateai_score['chr_org'].apply(lambda x:  x[3:])
#             self.primateai_score['chr'] =   self.primateai_score['chr'].astype(str)
#             variants_df = pd.merge(variants_df,self.primateai_score[['chr','nt_pos','nt_ref','nt_alt','aa_ref','aa_alt','primateai_score']],how = 'left')
#             etime = time.time()
#             print ("merge with primateai: " + str(variants_df.shape[0]) + " took %g seconds\n" % (etime - stime)) 
            
        ####***************************************************************************************************************************************************************
        #### aa_ref and aa_alt AA properties
        ####***************************************************************************************************************************************************************
        aa_properties = self.load_aa_properties()
        aa_properties_features = aa_properties.columns                
        aa_properties_ref_features = [x + '_ref' for x in aa_properties_features]
        aa_properties_alt_features = [x + '_alt' for x in aa_properties_features]   
        aa_properties_ref = aa_properties.copy()
        aa_properties_ref.columns = aa_properties_ref_features
        aa_properties_alt = aa_properties.copy()
        aa_properties_alt.columns = aa_properties_alt_features                
        variants_df = pd.merge(variants_df, aa_properties_ref, how='left')
        variants_df = pd.merge(variants_df, aa_properties_alt, how='left')
         
        for x in aa_properties_features:
            if x != 'aa':
                variants_df[x+'_delta'] = variants_df[x+'_ref'] - variants_df[x+'_alt']        
          
        alm_fun.show_msg(cur_log,self.verbose,"merge with aa properties: " + str(variants_df.shape[0]))
  
        ####***************************************************************************************************************************************************************
        #### flanking kmer AA column and properties  
        ####***************************************************************************************************************************************************************
        for i in range(1, k + 1):    
            aa_left = 'aa_ref_' + str(i) + '_l'
            aa_right = 'aa_ref_' + str(i) + '_r'
            variants_df[aa_left] = variants_df[['p_vid', 'aa_pos']].apply(lambda x: seq_dict[x['p_vid']][max(0, (x['aa_pos'] - i - 1)):max(0, (x['aa_pos'] - i))], axis=1)
            variants_df[aa_right] = variants_df[['p_vid', 'aa_pos']].apply(lambda x: seq_dict[x['p_vid']][(x['aa_pos'] + i - 1):(x['aa_pos'] + i)], axis=1)
            aa_properties_ref_kmer_features = [x + '_ref_' + str(i) + '_l' for x in aa_properties_features]
            aa_properties_ref_kmer = self.aa_properties.copy()
            aa_properties_ref_kmer.columns = aa_properties_ref_kmer_features
            variants_df = pd.merge(variants_df, aa_properties_ref_kmer, how='left')
            aa_properties_ref_kmer_features = [x + '_ref_' + str(i) + '_r' for x in aa_properties_features]
            aa_properties_ref_kmer = self.aa_properties.copy()
            aa_properties_ref_kmer.columns = aa_properties_ref_kmer_features
            variants_df = pd.merge(variants_df, aa_properties_ref_kmer, how='left')
        alm_fun.show_msg(cur_log,self.verbose,"merge with kmer properties: " + str(variants_df.shape[0]))
        
        ####***************************************************************************************************************************************************************
        #### One hot features for the aa and flanking aa 
        ####***************************************************************************************************************************************************************        
        lst_aa_20 = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q"]
        for aa in lst_aa_20:
            variants_df['aa_ref_' + aa] = variants_df['aa_ref'].apply(lambda x: int(x == aa))
            variants_df['aa_alt_' + aa] = variants_df['aa_alt'].apply(lambda x: int(x == aa))

        for i in range(1, k + 1):    
            aa_left = 'aa_ref_' + str(i) + '_l'
            aa_right = 'aa_ref_' + str(i) + '_r'

            for aa in lst_aa_20:
                variants_df['aa_ref_' + str(i) + '_l_' + aa] = variants_df['aa_ref_' + str(i) + '_l'].apply(lambda x: int(x == aa))
                variants_df['aa_ref_' + str(i) + '_r_' + aa] = variants_df['aa_ref_' + str(i) + '_r'].apply(lambda x: int(x == aa))

        
        ####***************************************************************************************************************************************************************
        #### merge with the blosum properties
        ####***************************************************************************************************************************************************************
        [df_blosums, dict_blosums]  = self.load_blosums()       
        variants_df = pd.merge(variants_df, df_blosums, how='left')
        alm_fun.show_msg(cur_log,self.verbose,"merge with blosums: " + str(variants_df.shape[0]))
        
        ####***************************************************************************************************************************************************************
        #### merge with the funsum properties
        ####***************************************************************************************************************************************************************
        funsum_dict = self.load_funsums()
        for funsum_key in funsum_dict.keys():
            variants_df = pd.merge(variants_df, funsum_dict[funsum_key], how='left')
            alm_fun.show_msg(cur_log,self.verbose,"merge with funsums - " + funsum_key + " :" + str(variants_df.shape[0]))
  
        ####***************************************************************************************************************************************************************
        ## Accessibility  
        ####***************************************************************************************************************************************************************        
        accsum_df = self.load_accsum()
        variants_df = pd.merge(variants_df,accsum_df,how = 'left')
        alm_fun.show_msg(cur_log,self.verbose,"merge with accsums: " + str(variants_df.shape[0]))
 
        ####*************************************************************************************************************************************************************
        #### Encode name features (one hot encode)
        ####*************************************************************************************************************************************************************        
        variants_df['aa_ref_encode'] = variants_df['aa_ref'].apply(lambda x: self.dict_aaencode.get(x,-1))
        variants_df['aa_alt_encode'] = variants_df['aa_alt'].apply(lambda x: self.dict_aaencode.get(x,-1))
        
        for aa in self.lst_aa_20:
            variants_df['aa_ref_' + aa] = variants_df['aa_ref'].apply(lambda x: int(x == aa))

        for i in range(1, k + 1):    
            aa_left = 'aa_ref_' + str(i) + '_l'
            aa_right = 'aa_ref_' + str(i) + '_r'
            variants_df[aa_left + '_encode'] = variants_df[aa_left].apply(lambda x: self.dict_aaencode.get(x,-1))
            variants_df[aa_right + '_encode'] = variants_df[aa_right].apply(lambda x: self.dict_aaencode.get(x,-1))
            
            for aa in self.lst_aa_20:
                variants_df['aa_ref_' + str(i) + '_l_' + aa] = variants_df['aa_ref_' + str(i) + '_l'].apply(lambda x: int(x == aa))
                variants_df['aa_ref_' + str(i) + '_r_' + aa] = variants_df['aa_ref_' + str(i) + '_r'].apply(lambda x: int(x == aa))
                          
        total_etime = time.time()
        alm_fun.show_msg(cur_log,self.verbose,"Variants processing took %g seconds\n" % (total_etime - total_stime))                     
        return (variants_df)    
    
    def add_psipred(self,input_df,single_id = ''):
        if single_id != '':
            psipred_file = self.db_path + 'psipred/bygene/' + single_id + '.ss2'        
            psipred = pd.read_csv(psipred_file, skiprows=[0, 1], header=None, sep='\s+')
            psipred = psipred.loc[:, [0, 1, 2]]
            psipred.columns = ['aa_pos', 'aa_ref', 'aa_psipred']
        else:    
            psipred = pd.read_csv(self.db_path + 'psipred/all/psipred_df.csv')
            psipred.columns = ['aa_pos','aa_ref','aa_psipred','p_vid']
            
        psipred = psipred.drop_duplicates()        
        input_df = pd.merge(input_df, psipred, how='left')
        psipred_lst = ['E','H','C']
        for ss in psipred_lst:
            input_df['aa_psipred' + '_' + ss] = input_df['aa_psipred'].apply(lambda x: int(x == ss))
        return(input_df)

    def add_pisa(self,input_df):
        pisa_df = pd.read_csv(self.db_path + 'pisa/all/all_pisa.csv')   
        input_df = pd.merge(input_df, pisa_df, how='left') 
        return(input_df)
    
    def add_pfam(self,input_df):
        pfam = pd.read_csv(self.db_path + 'pfam/org/9606.tsv', header=None, skiprows=3, sep='\t')
        pfam.columns = ['p_vid', 'a_start', 'a_end', 'e_start', 'e_end', 'hmm_id', 'hmm_name', 'type', 'hmm_start', 'hmm_end', 'hmm_length', 'bit_score', 'e_value', 'clan']
          
        p_vids = input_df['p_vid'].unique()
        cur_pfam = pfam.loc[(pfam['p_vid'].isin(p_vids)), :]
        input_df['pfam_id'] = np.nan
        count = 0
        for i in cur_pfam.index:
            cur_hmmid = cur_pfam.loc[i, "hmm_id"]
            cur_pvid = cur_pfam.loc[i, 'p_vid'] 
            cur_aa_start = cur_pfam.loc[i, 'a_start']
            cur_aa_end = cur_pfam.loc[i, 'a_end']
            input_df.loc[(input_df['p_vid'] == cur_pvid) & (input_df.aa_pos >= cur_aa_start) & (input_df.aa_pos <= cur_aa_end), 'pfam_id'] = cur_hmmid
            count += 1
            
        input_df['in_domain'] = np.nan
        input_df.loc[input_df['pfam_id'].notnull(), 'in_domain'] = 1
        input_df.loc[input_df['pfam_id'].isnull(), 'in_domain'] = 0    
        return(input_df)    
    
    def add_sublocation(self,input_df):
        sublocation = pd.read_csv(self.db_path + 'sublocation/all/uniprot_sublocation_processed.txt', sep='\t')
        sublocation.columns = ['p_vid', 'aa_pos', 'membrane']          
        input_df = pd.merge(input_df,sublocation,how = 'left')
        input_df.loc[input_df['membrane'].isnull(),'membrane'] = 0
        return(input_df)    
    
    def add_sift(self,input_df):
        sift_df = None
        p_vids = list(input_df['p_vid'].unique())
        for p_vid in p_vids:
            if os.path.isfile(self.db_path + 'sift/bygene/' + p_vid + '_sift.csv'):
                cur_sift = pd.read_csv(self.db_path + 'sift/bygene/' + p_vid + '_sift.csv')
                if sift_df is None:
                    sift_df = cur_sift
                else:
                    sift_df = pd.concat([sift_df,cur_sift])
        if sift_df is not None:
            input_df = input_df.merge(sift_df,how = 'left')
        else:
            input_df['sift_score'] = np.nan
        return(input_df)
    
    def add_mistic(self,input_df):
        mistic_df = pd.read_csv(self.db_path + '/mistic/all/MISTIC_GRCh37_avg_duplicated_scores.csv',dtype = {'chr':'str'}) 
        mistic_df.loc[mistic_df['chr'] == 'chrX','chr'] = 'X'
        input_df = pd.merge(input_df,mistic_df,how = 'left')
        return(input_df)
#         print ('mistic: '  + str(mistic_df['chr'].unique()))
    
    def add_provean(self,input_df):
        provean_df = None
        p_vids = list(input_df['p_vid'].unique())
        for p_vid in p_vids:
            if os.path.isfile(self.db_path + 'provean/bygene/' + p_vid + '_provean.csv'):
                cur_provean = pd.read_csv(self.db_path + 'provean/bygene/' + p_vid + '_provean.csv')
                if provean_df is None:
                    provean_df = cur_provean
                else:
                    provean_df = pd.concat([provean_df,cur_provean])
        if provean_df is not None:
            input_df = input_df.merge(provean_df,how = 'left')
        else:
            input_df['provean_score'] = np.nan            
        return(input_df)
            
    def load_blosums(self):
        blosum_rawdata_path = self.db_path + 'blosum/org/'
        df_blosums = None
        dict_blosums = {} 
        blosums = ['blosum30', 'blosum35', 'blosum40', 'blosum45', 'blosum50', 'blosum55', 'blosum60', 'blosum62', 'blosum65', 'blosum70', 'blosum75', 'blosum80', 'blosum85', 'blosum90', 'blosum95', 'blosum100']
        for blosum in blosums:
            blosum_raw = pd.read_csv(blosum_rawdata_path + "new_" + blosum + ".sij", sep='\t')
            col_names = blosum_raw.columns
            b = blosum_raw.replace(' ', 0).astype('float') 
            bv = b.values + b.transpose().values
            bv[np.diag_indices_from(bv)] = bv[np.diag_indices_from(bv)] / 2
            blosum_new = pd.DataFrame(bv, columns=col_names)
            blosum_new['aa_ref'] = col_names
            blosum_new = pd.melt(blosum_new, id_vars=['aa_ref'])
            blosum_new.columns = ['aa_ref', 'aa_alt', blosum]
#             blosum_new[blosum] = 0-blosum_new[blosum]
            if df_blosums is None:
                df_blosums = blosum_new.copy()
            else:
                df_blosums = df_blosums.join(blosum_new[blosum]) 
            dict_blosums[blosum] = blosum_new   
        return [df_blosums, dict_blosums]   
    
    def load_funsums(self):
        funsum_dict = {}
        for file in glob.glob(self.db_path + 'funsum/csv/*.csv'):
            key = file.split('/')[-1].split('.')[0]
            cur_funsum_df = pd.read_csv(file)
            funsum_dict[key] = cur_funsum_df
        return(funsum_dict)
    
    def load_accsum(self):
        accsum_df = pd.read_csv(self.db_path + 'accsum/csv/accsum.csv')
        return(accsum_df)    
    
    def get_polyphen_train_data(self):
        polyphen_train_deleterious_file = 'polyphen/org/humdiv-2011_12.deleterious.pph.input'
        polyphen_train_neutral_file = 'polyphen/org/humdiv-2011_12.neutral.pph.input'
        polyphen_train0 = pd.read_csv(self.db_path + polyphen_train_neutral_file, header=None, names=['p_vid', 'aa_pos', 'aa_ref', 'aa_alt'])        
        polyphen_train1 = pd.read_csv(self.db_path + polyphen_train_deleterious_file, header=None, names=['p_vid', 'aa_pos', 'aa_ref', 'aa_alt'])
        polyphen_train = pd.concat([polyphen_train0, polyphen_train1])
        polyphen_train['polyphen_train'] = 1
        polyphen_train.to_csv(self.db_path + 'polyphen/csv/polyphen_train_humdiv.csv', index=False)
#         