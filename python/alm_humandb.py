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


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)   

       
#*****************************************************************************************************************************
#file based human database 
#*****************************************************************************************************************************

class alm_humandb:
        
    def __init__(self,argvs):
        stime = time.time()  
        self.assembly = argvs['assembly']
        self.db_path = argvs['humandb_path']
        self.project_path = argvs['project_path']
        self.python_path = argvs['python_path']
        self.log = argvs['humandb_path'] + 'log/humandb.log'
        self.humandb_object_logs = {}
        self.verbose = 1
        self.flanking_k = 0
        self.cluster = 'galen'
        self.db_version = 'manuscript'  # or 'uptodate'
        self.argvs = argvs
        
        ####***************************************************************************************************************************************************************
        # Nucleotide and Amino Acids related
        ####***************************************************************************************************************************************************************
        self.lst_nt = ['A', 'T', 'C', 'G']
        self.lst_aa = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q", "U", "*", '_']
        self.lst_aa_21 = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q", "*"]
        self.lst_aa_20 = ["S", "A", "V", "R", "D", "F", "T", "I", "L", "K", "G", "Y", "N", "C", "P", "E", "M", "W", "H", "Q"]
        self.lst_aa3 = ["Ser", "Ala", "Val", "Arg", "Asp", "Phe", "Thr", "Ile", "Leu", "Lys", "Gly", "Tyr", "Asn", "Cys", "Pro", "Glu", "Met", "Trp", "His", "Gln", "Sec", "Ter", 'Unk']
        self.lst_aaname = ["Serine", "Alanine", "Valine", "Arginine", "Asparitic Acid", "Phenylalanine", "Threonine", "Isoleucine", "Leucine", "Lysine", "Glycine", "Tyrosine", "Asparagine", "Cysteine", "Proline", "Glutamic Acid", "Methionine", "Tryptophan", "Histidine", "Glutamine", "Selenocysteine", "Stop", "Unknown"]
        self.lst_chr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT']
        
        self.dict_aa3 = {}
        for i in range(len(self.lst_aa3)):
            self.dict_aa3[self.lst_aa3[i]] = self.lst_aa[i]
            
        self.dict_aa3_upper = {}
        for i in range(len(self.lst_aa3)):
            self.dict_aa3_upper[self.lst_aa3[i].upper()] = self.lst_aa[i]
            
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
         
    def humandb_action(self, argvs):
        if argvs['db_action'] == 'create_all':        
            self.create_hgnc_data()
            self.create_uniprot_data()
            self.create_ensembl66_data()
            self.create_matched_uniprot_mapping()
            self.create_pisa_data()
            self.create_pfam_data()
            self.create_sift_data()
            self.create_provean_data()
            self.create_clinvar_data()
            
        if 'single_job' in argvs['db_action']:
            cur_dbobject = argvs['db_action'].split("-")[2]
            cur_action = argvs['db_action'].split("-")[0]
            self.create_and_run_single_job(cur_action,cur_dbobject,argvs['parallel_id'],argvs['parallel_num']) 
#         if 'jobs' in argvs['db_action']
#             cur_dbobject = argvs['db_action'].split("-")[2]
#             cur_action = argvs['db_action'].split("-")[0]
#             self.create_and_run_jobs(cur_action,cur_dbobject,argvs['parallel_num']) 
#             
        if argvs['db_action'] == 'varity_check_data':
            self.varity_check_data(argvs['parallel_id'])            
        if argvs['db_action'] == 'varity_check_data_jobs':
            self.varity_check_data_jobs()                       
        if argvs['db_action'] == 'create_aa_properties_data':            
            self.create_aa_properties_data()
        if argvs['db_action'] == 'create_codon_usage_data':            
            self.create_codon_usage_data()            
        if argvs['db_action'] == 'create_blosum_data':
            self.create_blosum_data()
        if argvs['db_action'] == 'create_accsum_data':
            self.create_accsum_data()            
        if argvs['db_action']  == 'create_hgnc_data':
            self.create_hgnc_data()
        if argvs['db_action']  == 'create_uniprot_data':
            self.create_uniprot_data()   
        if argvs['db_action']  == 'create_pfam_data':
            self.create_pfam_data()    
        if argvs['db_action']  == 'initiate_pisa_data':
            self.initiate_pisa_data()             
        if argvs['db_action'] == 'retrieve_pisa_data':
            self.retrieve_pisa_data(argvs['parallel_id'], argvs['parallel_num']) 
        if argvs['db_action'] == 'retrieve_pisa_data_jobs':
            self.retrieve_pisa_data_jobs(argvs['parallel_num'])             
        if argvs['db_action'] == 'combine_pisa_data':
            self.combine_pisa_data()  
            
        if argvs['db_action']  == 'initiate_psipred_data':
            self.initiate_psipred_data()   
        
        if argvs['db_action'] == 'check_psipred_data':
            self.check_psipred_data()            
        if argvs['db_action'] == 'retrieve_psipred_data':
            self.retrieve_psipred_data(argvs['parallel_id'], argvs['parallel_num']) 
        if argvs['db_action'] == 'retrieve_psipred_data_jobs':
            self.retrieve_psipred_data_jobs(argvs['parallel_num'])             
        if argvs['db_action'] == 'combine_psipred_data':
            self.combine_psipred_data()  
            
        if argvs['db_action'] =='create_humsavar_data':
            self.create_humsavar_data()            

        if argvs['db_action'] =='create_hgmd_data':
            self.create_hgmd_data()   
                  
        if argvs['db_action']  == 'create_clinvar_data':
            self.create_clinvar_data()
        if argvs['db_action'] == 'create_mave_data':
            self.create_mave_data() 
        if argvs['db_action'] == 'create_funsum_data':
            self.create_funsum_data() 
        if argvs['db_action'] == 'create_varity_data':
            self.create_varity_data()                                                     
        if argvs['db_action']  == 'varity_dbnsfp_jobs':
            self.varity_dbnsfp_jobs()
        if argvs['db_action']  == 'varity_dbnsfp_process':
            self.varity_dbnsfp_process(argvs['parallel_id'])
        if argvs['db_action']  == 'varity_process_jobs': 
            self.varity_process_jobs()
        if argvs['db_action']  == 'varity_merge_data': 
            self.varity_merge_data(argvs['parallel_id'])
        if argvs['db_action']  == 'varity_merge_data_jobs':
            self.varity_merge_data_jobs()
        if argvs['db_action']  == 'varity_all_variant_process': 
            self.varity_all_variant_process(argvs['parallel_id'])
        if argvs['db_action']  == 'varity_all_variant_process_jobs':
            self.varity_all_variant_process_jobs()
            
        if argvs['db_action']  == 'varity_combine_train_data':                         
            self.varity_combine_train_data()
            
        if argvs['db_action'] == 'varity_combine_all_data':
            self.varity_combine_all_data()
            
        if argvs['db_action'] == 'varity_mave_final_data':
            self.varity_mave_final_data()
            
        if argvs['db_action'] == 'varity_count_all_data':
            self.varity_count_all_data()
            
        if argvs['db_action']  == 'varity_train_variant_process':                         
            self.varity_train_variant_process()    
            
        if argvs['db_action'] == 'varity_train_data_final_process':
            self.varity_train_data_final_process()    
              
        if argvs['db_action'] == 'varity_all_data_final_process':
            self.varity_all_data_final_process(argvs['parallel_id'])      
            
        if argvs['db_action'] == 'varity_final_process_jobs':
            self.varity_final_process_jobs()      

        if argvs['db_action']  == 'create_ensembl66_data':
            self.create_ensembl66_data()
        if argvs['db_action']  == 'create_matched_uniprot_mapping':            
            self.create_matched_uniprot_mapping()
            
        if argvs['db_action']  == 'create_sift_data':
            self.create_sift_data()
        if argvs['db_action']  == 'create_provean_data':
            self.create_provean_data()  
        if argvs['db_action']  == 'create_gnomad_data':
            self.create_gnomad_data()
        if argvs['db_action']  == 'create_evmutation_data':
            self.create_evmutation_data()   
            
        if argvs['db_action'] == 'create_varity_data_for_single_protein':
            self.create_varity_data_for_single_protein(argvs['single_id'],argvs['single_id_chr'])

        if argvs['db_action'] == 'run_psipred_by_uniprotid':
            self.run_psipred_by_uniprotid(argvs['single_id'])

    def create_varity_data_for_single_protein(self, id,chr):
        cur_log = self.project_path + '/log/create_varity_data_for_single_protein.log'        
        alm_fun.show_msg(cur_log,self.verbose,'Create VARITY recrod for ID: ' + id)
        
        #1: Run DBNSFP job
        if not os.path.isfile(self.db_path +'varity/csv/varity_dbnsfp_snv_' + id + '_0.out'):                      
            alm_fun.show_msg(cur_log,self.verbose,'Running DBNSFP job......')
            self.varity_dbnsfp_jobs(id,chr)
        
        while not os.path.isfile(self.db_path +'varity/csv/varity_dbnsfp_snv_' + id + '_0.out'):
            time.sleep(1)
        
        #2 RUN VARITY Process to process the DNBSFP output
        alm_fun.show_msg(cur_log,self.verbose,'Running VARITY DBNSFP process job......')
        self.varity_dbnsfp_process(parallel_id = '',single_id = id)
    
        #3 RUN VARITY Merge to merge with ClinVar, HumsaVar etc 
        alm_fun.show_msg(cur_log,self.verbose,'Running VARITY merge job......')
        self.varity_merge_data(parallel_id = '',single_id = id)
    
        #4 RUN VARITY Variant Process
        alm_fun.show_msg(cur_log,self.verbose,'Running VARITY Variant process job......')
        self.varity_single_protein_variant_process(id)
    
        # RUN VARITY Final Process
        alm_fun.show_msg(cur_log,self.verbose,'Running VARITY Final process job......')
        self.varity_single_protein_data_final_process(id)
                       
    def create_and_run_single_job(self,cur_action,cur_dbobject,parallel_id,parallel_num):
        self.init_humamdb_object(cur_dbobject)
        
        if self.cluster == 'galen':            
            exclusion_nodes_list = ''
            exclusion_nodes_log = '/home/rothlab/jwu/projects/humandb/log/exclusion_nodes.log'
            if os.path.isfile(exclusion_nodes_log):
                for line in  open(exclusion_nodes_log,'r'):
                    exclusion_nodes_list =  exclusion_nodes_list + line.rstrip()[5:] + ','
                exclusion_nodes_list = exclusion_nodes_list[:-1]
            #*************************************************************************************************
            #run single job on cluster for current action 
            #*************************************************************************************************
            cur_cmd = "python3 /home/rothlab/jwu/projects/humandb/python/humandb_debug.py '/home/rothlab/jwu/projects/ml/python' '/home/rothlab/jwu/projects/humandb/' '/home/rothlab/jwu/database/humandb_new/' 'run_humandb_action'" + " '" + cur_action + "' "  + "'" + parallel_id + "' " + str(parallel_num)  
            sh_file = open(self.db_path + cur_dbobject + '/bat/' + cur_action + '.sh','w')  
            sh_file.write('#!/bin/bash' + '\n')
            sh_file.write('# set the number of nodes' + '\n')
            sh_file.write('#SBATCH --nodes=1' + '\n')
            sh_file.write('# set the memory for each node' + '\n')
            sh_file.write('#SBATCH --mem=' + '10240' + '\n')   
            sh_file.write('# set name of job' + '\n')
            sh_file.write('#SBATCH --job-name=' + cur_action + '\n')
#             sh_file.write('# mail alert at start, end and abortion of execution' + '\n')
#             sh_file.write('#SBATCH --mail-type=ALL' + '\n')
#             sh_file.write('# send mail to this address' + '\n')
#             sh_file.write('#SBATCH --mail-user=joe.wu.ca@gmail.com' + '\n')      
            sh_file.write("srun " + cur_cmd)
            sh_file.close()
                            
            sbatch_cmd = 'sbatch --exclude=galen['  + exclusion_nodes_list + '] ' +  self.db_path + cur_dbobject + '/bat/' + cur_action + '.sh'
            print (sbatch_cmd)
            job_name = cur_action
                        
            job_id = '-1'
            while  job_id == '-1':
                return_process = subprocess.run(sbatch_cmd.split(" "), cwd = self.db_path + cur_dbobject + '/log/' ,capture_output = True,text=True)
                time.sleep(0.1)
                if return_process.returncode == 0:
                    job_id = return_process.stdout.rstrip().split(' ')[-1]
                else:
                    job_id = '-1'
                    print  (job_name + ' submit error,rescheduling......' )
            print (job_name + ' submitted id: ' + job_id) 

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
        aa_properties = pd.read_table(self.db_path + 'aa_properties/org/aa.txt', sep='\t')
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
                cur_df = pd.DataFrame.from_records([(aa_ref, aa_alt, self.get_aa_accessibility(aa_ref, aa_alt,dict_aa_codon,dict_codon_freq_aa, titv_ratio))], columns=['aa_ref', 'aa_alt', 'accessibility'])
                accsum_df = accsum_df.append(cur_df)
        accsum_df['accessibility_ste'] = 0
        output_path = self.db_path + 'accsum/csv/'
        accsum_df.to_csv(output_path + 'accsum.csv', index=False)
        alm_fun.show_msg(self.log,self.verbose,'accsum data created.')  
        
    def get_aa_accessibility(self, aa_ref, aa_alt, dict_aa_codon,dict_codon_freq_aa,titv_ratio=1):       
        aa_ref_codons = dict_aa_codon[aa_ref]
        aa_alt_codons = dict_aa_codon[aa_alt]
        access = 0
        
        for ref_codon in aa_ref_codons:
            for alt_codon in aa_alt_codons:
                if alm_fun.hamming_distance(ref_codon, alt_codon) == 1:
                    if alm_fun.get_codon_titv(ref_codon, alt_codon) == 'ti':
#                         access += dict_codon_freq_aa[ref_codon] * dict_codon_freq_aa[alt_codon]
                        access += dict_codon_freq_aa[ref_codon]/9
                    if alm_fun.get_codon_titv(ref_codon, alt_codon) == 'tv':
#                         access += dict_codon_freq_aa[ref_codon] * dict_codon_freq_aa[alt_codon] / titv_ratio
                        access += dict_codon_freq_aa[ref_codon]/9
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
       
        self.ebi_ftp_obj = alm_fun.create_ftp_object('ftp.ebi.ac.uk')
        return_info = alm_fun.download_ftp(self.ebi_ftp_obj, '/pub/databases/genenames/new/tsv/', 'hgnc_complete_set.txt', self.db_path + 'hgnc/org/hgnc_complete_set.txt')        
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
        id_lst = ['symbol','ensembl_gene_id','refseq_accession','uniprot_ids','ucsc_id']
        
        for id in id_lst:
            cur_hgnc_ids = hgnc.loc[hgnc[id].notnull(),['hgnc_id',id]] 
            cur_id_dict = {}
            cur_hgnc_ids.apply(lambda x: fill_dict(x['hgnc_id'],x[id],cur_id_dict),axis = 1)
            id2hgnc_dict[id] = cur_id_dict
        pass
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

        #******************************************************
        # All IDs maps to uniprot
        #******************************************************
#         id_maps = pd.read_table(self.db_path + 'uniprot/org/HUMAN_9606_idmapping_selected.tab', sep='\t', header=None) 
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
        alm_fun.show_msg(cur_log,self.verbose,'Created uniprot data.')
    
    def create_hgmd_data(self):
        self.init_humamdb_object("hgmd")
        cur_log = self.db_path + 'hgmd/log/hgmd.log'        
        if not os.path.isfile(self.db_path + 'hgmd/org/hmgd_2015.csv'):       
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

        hgmd_snv['hgmd_source'] = 1
        hgmd_snv['hgmd_label'] = 1     
        hgmd_snv.to_csv(self.db_path +'hgmd/all/hgmd_snv.csv',index = False)
        
#         clinvar_snv = pd.read_csv(self.db_path + 'clinvar/all/clinvar_snv.csv',dtype = {'chr':'str'})
#         clinvar_snv['clinvar_source'] = 1
#         
#         compare_snv = clinvar_snv.merge(hgmd_snv[['chr','nt_pos','hgmd_source','hgvs','nt_ref_hgmd','nt_alt_hgmd']],how= 'left')
#         
#         compare_snv.loc[(compare_snv['hgmd_source'] == 1) & (compare_snv['nt_alt'] != compare_snv['nt_alt_hgmd']),['chr','nt_pos','nt_alt','nt_ref','nt_alt_hgmd','nt_ref_hgmd','hgvs','clinvar_id']]
#         
#         clinvar_snv.loc[clinvar_snv['nt_pos'] == 5247843,:]
        
        
#         print('OK')
    
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
        clinvar_submission_raw = pd.read_table(self.db_path + self.clinvar_submission_file, sep='\t',skiprows = 15  )
        alm_fun.show_msg(cur_log,self.verbose,'Total number of CLINVAR submission records : ' + str(clinvar_submission_raw.shape[0]))
        
        clinvar_submission = clinvar_submission_raw[['#VariationID','CollectionMethod']]
        clinvar_submission.columns = ['clinvar_id','clinvar_collection_method']
        clinvar_submission_group = clinvar_submission.groupby(['clinvar_id']) ['clinvar_collection_method'].agg(list).reset_index()
        clinvar_submission_group.columns = ['clinvar_id','clinvar_collection_methods']
        clinvar_submission_group['clinvar_literature_only'] = clinvar_submission_group['clinvar_collection_methods'].apply(lambda x: chk_method(x))

        #load clinvar rawdata
        self.clinvar_raw_file = 'clinvar/org/variant_summary.txt'        
        clinvar_raw = pd.read_table(self.db_path + self.clinvar_raw_file, sep='\t',dtype={'Chromosome':'str'})
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
        
    def create_mave_data(self):    
        #******************************************************************************
        #Roth Lab MAVE data
        #******************************************************************************
        self.init_humamdb_object ('mave')  
        cur_log = self.db_path + 'mave/log/mave.log'
                
        if not os.path.isfile(self.db_path + 'mave/all/mave_missense.csv'): 
            if not os.path.isfile(self.db_path + 'mave/all/mave_missense.csv'):                   
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
    
    def initiate_pisa_data(self):
        ####***************************************************************************************************************************************************************    
        # EBI FTP
        ####***************************************************************************************************************************************************************
        # Uniprot ID to PDB ids 
        # (ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_uniprot.csv.gz)       
        ####***************************************************************************************************************************************************************
        self.init_humamdb_object('pisa') 
        cur_log = self.db_path + 'pisa/log/pisa.log' 
        self.ebi_ftp_obj = alm_fun.create_ftp_object('ftp.ebi.ac.uk')
        return_info = alm_fun.download_ftp(self.ebi_ftp_obj, '/pub/databases/msd/sifts/flatfiles/csv/', 'pdb_chain_uniprot.csv.gz', self.db_path + 'pisa/org/pdb_chain_uniprot.csv.gz')
        if (return_info == 'updated') | (return_info == 'downloaded'):        
            alm_fun.gzip_decompress(self.db_path + 'pisa/org/pdb_chain_uniprot.csv.gz', self.db_path + 'pisa/org/pdb_chain_uniprot.csv')  
        alm_fun.show_msg(self.log,self.verbose,'Initiated pisa data.')
        
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
        
    def retrieve_pisa_data_jobs(self,parallel_num):
        for parallel_id in range(parallel_num):
            if self.cluster == 'galen':            
                #*************************************************************************************************
                #run parallel jobs (but you need figure out the way to determine when all jobs are finished 
                #*************************************************************************************************
                pisa_cmd = "python3 /home/rothlab/jwu/projects/humandb/python/humandb_debug.py '/home/rothlab/jwu/projects/ml/python' '/home/rothlab/jwu/projects/humandb/' '/home/rothlab/jwu/database/humandb_new/' 'run_humandb_action' 'retrieve_pisa_data' " +  str(parallel_id) + " " + str(parallel_num)
                sh_file = open(self.db_path + 'pisa/bat/pisa_job_' + str(parallel_id) + '_' + str(parallel_num) + '.sh','w')  
                sh_file.write('#!/bin/bash' + '\n')
                sh_file.write('# set the number of nodes' + '\n')
                sh_file.write('#SBATCH --nodes=1' + '\n')
                sh_file.write('# set max wallclock time' + '\n')
                sh_file.write('#SBATCH --time=100:00:00' + '\n')
                sh_file.write('# set name of job' + '\n')
                sh_file.write('#SBATCH --job-name=' + 'pisa_job_' + str(parallel_id) + '_' + str(parallel_num) + '\n')
                sh_file.write('# mail alert at start, end and abortion of execution' + '\n')
                sh_file.write('#SBATCH --mail-type=ALL' + '\n')
                sh_file.write('# send mail to this address' + '\n')
                sh_file.write('#SBATCH --mail-user=joe.wu.ca@gmail.com' + '\n')      
                sh_file.write("srun " +  pisa_cmd)
                sh_file.close()
                                
                sbatch_cmd = "sbatch " + self.db_path + 'pisa/bat/pisa_job_' + str(parallel_id) + '_' + str(parallel_num) + '.sh'
                print (sbatch_cmd)
                subprocess.run(sbatch_cmd.split(" "), cwd = self.db_path)

    def retrieve_pisa_data(self,parallel_id,parallel_num):
        cur_log = self.db_path + 'pisa/log/pisa.log'
        parallel_id = int(parallel_id)
        self.pdb_to_uniprot = pd.read_csv(self.db_path + 'pisa/org/pdb_chain_uniprot.csv', skiprows = 1 ,dtype={"PDB": str})          
        uniprot_human_reviewed_ids = list(np.load(self.db_path + 'uniprot/npy/uniprot_human_reviewed_ids.npy'))        
        pisa_ids = list(set(uniprot_human_reviewed_ids).intersection(set(self.pdb_to_uniprot['SP_PRIMARY'].unique())))                
        total_gene_num = len(pisa_ids)        
        gene_index_array = np.linspace(0,total_gene_num,parallel_num+1 , dtype = int)        
        cur_parallel_indices = list(range(gene_index_array[parallel_id],gene_index_array[parallel_id+1]))
        cur_gene_ids = [pisa_ids[i] for i in cur_parallel_indices]        
        cur_parallel_log = self.db_path + 'pisa/log/pisa_data_parallel_' + str(parallel_id) +  '_' + str(parallel_num) + '.log'          
        for uniprot_id in cur_gene_ids:
            self.retrieve_pisa_by_uniprotid(uniprot_id,cur_parallel_log)
        alm_fun.show_msg(cur_log,self.verbose,'pisa_job_' + str(parallel_id) + '_' + str(parallel_num) + ' is done.')   
                              
    def retrieve_pisa_by_uniprotid(self,uniprot_id,cur_log):
        try: 
            if os.path.isfile(self.db_path + 'pisa/bygene/' + uniprot_id + '_pisa.csv'):
                return(0)
            if os.path.isfile(self.db_path + 'pisa/org/' + uniprot_id + '_interface.xml'):
                return(0)                                 
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
                        cur_molecule_df['aa_pos'] = cur_molecule_df['solv_ne'].astype(int)
                        
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
    
    def combine_pisa_data(self):
        pisa_csv_df = None
        for pisa_csv_file in glob.glob(self.db_path + 'pisa/bygene/*.csv'):
            cur_pisa_csv_df = pd.read_csv(pisa_csv_file)
            if pisa_csv_df is None:
                pisa_csv_df = cur_pisa_csv_df
            else:
                pisa_csv_df = pd.concat([pisa_csv_df,cur_pisa_csv_df])
            
        pisa_csv_df.to_csv(self.db_path + 'pisa/all/all_pisa.csv',index = False)
        
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
    
    def retrieve_psipred_data_jobs(self,parallel_num):
        for parallel_id in range(parallel_num):
            if self.cluster == 'galen':            
                #*************************************************************************************************
                #run parallel jobs (but you need figure out the way to determine when all jobs are finished 
                #*************************************************************************************************
                psipred_cmd = "python3 /home/rothlab/jwu/projects/humandb/python/humandb_debug.py '/home/rothlab/jwu/projects/ml/python' '/home/rothlab/jwu/projects/humandb/' '/home/rothlab/jwu/database/humandb_new/' 'run_humandb_action' 'retrieve_psipred_data' " +  str(parallel_id) + " " + str(parallel_num)
                sh_file = open(self.db_path + 'psipred/bat/psipred_job_' + str(parallel_id) + '_' + str(parallel_num) + '.sh','w')  
                sh_file.write('#!/bin/bash' + '\n')
                sh_file.write('# set the number of nodes' + '\n')
                sh_file.write('#SBATCH --nodes=1' + '\n')
                sh_file.write('# set max wallclock time' + '\n')
                sh_file.write('#SBATCH --time=100:00:00' + '\n')
                sh_file.write('# set name of job' + '\n')
                sh_file.write('#SBATCH --job-name=' + 'psipred_job_' + str(parallel_id) + '_' + str(parallel_num) + '\n')
                sh_file.write('# mail alert at start, end and abortion of execution' + '\n')
                sh_file.write('#SBATCH --mail-type=ALL' + '\n')
                sh_file.write('# send mail to this address' + '\n')
                sh_file.write('#SBATCH --mail-user=joe.wu.ca@gmail.com' + '\n')      
                sh_file.write("srun " +  psipred_cmd)
                sh_file.close()
                                
                sbatch_cmd = "sbatch " + self.db_path + 'psipred/bat/psipred_job_' + str(parallel_id) + '_' + str(parallel_num) + '.sh'
                print (sbatch_cmd)
                subprocess.run(sbatch_cmd.split(" "), cwd = self.db_path + 'psipred/log/')

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
            
            
    def  run_psipred_by_uniprotid(self,uniprot_id):
        #**********************************************************************************************
        # PISPRED : after installation, remeber to edit the runpsipred script to make it work!!!!!!!
        #**********************************************************************************************
        cur_log =  self.db_path + 'psipred/log/psipred_data_single_'+ uniprot_id +  '.log'   
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
        alm_fun.show_msg(cur_log,self.verbose, psipred_seq_str)
#         psipred_seq_file.write(psipred_seq_str)      
        psipred_seq_file.close()
        alm_fun.show_msg(cur_log,self.verbose, uniprot_id + ' psipred is done.')


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
#                         
    def create_varity_data(self):  
        self.init_humamdb_object('varity') 
        cur_log = self.db_path + 'varity/log/varity.log'
        hgnc2id_dict = np.load(self.db_path + 'hgnc/npy/hgnc2id_dict.npy').item()
        #************************************************************************************************************************************************************************
        #Combine clinvar and mave genes to varity genes 
        #************************************************************************************************************************************************************************
#         clinvar_snv = pd.read_csv(self.db_path + 'clinvar/all/clinvar_snv.csv',dtype = {'chr':'str'})
#         mave_missense = pd.read_csv(self.db_path + 'mave/all/mave_missense.csv')  
                                
        clinvar_disease_genes = list(pd.read_csv(self.db_path + 'clinvar/all/clinvar_disease_genes.txt',header = None)[0])         
        alm_fun.show_msg(self.log,self.verbose,'Total number of CLINVAR disease genes : ' + str(len(clinvar_disease_genes)))
        
        humsavar_disease_genes = list(pd.read_csv(self.db_path + 'humsavar/all/humsavar_disease_genes.txt',header = None)[0])         
        alm_fun.show_msg(self.log,self.verbose,'Total number of HUMSAVAR disease genes : ' + str(len(humsavar_disease_genes)))
        
       
        mave_genes = list(pd.read_csv(self.db_path + 'mave/all/mave_genes.txt',header = None)[0]) 
        alm_fun.show_msg(self.log,self.verbose,'Total number of MAVE genes : ' + str(len(mave_genes)))
        
        varity_genes = list(set(clinvar_disease_genes + mave_genes + humsavar_disease_genes))        
        alm_fun.show_msg(self.log,self.verbose,'Total number of VARITY genes : ' + str(len(varity_genes)))
        
        varity_genes_df = pd.DataFrame(varity_genes)
        varity_genes_df.columns = ['hgnc_id']
        varity_genes_df['uniprot_id'] = varity_genes_df.apply(lambda x: hgnc2id_dict['uniprot_ids'].get(x['hgnc_id'],np.nan),axis = 1)
        varity_genes_df = varity_genes_df.loc[varity_genes_df['uniprot_id'].notnull(),:]
        varity_genes_df['hgnc_symbol'] = varity_genes_df.apply(lambda x: hgnc2id_dict['symbol'].get(x['hgnc_id'],np.nan),axis = 1)
        varity_genes_df['chr'] = varity_genes_df.apply(lambda x: hgnc2id_dict['chr'].get(x['hgnc_id'],''),axis = 1)
        varity_genes_df['p_vid'] = varity_genes_df['uniprot_id'].apply(lambda x: x.split('|')[-1])
        alm_fun.show_msg(self.log,self.verbose,'Final number of VARITY genes with valid Uniprot ID : ' + str(varity_genes_df.shape[0]))
        varity_genes_df.to_csv(self.db_path + 'varity/all/varity_disease_genes.csv',index = False)
        alm_fun.show_msg(cur_log,self.verbose,'Varity data initiated.\n')

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
                        
#             varity_variant_process_cmd =  'nohup ' + varity_variant_process_cmd + ' >' +  job_name + '.out &\n'
#             print (varity_variant_process_cmd)
#             os.system(varity_variant_process_cmd)

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

#             varity_final_process_cmd =  'nohup ' + varity_final_process_cmd + ' >' +  job_name + '.out &\n'
#             print (varity_final_process_cmd)
#             os.system(varity_final_process_cmd)
                
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
        input_df = input_df.merge(sift_df,how = 'left')
        return(input_df)
    
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
        input_df = input_df.merge(provean_df,how = 'left')
        return(input_df)
            
    def load_blosums(self):
        blosum_rawdata_path = self.db_path + 'blosum/org/'
        df_blosums = None
        dict_blosums = {} 
        blosums = ['blosum30', 'blosum35', 'blosum40', 'blosum45', 'blosum50', 'blosum55', 'blosum60', 'blosum62', 'blosum65', 'blosum70', 'blosum75', 'blosum80', 'blosum85', 'blosum90', 'blosum95', 'blosum100']
        for blosum in blosums:
            blosum_raw = pd.read_table(blosum_rawdata_path + "new_" + blosum + ".sij", sep='\t')
            col_names = blosum_raw.columns.get_values()
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

    def load_aa_properties(self):
        aa_properties = pd.read_table(self.db_path + 'aa_properties/org/aa.txt', sep='\t')
        aa_properties.drop_duplicates(inplace=True)
        aa_properties.drop(['aa_name'], axis=1, inplace=True)
        # aa_properties_features = ['aa','mw','pka','pkb','pi', 'cyclic','charged','charge','hydropathy_index','hydrophobic','polar','ionizable','aromatic','aliphatic','hbond','sulfur','pbr','avbr','vadw','asa']
        aa_properties_features = ['aa', 'mw', 'pka', 'pkb', 'pi', 'hi', 'pbr', 'avbr', 'vadw', 'asa', 'pbr_10', 'avbr_100', 'vadw_100', 'asa_100', 'cyclic', 'charge', 'positive', 'negative', 'hydrophobic', 'polar', 'ionizable', 'aromatic', 'aliphatic', 'hbond', 'sulfur', 'essential', 'size']
        aa_properties.columns = aa_properties_features    
        return (aa_properties)
    
    def get_polyphen_train_data(self):
        polyphen_train_deleterious_file = 'polyphen/org/humdiv-2011_12.deleterious.pph.input'
        polyphen_train_neutral_file = 'polyphen/org/humdiv-2011_12.neutral.pph.input'
        polyphen_train0 = pd.read_table(self.db_path + polyphen_train_neutral_file, header=None, names=['p_vid', 'aa_pos', 'aa_ref', 'aa_alt'])        
        polyphen_train1 = pd.read_table(self.db_path + polyphen_train_deleterious_file, header=None, names=['p_vid', 'aa_pos', 'aa_ref', 'aa_alt'])
        polyphen_train = pd.concat([polyphen_train0, polyphen_train1])
        polyphen_train['polyphen_train'] = 1
        polyphen_train.to_csv(self.db_path + 'polyphen/csv/polyphen_train_humdiv.csv', index=False)
        