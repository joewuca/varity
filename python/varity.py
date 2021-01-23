import numpy as np
import pandas as pd
import sys
import os
import glob
import time
import subprocess
import traceback
import string
import random
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
import seaborn as sns
import alm_project
import alm_ml
import alm_fun
import alm_humandb


class varity:

    def __init__(self,runtime):   
        print ('Class: [varity] [__init__]...... @' + str(datetime.now()))
        argvs = self.read_config(runtime)         
        #1) create alm_project instance    
        self.proj = alm_project.alm_project(argvs,self)
        #2) create alm_ml instance                 
        self.ml = alm_ml.alm_ml(argvs['ml'])
        self.ml.varity_obj = self   
        self.ml.proj = self.proj
    
    def debug(self,runtime):
        
        key_cols = ['chr','nt_pos','nt_ref','nt_alt','p_vid','aa_pos','aa_ref','aa_alt']
        annotation_cols = ['clinvar_id','clinvar_source','hgmd_source','gnomad_source','humsavar_source','mave_source',
                           'clinvar_label','hgmd_label','gnomad_label','humsavar_label','mave_label','label',
                           'train_clinvar_source','train_hgmd_source','train_gnomad_source','train_humsavar_source','train_mave_source']                           
        score_cols = ['Polyphen2_selected_HVAR_score','Polyphen2_selected_HDIV_score','PROVEAN_selected_score','SIFT_selected_score',
                      'CADD_raw','PrimateAI_score','Eigen-raw_coding','GenoCanyon_score','integrated_fitCons_score','REVEL_score',
                      'M-CAP_score','LRT_score','MutationTaster_selected_score','MutationAssessor_selected_score',
                      'FATHMM_selected_score','MetaSVM_score','MetaLR_score','DANN_score','GERP++_RS',
                      'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','MPC_selected_score','mistic_score',
                      'mpc_score','DeepSequence_score','mave_input','mave_norm','mave_score','sift_score','provean_score']        
        feature_cols = ['PROVEAN_selected_score','SIFT_selected_score','evm_epistatic_score','integrated_fitCons_score','LRT_score','GERP++_RS',
                        'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','blosum100','in_domain','asa_mean','aa_psipred_E',
                        'aa_psipred_H','aa_psipred_C','bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min',
                        'mw_delta','pka_delta','pkb_delta','pi_delta','hi_delta','pbr_delta','avbr_delta','vadw_delta','asa_delta','cyclic_delta','charge_delta',
                        'positive_delta','negative_delta','hydrophobic_delta','polar_delta','ionizable_delta','aromatic_delta','aliphatic_delta','hbond_delta',
                        'sulfur_delta','essential_delta','size_delta']        
        qip_cols = ['gnomAD_exomes_AF','gnomAD_exomes_AC','gnomAD_exomes_nhomalt','mave_label_confidence','clinvar_review_star','accessibility']
        
        other_cols = ['mpc_obs_exp','mpc_mis_badness','mpc_fitted_score','RVIS_EVS','RVIS_percentile_EVS','LoF-FDR_ExAC','RVIS_ExAC','RVIS_percentile_ExAC']


        #**********************************************************************************************************************************************************
        # save core data
        #**********************************************************************************************************************************************************
        alm_predictor = self.proj.predictor[runtime['predictor']]
        alm_dataset = alm_predictor.data_instance
        
        core_df = alm_dataset.train_data_index_df
        extra_df = alm_dataset.extra_train_data_df_lst[0]
        
        
        core_df.to_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] + '_' + runtime['predictor'] + '_core.csv',index = False)
        extra_df.to_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] + '_' + runtime['predictor'] + '_extra.csv',index = False)   
        
        
        #**********************************************************************************************************************************************************
        # MAVE data
        #**********************************************************************************************************************************************************
#         mave_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_target_mave.csv')
#         mave_df['p_vid'].value_counts()
#         
        
       #**********************************************************************************************************************************************************
        # Check hp tuning (train/validation/test) performance
        #**********************************************************************************************************************************************************
#         test_hyperopt = {}
#         test_hyperopt['1213'] = pd.read_csv(runtime['project_path'] + 'output/csv/Revision1213_test_hyperopt_VARITY_R_tf0_test_hyperopt_all.csv')
#         test_hyperopt['1106'] = pd.read_csv(runtime['project_path'] + 'output/csv/Revision1106_test_hyperopt_VARITY_R_tf0_test_hyperopt_all.csv')
#         
#         fig = plt.figure(figsize=(15, 10))
#         i = 0
#         for key in test_hyperopt.keys():            
#             test_hyperopt[key]['overfit'] = test_hyperopt[key]['train_macro_cv_aubprc'] - test_hyperopt[key]['validation_macro_cv_aubprc']
#             i += 1
#             ax = plt.subplot(2, 1, i)
#             ax = sns.scatterplot(x = test_hyperopt[key]['validation_macro_cv_aubprc'],y = test_hyperopt[key]['train_macro_cv_aubprc'], hue = test_hyperopt[key]['aubprc'],s = 25)
#             
#             ax.set_xlim(0.5,1.05)
#             ax.set_ylim(0.5,1.05)
#             ax.set_xlabel('Validation', size=20,labelpad = 10)
#             ax.set_ylabel('Train', size=20,labelpad = 10)
#          fig.suptitle('VARITY_R 1213 Hyperopt Results' ,size = 35)      
#         fig.tight_layout()
#                              
#         plt.savefig(runtime['project_path'] + 'output/img/Revision_test_hyperopt_VARITY_R_tf0_test_hyperopt_all.png') 
#         

        #**********************************************************************************************************************************************************
        # Check denovo performance
        #**********************************************************************************************************************************************************
#         test_hyperopt = pd.read_csv(runtime['project_path'] + 'output/csv/Revision1213_test_hyperopt_VARITY_R_tf0_test_hyperopt_all.csv')
# 
#         hyperopt_denovo_auroc = pd.DataFrame(columns = ['col','spc','pcc'])
#         hyperopt_denovo_aubprc = pd.DataFrame(columns = ['col','spc','pcc'])
#         hyperopt_denovo_validation = pd.DataFrame(columns = ['col','spc','pcc'])
#         
#         
#         i  = 0
#         for col in test_hyperopt.columns:
#             if str(test_hyperopt[col].dtype) != 'object':
#                 auroc_spc = alm_fun.spc_cal(test_hyperopt['auroc'],test_hyperopt[col])
#                 auroc_pcc = alm_fun.pcc_cal(test_hyperopt['auroc'],test_hyperopt[col]) 
#                 hyperopt_denovo_auroc.loc[i,:] = [col,auroc_spc,auroc_pcc]
#                 
#                 aubprc_spc = alm_fun.spc_cal(test_hyperopt['aubprc'],test_hyperopt[col])
#                 aubprc_pcc = alm_fun.pcc_cal(test_hyperopt['aubprc'],test_hyperopt[col]) 
#                 hyperopt_denovo_aubprc.loc[i,:] = [col,aubprc_spc,aubprc_pcc]
#                 
#                 validation_spc = alm_fun.spc_cal(test_hyperopt['validation_macro_cv_aubprc'],test_hyperopt[col])
#                 validation_pcc = alm_fun.pcc_cal(test_hyperopt['validation_macro_cv_aubprc'],test_hyperopt[col]) 
#                 hyperopt_denovo_validation.loc[i,:] = [col,validation_spc,validation_pcc]
#                 
#                 
#                 i = i + 1
#         hyperopt_denovo_auroc.to_csv(runtime['project_path'] + 'output/csv/Revision1213_test_hyperopt_VARITY_R_tf0_test_hyperopt_all_auroc_correlation.csv')
#         hyperopt_denovo_aubprc.to_csv(runtime['project_path'] + 'output/csv/Revision1213_test_hyperopt_VARITY_R_tf0_test_hyperopt_all_aubprc_correlation.csv')
#         hyperopt_denovo_validation.to_csv(runtime['project_path'] + 'output/csv/Revision1213_test_hyperopt_VARITY_R_tf0_test_hyperopt_all_validation_correlation.csv')

        #**********************************************************************************************************************************************************
        # Update PISA
        #**********************************************************************************************************************************************************      
#         key_cols = ['p_vid','aa_ref','aa_pos']
#         pisa_cols = ['bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min','solv_ne_max','asa_mean']
#         pisa_cols = ['bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min','asa_mean']
#          
#         pisa_cols_1 = ['bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min','asa_mean','solv_ne_max','solv_ne_abs_max']
#  
#           
# #         all_pisa_df = pd.read_csv(runtime['db_path'] + 'pisa/all/all_pisa.csv')[key_cols + pisa_cols]
# #         all_pisa1_df = pd.read_csv(runtime['db_path'] + 'pisa/all/all_pisa1.csv')[key_cols + pisa_cols]
# #         all_pisa2_df = pd.read_csv(runtime['db_path'] + 'pisa/all/all_pisa2.csv')[key_cols + pisa_cols]
#         all_pisa3_df = pd.read_csv(runtime['db_path'] + 'pisa/all/all_pisa3.csv')[key_cols + pisa_cols_1]
#  
#                   
# #         input_file = runtime['db_path'] + 'varity/bygene/P01130_varity_snv.csv'
# #         input_file = runtime['db_path'] + 'varity/all/P01130_varity_snv.csv'
# #         input_file = runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_negative_added_snv.csv'
#         input_file = runtime['db_path'] + 'varity/all/varity_target_mave.csv'
#          
# #         input_file = runtime['project_path'] + 'output/csv/denovodb_snv_dn1.csv'         
#         input_df = pd.read_csv(input_file)        
#           
#         input_df['index'] = input_df.index                
#         input_df = input_df.drop(columns = pisa_cols)        
# #         input_pisa_df = pd.merge(input_df,all_pisa_df,how = 'left')
# #         input_pisa1_df = pd.merge(input_df,all_pisa1_df,how = 'left')
# #         input_pisa2_df = pd.merge(input_df,all_pisa2_df,how = 'left')
#         input_pisa3_df = pd.merge(input_df,all_pisa3_df,how = 'left')
#          
# #         input_nopisa_df = input_df.copy()
# #         for pisa_col in pisa_cols:
# #             input_nopisa_df[pisa_col] = np.nan
#                            
#         input_df.index = input_df['index']
#           
# #         input_pisa_df.to_csv(input_file.split('.')[0] + '_pisa.csv')
# #         input_pisa1_df.to_csv(input_file.split('.')[0] + '_pisa1.csv')
# #         input_pisa2_df.to_csv(input_file.split('.')[0] + '_pisa2.csv')
#         input_pisa3_df.to_csv(input_file.split('.')[0] + '_pisa3.csv',index = False)
# #         input_nopisa_df.to_csv(input_file.split('.')[0] + '_nopisa.csv')
#          
#           
#          
# #         print(input_df[pisa_cols + [x + '1' for x in pisa_cols] + [x + '2' for x in pisa_cols]].count())
# #         input_df.to_csv(input_file.split('.')[0] + '_pisa.csv')
# 



        #**********************************************************************************************************************************************************
        # Compare pisa1 pisa2 pisa3
        #**********************************************************************************************************************************************************
#         pisa1_df = pd.read_csv(runtime['db_path'] + 'pisa1/bygene/P01130_pisa.csv')[['p_vid','aa_pos','aa_ref','asa_mean']]
#         pisa1_df = pisa1_df.rename(columns = {'asa_mean':'asa_mean1'})        
#         pisa2_df = pd.read_csv(runtime['db_path'] + 'pisa2/bygene/P01130_pisa.csv')[['p_vid','aa_pos','aa_ref','asa_mean']]
#         pisa2_df = pisa2_df.rename(columns = {'asa_mean':'asa_mean2'})
#         pisa3_df = pd.read_csv(runtime['db_path'] + 'pisa3/bygene/P01130_pisa.csv')[['p_vid','aa_pos','aa_ref','asa_mean']]
#         pisa3_df = pisa3_df.rename(columns = {'asa_mean':'asa_mean3'})
#         
#         pisa_df = pisa3_df.merge(pisa2_df,how = 'left')
#         pisa_df = pisa_df.merge(pisa1_df,how = 'left')
#         
#         pisa_df.to_csv(runtime['db_path'] + 'pisa3/bygene/P01130_pisa_compare.csv',index = False)
#         
#         pisa_df.loc[pisa_df['asa_mean2'] != pisa_df['asa_mean3'],: ]

        #**********************************************************************************************************************************************************
        # Check denovo variatns between 1106 and 1213
        #**********************************************************************************************************************************************************
#         uniprot_pdb_chain = pd.read_csv(runtime['db_path'] + 'pisa/all/uniprot_pdb_chain_dbref.csv')
#         
#         pdb_uniprot_chain = pd.read_csv(runtime['db_path'] + 'pisa/all/pdb_chain_uniprot.csv')
#         pdb_uniprot_chain.columns = ['pdb_id','chain_id','p_vid','res_beg','res_end','pdb_beg_1','pdb_end_1','uniprot_beg','uniprot_end']
#         uniprot_human_reviewed_ids = list(np.load(runtime['db_path'] + 'uniprot/npy/uniprot_human_reviewed_ids.npy'))
#         pdb_uniprot_chain = pdb_uniprot_chain.loc[pdb_uniprot_chain['p_vid'].isin(uniprot_human_reviewed_ids),:]
#         pdb_uniprot_chain = pdb_uniprot_chain.loc[(pdb_uniprot_chain['pdb_beg_1'] != 'None') & (pdb_uniprot_chain['pdb_end_1'] != 'None') ,:]

#         pdb_uniprot_chain['flag'] = 1
#         
#         uniprot_pdb_chain_compare = pd.merge(uniprot_pdb_chain,pdb_uniprot_chain,how = 'left')
#         
#         uniprot_pdb_chain_compare = uniprot_pdb_chain_compare.loc[uniprot_pdb_chain_compare['flag'] == 1,:]
#         uniprot_pdb_chain_compare['pdb_beg'] = uniprot_pdb_chain_compare['pdb_beg'].astype(str)
#         uniprot_pdb_chain_compare['pdb_beg_1'] = uniprot_pdb_chain_compare['pdb_beg_1'].astype(str)
#         uniprot_pdb_chain_compare['uniprot_beg'] = uniprot_pdb_chain_compare['uniprot_beg'].astype(str)
         
#         uniprot_pdb_chain_compare.loc[uniprot_pdb_chain_compare['pdb_beg'] != uniprot_pdb_chain_compare['pdb_beg_1'] ,:].shape
      
#         uniprot_pdb_chain_compare.to_csv(runtime['db_path'] + 'pisa/all/uniprot_pdb_chain_dbref_compare.csv' ,index = False)

#         denovo_1213_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_negative_added_snv_1213.csv')        
#         denovo_1106_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_negative_added_snv_1106_org.csv')    
#         denovo_1213_extra_df = denovo_1213_df[['p_vid','unique_snv_id','asa_mean','bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min']]
#         denovo_1213_extra_df.columns = ['p_vid','unique_snv_id','asa_mean_1213','bsa_max_1213','h_bond_max_1213','salt_bridge_max_1213','disulfide_bond_max_1213','covelent_bond_max_1213','solv_ne_min_1213']
#         denovo_compare_df = pd.merge(denovo_1106_df,denovo_1213_extra_df,how = 'left')   
#                      
#         denovo_compare_df['diff'] = 0
#         denovo_compare_df.loc[(denovo_compare_df['asa_mean'] != denovo_compare_df['asa_mean_1213']) & denovo_compare_df['asa_mean'].notnull(), 'diff'] = 1
#         denovo_compare_df.to_csv (runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_negative_added_snv_compare.csv',index = False)
        
#         denovo_1213_extra_df.loc[denovo_1213_extra_df['p_vid'] == 'Q9BUQ8',['unique_snv_id','asa_mean_1213']]
#         denovo_1106_df.loc[denovo_1106_df['p_vid'] == 'Q9BUQ8',['unique_snv_id','asa_mean']]
#         denovo_compare_df.loc[denovo_compare_df['p_vid'] == 'Q9BUQ8',['unique_snv_id','asa_mean',

#         train_1213_df = pd.read_csv(runtime['project_path'] + 'output/csv/train_data_Revision1213.csv')        
#         train_1106_df = pd.read_csv(runtime['project_path'] + 'output/csv/train_data_Revision1106_org.csv')        
#         train_1213_df = train_1213_df.rename(columns = {'asa_mean': 'asa_mean_1213','bsa_max': 'bsa_max_1213'})
#         train_1106_df = train_1106_df.merge(train_1213_df,how = 'left')        
#         train_1106_df['asa_diff'] = np.abs(train_1106_df['asa_mean'] - train_1106_df['asa_mean_1213'])
#         train_1106_df['diff'] = 0
#         train_1106_df.loc[(train_1106_df['asa_diff'] > 5) & train_1106_df['asa_mean'].notnull(), 'diff'] = 1  
#         train_1106_df.to_csv(runtime['project_path'] + 'output/csv/train_data_pisa_compare.csv',index = False)      
# #         
#         extra_train_1213_df = pd.read_csv(runtime['project_path'] + 'output/csv/extra_train_data_Revision1213.csv')        
#         extra_train_1106_df = pd.read_csv(runtime['project_path'] + 'output/csv/extra_train_data_Revision1106.csv')        
#         extra_train_1213_df = extra_train_1213_df.rename(columns = {'asa_mean': 'asa_mean_1213','bsa_max': 'bsa_max_1213'})
#         extra_train_1106_df = extra_train_1106_df.merge(extra_train_1213_df,how = 'left')
#         extra_train_1106_df['asa_diff'] = np.abs(extra_train_1106_df['asa_mean'] - extra_train_1106_df['asa_mean_1213'])
#         extra_train_1106_df['diff'] = 0
#         extra_train_1106_df.loc[(extra_train_1106_df['asa_diff'] > 5) & extra_train_1106_df['asa_mean'].notnull(), 'diff'] = 1        
#         extra_train_1106_df.to_csv(runtime['project_path'] + 'output/csv/train_data_pisa_compare.csv',index = False)        
# 
#         train_1106_df.loc[train_1106_df['asa_mean'].notnull() & train_1106_df['asa_mean_1213'].isnull(), :].shape        
#         train_1106_df.loc[train_1106_df['asa_mean'].notnull(),].shape
#         train_1106_df.loc[train_1106_df['asa_mean_1213'].notnull(),].shape
# 
#         extra_train_1106_df.loc[extra_train_1106_df['asa_mean'].notnull() & extra_train_1106_df['asa_mean_1213'].isnull(), :].shape        
#         extra_train_1106_df.loc[extra_train_1106_df['asa_mean'].notnull(),].shape
#         extra_train_1106_df.loc[extra_train_1106_df['asa_mean_1213'].notnull(),].shape
#         
#         train_1213_extra_df = train_1213_df[['unique_snv_id','asa_mean','bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min']]
#         train_1213_extra_df.columns = ['unique_snv_id','asa_mean_1213','bsa_max_1213','h_bond_max_1213','salt_bridge_max_1213','disulfide_bond_max_1213','covelent_bond_max_1213','solv_ne_min_1213']
#         train_1106_df = train_1106_df.merge(train_1213_extra_df,how = 'left')    


#         denovo_snv = pd.read_csv(runtime['db_path'] + 'denovodb/all/denovodb_enriched_snv.csv',dtype = {'chr':'str','aa_pos':'int'})
#         x_df = pd.read_csv(runtime['db_path'] + 'varity/bygene/Q9BUQ8_varity_snv.csv',dtype = {'chr':'str','aa_pos':'int'})                                
#         denovo_snv_new = pd.merge(denovo_snv,x_df,how = 'left')         
#         denovo_snv.loc[denovo_snv['p_vid'] == 'Q9BUQ8',['asa_mean']]
#         x_df.loc[x_df['asa_mean'].notnull(),:]
#         
#         all_pisa_1213 = pd.read_csv(runtime['db_path'] + 'pisa/all/all_pisa.csv')                                    
#         all_pisa_1213.loc[all_pisa_1213['p_vid'] == 'Q9BUQ8',:]
                

#         
#         denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'asa_mean'] = denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'asa_mean_1213']
#         denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'bsa_max'] = denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'bsa_max_1213']
#         denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'h_bond_max'] = denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'h_bond_max_1213']
#         denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'salt_bridge_max'] = denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'salt_bridge_max_1213']
#         denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'disulfide_bond_max'] = denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'disulfide_bond_max_1213']
#         denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'covelent_bond_max'] = denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'covelent_bond_max_1213']
#         denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'solv_ne_min'] = denovo_1106_df.loc[denovo_1106_df['asa_mean'].notnull(),'solv_ne_min_1213']
#         
#         denovo_1106_df.to_csv (runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_negative_added_snv_1106.csv',index = False)    
# 
#      
        print ("debug done.")


    def plot_score_versus_probability(self,runtime):
        
#        **********************************************************************************************************************************************************
#         Convert scores to precision
#        **********************************************************************************************************************************************************
        score = runtime['predictor']            
        core_set = pd.read_csv(runtime['project_path'] + 'output/csv/Revision1230_1_Revision1230_1_VARITY_R_core_VARITY_R.csv')[[score,'label']]
     
        score_boundaries = list(np.round(np.linspace(0, 1, 21),2))
        probabilities = []
        scores = []
        all_nums = []
        
        for i in range(len(score_boundaries)-1):            
            score_bs = score_boundaries[i]
            score_es = score_boundaries[i+1]
            
            all_num = core_set.loc[(core_set[score] >= score_bs) & (core_set[score] <= score_es),:].shape[0]
            p_num = core_set.loc[(core_set[score] >= score_bs) & (core_set[score] <= score_es) & (core_set['label'] == 1),:].shape[0]
            all_nums.append(all_num)
            probabilities.append(p_num/all_num)
            scores.append(str(score_bs) + ' ~ ' + str(score_es))
        print (all_nums)            
        fig = plt.figure(figsize=(10, 10))
        plt.clf()
        plt.rcParams["font.family"] = "Helvetica"    
        ax = plt.subplot()                  
        ax.scatter(scores,probabilities,s = 50)
        ax.set_title(score + ' score to probability' ,size = 25,pad = 10)
        ax.set_xlabel(score + ' score range', size=20,labelpad = 10)
        ax.set_ylabel('Probability', size=20,labelpad = 10)
        ax.tick_params(labelsize=15) 
        ax.set_xticklabels(scores,rotation = 70)
        fig.tight_layout()
        fig.savefig(runtime['project_path'] + 'output/img/' + score + '_to_probability.png')  
#                    
            
            

#         y = list(core_set['label'])
#         y_predicted = list(core_set[score])               
#         core_set['precision'] = core_set.apply(lambda x: alm_fun.score_to_precision(y,y_predicted,x[score]),axis = 1)
#          
#         fig = plt.figure(figsize=(30, 20))
#         plt.clf()
#         plt.rcParams["font.family"] = "Helvetica"    
#         ax = plt.subplot()                  
#         ax.scatter(core_set[score],core_set['precision'])
#         ax.set_title(score + ' score to balanced precesion' ,size = 35,pad = 20)
#         ax.set_xlabel(score + ' score', size=30,labelpad = 20)
#         ax.set_ylabel('Balanced Precsion', size=30,labelpad = 20) 
#         fig.tight_layout()
#         fig.savefig(runtime['project_path'] + 'output/img/' + score + '_to_precison.png')  
# #                  
        
    def set_job_name(self,runtime):
        job_name = ''
        if runtime['qip'] == '':
            job_name = runtime['session_id'] + '_' + runtime['action'] + '_' + runtime['predictor'] + '_tf' + str(runtime['cur_test_fold']) + '_' + runtime['batch_id']
        else:
            job_name = runtime['session_id'] + '_' + runtime['action'] + '_' + runtime['predictor'] + '_tf' + str(runtime['cur_test_fold']) + '_' + runtime['qip'] + '_' + runtime['batch_id']
        return (job_name)
    
    def varity_action(self,runtime):       
         
        result_dict = None
        if runtime['batch_id'] == '':
            runtime['batch_id'] = str(alm_fun.get_random_id(10))                            
        #### assign job_name to current job if empty
        if runtime['job_name'] == '':
            runtime['job_name'] = self.set_job_name(runtime)
        
        runtime['log'] = runtime['project_path'] + '/output/log/' + runtime['job_name'] + '.log'
            
        #### if run_on_node then submit the current job to cluster                  
        if runtime['run_on_node'] == 1:           
            [runtime['job_id'],runtime['job_name']] = self.varity_action_cluster(runtime)
            print('\nCommand [' + runtime['action'] + '] on predictor [' + runtime['predictor'] + '] is running on cluster......' )            
        else:                
#             ##### disable commands for 'Final' session            
#             if (runtime['action'] in['init_session','mv_analysis','hp_tuning','save_best_hp','test_cv_prediction']) & (runtime['session_id'] == 'Final'):
#                 print (runtime['action'] + ' command is disabled for  ' + '[' + runtime['session_id'] + '] session, this command may change the existing VARITY models in this session. To create new VARITY models, please initiate a new session.')
#                 sys.exit()
            if runtime['action'] == 'debug':
                self.debug(runtime)

            if  runtime['action'] == 'mv_analysis':                
                self.ml.weights_opt_sp(runtime)    
                
            if runtime['action'] == 'validation_cv_prediction_sp':
                self.ml.fun_validation_cv_prediction_sp(runtime)                   
                
            if runtime['action'] == 'single_validation_fold_prediction':
                self.ml.fun_single_validation_fold_prediction(runtime)        
                
            if runtime['action'] == 'single_fold_prediction':
                self.ml.fun_single_fold_prediction(runtime)                        
                
            if runtime['action'] == 'plot_mv_result':  
                self.ml.plot_sp_result(runtime)                
                                
            if runtime['action'] == 'hp_tuning':                
                self.ml.weights_opt_hyperopt(runtime)     
                
            if runtime['action'] == 'test_hyperopt':
                self.ml.fun_test_hyperopt(runtime)
            
            if runtime['action'] == 'save_best_hp':
                self.ml.save_best_hp_dict_from_trials(runtime)
                                        
            if runtime['action'] == 'plot_hp_weight':
                self.ml.plot_data_weight(runtime)
                
            if runtime['action'] == 'plot_all_hp_weight':
                
                alm_predictor = self.proj.predictor[runtime['predictor']]
                for qip_name in alm_predictor.qip.keys():
                    runtime['plot_qip'] = qip_name
                    runtime['plot_sets'] = alm_predictor.qip[qip_name]['set_list']
                    self.ml.plot_data_weight(runtime)                
                                
            if runtime['action'] == 'test_cv_prediction':
                self.ml.fun_test_cv_prediction(runtime)
                
            if runtime['action'] == 'test_cv_prediction':
                self.ml.fun_test_cv_prediction(runtime)
                                                                 
            if runtime['action'] == 'plot_test_result':
                self.ml.plot_test_result(runtime)
                
            if runtime['action'] == 'plot_mave_result':
                self.ml.plot_mave_result(runtime)
                
            if runtime['action'] == 'plot_ldlr_result':
                self.ml.plot_ldlr_result(runtime)                
                
            if runtime['action'] == 'plot_feature_shap_interaction':
                self.ml.plot_feature_shap_interaction(runtime)     

            if runtime['action'] == 'plot_score_versus_probability':
                self.plot_score_versus_probability(runtime)                                               
                                                                                
            if runtime['action'] == 'target_prediction':
                result_dict = self.ml.fun_target_prediction(runtime)
                
            if runtime['action'] == 'merge_prediction':
                result_dict = self.ml.fun_merge_target_prediction(runtime)
                                                        
            if runtime['action'] == 'update_data':
                self.update_data(runtime)
                
            if runtime['action'] == 'process_denovo_variants':
                self.process_denovo_variants(runtime)  
                
            if runtime['action'] == 'ldlr_analysis':
                self.ldlr_analysis(runtime)      
                
            if runtime['action'] == 'fun_loo_predictions':
                result_dict = self.ml.fun_loo_predictions(runtime)                    
                                         
            if runtime['action'] == 'fun_combine_loo_predictions':
                result_dict = self.ml.fun_combine_loo_predictions(runtime)  
                
            if runtime['action'] == 'add_loo_predictions':
                result_dict = self.add_loo_predictions(runtime)     
                
            if runtime['action'] == 'add_deepsequence':
                result_dict = self.add_deepsequence(runtime)                  
                                                                                                        
            if runtime['action'] == 'run_batch_id_jobs':
                result_dict = self.ml.run_batch_id_jobs(runtime)                    
                                                                               
 
            alm_fun.show_msg(self.ml.log, self.ml.verbose, '\nCommand [' + runtime['action'] + '] on predictor [' + runtime['predictor'] + '] is finished.' )
                             
        return([runtime['job_id'],runtime['job_name'],result_dict])    

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
        log_file = runtime['project_path'] + '/output/log/varity.log'
        
        argvs['project'] = {}
        argvs['ml'] = {}
        
        argvs['project']['project_paht'] = runtime['project_path']
        argvs['project']['log'] = log_file
        argvs['project']['session_id'] = runtime['session_id'] 
        
        for object in ['project','ml']:
            argvs[object]['log'] = log_file
            argvs[object]['verbose'] = 1
            argvs[object]['project_path'] = runtime['project_path']
            argvs[object]['session_id'] = runtime['session_id'] 
        
        for object in ['data','estimator','predictor']:    
            for i in argvs[object].keys():        
                argvs[object][i]['log'] = log_file
                argvs[object][i]['verbose'] = 1
                argvs[object][i]['project_path'] = runtime['project_path']
                argvs[object][i]['session_id'] = runtime['session_id'] 
                
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
        
    def filter_test(self,alm_predictor,cur_test_df,runtime):
        alm_dataset = alm_predictor.data_instance  
        cur_test_df.loc[cur_test_df['gnomAD_exomes_AF'].isnull(),'gnomAD_exomes_AF'] = 1e-06
        
        hmgd_2018 = pd.read_csv(runtime['db_path'] + 'hgmd/org/hgmd_2018.csv')
        hmgd_2018['unique_snv_id'] = hmgd_2018.apply(lambda x: 'chr' + str(x['chr']) + ':' + str(x['nt_pos']) + x['nt_ref'] +  x['nt_alt'],axis = 1)                
        cur_test_df['unique_snv_id'] = cur_test_df.apply(lambda x: 'chr' + str(x['chr']) + ':' + str(x['nt_pos']) + x['nt_ref'] +  x['nt_alt'],axis = 1)
        cur_test_df['hgmd_2018'] = 0
        cur_test_df.loc[cur_test_df['unique_snv_id'].isin(hmgd_2018['unique_snv_id']),'hgmd_2018'] = 1
        
        
        #### add loo scores
        
        if 'VARITY_R_LOO' not in cur_test_df.columns:
            all_loo_predictions = pd.read_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_all_loo_predictions_with_keycols.csv')
            cur_test_df = pd.merge(cur_test_df,all_loo_predictions,how = 'left')
                
        # make sure all predictors have available scores
        print('# of test variants: ' + str(cur_test_df.shape[0]))

        for plot_predictor in runtime['compare_predictors']:
            alm_plot_predictor = self.proj.predictor[plot_predictor]
            if plot_predictor in cur_test_df.columns:
                score_name = plot_predictor
            else:
                score_name = alm_plot_predictor.features[0]
            cur_test_df = cur_test_df.loc[cur_test_df[score_name].notnull(), :]
            print('Removing variants that are missing for' + '[' + plot_predictor + ']: ' + str(cur_test_df.shape[0]))
             
        positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
        negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)
            
        print('# of test variants after filtering for score availability to all predictors: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
                   

        if runtime['filter_test_score'] == 0: # original_test_data 
            print ("No filtering......")
                        
        if runtime['filter_test_score'] == 1: # original_test_data with hgmd variants removed   
            cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_source']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]  

#             
        if runtime['filter_test_score'] == 2: # original_test_data with hgmd variants removed   
            cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_source']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]
            cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_2018']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]   
          
            
        if runtime['filter_test_score'] == 3:
            cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_source']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]  
            cur_test_df = cur_test_df.loc[~((cur_test_df['gnomad_source']== 1) & (cur_test_df[runtime['dependent_variable']]== 0)) ,:]
                      
#             negative_genes = list(cur_test_df.loc[cur_test_df[runtime['dependent_variable']] == 0,'p_vid'].unique())
#             cur_test_df = cur_test_df.loc[cur_test_df['p_vid'].isin(negative_genes),:]
                  
        if runtime['filter_test_score'] == 'ds': # add deep sequence score 
            print ("No filtering......")
            
        if runtime['filter_test_score'] == 'ev': # add deep sequence score 
#             cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_source']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]
            print ("No filtering......")            
            

        if runtime['filter_test_score'] == 'dn0': # original_test_data with hgmd variants removed       
            
#             cur_test_df = cur_test_df.loc[cur_test_df['gnomAD_exomes_AF'] <= 1e-06,:]   
#             positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
#             negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)
#             print('# of test variants after removing higher MAF than 1E-06: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')         
            #### fdr_cutoff                 
            #### make sure all variants are MAF <0.005
#             cur_test_df = cur_test_df.loc[cur_test_df['gnomAD_exomes_AF'] <= 0.005,:]   
#             positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
#             negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)
#             print('# of test variants after removing higher MAF than 0.005: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')         
            #### fdr_cutoff
            cur_test_df = cur_test_df.loc[cur_test_df['fdr_cutoff'] <= 0.05,:]
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)            
            print('# of test variants after removing FDR > 0.05: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
            #### control how many negative variants want to add 
            cur_test_df = cur_test_df.loc[~((cur_test_df['from_denovodb'] == 0) & (cur_test_df[runtime['dependent_variable']]== 0) & (cur_test_df['gnomAD_exomes_nhomalt'] < 1000)) ,:]
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
            print('# of test variants after removing added negative variants : ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
            
            
            # remove VARITY training variants   
#             cur_test_df = cur_test_df.loc[cur_test_df['VARITY_R_LOO'].isnull(),:]
            
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df[runtime['dependent_variable']]== 1)),:]
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df['train_humsavar_source']== 1)),:]
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df['train_clinvar_source']== 1)),:]
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df['train_mave_source']== 1)),:]
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df['train_hgmd_source']== 1)),:]
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df['train_gnomad_source']== 1)),:]            
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
            print('# of test variants after removing VARITY_R training data: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
#             
#             #### remove variants existed in ClinVAR, HumsaVAR, MAVE and HGMD(2015, and current Version)             
            cur_test_df = cur_test_df.loc[~((cur_test_df['new_hgmd']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:] 
            cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_2018']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]               
            cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_source']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]
#             
#             positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
#             negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
#             print('# of test variants after removing HGMD appearance : ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
#             cur_test_df = cur_test_df.loc[cur_test_df['gnomad_source']!= 1,:]
            cur_test_df = cur_test_df.loc[cur_test_df['clinvar_source']!= 1,:]
            cur_test_df = cur_test_df.loc[cur_test_df['humsavar_source']!= 1,:]
            cur_test_df = cur_test_df.loc[cur_test_df['mave_source']!= 1,:]
#             
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
            print('# of test variants after removing ClinVAR, HumsaVAR and MAVE appearance: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')

            cur_test_df.to_csv(runtime['project_path'] + 'output/csv/denovodb_snv_dn0.csv')  
                                                                
            
        if runtime['filter_test_score'] == 'dn1': # original_test_data with hgmd variants removed            
            #### make sure all variants are MAF <0.005
            cur_test_df = cur_test_df.loc[cur_test_df['gnomAD_exomes_AF'] <= 0.005,:]   
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)
            print('# of test variants after removing higher MAF than 0.005: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')         
            #### fdr_cutoff
            cur_test_df = cur_test_df.loc[cur_test_df['fdr_cutoff'] <= 0.05,:]
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)            
            print('# of test variants after removing FDR > 0.05: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
            #### control how many negative variants want to add 
            cur_test_df = cur_test_df.loc[~((cur_test_df['from_denovodb'] == 0) & (cur_test_df[runtime['dependent_variable']]== 0) & (cur_test_df['gnomAD_exomes_nhomalt'] < 1000)) ,:]
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
            print('# of test variants after removing added negative variants : ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
            
            
#             remove VARITY training variants   
            cur_test_df = cur_test_df.loc[cur_test_df['VARITY_R_weight'].isnull(),:]
            cur_test_df = cur_test_df.loc[cur_test_df['VARITY_ER_weight'].isnull(),:]
            
            
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df[runtime['dependent_variable']]== 1)),:]
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df['train_humsavar_source']== 1)),:]
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df['train_clinvar_source']== 1)),:]
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df['train_mave_source']== 1)),:]
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df['train_hgmd_source']== 1)),:]
#             cur_test_df = cur_test_df.loc[~(cur_test_df['VARITY_R_LOO'].notnull() & (cur_test_df['train_gnomad_source']== 1)),:]            
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
            print('# of test variants after removing VARITY_R and VARITY_ER training data: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
#             
            #### remove variants existed in ClinVAR, HumsaVAR, MAVE and HGMD(2015, and current Version)             
            cur_test_df = cur_test_df.loc[~((cur_test_df['new_hgmd']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:] 
#             cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_2018']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]               
            cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_source']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]
            
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
            print('# of test variants after removing HGMD appearance : ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
#             cur_test_df = cur_test_df.loc[cur_test_df['gnomad_source']!= 1,:]
#             cur_test_df = cur_test_df.loc[cur_test_df['clinvar_source']!= 1,:]
#             cur_test_df = cur_test_df.loc[cur_test_df['humsavar_source']!= 1,:]
#             cur_test_df = cur_test_df.loc[cur_test_df['mave_source']!= 1,:]
             
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
            print('# of test variants after removing ClinVAR, HumsaVAR and MAVE appearance: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')

            cur_test_df.to_csv(runtime['project_path'] + 'output/csv/denovodb_snv_dn1.csv')        
            
        if runtime['filter_test_score'] == 'dn2': # original_test_data with hgmd variants removed            
            #### make sure all variants are MAF <0.005
            cur_test_df = cur_test_df.loc[cur_test_df['gnomAD_exomes_AF'] <= 1e-06,:]   
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)
            print('# of test variants after removing higher MAF than 1E-06: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')         
            #### fdr_cutoff
            cur_test_df = cur_test_df.loc[cur_test_df['fdr_cutoff'] <= 0.05,:]
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)            
            print('# of test variants after removing FDR > 0.05: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
            #### control how many negative variants want to add 
            cur_test_df = cur_test_df.loc[~((cur_test_df['from_denovodb'] == 0) & (cur_test_df[runtime['dependent_variable']]== 0) & (cur_test_df['gnomAD_exomes_nhomalt'] < 1000)) ,:]
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
            print('# of test variants after removing added negative variants : ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')

            #### remove variants existed in ClinVAR, HumsaVAR, MAVE and HGMD(2015, and current Version)             
            cur_test_df = cur_test_df.loc[~((cur_test_df['new_hgmd']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:] 
#             cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_2018']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]               
            cur_test_df = cur_test_df.loc[~((cur_test_df['hgmd_source']== 1) & (cur_test_df[runtime['dependent_variable']]== 1)) ,:]
            
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
            print('# of test variants after removing HGMD appearance : ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
#             cur_test_df = cur_test_df.loc[cur_test_df['gnomad_source']!= 1,:]
            cur_test_df = cur_test_df.loc[cur_test_df['clinvar_source']!= 1,:]
            cur_test_df = cur_test_df.loc[cur_test_df['humsavar_source']!= 1,:]
            cur_test_df = cur_test_df.loc[cur_test_df['mave_source']!= 1,:]
            
            positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
            negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)              
            print('# of test variants after removing ClinVAR, HumsaVAR and MAVE appearance: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')

            cur_test_df.to_csv(runtime['project_path'] + 'output/csv/denovodb_snv_dn2.csv')   

        if runtime['filter_test_score'] == 'MAF_1E06':

            cur_test_df = cur_test_df.loc[cur_test_df['gnomAD_exomes_AF'] <= 1e-06,:]  
            cur_test_df = cur_test_df.loc[cur_test_df['hgmd_source']!= 1 ,:]  
            cur_test_df = cur_test_df.loc[cur_test_df['clinvar_source']!= 1,:]
            cur_test_df = cur_test_df.loc[cur_test_df['humsavar_source']!= 1,:]                           
            
        if runtime['filter_test_score'] == 'MAF_0.005':
#             remove VARITY training variants   
#             cur_test_df = cur_test_df.loc[cur_test_df['VARITY_R_weight'].isnull(),:]
#             cur_test_df = cur_test_df.loc[cur_test_df['VARITY_ER_weight'].isnull(),:]
            cur_test_df = cur_test_df.loc[cur_test_df['gnomAD_exomes_AF'] <= 0.005,:]
#             cur_test_df = cur_test_df.loc[cur_test_df['hgmd_source']!= 1 ,:]  
#             cur_test_df = cur_test_df.loc[cur_test_df['clinvar_source']!= 1,:]
#             cur_test_df = cur_test_df.loc[cur_test_df['humsavar_source']!= 1,:]   
#             cur_test_df = cur_test_df.loc[cur_test_df['hgmd_2018']!= 1,:]                  
            
        if runtime['filter_test_score'] == 'MAF_1':

            cur_test_df = cur_test_df.loc[cur_test_df['gnomAD_exomes_AF'] <= 1,:] 
            cur_test_df = cur_test_df.loc[cur_test_df['hgmd_source']!= 1 ,:]  
            cur_test_df = cur_test_df.loc[cur_test_df['clinvar_source']!= 1,:]
            cur_test_df = cur_test_df.loc[cur_test_df['humsavar_source']!= 1,:]         

        if runtime['filter_test_score'] == 4: # original_test_data with hgmd variants removed , plus gnomad variants
            cur_test_df = cur_test_df.loc[cur_test_df['hgmd_source']!= 1,:]  
            extra_test_df = alm_dataset.extra_train_data_df_lst[0].copy()            
            if '_R_' in alm_predictor.name:
#                 gnomAD_exomes_nhomalt
                extra_test_df = extra_test_df.loc[(extra_test_df['train_gnomad_source'] == 1) & (extra_test_df['weight']==0) & (extra_test_df['gnomAD_exomes_AF'] <= 0.005),:]
#                 extra_test_df = extra_test_df.loc[(extra_test_df['train_gnomad_source'] == 1) & (extra_test_df['gnomAD_exomes_AF'] <= 0.005),:]
            if '_ER_' in alm_predictor.name:
                extra_test_df = extra_test_df.loc[(extra_test_df['train_gnomad_source'] == 1) & (extra_test_df['weight']==0) & (extra_test_df['gnomAD_exomes_AF'] <= 1e-06),:]
            cur_test_df = pd.concat([cur_test_df,extra_test_df])
            
        if runtime['filter_test_score'] == 5: # original_test_data with hgmd variants removed , plus gnomad variants
            cur_test_df = cur_test_df.loc[cur_test_df['hgmd_source']!= 1,:]  
            extra_test_df = alm_dataset.extra_train_data_df_lst[0].copy()            
            if '_R_' in alm_predictor.name:
#                 gnomAD_exomes_nhomalt
                extra_test_df = extra_test_df.loc[(extra_test_df['train_gnomad_source'] == 1) & (extra_test_df['gnomAD_exomes_AF'] <= 0.005),:]
            if '_ER_' in alm_predictor.name:
                extra_test_df = extra_test_df.loc[(extra_test_df['train_gnomad_source'] == 1) & (extra_test_df['gnomAD_exomes_AF'] <= 1e-06),:]
            cur_test_df = pd.concat([cur_test_df,extra_test_df])
         
 
        positive_num = np.sum(cur_test_df[runtime['dependent_variable']]== 1)
        negative_num = np.sum(cur_test_df[runtime['dependent_variable']]== 0)
            
        print('# of test variants after final filtering: ' + str(cur_test_df.shape[0]) + '[P:' + str(positive_num) + ',' + 'N:' + str(negative_num) + ']')
                       
                                                                                    
        remain_test_indices = list(cur_test_df.index)
        return(remain_test_indices)
    
    def add_test(self,alm_dataset,runtime):
        org_test_df = alm_dataset.train_data_index_df.loc[alm_dataset.test_splits_df[runtime['cur_test_fold']][runtime['cur_gradient_key']],:]
        cur_train_df = alm_dataset.train_data_index_df.loc[alm_dataset.train_splits_df[runtime['cur_test_fold']][runtime['cur_gradient_key']],:]
        extra_data_df = None
        if len(alm_dataset.extra_train_data_df_lst) != 0:    
            extra_data_df = alm_dataset.extra_train_data_df_lst[alm_dataset.extra_data_index].copy()                 
        pass    
        if extra_data_df is not None:    
#             added_test_df = pd.concat([org_test_df, extra_data_df.loc[add_test_index,:]])
            added_test_df = pd.concat([org_test_df, extra_data_df])
        else:
            add_test_df = org_test_df                        
        return(added_test_df)
    
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
                    #there might be slightly delay between the job ending and saving the results, so wait 1 seconds ? really?                     
                    if not os.path.isfile(cur_job_result): # no result
                        time.sleep(1)
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
                                                                
                                                                
            if runtime['show_detail_monitor_jobs'] == 1:                                                                
                alm_fun.show_msg (cur_log,1, str(running_jobs_num) + '/' +  str(len(cur_jobs_dict.keys()))  + ' ' +  str(running_jobs) +  ' jobs are still running......')
            else:
                alm_fun.show_msg (cur_log,1, str(running_jobs_num) + '/' +  str(len(cur_jobs_dict.keys())) +  ' jobs are still running......')
        return (all_parallel_jobs_done)
            
    def varity_action_cluster(self,runtime):        
        runtime['run_on_node'] = 0                             
        job_command = 'python3 ' + runtime['varity_command']          
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
    

        cpus = '1'
        sh_file = open(runtime['project_path'] + '/output/bat/' + str(runtime['job_name']  )   + '.sh','w')  
        sh_file.write('#!/bin/bash' + '\n')
        sh_file.write('# set the number of nodes' + '\n')
        sh_file.write('#SBATCH --nodes=1' + '\n')
        sh_file.write('# set the number of tasks' + '\n')
        sh_file.write('#SBATCH --ntasks=1' + '\n')
        sh_file.write('# set the number of cpus per task' + '\n')
        sh_file.write('#SBATCH --cpus-per-task=' + cpus + '\n')        
        sh_file.write('# set the memory for each node' + '\n')
        sh_file.write('#SBATCH --mem=' + str(runtime['mem']) + '\n')    
        sh_file.write('# set name of job' + '\n')
        sh_file.write('#SBATCH --job-name=' + str(runtime['job_name']  )  + '\n')
        sh_file.write("srun " + job_command)
        sh_file.close()
        
        if runtime['node'] != '':
            sbatch_cmd = 'sbatch --nodelist=' + runtime['node'] + ' ' +  runtime['project_path'] + '/output/bat/' + str(runtime['job_name']  )  + '.sh'
        else:
        
            if exclusion_nodes_list == '':
                sbatch_cmd = 'sbatch ' + runtime['project_path'] + '/output/bat/' + str(runtime['job_name']  )  + '.sh'
            else:
                sbatch_cmd = 'sbatch --exclude=galen['  + exclusion_nodes_list + '] ' + runtime['project_path'] + '/output/bat/' + str(runtime['job_name']) + '.sh'
            
        print(sbatch_cmd)
        #check if number of pending jobs
        chk_pending_cmd = 'squeue -u jwu -t PENDING'  
        return_process =  subprocess.run(chk_pending_cmd.split(" "), cwd = runtime['project_path'] + '/output/log/',capture_output = True,text=True)
        pending_list = return_process.stdout                                 
        pending_num = len(pending_list.split('\n'))
        print ('Current number of pending jobs:' + str(pending_num))            
        runtime['job_id'] = '-1'
        
        if pending_num < 100:      
            retry_num = 0
            while (runtime['job_id'] == '-1') & (retry_num < 10):
                try:
                    return_process = subprocess.run(sbatch_cmd.split(" "), cwd = runtime['project_path'] + '/output/log/',capture_output = True,text=True)
                    time.sleep(0.1)
                    if return_process.returncode == 0:
                        runtime['job_id'] = return_process.stdout.rstrip().split(' ')[-1]
                    else:
                        runtime['job_id'] = '-1'
                        retry_num = retry_num + 1
                        print  (runtime['job_name'] + ' submit error,rescheduling......'  + str(retry_num))
                except:
                    runtime['job_id'] = '-1'
                    print(traceback.format_exc())
                    print (runtime['job_name'] + ' expected error,rescheduling......')
            print (runtime['job_name'] + ' submitted id: ' + runtime['job_id']) 

                 
        return([runtime['job_id'],runtime['job_name']])
    
    def fun_run_subprocess(self,cmd,runtime):
        return_code = subprocess.run(cmd.split(" "), cwd = runtime['project_path'] + '/output/log/',capture_output = True,text=True)        
        return (return_code)                       
    
    def update_data(self,runtime):        
        alm_predictor = self.proj.predictor[runtime['predictor']]
        alm_dataset = alm_predictor.data_instance
        
        if runtime['check_data'] == 1:
#             key_cols = ['chr','nt_pos','nt_ref','nt_alt','p_vid','aa_pos','aa_ref','aa_alt']
#             alm_dataset.train_data_index_df[key_cols + ['asa_mean','bsa_max']].to_csv(runtime['project_path'] + 'output/csv/train_data_' + runtime['session_id'] + '.csv',index = False)
#             alm_dataset.extra_train_data_df_lst[0][key_cols + ['asa_mean','bsa_max']].to_csv(runtime['project_path'] + 'output/csv/extra_train_data_' + runtime['session_id'] + '.csv',index = False)
#             
            print(alm_dataset.train_data_index_df[['asa_mean','bsa_max']].count())#
            print(alm_dataset.extra_train_data_df_lst[0][['asa_mean','bsa_max']].count())#
             
                    
        if runtime['add_deepsequence_scores'] == 1:                    
            deepsequence_df = pd.read_csv(runtime['db_path'] + 'deepsequence/all/all_deepsequence_scores.csv')            
            #Remove deepsequence score if exists
            if 'deepsequence_score' in alm_dataset.train_data_index_df.columns:
                alm_dataset.train_data_index_df = alm_dataset.train_data_index_df.drop(columns = ['deepsequence_score'])
                alm_dataset.extra_train_data_df_lst[0] = alm_dataset.extra_train_data_df_lst[0].drop(columns = ['deepsequence_score'])
            
            
            alm_dataset.train_data_index_df['index'] =  alm_dataset.train_data_index_df.index
            alm_dataset.train_data_index_df = pd.merge(alm_dataset.train_data_index_df,deepsequence_df,how = 'left')
            alm_dataset.train_data_index_df.index = alm_dataset.train_data_index_df['index']
            
            alm_dataset.extra_train_data_df_lst[0]['index'] = alm_dataset.extra_train_data_df_lst[0].index
            alm_dataset.extra_train_data_df_lst[0] = pd.merge(alm_dataset.extra_train_data_df_lst[0],deepsequence_df,how = 'left')
            alm_dataset.extra_train_data_df_lst[0].index = alm_dataset.extra_train_data_df_lst[0]['index']

            alm_dataset.save_data()
            print ('deepsquence score added......')
                
        if runtime['add_core_setname'] == 1:
            
            alm_dataset.train_data_df.loc[alm_dataset.train_data_df['label'] == 1,'set_name'] = 'core_clinvar_1'  
            alm_dataset.train_data_df.loc[alm_dataset.train_data_df['label'] == 0,'set_name'] = 'core_clinvar_0'
            alm_dataset.train_data_index_df = alm_dataset.train_data_df
            alm_dataset.validation_data_index_df = alm_dataset.train_data_df
            alm_dataset.test_data_index_df = alm_dataset.train_data_df
            alm_dataset.save_data()
            print ('core set name added......')
        
        if runtime['old_system'] == 1: # convert the old data format to new (the manscript data format is old)
            alm_dataset.train_data_original_df = alm_dataset.convert_old_data(alm_dataset.train_data_original_df)
            alm_dataset.extra_train_data_df_lst[0] = alm_dataset.convert_old_data(alm_dataset.extra_train_data_df_lst[0])
            alm_dataset.extra_train_data_df_lst[0]['extra_data'] = 1
            alm_dataset.train_data_df = alm_dataset.convert_old_data(alm_dataset.train_data_df)
            alm_dataset.train_data_df['extra_data'] = 0
            
            alm_dataset.train_data_index_df = alm_dataset.train_data_df
            alm_dataset.validation_data_index_df = alm_dataset.train_data_df
            alm_dataset.test_data_index_df = alm_dataset.train_data_df
            alm_dataset.save_data()
            print ('old system data converted......')
        
        if runtime['add_new_scores'] == 1:

            #Remove mpc and mistic columns if exist already
            alm_dataset.train_data_index_df = alm_dataset.train_data_index_df.drop(columns = ['mpc_obs_exp','mpc_mis_badness','mpc_fitted_score','mpc_score','mistic_score'])
            alm_dataset.extra_train_data_df_lst[0] = alm_dataset.extra_train_data_df_lst[0].drop(columns = ['mpc_obs_exp','mpc_mis_badness','mpc_fitted_score','mpc_score','mistic_score'])
            
            #take care of the chr column (the problem came from merge different training data from each chromosome)
            alm_dataset.train_data_index_df['chr'] = alm_dataset.train_data_index_df['chr'].astype(str)                
            alm_dataset.train_data_index_df['chr'] = alm_dataset.train_data_index_df['chr'].apply(lambda x: x.split('.')[0])
            alm_dataset.train_data_index_df['nt_pos'] = alm_dataset.train_data_index_df['nt_pos'].astype(int)
            alm_dataset.train_data_index_df['index'] = alm_dataset.train_data_index_df.index
                                                            
            alm_dataset.extra_train_data_df_lst[0]['chr'] = alm_dataset.extra_train_data_df_lst[0]['chr'].astype(str)                 
            alm_dataset.extra_train_data_df_lst[0]['chr'] = alm_dataset.extra_train_data_df_lst[0]['chr'].apply(lambda x: x.split('.')[0])
            alm_dataset.extra_train_data_df_lst[0].loc[alm_dataset.extra_train_data_df_lst[0]['nt_pos'].isnull(),'nt_pos'] = -1
#                 alm_dataset.extra_train_data_df_lst[0]['nt_pos'] = alm_dataset.extra_train_data_df_lst[0]['nt_pos'].astype(float)
            alm_dataset.extra_train_data_df_lst[0]['nt_pos'] = alm_dataset.extra_train_data_df_lst[0]['nt_pos'].astype(int)
            alm_dataset.extra_train_data_df_lst[0]['index'] = alm_dataset.extra_train_data_df_lst[0].index

            print ('train_data: '  + str(alm_dataset.train_data_index_df['chr'].unique()))
            print ('extra train data: ' + str(alm_dataset.extra_train_data_df_lst[0]['chr'].unique()))

            mpc_df = pd.read_csv(alm_dataset.db_path + '/mpc/all/mpc_values_v2_avg_duplicated_scores.csv',dtype = {'chr':'str'})  
            print ('mpc: '  + str(mpc_df['chr'].unique()))
            mistic_df = pd.read_csv(alm_dataset.db_path + '/mistic/all/MISTIC_GRCh37_avg_duplicated_scores.csv',dtype = {'chr':'str'}) 
            mistic_df.loc[mistic_df['chr'] == 'chrX','chr'] = 'X'
            print ('mistic: '  + str(mistic_df['chr'].unique()))
            
            print ('Before add new score, extra train: ' + str(alm_dataset.extra_train_data_df_lst[0].shape) + 'train: ' + str(alm_dataset.train_data_index_df.shape))
            print('extra train dtypes:' + str(alm_dataset.extra_train_data_df_lst[0][['chr','nt_pos','nt_ref','nt_alt']].dtypes))
            print('train dtypes:' + str(alm_dataset.extra_train_data_df_lst[0][['chr','nt_pos','nt_ref','nt_alt']].dtypes))
            print('mistic dtypes:' + str(mistic_df[['chr','nt_pos','nt_ref','nt_alt']].dtypes))
            print('mpc dtypes:' + str(mpc_df[['chr','nt_pos','nt_ref','nt_alt']].dtypes))

            new_extra_train_data_df = pd.merge(alm_dataset.extra_train_data_df_lst[0],mpc_df,how = 'left')
            new_extra_train_data_df = pd.merge(new_extra_train_data_df,mistic_df,how = 'left')
            new_extra_train_data_df.index = new_extra_train_data_df['index']
                            
            new_train_data_index_df = pd.merge(alm_dataset.train_data_index_df,mpc_df,how = 'left')
            new_train_data_index_df = pd.merge(new_train_data_index_df,mistic_df,how = 'left')
            new_train_data_index_df.index = new_train_data_index_df['index']
                                                                               
            alm_dataset.extra_train_data_df_lst[0] = new_extra_train_data_df
            alm_dataset.train_data_index_df = new_train_data_index_df                
                            
            print ('after add new score, extra train: ' + str(alm_dataset.extra_train_data_df_lst[0].shape) + 'train: ' + str(alm_dataset.train_data_index_df.shape))                
            print ('train data mistic available: ' + str(alm_dataset.train_data_index_df.loc[alm_dataset.train_data_index_df['mistic_score'].notnull(),:].shape[0]))
            print ('train data mpc available: ' + str(alm_dataset.train_data_index_df.loc[alm_dataset.train_data_index_df['mpc_score'].notnull(),:].shape[0]))                
            print ('extra train data mistic available: ' + str(alm_dataset.extra_train_data_df_lst[0].loc[alm_dataset.extra_train_data_df_lst[0]['mistic_score'].notnull(),:].shape[0]))
            print ('extra train data mpc available: ' + str(alm_dataset.extra_train_data_df_lst[0].loc[alm_dataset.extra_train_data_df_lst[0]['mpc_score'].notnull(),:].shape[0]))
                            
            alm_dataset.save_data()
            print ('new scores MISTIC and MPC added.....')   
            
        if runtime['shrink_data'] == 1:
            
            key_cols = ['chr','nt_pos','nt_ref','nt_alt','p_vid','aa_pos','aa_ref','aa_alt']
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
                            'aa_psipred_H','aa_psipred_C','bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min',
                            'mw_delta','pka_delta','pkb_delta','pi_delta','hi_delta','pbr_delta','avbr_delta','vadw_delta','asa_delta','cyclic_delta','charge_delta',
                            'positive_delta','negative_delta','hydrophobic_delta','polar_delta','ionizable_delta','aromatic_delta','aliphatic_delta','hbond_delta',
                            'sulfur_delta','essential_delta','size_delta']
            
            qip_cols = ['gnomAD_exomes_AF','gnomAD_exomes_AC','gnomAD_exomes_nhomalt','mave_label_confidence','clinvar_review_star','accessibility']
            
            other_cols = ['set_name','log_af']
            
            all_cols = list(set(key_cols + annotation_cols + score_cols + feature_cols + qip_cols + other_cols))
                    
            alm_dataset.train_data_df = None                                 
            alm_dataset.train_data_original_df = None
            alm_dataset.test_data_df = None 
            alm_dataset.target_data_df = None 
            
            
            alm_dataset.validation_data_index_df = None
            alm_dataset.test_data_index_df = None 
            alm_dataset.target_data_index_df = None 
            
            alm_dataset.train_data_for_target_df = None 
            alm_dataset.target_data_for_target_df = None 
            alm_dataset.validation_data_for_target_df = None                 
            
            alm_dataset.train_data_index_df = alm_dataset.train_data_index_df.loc[:,all_cols]
            alm_dataset.extra_train_data_df_lst[0] = alm_dataset.extra_train_data_df_lst[0].loc[:,all_cols]
                            
            alm_dataset.save_data()
            print ('Data shrinked.....')   
            
        if runtime['add_complete_sift_provean_scores'] == 1:
                                   
            alm_dataset.train_data_index_df = self.add_sift_provean(alm_dataset.train_data_index_df,runtime) 
            alm_dataset.extra_train_data_df_lst[0] = self.add_sift_provean(alm_dataset.extra_train_data_df_lst[0],runtime)
            alm_dataset.save_data()
            print ('complete sift provean scores added......')  
            
        if runtime['update_pisa'] == 1:
            
            all_pisa_df = pd.read_csv(runtime['db_path'] + 'pisa/all/all_' + runtime['pisa_folder'] + '.csv')            
            key_cols = ['p_vid','aa_ref','aa_pos']
            pisa_cols = ['bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min','asa_mean']
            
            pisa_cols_1 = ['bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min','asa_mean','solv_ne_max','solv_ne_abs_max']
            
            print(alm_dataset.train_data_index_df[['asa_mean','bsa_max','solv_ne_min']].count())
            print(alm_dataset.extra_train_data_df_lst[0][['asa_mean','bsa_max','solv_ne_min']].count())
 
            alm_dataset.train_data_index_df['index'] = alm_dataset.train_data_index_df.index
            alm_dataset.extra_train_data_df_lst[0]['index'] = alm_dataset.extra_train_data_df_lst[0].index
 
            alm_dataset.train_data_index_df = alm_dataset.train_data_index_df.drop(columns = pisa_cols)
            alm_dataset.extra_train_data_df_lst[0] = alm_dataset.extra_train_data_df_lst[0].drop(columns = pisa_cols)
            alm_dataset.train_data_index_df = pd.merge(alm_dataset.train_data_index_df,all_pisa_df[key_cols + pisa_cols_1],how = 'left')
            alm_dataset.extra_train_data_df_lst[0] = pd.merge(alm_dataset.extra_train_data_df_lst[0],all_pisa_df[key_cols + pisa_cols_1],how = 'left')
             
            alm_dataset.train_data_index_df.index = alm_dataset.train_data_index_df['index']
            alm_dataset.extra_train_data_df_lst[0].index = alm_dataset.extra_train_data_df_lst[0]['index']
            
            print(alm_dataset.train_data_index_df[['asa_mean','bsa_max','solv_ne_min']].count())#
            print(alm_dataset.extra_train_data_df_lst[0][['asa_mean','bsa_max','solv_ne_min']].count())#
                                                     
            alm_dataset.save_data()
            print ('complete updating pisa......')
            
        if runtime['add_loo'] == 1:
            all_loo_predictions = pd.read_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_all_loo_predictions_with_keycols.csv')
            
            if 'VARITY_R_LOO' not in alm_dataset.train_data_index_df.columns:
                alm_dataset.train_data_index_df['index'] = alm_dataset.train_data_index_df.index
                alm_dataset.extra_train_data_df_lst[0]['index'] = alm_dataset.extra_train_data_df_lst[0].index
                
                alm_dataset.train_data_index_df = pd.merge(alm_dataset.train_data_index_df,all_loo_predictions,how = 'left')
                alm_dataset.extra_train_data_df_lst[0] = pd.merge(alm_dataset.extra_train_data_df_lst[0],all_loo_predictions,how = 'left')
                                 
                alm_dataset.train_data_index_df.index = alm_dataset.train_data_index_df['index']
                alm_dataset.extra_train_data_df_lst[0].index = alm_dataset.extra_train_data_df_lst[0]['index']
                
                alm_dataset.save_data()    
                print ('complete adding loo preditions......')
            else:             
                print ('loo preditions exist......')                          
    
    def process_denovo_variants(self,runtime):        
        #**********************************************************************************************************************************************************
        # Add negative examples from gnomAD  
        #**********************************************************************************************************************************************************                
        varity_target_denovodb_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_all.csv')        
        varity_target_denovodb_homo_df = varity_target_denovodb_df.loc[(varity_target_denovodb_df['gnomAD_exomes_AF'] < 0.005) & (varity_target_denovodb_df['gnomAD_exomes_nhomalt'] >= 1) & (varity_target_denovodb_df['label'] == 0),:]
        varity_target_denovodb_homo_df['denovo_label'] = varity_target_denovodb_homo_df['label']
        varity_target_denovodb_homo_df = varity_target_denovodb_homo_df.drop(columns = ['label'])
        varity_target_denovodb_homo_df['unique_snv_id'] = varity_target_denovodb_homo_df.apply(lambda x: 'chr' + str(x['chr']) + ':' + str(x['nt_pos']) + x['nt_ref'] +  x['nt_alt'],axis = 1)
        varity_target_denovodb_homo_df['from_denovodb'] = 0
        varity_target_denovodb_homo_df = varity_target_denovodb_homo_df.loc[varity_target_denovodb_homo_df['p_vid'] != 'P62805',:]
        
                   
        varity_target_denovodb_homo_df.to_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_homo.csv',index = False)
         
        print ('Total negative variants to add: ' + str(varity_target_denovodb_homo_df.shape[0]))
           
        #### adding negative examples to denovodb enriched snvs 
        denovo_enriched_snv_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_snv.csv')
        
        denovo_enriched_snv_df = denovo_enriched_snv_df.loc[denovo_enriched_snv_df['p_vid'] != 'P62805',:]
        denovo_enriched_snv_df['unique_snv_id'] = denovo_enriched_snv_df.apply(lambda x: 'chr' + str(x['chr']) + ':' + str(x['nt_pos']) + x['nt_ref'] +  x['nt_alt'],axis = 1)
        denovo_enriched_snv_df['from_denovodb'] = 1
        denovo_enriched_snv_df = denovo_enriched_snv_df[varity_target_denovodb_homo_df.columns]              
        varity_target_denovodb_homo_extra_df = varity_target_denovodb_homo_df.loc[~varity_target_denovodb_homo_df['unique_snv_id'].isin(denovo_enriched_snv_df['unique_snv_id']),:]
        denovo_enriched_negative_added_snv_df = pd.concat([denovo_enriched_snv_df,varity_target_denovodb_homo_extra_df]) 
        
        #**********************************************************************************************************************************************************
        # Add new HGMD 
        #**********************************************************************************************************************************************************                    
        denovo_hgmd_df = pd.read_csv(runtime['db_path'] + 'hgmd/org/denovodb_hgmd_2020.csv')      
        hgmd_existed_ids = list(denovo_hgmd_df.loc[denovo_hgmd_df['annotation'].notnull(),'unique_snv_id'].unique())
        denovo_enriched_negative_added_snv_df['new_hgmd'] = 0   
        denovo_enriched_negative_added_snv_df.loc[denovo_enriched_negative_added_snv_df['unique_snv_id'].isin(hgmd_existed_ids) ,'new_hgmd'] = 1
                        
        #**********************************************************************************************************************************************************
        # add FDR_cutoff denovodb gene
        #**********************************************************************************************************************************************************
        denovodb_enriched_genes_df = pd.read_csv(runtime['db_path'] + 'denovodb/all/denovodb_enriched_genes.csv') [['p_vid','fdr_cutoff']]
        
                                       
        denovo_enriched_negative_added_snv_df = pd.merge(denovo_enriched_negative_added_snv_df,denovodb_enriched_genes_df[['p_vid','fdr_cutoff']],how = 'left')
    
               
        denovo_enriched_negative_added_snv_df.to_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_negative_added_snv.csv',index = False)  
        print ('Total variants : ' + str(denovo_enriched_negative_added_snv_df.shape[0]))
        
    def add_sift_provean(self,input_df,runtime):
        output_df = None
        org_index = input_df.index
        input_df['org_index'] = input_df.index
        p_vids = list(input_df['p_vid'].unique())
        print ("Total number of p_vids: " + str(len(p_vids)))
        i = 0
        for p_vid in p_vids:
            i = i + 1       
            print (str(i) + '-' + p_vid)      
            cur_input_df = input_df.loc[input_df['p_vid'] == p_vid,:]            
            if os.path.isfile(runtime['db_path'] + 'sift/bygene/' + p_vid + '_sift.csv'):
                cur_sift = pd.read_csv(runtime['db_path'] + 'sift/bygene/' + p_vid + '_sift.csv')
                cur_input_df = pd.merge(cur_input_df,cur_sift,how = 'left')
            else:
                cur_input_df['sift_score'] = np.nan
                
            if os.path.isfile(runtime['db_path'] + 'provean/bygene/' + p_vid + '_provean.csv'):
                cur_provean = pd.read_csv(runtime['db_path'] + 'provean/bygene/' + p_vid + '_provean.csv')
                cur_input_df = pd.merge(cur_input_df,cur_provean,how = 'left')
            else:
                cur_input_df['provean_score'] = np.nan                
                                            
            
            if output_df is None:
                output_df = cur_input_df
            else:
                output_df = pd.concat([output_df,cur_input_df])                
        output_df.index = output_df['org_index'] 
        output_df = output_df.loc[org_index,:]
           
        return(output_df)    
        
    def ldlr_analysis(self,runtime):
        
        def find_variants(x,ukb_variants_dict,score):
            result_variant = ''
            result_variant_score = np.nan
            result_circularity = np.nan
            result_maf = np.nan
            for cur_variant in x:
                cur_variant_score = ukb_variants_dict[cur_variant][score]
                if not np.isnan(cur_variant_score):                    
                    if (cur_variant_score > result_variant_score) | np.isnan(result_variant_score) :
                        result_variant_score = cur_variant_score
                        result_variant = cur_variant
            if result_variant != '':
                result_circularity = ukb_variants_dict[result_variant]['circularity']
                result_maf = ukb_variants_dict[result_variant]['ukb_af']
            
            return([result_variant,result_variant_score,result_circularity,result_maf])
                        
        ####***************************************************************************************************************************************************************    
        # UKB LDLR variants (files are from Roujia)
        ####***************************************************************************************************************************************************************                    
        key_cols = ['chr','nt_pos','nt_ref','nt_alt','p_vid','aa_pos','aa_ref','aa_alt']
        annotation_cols = ['clinvar_id','clinvar_source','hgmd_source','gnomad_source','humsavar_source','mave_source',
                           'clinvar_label','hgmd_label','gnomad_label','humsavar_label','mave_label','label',
                           'train_clinvar_source','train_hgmd_source','train_gnomad_source','train_humsavar_source','train_mave_source']                           
        score_cols = ['Polyphen2_selected_HVAR_score','Polyphen2_selected_HDIV_score','PROVEAN_selected_score','SIFT_selected_score',
                      'CADD_raw','PrimateAI_score','Eigen-raw_coding','GenoCanyon_score','integrated_fitCons_score','REVEL_score',
                      'M-CAP_score','LRT_score','MutationTaster_selected_score','MutationAssessor_selected_score',
                      'FATHMM_selected_score','MetaSVM_score','MetaLR_score','DANN_score','GERP++_RS',
                      'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','MPC_selected_score','mistic_score',
                      'mpc_score','deepsequence_score','mave_input','mave_norm','mave_score','sift_score','provean_score','VARITY_R','VARITY_ER','VARITY_R_LOO','VARITY_ER_LOO']        
        feature_cols = ['provean_score','sift_score','evm_epistatic_score','integrated_fitCons_score','LRT_score','GERP++_RS',
                        'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','blosum100','in_domain','asa_mean','aa_psipred_E',
                        'aa_psipred_H','aa_psipred_C','bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min','solv_ne_max','solv_ne_abs_max',
                        'mw_delta','pka_delta','pkb_delta','pi_delta','hi_delta','pbr_delta','avbr_delta','vadw_delta','asa_delta','cyclic_delta','charge_delta',
                        'positive_delta','negative_delta','hydrophobic_delta','polar_delta','ionizable_delta','aromatic_delta','aliphatic_delta','hbond_delta',
                        'sulfur_delta','essential_delta','size_delta']        
        qip_cols = ['gnomAD_exomes_AF','gnomAD_exomes_AC','gnomAD_exomes_nhomalt','mave_label_confidence','clinvar_review_star','accessibility']        
        other_cols = ['mpc_obs_exp','mpc_mis_badness','mpc_fitted_score','RVIS_EVS','RVIS_percentile_EVS','LoF-FDR_ExAC','RVIS_ExAC','RVIS_percentile_ExAC','VARITY_R_weight','VARITY_ER_weight']
                     
        all_cols = list(key_cols + annotation_cols + list(set(score_cols + feature_cols)) + qip_cols + other_cols)         
                   
#         ldlr_ukb_df = pd.read_csv(runtime['db_path'] + 'ukb/org/merged_raw_v.csv')        
#         ldlr_ukb_df = ldlr_ukb_df[['pos','chr','ref_n','alt','eid']]
#         ldlr_ukb_df.columns = ['nt_pos_hg38','chr','nt_ref','nt_alt','eid']
#         ldlr_ukb_df['nt_pos'] = ldlr_ukb_df['nt_pos_hg38'] + 110676
#         ldlr_ukb_df['chr'] = ldlr_ukb_df['chr'].apply(lambda x: str(x[3:]))
#                                            
#         ldlr_snv_df = pd.read_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] + '_P01130_varity_snv_pisa3_predicted_loo.csv')[all_cols] 
#         ldlr_snv_df['chr'] =  ldlr_snv_df['chr'].astype(str)      
#                                            
#         ldlr_ukb_df = pd.merge(ldlr_ukb_df,ldlr_snv_df,how  = 'left')
#                                                    
#         ldlr_unique_variants_df = pd.read_csv(runtime['db_path'] + 'ukb/org/allv_withweights_new.csv')        
#         ldlr_unique_variants_df = ldlr_unique_variants_df[['pos_ukb','chr','ref_n','alt','type','af','hgvsg_maf']]
#         ldlr_unique_variants_df.columns = ['nt_pos_hg38','chr','nt_ref','nt_alt','variant_type','gnomad_af','ukb_af']
#         ldlr_unique_variants_df['nt_pos'] = ldlr_unique_variants_df['nt_pos_hg38'] + 110676
#         ldlr_unique_variants_df['chr'] = ldlr_unique_variants_df['chr'].apply(lambda x: str(x[3:]))    
#         ldlr_unique_variants_df['chr'] =  ldlr_unique_variants_df['chr'].astype(str)            
#         ldlr_ukb_df = pd.merge(ldlr_ukb_df,ldlr_unique_variants_df,how = 'left')
#                                
#         phenotype_df = pd.read_csv(runtime['db_path'] + 'ukb/org/phenotype_list.csv')
#         phenotype_df['ID'] = phenotype_df['ID'].apply(lambda x: str(x) + '-0.0')
#         filtered_phenotype_df = pd.read_csv(runtime['db_path'] + 'ukb/org/filtered_phenotype.csv')        
#         selected_phenotype = list(filtered_phenotype_df.columns[filtered_phenotype_df.columns.isin(list(phenotype_df['ID']))])        
#         phenotype_abbr = phenotype_df['phenotype_abbr']
#         phenotype_abbr.index = phenotype_df['ID']
#         phenotype_abbr_dict = phenotype_abbr.to_dict()         
#         selected_phenotype_df = filtered_phenotype_df.loc[:,selected_phenotype + ['eid']]        
#         selected_phenotype_df = selected_phenotype_df.rename(columns = phenotype_abbr_dict)
#         selected_phenotype_df.loc[selected_phenotype_df['medication'].isnull(),'medication'] = selected_phenotype_df.loc[selected_phenotype_df['medication'].isnull(),'medication1']
#         selected_phenotype_df['ldl_medication'] = selected_phenotype_df['medication']
#         selected_phenotype_df.loc[selected_phenotype_df['ldl_medication'] != 1,'ldl_medication'] = 0#  
#         selected_phenotype_df.loc[selected_phenotype_df['medication'].isnull(),'ldl_medication'] = selected_phenotype_df.loc[selected_phenotype_df['medication'].isnull(),'medication1']
#                                 
#                                
#         ldlr_ukb_df = pd.merge(ldlr_ukb_df,selected_phenotype_df,how = 'left')
#         ldlr_ukb_df['unique_snv_id'] = ldlr_ukb_df.apply(lambda x: 'chr' + str(x['chr']) + ':' + str(x['nt_pos']) + x['nt_ref'] +  x['nt_alt'],axis = 1)
#         ldlr_ukb_df.loc[ldlr_ukb_df['gnomAD_exomes_AF'].isnull() & (ldlr_ukb_df['variant_type'] == 'missense'), 'gnomAD_exomes_AF'] = 1e-06    
#                                      
#         ldlr_ukb_df['neg_log_af'] = 0- np.log10(ldlr_ukb_df['gnomAD_exomes_AF'])
#         ldlr_ukb_df['neg_log_ukb_af'] = 0- np.log10(ldlr_ukb_df['ukb_af'])
#         ldlr_ukb_df['age'] = 2020 - ldlr_ukb_df['birth_year']         
#         ldlr_ukb_df.to_csv(runtime['db_path'] + 'ukb/csv/ldlr_ukb_variants_pisa3.csv',index = False)
#                       
#         #### create ldlr ukb unique variants dataframe 
#         ldlr_unique_variants_df = ldlr_ukb_df[all_cols + ['variant_type','unique_snv_id','ukb_af','neg_log_af','neg_log_ukb_af']].drop_duplicates()             
#         ldlr_unique_variants_eid_count_df = ldlr_ukb_df.groupby(['chr','nt_pos','nt_ref','nt_alt'])['eid'].agg('count').reset_index().rename(columns = {'eid':'eid_count'}) 
#         ldlr_unique_variants_df = pd.merge(ldlr_unique_variants_df,ldlr_unique_variants_eid_count_df,how = 'left')      
#                                
#         ldlr_unique_variants_df.to_csv(runtime['db_path'] + 'ukb/csv/ldlr_ukb_unique_variants_pisa3.csv',index = False) 
#                   
#                   
# #         ####***************************************************************************************************************************************************************    
# #         # Create unique patient data and variant data 
# #         ####***************************************************************************************************************************************************************
#         phenotype_list = ['LDL']
#         covariate_list = ['sex','age','ldl_medication']        
#         selected_score_cols = ['Polyphen2_selected_HVAR_score','Polyphen2_selected_HDIV_score',
#               'CADD_raw','PrimateAI_score','Eigen-raw_coding','GenoCanyon_score','integrated_fitCons_score','REVEL_score',
#               'M-CAP_score','LRT_score','MutationTaster_selected_score','MutationAssessor_selected_score',
#               'FATHMM_selected_score','MetaSVM_score','MetaLR_score','DANN_score','GERP++_RS',
#               'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','mistic_score',
#               'mpc_score','mpc_obs_exp','mpc_mis_badness','mpc_fitted_score','blosum100','sift_score','provean_score','VARITY_R','VARITY_ER','VARITY_R_LOO','VARITY_ER_LOO']
#                                 
#                        
#         #load variants    
#         ldlr_ukb_df = pd.read_csv(runtime['db_path'] + 'ukb/csv/ldlr_ukb_variants_pisa3.csv')
#                                     
#         #flip variant scores
#         pcc_score_df = pd.DataFrame(selected_score_cols,columns = ['score'])
#         pcc_score_df['ldl_pcc'] = [alm_fun.pcc_cal(ldlr_ukb_df.loc[(ldlr_ukb_df['variant_type'] == 'missense') & (ldlr_ukb_df['gnomAD_exomes_AF'] == 1e-06),score],ldlr_ukb_df.loc[(ldlr_ukb_df['variant_type'] == 'missense') & (ldlr_ukb_df['gnomAD_exomes_AF'] == 1e-06),'LDL'])  for score in selected_score_cols]
#         flip_scores = list(pcc_score_df.loc[pcc_score_df['ldl_pcc'] < 0, 'score'])
#         print ('Flip scores : ' + str(flip_scores))
#         for score in flip_scores:
#             ldlr_ukb_df.loc[:,score] = 0 -ldlr_ukb_df.loc[:,score] 
#         pass 
#                                         
#         #### variant dictionary
#         ukb_variant_dict = {}
#         ldlr_unique_variants_df = pd.read_csv(runtime['db_path'] + 'ukb/csv/ldlr_ukb_unique_variants_pisa3.csv')
#         ldlr_unique_variants_flipped_df = ldlr_unique_variants_df.copy()        
#         ldlr_unique_variants_flipped_df['circularity'] = ldlr_unique_variants_flipped_df.apply(lambda x: np.nansum(x[['clinvar_source','hgmd_source','mave_source','humsavar_source']]),axis = 1)        
#         for score in flip_scores:
#             ldlr_unique_variants_flipped_df.loc[:,score] = 0 -ldlr_unique_variants_flipped_df.loc[:,score]                  
#         pass             
#                           
#         for snv_id in list(ldlr_unique_variants_flipped_df['unique_snv_id']):
#             ukb_variant_dict[snv_id] = {}
#             ukb_variant_dict[snv_id]['circularity'] = ldlr_unique_variants_flipped_df.loc[ldlr_unique_variants_flipped_df['unique_snv_id'] == snv_id,'circularity'].values[0]
#             ukb_variant_dict[snv_id]['ukb_af'] = ldlr_unique_variants_flipped_df.loc[ldlr_unique_variants_flipped_df['unique_snv_id'] == snv_id,'ukb_af'].values[0]
#             for score in selected_score_cols:
#                 ukb_variant_dict[snv_id][score] = ldlr_unique_variants_flipped_df.loc[ldlr_unique_variants_flipped_df['unique_snv_id'] == snv_id,score].values[0]
#         pass
#                                    
#         ldlr_unique_variants_flipped_df.to_csv(runtime['db_path'] + 'ukb/csv/ldlr_ukb_unique_variants_flipped_pisa3.csv',index = False)    
#                            
#                                       
#         ####eid dataframe
#         ldlr_unqiue_eid_df = ldlr_ukb_df[['eid'] + phenotype_list + covariate_list].drop_duplicates()
#                                    
#         #list of variants for each individual, take variant that has maximum score 
#         ldlr_eid_variants_df = ldlr_ukb_df.groupby(['eid'])['unique_snv_id'].agg(list).reset_index()        
#         for score in selected_score_cols:       
#             ldlr_eid_variants_df[score+'_lst'] = ldlr_eid_variants_df.apply(lambda x: find_variants(x['unique_snv_id'],ukb_variant_dict,score),axis = 1)
#             ldlr_eid_variants_df[score+'_variant'] = ldlr_eid_variants_df[score+'_lst'].apply(lambda x: x[0])
#             ldlr_eid_variants_df[score] = ldlr_eid_variants_df[score+'_lst'].apply(lambda x: x[1])
#             ldlr_eid_variants_df[score+'_circularity'] = ldlr_eid_variants_df[score+'_lst'].apply(lambda x: x[2])
#             ldlr_eid_variants_df[score+'_ukb_af'] = ldlr_eid_variants_df[score+'_lst'].apply(lambda x: x[3])
#         ldlr_eid_variants_df['circularity'] = ldlr_eid_variants_df.apply(lambda x: np.nansum([x[score + '_circularity'] for score in selected_score_cols]), axis = 1)
#                                    
#         #variant_type 
#         ldlr_eid_variant_type = ldlr_ukb_df.groupby(['eid'])['variant_type'].agg(set).reset_index()
#         ldlr_eid_variant_type['variant_type'] =  ldlr_eid_variant_type['variant_type'].astype(str)   
#         print(ldlr_eid_variant_type['variant_type'].value_counts())
#                                      
#         #lowest_maf
#         ldlr_eid_lowest_maf = ldlr_ukb_df.groupby(['eid'])['ukb_af','gnomAD_exomes_AF'].agg(np.nanmin).reset_index()
#         ldlr_eid_lowest_maf = ldlr_eid_lowest_maf.rename(columns = {'ukb_af':'lowest_ukb_af','gnomAD_exomes_AF':'lowest_gnomad_af'})
#                                                      
#         #####remove individual with invalid phenotype
#         print ('number of individuals: ' + str(ldlr_unqiue_eid_df.shape[0]))
#         for phenotype in phenotype_list + covariate_list:
#             ldlr_unqiue_eid_df = ldlr_unqiue_eid_df.loc[ldlr_unqiue_eid_df[phenotype].notnull(),:]
#             print ('number of individuals with valid ' + phenotype + ': ' + str(ldlr_unqiue_eid_df.shape[0]))
#                                     
#         #### covariate correction
#         for phenotype in phenotype_list:
#             ldlr_unqiue_eid_df[phenotype + '_z'] = ldlr_unqiue_eid_df[phenotype]          
#             for covariate in ['sex','ldl_medication']:
#                 ldlr_unqiue_eid_df.loc[(ldlr_unqiue_eid_df[covariate] == 1),phenotype + '_z'] = stats.zscore(ldlr_unqiue_eid_df.loc[(ldlr_unqiue_eid_df[covariate] == 1),phenotype + '_z'])
#                 ldlr_unqiue_eid_df.loc[(ldlr_unqiue_eid_df[covariate] == 0),phenotype + '_z'] = stats.zscore(ldlr_unqiue_eid_df.loc[(ldlr_unqiue_eid_df[covariate] == 0),phenotype + '_z'])
#                                                             
#             x = np.array(ldlr_unqiue_eid_df['age'])
#             y = np.array(ldlr_unqiue_eid_df[phenotype + '_z'])
#             reg = sklearn.linear_model.LinearRegression().fit(x.reshape(-1,1),y)            
#             ldlr_unqiue_eid_df[phenotype + '_z'] = y - reg.coef_*x        
#                         
#         #merge eid dataframe and save
#         ldlr_unqiue_eid_df = pd.merge(ldlr_unqiue_eid_df,ldlr_eid_variant_type,how = 'left')
#         ldlr_unqiue_eid_df = pd.merge(ldlr_unqiue_eid_df,ldlr_eid_lowest_maf,how = 'left')                
#         ldlr_unqiue_eid_df = pd.merge(ldlr_unqiue_eid_df,ldlr_eid_variants_df,how = 'left')          
#         ldlr_unqiue_eid_df.to_csv(runtime['db_path'] + 'ukb/csv/ldlr_unique_eid_pisa3.csv',index = False)   
#            
#         
        #####*************************************************************************************************************************************************
        #Variant based PCC/SPC Analysis
        #####*************************************************************************************************************************************************
        phenotype_list = ['LDL']
        covariate_list = ['sex','age','ldl_medication']    
        selected_score_cols = ['Polyphen2_selected_HVAR_score','Polyphen2_selected_HDIV_score',
              'CADD_raw','PrimateAI_score','Eigen-raw_coding','GenoCanyon_score','integrated_fitCons_score','REVEL_score',
              'M-CAP_score','LRT_score','MutationTaster_selected_score','MutationAssessor_selected_score',
              'FATHMM_selected_score','MetaSVM_score','MetaLR_score','DANN_score','GERP++_RS',
              'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','mistic_score',
              'mpc_score','sift_score','provean_score','VARITY_ER','VARITY_R','VARITY_ER_LOO','VARITY_R_LOO']
        
#         selected_score_cols = ['Polyphen2_selected_HVAR_score','Polyphen2_selected_HDIV_score',
#               'CADD_raw','PrimateAI_score','Eigen-raw_coding','GenoCanyon_score','integrated_fitCons_score','REVEL_score',
#               'M-CAP_score','LRT_score','MutationTaster_selected_score','MutationAssessor_selected_score',
#               'FATHMM_selected_score','MetaSVM_score','MetaLR_score','DANN_score','GERP++_RS',
#               'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','mistic_score',
#               'mpc_score','sift_score','provean_score','VARITY_ER','VARITY_R']     
#         
        
        rank_score_cols = ['Polyphen2_selected_HVAR_score','Polyphen2_selected_HDIV_score',
              'CADD_raw','PrimateAI_score','Eigen-raw_coding','GenoCanyon_score','integrated_fitCons_score','REVEL_score',
              'M-CAP_score','LRT_score','MutationTaster_selected_score','MutationAssessor_selected_score',
              'FATHMM_selected_score','MetaSVM_score','MetaLR_score','DANN_score','GERP++_RS',
              'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','mistic_score',
              'mpc_score','sift_score','provean_score']
        
#         rank_score_cols = ['sift_score','provean_score','LRT_score','GERP++_RS',
#               'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds']
#                              


#         out_cols = ['chr','nt_pos','nt_ref','nt_alt','p_vid','aa_pos','aa_ref','aa_alt','VARITY_R_num_better_rank','VARITY_R_better_rank','VARITY_R','mpc_score','mistic_score','sift_score','provean_score','LDL_z','LDL_z_rank','bsa_max','asa_mean','in_domain','eid_count','circularity','clinvar_id','clinvar_label','total_shap','provean_score_shap','sift_score_shap','evm_epistatic_score_shap','integrated_fitCons_score_shap','LRT_score_shap','GERP++_RS_shap','phyloP30way_mammalian_shap','phastCons30way_mammalian_shap','SiPhy_29way_logOdds_shap','blosum100_shap','in_domain_shap','asa_mean_shap','aa_psipred_E_shap','aa_psipred_H_shap','aa_psipred_C_shap','bsa_max_shap','h_bond_max_shap','salt_bridge_max_shap','disulfide_bond_max_shap','covelent_bond_max_shap','solv_ne_min_shap','mw_delta_shap','pka_delta_shap','pkb_delta_shap','pi_delta_shap','hi_delta_shap','pbr_delta_shap','avbr_delta_shap','vadw_delta_shap','asa_delta_shap','cyclic_delta_shap','charge_delta_shap','positive_delta_shap','negative_delta_shap','hydrophobic_delta_shap','polar_delta_shap','ionizable_delta_shap','aromatic_delta_shap','aliphatic_delta_shap','hbond_delta_shap','sulfur_delta_shap','essential_delta_shap','size_delta_shap','base_shap']
        
        out_cols = ['chr','nt_pos','nt_ref','nt_alt','p_vid','aa_pos','aa_ref','aa_alt','VARITY_R_num_better_rank','VARITY_R_better_rank','VARITY_R','VARITY_ER_num_better_rank','VARITY_ER_better_rank','VARITY_ER','mpc_score','mistic_score','sift_score','provean_score','LDL','LDL_z','LDL_z_rank','bsa_max','asa_mean','solv_ne_min','solv_ne_max','solv_ne_abs_max','in_domain','eid_count','circularity','clinvar_id','clinvar_label']


        covariate_list = ['sex','age','ldl_medication']     
        compare_score = 'VARITY_R'   
        bootstrap_num = 2000   
        remove_circularity = 0
        maf_cutoff = 0.005
        phenotype = 'LDL_z'

        ldlr_ukb_df = pd.read_csv(runtime['db_path'] + 'ukb/csv/ldlr_ukb_variants_pisa3.csv')        
        ldlr_unqiue_eid_df  = pd.read_csv(runtime['db_path'] + 'ukb/csv/ldlr_unique_eid_pisa3.csv')
        ldlr_ukb_df = pd.merge(ldlr_ukb_df,ldlr_unqiue_eid_df[['eid',phenotype]], how = 'left')           
        ldlr_ukb_missense_df = ldlr_ukb_df.loc[(ldlr_ukb_df['variant_type'] == 'missense') & ldlr_ukb_df[phenotype].notnull(),:]                
#         ldlr_variant_phenotype_df = ldlr_ukb_missense_df.groupby(['unique_snv_id'])[phenotype,'LDL'].agg('mean').reset_index()        
        ldlr_variant_phenotype_df = ldlr_ukb_missense_df.groupby(['unique_snv_id'])[phenotype,'LDL'].agg('mean').reset_index()
 
        ldlr_variants_flipped_df = pd.read_csv(runtime['db_path'] + 'ukb/csv/ldlr_ukb_unique_variants_flipped_pisa3.csv')                    
        print ('number of variants: ' + str(ldlr_variants_flipped_df.shape[0]))
            
#         remove variants that are not missense           
        ldlr_missense_variants_df = ldlr_variants_flipped_df.loc[ldlr_variants_flipped_df['variant_type'] == 'missense',:]
        print ('number of missense variants : ' + str(ldlr_missense_variants_df.shape[0]))
         
 
#         remove circularity
        if remove_circularity == 1:             
            ldlr_missense_variants_df = ldlr_missense_variants_df.loc[ldlr_missense_variants_df['VARITY_R_weight'].isnull() & ldlr_missense_variants_df['VARITY_ER_weight'].isnull(),:]           
#             ldlr_missense_variants_df = ldlr_variants_flipped_df.loc[(ldlr_variants_flipped_df['variant_type'] == 'missense') & (ldlr_variants_flipped_df['circularity'] == 0),:]
            print ('number of variants after remove circularity: ' + str(ldlr_missense_variants_df.shape[0]))
     
#         remove invalid scores
        for score in selected_score_cols:
            ldlr_missense_variants_df = ldlr_missense_variants_df.loc[ldlr_missense_variants_df[score].notnull(),:]
        print ('number of variants after remove vairnats with invalid scores: ' + str(ldlr_missense_variants_df.shape[0]))
     
#         remove high MAF variants 
        ldlr_missense_variants_df = ldlr_missense_variants_df.loc[ldlr_missense_variants_df['gnomAD_exomes_AF'] <= maf_cutoff,:]
        print ('number of variants after remove high MAF variants: ' + str(ldlr_missense_variants_df.shape[0]))
     
     
        ldlr_missense_variants_df = pd.merge(ldlr_missense_variants_df, ldlr_variant_phenotype_df ,how ='left')
        ldlr_missense_variants_df.to_csv(runtime['db_path'] + 'ukb/csv/ldlr_missense_variants_' + str(remove_circularity) + '_maf_' + str(maf_cutoff) + '_pisa3.csv',index = False)        
        ldlr_missense_variants_df[key_cols + feature_cols].to_csv(runtime['db_path'] + 'ukb/csv/ldlr_missense_variants_pisa3.csv',index = False)
         
#         remove invalid phenotype        
        ldlr_missense_variants_df = ldlr_missense_variants_df.loc[ldlr_missense_variants_df[phenotype].notnull(),:]            
        print ('number of variants after remove invalid pthneotype: ' + str(ldlr_missense_variants_df.shape[0]))        
            
         
         
        #add ranks to each score and phenotype
         
        ldlr_missense_variants_df = ldlr_missense_variants_df.sort_values([phenotype]).reset_index(drop = True)
        ldlr_missense_variants_df[phenotype + '_rank'] = ldlr_missense_variants_df.index
         
        for score in selected_score_cols:
            ldlr_missense_variants_df = ldlr_missense_variants_df.sort_values([score]).reset_index(drop = True)
            ldlr_missense_variants_df[score + '_rank'] = ldlr_missense_variants_df.index
            ldlr_missense_variants_df[score + '_rank_diff'] = np.abs(ldlr_missense_variants_df[score + '_rank'] - ldlr_missense_variants_df[phenotype + '_rank'])
                         
        for i in ldlr_missense_variants_df.index:
            ldlr_missense_variants_df.loc[i,'VARITY_R_num_better_rank'] = 0
            ldlr_missense_variants_df.loc[i,'VARITY_R_better_rank'] = 0
            ldlr_missense_variants_df.loc[i,'VARITY_ER_num_better_rank'] = 0
            ldlr_missense_variants_df.loc[i,'VARITY_ER_better_rank'] = 0            
            for score in rank_score_cols:
                diff_r  = ldlr_missense_variants_df.loc[i,score + '_rank_diff'] - ldlr_missense_variants_df.loc[i,'VARITY_R_rank_diff']
                ldlr_missense_variants_df.loc[i,'VARITY_R_better_rank'] = ldlr_missense_variants_df.loc[i,'VARITY_R_better_rank'] + diff_r
                if diff_r > 0 :  
                    ldlr_missense_variants_df.loc[i,'VARITY_R_num_better_rank'] =  ldlr_missense_variants_df.loc[i,'VARITY_R_num_better_rank'] + 1
                    
                diff_er  = ldlr_missense_variants_df.loc[i,score + '_rank_diff'] - ldlr_missense_variants_df.loc[i,'VARITY_ER_rank_diff']
                ldlr_missense_variants_df.loc[i,'VARITY_ER_better_rank'] = ldlr_missense_variants_df.loc[i,'VARITY_ER_better_rank'] + diff_er
                if diff_er > 0 :  
                    ldlr_missense_variants_df.loc[i,'VARITY_ER_num_better_rank'] =  ldlr_missense_variants_df.loc[i,'VARITY_ER_num_better_rank'] + 1
             
        pass
          
#         merge with shap values  
        shap_er_df = pd.read_csv(runtime['project_path'] + 'output/csv/Revision1230_1_target_prediction_VARITY_ER_tf0_Revision1230_1_P01130_varity_snv_pisa3_target_shap.csv')
        shap_er_df = shap_er_df.drop(columns = ['VARITY_ER'])
        er_columns = {x: x + '_er' for x in shap_er_df.columns if 'shap' in x}
        shap_er_df = shap_er_df.rename(columns = er_columns)
        
        shap_r_df = pd.read_csv(runtime['project_path'] + 'output/csv/Revision1230_1_target_prediction_VARITY_R_tf0_Revision1230_1_P01130_varity_snv_pisa3_target_shap.csv')
        shap_r_df = shap_r_df.drop(columns = ['VARITY_R'])
        r_columns = {x: x + '_r' for x in shap_r_df.columns if 'shap' in x}
        shap_r_df = shap_r_df.rename(columns = r_columns)
        
        
        ldlr_missense_variants_output_df = ldlr_missense_variants_df[out_cols]
        ldlr_missense_variants_output_df = pd.merge(ldlr_missense_variants_output_df,shap_er_df,how = 'left')
        ldlr_missense_variants_output_df = pd.merge(ldlr_missense_variants_output_df,shap_r_df,how = 'left')
        ldlr_missense_variants_output_df.to_csv(runtime['db_path'] + 'ukb/csv/ldlr_missense_variants_' + str(remove_circularity) + '_maf_' + str(maf_cutoff) + '_withrank_pisa3.csv',index = False)
        
               
        ldlr_score_df = pd.DataFrame(selected_score_cols,columns = ['score'])
                        
        ldlr_score_df[phenotype+ '_pcc'] = [alm_fun.pcc_cal(ldlr_missense_variants_df[score],ldlr_missense_variants_df[phenotype])  for score in selected_score_cols]      
        ldlr_score_df[phenotype+ '_spc'] = [alm_fun.spc_cal(ldlr_missense_variants_df[score],ldlr_missense_variants_df[phenotype])  for score in selected_score_cols]
         
        if bootstrap_num == 1:
            boostrap_indices = list(ldlr_missense_variants_df.index)
            ldlr_score_df[phenotype+ '_pcc' + '_' + str(0)] = [alm_fun.pcc_cal(ldlr_missense_variants_df.loc[boostrap_indices,score],ldlr_missense_variants_df.loc[boostrap_indices,phenotype])  for score in selected_score_cols]
            ldlr_score_df[phenotype+ '_spc' + '_' + str(0)] = [alm_fun.spc_cal(ldlr_missense_variants_df.loc[boostrap_indices,score],ldlr_missense_variants_df.loc[boostrap_indices,phenotype])  for score in selected_score_cols]
                
        else:
            for i in range(bootstrap_num):
                boostrap_indices= sklearn.utils.resample(list(ldlr_missense_variants_df.index))
                ldlr_score_df[phenotype+ '_pcc' + '_' + str(i)] = [alm_fun.pcc_cal(ldlr_missense_variants_df.loc[boostrap_indices,score],ldlr_missense_variants_df.loc[boostrap_indices,phenotype])  for score in selected_score_cols]
                ldlr_score_df[phenotype+ '_spc' + '_' + str(i)] = [alm_fun.spc_cal(ldlr_missense_variants_df.loc[boostrap_indices,score],ldlr_missense_variants_df.loc[boostrap_indices,phenotype])  for score in selected_score_cols]
        
#         ldlr_score_df.to_csv(runtime['db_path'] + 'ukb/csv/ldlr_scores_bootstrapped_' +str(remove_circularity) + '_maf_' + str(maf_cutoff) + '_pisa3.csv',index = False)
           
        ldlr_score_df = pd.read_csv(runtime['db_path'] + 'ukb/csv/ldlr_scores_bootstrapped_' +str(remove_circularity) + '_maf_' + str(maf_cutoff) + '_pisa3.csv')
        
#         for metric in  ['spc','pcc']:
        for metric in  ['spc']:
            perfix = phenotype + '_' + metric            
            boostrapped_cols = []
            for i in range(bootstrap_num):
                boostrapped_cols.append(phenotype+ '_' + metric + '_' + str(i))
            pass                
            compare_scores = ldlr_score_df.loc[ldlr_score_df['score'] == compare_score,boostrapped_cols].values
            ldlr_score_df[perfix + '_mean'] = ldlr_score_df[boostrapped_cols].apply(lambda x: np.mean(x),axis = 1)
            ldlr_score_df[perfix + '_se'] = ldlr_score_df[boostrapped_cols].apply(lambda x: np.std(x,ddof=1),axis = 1)
            ldlr_score_df[perfix + '_effect_size'] = ldlr_score_df[boostrapped_cols].apply(lambda x: np.mean(np.subtract(compare_scores,x.values)),axis = 1)
            ldlr_score_df[perfix + '_effect_size_se'] = ldlr_score_df[boostrapped_cols].apply(lambda x: np.std(np.subtract(compare_scores,x.values),ddof=1),axis = 1)
            ldlr_score_df[perfix + '_zscore'] = ldlr_score_df[perfix + '_effect_size']/ldlr_score_df[perfix + '_effect_size_se']
            ldlr_score_df[perfix + '_pvalue'] =  stats.norm.sf(abs(ldlr_score_df[perfix + '_zscore']))
            ldlr_score_df[perfix + '_ci'] = ldlr_score_df.apply(lambda x: "{:.3f}".format(x[phenotype + '_' + metric + '_effect_size'] - 1.65*x[phenotype + '_' + metric + '_effect_size_se']) + ' ~ inf', axis =1 )         
            selected_cols = ['score',perfix + '_mean',perfix + '_se',perfix + '_effect_size',perfix + '_ci',perfix + '_pvalue']
            ldlr_score_df.sort_values([perfix],ascending = False)[selected_cols].to_csv(runtime['db_path'] + 'ukb/csv/ldlr_variants_remove_circularity_' + str(remove_circularity) + '_maf_' + str(maf_cutoff) +  '_' + metric + '_pisa3.csv',index = False)
        pass
        print ('OK')    
        
           
          
#         #####*************************************************************************************************************************************************
#         #phenotype shift Analysis
#         #####*************************************************************************************************************************************************
      
#                                     
#         ldlr_unqiue_eid_df = pd.read_csv(runtime['db_path'] + 'ukb/csv/ldlr_unique_eid.csv')  
#         #eid (only carry missense or syn)
#         ldlr_unqiue_eid_syn_df = ldlr_unqiue_eid_df.loc[(ldlr_unqiue_eid_df['variant_type'] == "{'syn'}"),:]
#         ldlr_unqiue_eid_missense_df = ldlr_unqiue_eid_df.loc[(ldlr_unqiue_eid_df['variant_type'] == "{'missense', 'syn'}") | (ldlr_unqiue_eid_df['variant_type'] == "{'syn', 'missense'}"),:]
#                                       
#         #make pathogenicity predictions on each variant, assign pathogenicity to  x% of most pathogenic variants       
#         percentile = 0.3
#         remove_circularity = 0
#         phenotype = 'LDL_z'
#         prediction_type = 'percentile'
#         
#         print ('percentile: ' + str(percentile))  
#         print ('remove circularity: ' + str(remove_circularity))
#         print ('phenotype: ' + str(phenotype))
#                 
#         ldlr_variants_flipped_df = pd.read_csv(runtime['db_path'] + 'ukb/csv/ldlr_ukb_unique_variants_flipped.csv')        
#         if remove_circularity == 1: 
#             ldlr_missense_variants_df = ldlr_variants_flipped_df.loc[(ldlr_variants_flipped_df['variant_type'] == 'missense') & (ldlr_variants_flipped_df['circularity'] == 0),:]
#         else:
#             ldlr_missense_variants_df = ldlr_variants_flipped_df.loc[ldlr_variants_flipped_df['variant_type'] == 'missense',:]
#         num_missense_variants = ldlr_missense_variants_df.shape[0]                                                                         
#         num_pathogenic_variants =  np.int(num_missense_variants*percentile)
#         print ('total number of missense variants: ' + str(num_missense_variants) + '[' + str(num_pathogenic_variants) +']')
#         
#         cutoff_score_df = pd.DataFrame(selected_score_cols,columns = ['score'])
#         for score in selected_score_cols:            
#            ldlr_missense_variants_df = ldlr_missense_variants_df.sort_values([score],ascending = False)
#            ldlr_missense_variants_df[score+ '_percentile_label'] = 0               
#            ldlr_missense_variants_df.loc[ldlr_missense_variants_df.index[0:num_pathogenic_variants],score+ '_percentile_label'] = 1               
#            cutoff_score_df[score + '_percentile_cutoff'] =  ldlr_missense_variants_df.loc[ldlr_missense_variants_df.index[num_pathogenic_variants-1],score]
# 
#            cur_missense_variants_df = ldlr_missense_variants_df[['unique_snv_id',score+ '_percentile_label']]
#            cur_missense_variants_df.columns = [score + '_variant',score+ '_percentile_label'] 
#            ldlr_unqiue_eid_missense_df = pd.merge(ldlr_unqiue_eid_missense_df,cur_missense_variants_df,how = 'left')
#            
#         ldlr_unqiue_eid_missense_df.to_csv(runtime['db_path'] + 'ukb/csv/ldlr_unique_eid_missense.csv',index = False)
#         ldlr_missense_variants_df.to_csv(runtime['db_path'] + 'ukb/csv/ldlr_ukb_unique_variants_missense.csv',index = False)                
#         #Plot the figure
#         fig = plt.figure(figsize=(20, 10))
#         plt.clf()
#         plt.rcParams["font.family"] = "Helvetica"    
#         ax = plt.subplot()                  
#         legend_data = pd.DataFrame(columns = ['Score','Mean','SE','Median','Size','Cutoff'] )          
#         ax.hist(ldlr_unqiue_eid_missense_df[phenotype],bins = 'auto',density = True, histtype = 'step',cumulative = False, color = 'black')
#         #         plt.hist(ldlr_unqiue_eid_df[phenotype],bins = 'auto',density = False, histtype = 'step',cumulative = False, color = 'black')
#         all_size = ldlr_unqiue_eid_missense_df[phenotype].shape[0]
#         all_mean = np.round(ldlr_unqiue_eid_missense_df[phenotype].mean(),4)
#         all_ste = np.round(ldlr_unqiue_eid_missense_df[phenotype].mean()/np.sqrt(all_size),4)
#         all_median = np.round(ldlr_unqiue_eid_missense_df[phenotype].median(),4)
#         legend_data_index = 0
#          
#         legend_data.loc[legend_data_index,:] = ['NULL Distribution',all_mean,all_ste,all_median,all_size,np.nan]
#          
#         for score in selected_score_cols:     
#             print (score)              
#             positive_index = ldlr_unqiue_eid_missense_df[score + '_' + prediction_type + '_label'] == 1
#             ax.hist(ldlr_unqiue_eid_missense_df.loc[positive_index,phenotype],bins = 'auto',density = True, histtype = 'step',cumulative = False,linewidth = 2)
#             cur_size =  ldlr_unqiue_eid_missense_df.loc[positive_index,phenotype].shape[0]                       
#             cur_mean = np.round(ldlr_unqiue_eid_missense_df.loc[positive_index,phenotype].mean(),4)
#             cur_ste = np.round(ldlr_unqiue_eid_missense_df.loc[positive_index,phenotype].std()/np.sqrt(cur_size),4)
#             cur_median = np.round(ldlr_unqiue_eid_missense_df.loc[positive_index,phenotype].median(),4)#
#             cur_cutoff = cutoff_score_df.loc[cutoff_score_df['score'] == score,score + '_' + prediction_type + '_cutoff'].values[0]                                                                                                         
#             legend_data_index = legend_data_index + 1
#             legend_data.loc[legend_data_index,:] = [score,cur_mean,cur_ste,cur_median,cur_size,cur_cutoff]
# 
#         pass
#         legend_data = legend_data.sort_values(['Mean'],ascending= False)
#         legend_table = ax.table(cellText = legend_data.values.tolist(),colLabels = legend_data.columns,loc ='upper right')
#         for i in range(len(legend_data.columns)):    
#             legend_table.auto_set_column_width(i)
#         pass
#         fig.tight_layout()
#         fig.savefig(runtime['db_path'] + 'ukb/img/' + phenotype + '_percentile_' + str(percentile) + '_circularity_removed_' + str(remove_circularity) + '.png')  
#  
#         print ('OK')
#         

           
           



# 
#             ldlr_unqiue_eid_missense_df[score + '_label'] = 0
#             #### use cutoff for each score to make binary predictions                 
#             if prediction_type == 'cutoff':                                
#                 cur_cutoff = cutoff_score_df.loc[cutoff_score_df['score'] == score,cutoff_type].values[0]
#                 ldlr_unqiue_eid_missense_df.loc[ldlr_unqiue_eid_missense_df[score] >= cur_cutoff, score + '_label'] = 1
#              
#             #### use the x% percentile to make binary predictions
#             if prediction_type == 'percentile':   
#                 ldlr_unqiue_eid_missense_df = ldlr_unqiue_eid_missense_df.sort_values([score,phenotype])
#                 indices = ldlr_unqiue_eid_missense_df.index                        
#                 ldlr_unqiue_eid_missense_df.loc[indices[(index_boundaries[num_splits - 1]-1):index_boundaries[num_splits]],score + '_label'] = 1
#                 cur_cutoff = ldlr_unqiue_eid_missense_df.loc[indices[(index_boundaries[num_splits - 1]-1)],score]        
        
#         total_num = ldlr_unqiue_eid_missense_df.shape[0]
#         phenotype = 'LDL_z'
#         prediction_type ='percentile'
#         num_splits = 5       
# 
#         index_boundaries = np.linspace(0,total_num-1,num_splits + 1,dtype = int)            
#         ####plot x% most deletious predictions 
#         ldlr_unqiue_eid_missense_df = ldlr_unqiue_eid_missense_df.loc[np.random.permutation(ldlr_unqiue_eid_missense_df.index),:]        
#         ldlr_unqiue_eid_missense_df = ldlr_unqiue_eid_missense_df.reset_index(drop = True)
#         ldlr_unqiue_eid_missense_df['index'] = ldlr_unqiue_eid_missense_df.index        
#         
         
#             for i in range(num_splits):
    #             if i == 0:
    #                 plt.hist(ldlr_unqiue_eid_missense_df.loc[indices[index_boundaries[i]:index_boundaries[i+1]],phenotype],bins = 'auto',density = False, histtype = 'step',cumulative = False)
    #             else:
#                     if i == num_splits - 1: 
#                         ax.hist(ldlr_unqiue_eid_missense_df.loc[indices[(index_boundaries[i]-1):index_boundaries[i+1]],phenotype],bins = 'auto',density = False, histtype = 'step',cumulative = False,linewidth = 2)
#                         cur_size =  ldlr_unqiue_eid_missense_df.loc[indices[(index_boundaries[i]-1):index_boundaries[i+1]],phenotype].shape[0]                       
#                         cur_mean = np.round(ldlr_unqiue_eid_missense_df.loc[indices[(index_boundaries[i]-1):index_boundaries[i+1]],phenotype].mean(),4)
#                         cur_ste = np.round(ldlr_unqiue_eid_missense_df.loc[indices[(index_boundaries[i]-1):index_boundaries[i+1]],phenotype].std()/np.sqrt(cur_size),4)
#                         cur_median = np.round(ldlr_unqiue_eid_missense_df.loc[indices[(index_boundaries[i]-1):index_boundaries[i+1]],phenotype].median(),4)                         
#                         legend_data_index = legend_data_index + 1
#                         legend_data.loc[legend_data_index,:] = [score,cur_mean,cur_ste,cur_median,cur_size,cur_cutoff]
#                     else:                    
#                         plt.hist(ldlr_unqiue_eid_missense_df.loc[indices[(index_boundaries[i]-1):index_boundaries[i+1]],phenotype],bins = 'auto',density = False, histtype = 'step',cumulative = False)                
#             pass
        #####*************************************************************************************************************************************************
        #PCC/SPC Analysis
        #####*************************************************************************************************************************************************
#         maf_cutoff = 0.001
#         print ('number of individuals: ' + str(ldlr_unqiue_eid_missense_df['eid'].shape[0]))
#         ldlr_unqiue_eid_missense_filtered_df = ldlr_unqiue_eid_missense_df.loc[ldlr_unqiue_eid_missense_df['lowest_gnomad_af'] <= maf_cutoff,:]                     
#         print ('number of individuals contains MAF <= ' + str(maf_cutoff) + ': ' +  str(ldlr_unqiue_eid_missense_filtered_df.shape[0])) 
#         
#         # remove individual who has variants with circularity        
# #         ldlr_unqiue_eid_missense_filtered_df = ldlr_unqiue_eid_missense_filtered_df.loc[ldlr_unqiue_eid_missense_filtered_df['circularity'] == 0,:]
# #         print ('number of individuals after remove circularity: ' + str(ldlr_unqiue_eid_missense_filtered_df.shape[0])) 
# #         
#         # remove inidividual that has invalid scores
#         for score in selected_score_cols + covariate_list:
#             ldlr_unqiue_eid_missense_filtered_df = ldlr_unqiue_eid_missense_filtered_df.loc[ldlr_unqiue_eid_missense_filtered_df[score].notnull(),:]
#             print ('number of individuals after remove invalid ' + score + ': ' + str(ldlr_unqiue_eid_missense_filtered_df.shape[0]))
#         pass
#         
#                                     
#         ####analysis (PCC,SPC)      
#         ldlr_score_df = pd.DataFrame(selected_score_cols + covariate_list,columns = ['score'])        
#         for phenotype in phenotype_list:        
#             ldlr_score_df[phenotype+ '_pcc_z'] = [alm_fun.pcc_cal(ldlr_unqiue_eid_missense_filtered_df[score],ldlr_unqiue_eid_missense_filtered_df[phenotype + '_z'])  for score in selected_score_cols + covariate_list]
#             ldlr_score_df[phenotype+ '_pcc'] = [alm_fun.pcc_cal(ldlr_unqiue_eid_missense_filtered_df[score],ldlr_unqiue_eid_missense_filtered_df[phenotype])  for score in selected_score_cols + covariate_list]
#             ldlr_score_df[phenotype+ '_spc_z'] = [alm_fun.spc_cal(ldlr_unqiue_eid_missense_filtered_df[score],ldlr_unqiue_eid_missense_filtered_df[phenotype + '_z'])  for score in selected_score_cols + covariate_list]
#             ldlr_score_df[phenotype+ '_spc'] = [alm_fun.spc_cal(ldlr_unqiue_eid_missense_filtered_df[score],ldlr_unqiue_eid_missense_filtered_df[phenotype])  for score in selected_score_cols + covariate_list]            
#         pass
#         ldlr_score_df.sort_values(['LDL_pcc_z'],ascending = False).to_csv(runtime['db_path'] + 'ukb/csv/ldlr_score_comparision_maf_' + str(maf_cutoff) + '.csv',index = False)
# 
#         print ('OK')        
        
        
        
        
        
#         #####*************************************************************************************************************************************************
#         #PCC/SPC Analysis
#         #####*************************************************************************************************************************************************        
#         maf_cutoff = 0.005
#         ldlr_ukb_filtered_df = ldlr_ukb_df.loc[(ldlr_ukb_df['variant_type'] == 'missense') & (ldlr_ukb_df['gnomAD_exomes_AF'] <= maf_cutoff),:]             
#         print ("number of missense variants: " + str(ldlr_ukb_filtered_df.shape[0]))
#         print ("number of individuals: " + str(len(ldlr_ukb_filtered_df['eid'].unique())))
#         
#         #check correlation and flip the scores 
#         pcc_score_df = pd.DataFrame(selected_score_cols,columns = ['score'])
#         pcc_score_df['ldl_pcc'] = [alm_fun.pcc_cal(ldlr_ukb_filtered_df[score],ldlr_ukb_filtered_df['LDL'])  for score in selected_score_cols]        
#         flip_scores = list(pcc_score_df.loc[pcc_score_df['ldl_pcc'] < 0, 'score'])
#         for score in flip_scores:
#             ldlr_ukb_filtered_df.loc[:,score] = 0 -ldlr_ukb_filtered_df.loc[:,score]
# 
#         #selected phenotypes     
#         ldlr_unqiue_eid_df = ldlr_ukb_filtered_df[['eid'] + phenotype_list + covariate_list].drop_duplicates()
#         
#         #####remove individual with invalid phenotype
#         for phenotype in phenotype_list + covariate_list:
#             ldlr_unqiue_eid_df = ldlr_unqiue_eid_df.loc[ldlr_unqiue_eid_df[phenotype].notnull(),:]
#             
#         print ("number of individuals after remove invalid phenotypes: " + str(len(ldlr_ukb_filtered_df['eid'].unique())))            
#             
#         #####remove covariates            
#         #age and ldl_medication (z score)
#         for phenotype in phenotype_list:
#             ldlr_unqiue_eid_df[phenotype + '_z'] = ldlr_unqiue_eid_df[phenotype]          
#             for covariate in ['sex','ldl_medication']:
#                 ldlr_unqiue_eid_df.loc[(ldlr_unqiue_eid_df[covariate] == 1),phenotype + '_z'] = stats.zscore(ldlr_unqiue_eid_df.loc[(ldlr_unqiue_eid_df[covariate] == 1),phenotype + '_z'])
#                 ldlr_unqiue_eid_df.loc[(ldlr_unqiue_eid_df[covariate] == 0),phenotype + '_z'] = stats.zscore(ldlr_unqiue_eid_df.loc[(ldlr_unqiue_eid_df[covariate] == 0),phenotype + '_z'])
#                 
#                 
#             x = np.array(ldlr_unqiue_eid_df['age'])
#             y = np.array(ldlr_unqiue_eid_df[phenotype + '_z'])
#             reg = sklearn.linear_model.LinearRegression().fit(x.reshape(-1,1),y)            
#             ldlr_unqiue_eid_df[phenotype + '_z'] = y - reg.coef_*x
#             
#         #add cirucularity
#         ldlr_ukb_filtered_df['circularity'] = ldlr_ukb_filtered_df.apply(lambda x: (x[['hgmd_source','clinvar_source','humsavar_source','mave_source']] == 1).sum(),axis = 1)                 
#         ldlr_unqiue_eid_circularity_df = ldlr_ukb_filtered_df.groupby(['eid'])['circularity'].sum(min_count = 1).reset_index()
#         ldlr_unqiue_eid_df = pd.merge(ldlr_unqiue_eid_df,ldlr_unqiue_eid_circularity_df,how = 'left') 
# 
#         #add burden scores for each patient 
#         ldlr_unqiue_eid_scores_df = ldlr_ukb_filtered_df.groupby(['eid'])[selected_score_cols + covariate_list].agg(['max','sum'],min_count = 1).reset_index()
#         ldlr_unqiue_eid_scores_df.columns = ['_'.join(col).strip() for col in ldlr_unqiue_eid_scores_df.columns]
#         ldlr_unqiue_eid_scores_df = ldlr_unqiue_eid_scores_df.rename(columns = {'eid_':'eid'})        
#         ldlr_unqiue_eid_df = pd.merge(ldlr_unqiue_eid_df,ldlr_unqiue_eid_scores_df,how = 'left')  
#         
#         # remove inidividual that has invalid scores
#         for score in selected_score_cols + covariate_list:
#             ldlr_unqiue_eid_df = ldlr_unqiue_eid_df.loc[ldlr_unqiue_eid_df[score+'_max'].notnull(),:]
#             ldlr_unqiue_eid_df = ldlr_unqiue_eid_df.loc[ldlr_unqiue_eid_df[score+'_sum'].notnull(),:]
#             
#         print ("number of individuals after remove invalid scores: " + str(len(ldlr_ukb_filtered_df['eid'].unique())))                                
#         ldlr_unqiue_eid_df.to_csv(runtime['db_path'] + 'ukb/csv/ldlr_unique_eid.csv',index = False)              
#                                    
#         # remove individual who has variants with circularity        
# #         ldlr_unqiue_eid_df = ldlr_unqiue_eid_df.loc[ldlr_unqiue_eid_df['circularity'] == 0]
# #         print ('number of individuals after remove circularity: ' + str(ldlr_unqiue_eid_df.shape[0])) 
#         
#         ####analysis (PCC,SPC)      
#         ldlr_score_df = pd.DataFrame(selected_score_cols + covariate_list,columns = ['score'])        
#         for phenotype in ['cholesterol','HDL','LDL','triglycerides']:        
#             ldlr_score_df[phenotype+ '_pcc_z_max'] = [alm_fun.pcc_cal(ldlr_unqiue_eid_df[score + '_max'],ldlr_unqiue_eid_df[phenotype + '_z'])  for score in selected_score_cols + covariate_list]
#             ldlr_score_df[phenotype+ '_pcc_max'] = [alm_fun.pcc_cal(ldlr_unqiue_eid_df[score + '_max'],ldlr_unqiue_eid_df[phenotype])  for score in selected_score_cols + covariate_list]
#             ldlr_score_df[phenotype+ '_spc_z_max'] = [alm_fun.spc_cal(ldlr_unqiue_eid_df[score + '_max'],ldlr_unqiue_eid_df[phenotype + '_z'])  for score in selected_score_cols + covariate_list]
#             ldlr_score_df[phenotype+ '_spc_max'] = [alm_fun.spc_cal(ldlr_unqiue_eid_df[score + '_max'],ldlr_unqiue_eid_df[phenotype])  for score in selected_score_cols + covariate_list]            
# 
#         ldlr_score_df.sort_values(['LDL_pcc_max'],ascending = False).to_csv(runtime['db_path'] + 'ukb/csv/ldlr_score_comparision_maf_' + str(maf_cutoff) + '.csv',index = False)
# 
#         print ('OK')
# #             reg_theta = []
# #             for score in selected_score_cols + covariate_list:
# #                 x = np.array(ldlr_unqiue_eid_df[[score + '_max']])
# #                 y = np.array(ldlr_unqiue_eid_df[phenotype + '_z'])
# #                 reg = sklearn.linear_model.LinearRegression().fit(x,y)
# #                 reg_theta.append(reg.coef_[-1])
# #              
# #                  
# #             ldlr_score_df[phenotype+ '_reg_theta'] = reg_theta
#             
# #             ldlr_score_df[phenotype+ '_spc_max'] = [alm_fun.spc_cal(ldlr_unqiue_eid_df[score + '_max'],ldlr_unqiue_eid_df[phenotype])  for score in selected_score_cols]
# #             ldlr_score_df[phenotype+ '_pcc_sum'] = [alm_fun.pcc_cal(ldlr_unqiue_eid_df[score + '_sum'],ldlr_unqiue_eid_df[phenotype])  for score in selected_score_cols]
# #             ldlr_score_df[phenotype+ '_spc_sum'] = [alm_fun.spc_cal(ldlr_unqiue_eid_df[score + '_sum'],ldlr_unqiue_eid_df[phenotype])  for score in selected_score_cols]
#         
# #             ldlr_score_df[phenotype+ '_pcc_max'] = [alm_fun.pcc_cal(ldlr_unqiue_eid_df[score + '_max'],ldlr_unqiue_eid_df[phenotype] + '_z')  for score in selected_score_cols]
# #             ldlr_score_df[phenotype+ '_spc_max'] = [alm_fun.spc_cal(ldlr_unqiue_eid_df[score + '_max'],ldlr_unqiue_eid_df[phenotype])  for score in selected_score_cols]
# #             ldlr_score_df[phenotype+ '_pcc_sum'] = [alm_fun.pcc_cal(ldlr_unqiue_eid_df[score + '_sum'],ldlr_unqiue_eid_df[phenotype])  for score in selected_score_cols]
# #             ldlr_score_df[phenotype+ '_spc_sum'] = [alm_fun.spc_cal(ldlr_unqiue_eid_df[score + '_sum'],ldlr_unqiue_eid_df[phenotype])  for score in selected_score_cols]








        
      
#         denovo_1213_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_negative_added_snv.csv')        
#         denovo_1213_df[['unique_snv_id','fdr_cutoff', 'deepsequence_score', 'new_hgmd']].shape
#         
#         denovo_1106_df = pd.read_csv(runtime['db_path'] + 'varity/all/Revision1106_varity_target_denovodb_enriched_negative_added_snv_backup.csv')
#         denovo_1106_df = denovo_1106_df.loc[denovo_1106_df['p_vid'] != 'P62805',:]
#                 
#         
#         
#         denovo_1213_extra_df = denovo_1213_df[['unique_snv_id','fdr_cutoff', 'deepsequence_score', 'new_hgmd','asa_mean','bsa_max']]
#         denovo_1213_extra_df.columns = ['unique_snv_id','fdr_cutoff', 'deepsequence_score', 'new_hgmd','asa_mean_1213','bsa_max_1213']
#         denovo_1106_df = denovo_1106_df.merge(denovo_1213_extra_df,how = 'left')
#         denovo_1106_df[list(denovo_1213_df.columns) + ['asa_mean_1213','bsa_max_1213']].to_csv(runtime['db_path'] + 'varity/all/Revision1106_varity_target_denovodb_enriched_negative_added_snv.csv',index = False)
#         
#         
#         denovo_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_negative_added_snv_predicted.csv')
#         denovo_df = denovo_df.loc[denovo_df['p_vid'] != 'P62805',:]
#         denovo_df.to_csv (runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_negative_added_snv_predicted.csv',index = False)

        
        #**********************************************************************************************************************************************************
        # Save core and extra data set
        #**********************************************************************************************************************************************************
#         alm_predictor = self.proj.predictor[runtime['predictor']]
#         alm_dataset = alm_predictor.data_instance       
#         core_set =  alm_dataset.train_data_index_df[key_cols + annotation_cols + list(set(score_cols + feature_cols)) + qip_cols]
#         extra_set = alm_dataset.extra_train_data_df_lst[0][key_cols + annotation_cols + list(set(score_cols + feature_cols)) + qip_cols]        
#         all_set = pd.concat([core_set,extra_set])
#         all_set.to_csv(runtime['project_path'] + 'data/' + alm_predictor.name + '_all_traindata.csv',index = False)        
#         core_set.to_csv(runtime['project_path'] + 'data/' + alm_predictor.name + '_core_data.csv',index = False)
#         cor_set_no_structure = core_set.copy()
#         structural_cols = ['asa_mean','aa_psipred_E','aa_psipred_H','aa_psipred_C','bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_min']
#         for col in structural_cols:
#              cor_set_no_structure[col] = np.nan  
#               
#         cor_set_no_structure.to_csv(runtime['project_path'] + 'data/' + alm_predictor.name + '_core_data_no_structure.csv',index = False)   
#          


           
        
#         db_obj = alm_humandb.alm_humandb({})
        #**********************************************************************************************************************************************************
        # LDLR ukb variants
        #**********************************************************************************************************************************************************
        
#         ldlr_snv_df = pd.read_csv(runtime['db_path'] + 'varity/bygene/P01130_varity_snv_predicted.csv')                
#         ldlr_snv_df[key_cols+ selected_score_cols].to_csv(runtime['db_path'] + 'varity/bygene/P01130_varity_snv_predicted_20201201.csv',index = False)
        
#         #**********************************************************************************************************************************************************
#         # add FDR_cutoff denovodb gene
#         #**********************************************************************************************************************************************************
#         denovodb_enriched_genes_df = pd.read_csv(runtime['db_path'] + 'denovodb/all/denovodb_enriched_genes.csv')                
#         denovo_enriched_snv_predicted_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_negative_added_snv_predicted.csv')        
#         denovo_enriched_snv_predicted_df = pd.merge(denovo_enriched_snv_predicted_df,denovodb_enriched_genes_df[['p_vid','fdr_cutoff']],how = 'left')
#         denovo_enriched_snv_predicted_df.to_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_negative_added_snv_predicted.csv',index = False) 
#           
         
        #**********************************************************************************************************************************************************
        # Check Coe enriched genes
        #**********************************************************************************************************************************************************
#         q_cutoff = 0.1
#         coe_enriched_df = pd.read_csv(runtime['db_path'] + 'denovodb/all/denovodb_coe_genes.csv')
# #         coe_enriched_df = coe_enriched_df.replace('no model',np.nan)
# #         coe_enriched_df.to_csv(runtime['db_path'] + 'denovodb/all/denovodb_coe_genes.csv')
#                 
#         coe_enriched_df.loc[((coe_enriched_df['ch_lgd_q'] < q_cutoff) & (coe_enriched_df['lgd_count'] > 1)) |
#                             ((coe_enriched_df['ch_missense_q'] < q_cutoff) & (coe_enriched_df['missense_count'] > 1)) |
#                             ((coe_enriched_df['ch_cadd_q'] < q_cutoff) & (coe_enriched_df['missense_count'] > 1)) |                            
#                             ((coe_enriched_df['dr_lgd_q'] < q_cutoff) & (coe_enriched_df['lgd_count'] > 1)) |
#                             ((coe_enriched_df['dr_missense_q'] < q_cutoff) & (coe_enriched_df['missense_count'] > 1)),:].shape                              

        #**********************************************************************************************************************************************************
        # Add LOO flag
        #**********************************************************************************************************************************************************        
#         denovo_snv_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_negative_added_snv_predicted.csv')
#         denovo_snv_df['VARITY_R_training_flag'] = 0
#         denovo_snv_df.loc[denovo_snv_df['VARITY_R_LOO'] != denovo_snv_df['VARITY_R'],'VARITY_R_training_flag'] = 1        
#         denovo_snv_df.to_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_negative_added_snv_predicted.csv',index = False)
#         
        #**********************************************************************************************************************************************************
        # Check VARITY_R weights 
        #**********************************************************************************************************************************************************
#         extra_weight_df = pd.read_csv(runtime['project_path'] + 'output/csv/VARITY_R_extra_weight.csv')
#         core_weight_df = pd.read_csv(runtime['project_path'] + 'output/csv/VARITY_R_core_weight.csv')
#         denovo_snv_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_negative_added_snv_predicted.csv')

#         p_vids = list(denovo_snv_df.loc[(denovo_snv_df['VARITY_R_LOO'].isnull()) & (denovo_snv_df['from_denovodb'] ==0),'p_vid'].unique())
#         denovo_snv_df.loc[(denovo_snv_df['VARITY_R_LOO'].isnull()) & (denovo_snv_df['from_denovodb'] ==0),['chr','p_vid','nt_pos','nt_ref','nt_alt','gnomAD_exomes_AF','gnomAD_exomes_nhomalt']]
                        
#         denovo_snv_df.loc[(denovo_snv_df['from_denovodb'] ==0) & (denovo_snv_df['p_vid'] == 'Q9C0F0' ),['chr','p_vid','nt_pos','nt_ref','nt_alt','gnomAD_exomes_AF','gnomAD_exomes_nhomalt']]                
#         extra_weight_df.loc[extra_weight_df['p_vid'] == 'Q9C0F0' , ['p_vid','chr','nt_pos','nt_ref','nt_alt','gnomAD_exomes_AF','gnomAD_exomes_nhomalt','set_name','weight']]                
        #**********************************************************************************************************************************************************
        # Generate VCF file for HGMD query 
        #**********************************************************************************************************************************************************
        
#         alm_predictor = self.proj.predictor[runtime['predictor']]
#         alm_dataset = alm_predictor.data_instance        
#         alm_dataset.train_data_index_df[['chr','nt_pos','nt_ref','nt_alt','clinvar_id']].to_csv(runtime['project_path'] + 'output/csv/hgmd_query_on_clinvar.csv',index = False)
#         

#         denovo_enriched_snv_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_snv.csv')
#         denovo_enriched_snv_df[['chr','nt_pos','nt_ref','nt_alt','StudyName']].to_csv(runtime['project_path'] + 'output/csv/hgmd_query_on_denovodb.csv',index = False)
#         denovo_enriched_snv_df['unique_snv_id'] = denovo_enriched_snv_df.apply(lambda x: 'chr' + str(x['chr']) + ':' + str(x['nt_pos']) + x['nt_ref'] +  x['nt_alt'],axis = 1)
#         denovo_enriched_vcg_df = denovo_enriched_snv_df[['chr','nt_pos','unique_snv_id','nt_ref','nt_alt']]
#         denovo_enriched_vcg_df.columns = ['CHROM','POS','ID','REF','ALT']
#         denovo_enriched_vcg_df = denovo_enriched_vcg_df.reset_index(drop = True)
#         denovo_enriched_vcg_df.loc[0:399,:].to_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_snv_1.vcf',sep = '\t', index = False, header = None)
#         denovo_enriched_vcg_df.loc[400:799,:].to_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_snv_2.vcf',sep = '\t', index = False, header = None)
#         denovo_enriched_vcg_df.loc[800:,:].to_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_snv_3.vcf',sep = '\t', index = False, header = None)
#         
#         denovo_hgmd_df = pd.read_csv(runtime['db_path'] + 'hgmd/org/denovodb_hgmd_2020.csv')        
#         denovo_hgmd_df['unique_snv_id'] = denovo_hgmd_df['vcf_input'].apply(lambda x:x.split(' ')[2])    
#         denovo_hgmd_df.to_csv(runtime['project_path'] + 'output/csv/denovodb_hgmd_2020.csv',index = False)       
                
        #**********************************************************************************************************************************************************
        # Add negative examples from gnomAD  
        #**********************************************************************************************************************************************************                
#         varity_target_denovodb_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_all.csv')        
#         varity_target_denovodb_homo_df = varity_target_denovodb_df.loc[(varity_target_denovodb_df['gnomAD_exomes_AF'] < 0.005) & (varity_target_denovodb_df['gnomAD_exomes_nhomalt'] >= 1) & (varity_target_denovodb_df['label'] == 0),:]
#         varity_target_denovodb_homo_df['denovo_label'] = varity_target_denovodb_homo_df['label']
#         varity_target_denovodb_homo_df = varity_target_denovodb_homo_df.drop(columns = ['label'])
#         varity_target_denovodb_homo_df['unique_snv_id'] = varity_target_denovodb_homo_df.apply(lambda x: 'chr' + str(x['chr']) + ':' + str(x['nt_pos']) + x['nt_ref'] +  x['nt_alt'],axis = 1)
#         varity_target_denovodb_homo_df['from_denovodb'] = 0
#                   
#         varity_target_denovodb_homo_df.to_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_homo.csv',index = False)
#         
#         print ('Total negative variants to add: ' + str(varity_target_denovodb_homo_df.shape[0]))
#           
#         #### adding negative examples to denovodb enriched snvs 
#         denovo_enriched_snv_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_snv.csv')
#         denovo_enriched_snv_df['unique_snv_id'] = denovo_enriched_snv_df.apply(lambda x: 'chr' + str(x['chr']) + ':' + str(x['nt_pos']) + x['nt_ref'] +  x['nt_alt'],axis = 1)
#         denovo_enriched_snv_df['from_denovodb'] = 1
#         denovo_enriched_snv_df = denovo_enriched_snv_df[varity_target_denovodb_homo_df.columns]              
#         varity_target_denovodb_homo_extra_df = varity_target_denovodb_homo_df.loc[~varity_target_denovodb_homo_df['unique_snv_id'].isin(denovo_enriched_snv_df['unique_snv_id']),:]
#         denovo_enriched_negative_added_snv_df = pd.concat([denovo_enriched_snv_df,varity_target_denovodb_homo_extra_df]) 
#           
#         denovo_enriched_negative_added_snv_df.to_csv(runtime['db_path'] + 'varity/all/varity_target_denovodb_enriched_negative_added_snv.csv',index = False) 
# 
#         print ('Total variants : ' + str(denovo_enriched_negative_added_snv_df.shape[0]))

        #**********************************************************************************************************************************************************
        # Add new HGMD 
        #**********************************************************************************************************************************************************            
#         denovo_enriched_negative_added_snv_predicted_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_negative_added_snv_predicted.csv')
#         denovo_hgmd_df = pd.read_csv(runtime['db_path'] + 'hgmd/org/denovodb_hgmd_2020.csv')        
#         hgmd_existed_ids = list(denovo_hgmd_df.loc[denovo_hgmd_df['annotation'].notnull(),'unique_snv_id'].unique())
#         denovo_enriched_negative_added_snv_predicted_df['new_hgmd'] = 0   
#         denovo_enriched_negative_added_snv_predicted_df.loc[denovo_enriched_negative_added_snv_predicted_df['unique_snv_id'].isin(hgmd_existed_ids) ,'new_hgmd'] = 1                               
#         denovo_enriched_negative_added_snv_predicted_df.to_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_negative_added_snv_predicted.csv',index = False) 
#   
# 
#         denovo_enriched_snv_predicted_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_snv_predicted.csv')
#         denovo_enriched_snv_predicted_df['unique_snv_id'] = denovo_enriched_snv_predicted_df.apply(lambda x: 'chr' + str(x['chr']) + ':' + str(x['nt_pos']) + x['nt_ref'] +  x['nt_alt'],axis = 1)
#         denovo_hgmd_df = pd.read_csv(runtime['db_path'] + 'hgmd/org/denovodb_hgmd_2020.csv')     
#         hgmd_existed_ids = list(denovo_hgmd_df.loc[denovo_hgmd_df['annotation'].notnull(),'unique_snv_id'].unique())
#         denovo_enriched_snv_predicted_df['new_hgmd'] = 0   
#         denovo_enriched_snv_predicted_df.loc[denovo_enriched_snv_predicted_df['unique_snv_id'].isin(hgmd_existed_ids) ,'new_hgmd'] = 1                               
#         denovo_enriched_snv_predicted_df.to_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_snv_predicted.csv',index = False) 
#           
#           


        
#         varity_disease_genes_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_disease_genes.csv')
#         varity_disease_genes = list(varity_disease_genes_df['p_vid'].unique())
#         
#         denovo_pn_snv_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_pn_snv_predicted.csv')        
#         denovo_pn_snv_df = denovo_pn_snv_df.loc[denovo_pn_snv_df['VARITY_R_LOO'].isnull(),:]
#         denovo_pn_snv_negative_df = denovo_pn_snv_df.loc[denovo_pn_snv_df['denovo_label'] == 0,:]
#         denovo_pn_snv_negative_df = denovo_pn_snv_negative_df.loc[denovo_pn_snv_negative_df['p_vid'].isin(varity_disease_genes),:]
#         negative_add_indices = np.random.permutation(denovo_pn_snv_negative_df.index)[0:negative_add_count]
#         
# #         denovo_pn_snv_negative_df.loc[negative_add_indices,:]
#         
#         denovo_enriched_snv_no_hgmd_negative_added_df =  pd.concat([denovo_pn_snv_negative_df.loc[negative_add_indices,:],denovo_enriched_snv_no_hgmd_df])
#         denovo_enriched_snv_no_hgmd_negative_added_df['denovo_label'].value_counts()
#         denovo_enriched_snv_no_hgmd_negative_added_df.to_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_snv_predicted_no_hgmd_negative_added.csv',index = False)
#         
# #         ukb_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_ukb_predicted.csv')  
# #                       
#         key_cols = ['chr','nt_pos','nt_ref','nt_alt','symbol','p_vid','aa_pos','aa_ref','aa_alt']
# #                          
#         score_cols = ['Polyphen2_selected_HVAR_score','Polyphen2_selected_HDIV_score','PROVEAN_selected_score','SIFT_selected_score',
#                       'CADD_raw','PrimateAI_score','Eigen-raw_coding','GenoCanyon_score','integrated_fitCons_score','REVEL_score',
#                       'M-CAP_score','LRT_score','MutationTaster_selected_score','MutationAssessor_selected_score',
#                       'FATHMM_selected_score','MetaSVM_score','MetaLR_score','DANN_score','GERP++_RS',
#                       'phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','MPC_selected_score','mistic_score',
#                       'mpc_score','sift_score','provean_score','Final_VARITY_R','Final_VARITY_ER','Final_VARITY_ER_LOO','Final_VARITY_R_LOO',
#                       'VARITY_ER_mv_25','VARITY_ER_mv_50','VARITY_ER_mv_75','VARITY_ER_mv_100','VARITY_ER_mv_125',
#                       'VARITY_R_mv_25','VARITY_R_mv_50','VARITY_R_mv_75','VARITY_R_mv_100','VARITY_R_mv_125']        
#         
#         ukb_to_roujia_df = ukb_df.loc[:,key_cols+ score_cols]
#         ukb_to_roujia_df.to_csv(runtime['project_path'] + 'output/csv/varity_target_ukb_predicted_to_roujia.csv',index = False)
#         
        
#         ukb_to_roujia_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_ukb_predicted_to_roujia.csv')
#         ukb_to_roujia_old_df = pd.read_csv(runtime['db_path'] + 'varity/all/varity_target_ukb_to_roujia.csv')
#           


#         denovo_variants_df_clean = denovo_variants_df.loc[(denovo_variants_df['train_clinvar_source'] != 1) & 
#                                                           (denovo_variants_df['train_hgmd_source'] != 1) &
#                                                           (denovo_variants_df['train_humsavar_source'] != 1) &
#                                                           (denovo_variants_df['train_mave_source'] != 1) &
#                                                           (denovo_variants_df['train_gnomad_source'] != 1),:]
#         denovo_variants_df_clean.to_csv(runtime['project_path'] + 'output/csv/Revision1106_varity_target_denovodb_enriched_3_clean.csv',index  = False)
#         


# 
#         ldlr_variants_df = pd.read_csv(runtime['project_path'] + 'output/csv/P01130_varity_snv.csv')                                
#         ldlr_variants_df['hg19_coordinate'] = ldlr_variants_df.apply(lambda x: 'chr' + str(x['chr']) + ':' + str(x['nt_pos']) ,axis = 1)
#         ldlr_variants_df.to_csv(runtime['project_path'] + 'output/csv/P01130_varity_snv_1.csv',index = False)
#         
#         pd.Series(ldlr_variants_df['hg19_coordinate'].unique()).to_csv(runtime['project_path'] + 'output/csv/P01130_hg19_coordinates.csv',index = False)
# 
#         
#         denovo_variants_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_snv_predicted.csv')
#         denovo_variants_df['hg19_coordinate'] = denovo_variants_df.apply(lambda x: 'chr' + x['chr'] + ':' + str(x['nt_pos']) ,axis = 1)
#         denovo_variants_df.to_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_snv_predicted_1.csv',index = False)
#         denovo_variants_df = pd.read_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_snv_predicted_1.csv')
#         denovo_hgmd_df = pd.read_csv(runtime['db_path'] + 'hgmd/org/denovodb_hgmd_2020.csv')       
#         hgmd_hg19_coor = list(denovo_hgmd_df.loc[denovo_hgmd_df['annotation'].notnull(),'hg19_coordinate'].unique())
#         denovo_variants_no_hgmd_df = denovo_variants_df.loc[~(denovo_variants_df['hg19_coordinate'].isin(hgmd_hg19_coor) & (denovo_variants_df['denovo_label'] == 1)),:]        
#         denovo_variants_no_hgmd_df.to_csv(runtime['project_path'] + 'output/csv/varity_target_denovodb_enriched_snv_predicted_no_hgmd.csv',index = False)



#         denovo_targets_df = pd.read_csv(runtime['project_path'] + 'output/csv/Revision_varity_target_denovodb_predicted.csv')
#         denovo_targets_df.loc[denovo_targets_df['SIFT_selected_score'].isnull(),'p_vid'].value_counts()
#         
#         denovodb_enriched_gene_df = pd.read_csv(runtime['db_path'] + 'denovodb/all/denovodb_enriched_genes.csv')[['Gene','Enrich_type']]
#         denovodb_enriched_gene_df.columns = ['symbol','enrich_type']
#         
#         denovo_targets_df = pd.merge(denovo_targets_df,denovodb_enriched_gene_df,how = 'left')
#         denovo_targets_df['enrich_type'].value_counts()
#         
#         denovo_targets_df.to_csv(runtime['project_path'] + 'output/csv/Revision_varity_target_denovodb_predicted.csv',index = False)

    def add_loo_predictions(self,runtime):
        
        
#         r_loo_predictions = pd.read_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_VARITY_R_loo_predictions_with_keycols.csv')
#         er_loo_predictions = pd.read_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_VARITY_ER_loo_predictions_with_keycols.csv')
#         
#         all_loo_predictions = pd.merge(r_loo_predictions,er_loo_predictions,how = 'left')
#         all_loo_predictions.to_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_all_loo_predictions_with_keycols.csv',index = False)
#         
        
        all_loo_predictions = pd.read_csv(runtime['project_path'] + 'output/csv/' + runtime['session_id'] +  '_all_loo_predictions_with_keycols.csv')
                
        input_file  = runtime['add_loo_input_file']        
        input_file_df  = pd.read_csv(input_file)
                    
        output_file_df = pd.merge(input_file_df,all_loo_predictions,how = 'left')     
        
        
        r_no_loo_indices = output_file_df['VARITY_R_LOO'].isnull() | (output_file_df['VARITY_R_weight'] < runtime['no_loo_weight_cutoff'])
        output_file_df.loc[r_no_loo_indices,'VARITY_R_LOO'] = output_file_df.loc[r_no_loo_indices,'VARITY_R']        
        
        
        er_no_loo_indices = output_file_df['VARITY_ER_LOO'].isnull() | (output_file_df['VARITY_ER_weight'] < runtime['no_loo_weight_cutoff'])
        output_file_df.loc[er_no_loo_indices,'VARITY_ER_LOO'] = output_file_df.loc[er_no_loo_indices,'VARITY_ER']
           
        output_file_df.to_csv(input_file.split('.')[0] + '_loo.csv',index = False)
        
    def add_deepsequence(self,runtime):                        
        deepsequence_df = pd.read_csv(runtime['db_path'] + 'deepsequence/all/all_deepsequence_scores.csv')            
        #Remove deepsequence score if exists
        input_file  = runtime['add_deepsequence_input_file']        
        input_file_df  = pd.read_csv(input_file)
        
        if 'deepsequence_score' in input_file_df.columns:
            input_file_df = input_file_df.drop(columns = ['deepsequence_score'])
        
        output_file_df = pd.merge(input_file_df,deepsequence_df,how = 'left')        
        output_file_df.to_csv(input_file.split('.')[0] + '_with_deepsequence.csv',index = False)
        
        
        
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
               