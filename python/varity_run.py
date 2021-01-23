import sys
import varity
import glob

def create_runtime_dict(sys_argv):
    #defaut runtime values
    runtime = {}
    runtime['db_path'] = '..'
    runtime['project_path'] = '..'  
    runtime['action'] = ''
    runtime['session_id']= ''
    runtime['predictors']= ''
    runtime['compare_predictors'] = []
    runtime['predictors'] = []
    runtime['init_hp_config'] = 0
    runtime['load_from_disk'] = 1
    runtime['save_to_disk'] = 1
    runtime['cur_test_fold']= 0
    runtime['cur_validation_fold']= 0
    runtime['cur_gradient_key']= 'no_gradient'
    runtime['hp_tune_type'] = 'hyperopt_logistic'
    runtime['trials_mv_size'] = -1
    runtime['trials_max_num'] = 1000
    runtime['hp_select_strategy'] = 'first_descent_mv_validation_selected_index'
#     runtime['plot_metric']= ['aubprc_interp','aubprc_org','auroc_interp','auroc_org']
#     runtime['plot_metric_order']= ['interp_aubprc','org_aubprc','interp_auroc','org_auroc']
#     runtime['plot_metric']= ['aubprc_interp','auroc_org','aubprc_org']
#     runtime['plot_metric_order']= ['interp_aubprc','org_auroc','org_aubprc']

    runtime['plot_metric']= ['interp_aubprc','org_auroc']
    runtime['plot_metric_order']= ['interp_aubprc','org_auroc']  
    
#     runtime['plot_metric']= ['org_auroc']
#     runtime['plot_metric_order']= ['org_auroc']   
#     
    runtime['filter_test_score'] = 1        
    runtime['num_bootstrap'] = 2000

    runtime['target_type'] = 'file'
    runtime['target_dataframe'] = ''
    runtime['target_file'] = '' 
    runtime['save_target_csv_name'] = ''
    runtime['save_target_npy_name'] = ''
    runtime['target_files'] = []
    runtime['target_predicted_files'] = []    
    runtime['target_dependent_variable'] = 'label'
    runtime['loo'] = 0
    runtime['loo_key_cols'] = ['p_vid','aa_pos','aa_ref','aa_alt']
    runtime['prediction_ouput_cols'] = ['chr','nt_pos','nt_ref','nt_alt','p_vid','aa_pos','aa_ref','aa_alt']
    runtime['prediction_ouput_with_features'] = 0
    runtime['prediction_ouput_with_input_cols'] = 0       
    runtime['shap_train_interaction'] = 0
    runtime['shap_test_interaction'] = 0
    runtime['load_existing_model'] = 0
    
    runtime['run_all_target_predictions'] = 0
        
    runtime['mv_hp'] = ''
    runtime['old_system'] = 0
    runtime['add_new_scores'] = 0
    runtime['add_core_setname'] = 0
    runtime['shrink_data'] = 0
    runtime['add_complete_sift_provean_scores'] = 0
    runtime['add_deepsequence_scores'] = 0
    runtime['add_test_data'] = 0  
    runtime['update_pisa'] = 0
    runtime['check_data'] = 0
    runtime['fig_x'] = 30
    runtime['fig_y'] = 20
    runtime['dpi'] = 300
    

    runtime['load_from_disk'] = 1 
    runtime['qip'] = ''
    runtime['cluster'] = 0
    runtime['run_on_node'] = 0
    runtime['job_id'] = -1
    runtime['job_name']=''
    runtime['batch_id']=''
 
    runtime['reinitiate_session'] = 0
    runtime['predictor']='' 

    runtime['independent_test_file'] = ''
    runtime['dependent_variable'] = 'label'

    runtime['parallel_batches'] = 100
    
    runtime['hp_logistic'] = 1
    
    runtime['show_detail_monitor_jobs'] = 0
    
    runtime['additional_features'] = []
    runtime['mem'] = 10240
    runtime['node'] = ''
    
    runtime['hp_dict_file'] = ''
    
    runtime['pisa_folder'] = 'pisa'
    
    runtime['test_hyperopt_type'] = 'mv'
    runtime['test_hyperopt_mvs'] = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
#     runtime['test_hyperopt_mvs'] = [50,100,150,200]
#     runtime['test_hyperopt_mvs'] = [10,20,30,40,50,100]
    
    
    
    runtime['overfit_epsilon'] = 0.01
    runtime['selected_tid'] = -1

    
    runtime['all_R_CV_preditors'] = ['VARITY_R_CV','SIFT_R_CV','Polyphen2_HDIV_R_CV','Polyphen2_HVAR_R_CV','PROVEAN_R_CV','CADD_R_CV','PrimateAI_R_CV','Eigen_R_CV',
             'REVEL_R_CV','M-CAP_R_CV','LRT_R_CV','MutationTaster_R_CV','MutationAssessor_R_CV','FATHMM_R_CV','MetaSVM_R_CV',
             'MetaLR_R_CV','GenoCanyon_R_CV','DANN_R_CV','GERP++_R_CV','phyloP_R_CV','PhastCons_R_CV','SiPhy_R_CV','fitCons_R_CV','MISTIC_R_CV','MPC_R_CV']
    runtime['all_ER_CV_preditors'] = ['VARITY_ER_CV','SIFT_ER_CV','Polyphen2_HDIV_ER_CV','Polyphen2_HVAR_ER_CV','PROVEAN_ER_CV','CADD_ER_CV','PrimateAI_ER_CV','Eigen_ER_CV',
             'REVEL_ER_CV','M-CAP_ER_CV','LRT_ER_CV','MutationTaster_ER_CV','MutationAssessor_ER_CV','FATHMM_ER_CV','MetaSVM_ER_CV',
             'MetaLR_ER_CV','GenoCanyon_ER_CV','DANN_ER_CV','GERP++_ER_CV','phyloP_ER_CV','PhastCons_ER_CV','SiPhy_ER_CV','fitCons_ER_CV','MISTIC_ER_CV','MPC_ER_CV']
    
    
    runtime['remove_structural_features'] = 0
    
    #read input runtime parameters      

    if isinstance(sys_argv,list):
        runtime['varity_command'] = sys_argv[0]
        for i in range(1,len(sys_argv)):
            key = sys_argv[i].split('=')[0]
            value = sys_argv[i].split('=')[1]
            if '[' in value:
                if value[1:-1] == '':
                    value_lst = []
                else:
                    value_lst = value[1:-1].split(',')   
                runtime[key] = value_lst    
            else:    
                if value.replace('.','0').replace('-','0'):                                            
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
            
    return(runtime)

def varity_run(sys_argv):
    runtime = create_runtime_dict(sys_argv)
    varity_obj = varity.varity(runtime)
    
    if len(runtime['predictors']) == 0:        
        runtime['predictors'] = [runtime['predictor']]

    for predictor in runtime['predictors']:
        cur_runtime = runtime.copy()
        cur_runtime['predictor'] = predictor
        cur_runtime['predictors'] = []            
        [job_id,job_name,result_dict] = varity_obj.varity_action(cur_runtime)
    return([job_id,job_name,result_dict])
 
if __name__ == "__main__":    
    varity_run(sys.argv)