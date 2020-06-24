import sys
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import time
import itertools
import pickle
import copy
from sklearn import neighbors as knn
from sklearn import linear_model as lm
from sklearn import feature_selection as fs
from sklearn import model_selection as ms
from sklearn.decomposition import nmf
from sklearn import svm
from sklearn import ensemble
from sklearn import tree
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics
import alm_data
import alm_es
import alm_fun
import alm_predictor
from datetime import datetime
from numpy import gradient
import varity


class alm_project:

    def __init__(self, argvs,varity_obj):
        #***************************************************************************************************************************************************************
        #Initiate project level parameters and ml object                     
        #***************************************************************************************************************************************************************
        project_init_params = argvs['project']
        for key in project_init_params:
            setattr(self, key, project_init_params[key])

        #parameters that are not open for configuration yet.
        self.verbose = 1
                           
        alm_fun.show_msg(self.log,self.verbose,'Class: [alm_project] [__init__]...... @' + str(datetime.now()))        

        #***************************************************************************************************************************************************************
        #Initiate data and es objects  (using runtime relevant predictors)                      
        #***************************************************************************************************************************************************************
        self.estimator = {}
        self.data = {}
        self.predictor = {}
        init_datas = []
        init_estimators = []
        
        if argvs['runtime']['action'] == 'init_session':        
            init_predictors = argvs['predictor'].keys()
        else:
            init_predictors = [argvs['runtime']['predictor']] + argvs['runtime']['compare_predictors']
            
        session_id = argvs['runtime']['session_id']
        for cur_predictor in init_predictors:
            if cur_predictor != '':
                cur_data = argvs['predictor'][cur_predictor]['data']
                if cur_data not in init_datas:                
                    init_datas.append(cur_data)
                cur_es = argvs['predictor'][cur_predictor]['estimator']
                if cur_es not in init_estimators:                
                    init_estimators.append(cur_es)            
                
                                        
        for cur_data in init_datas:
            if cur_data != 'None':
                cur_data_init_params = argvs['data'][cur_data] 
                cur_data_init_params['session_id'] = session_id
                cur_data_init_params['old_system'] = argvs['runtime']['old_system']
                cur_data_init_params['load_from_disk'] = argvs['runtime']['load_from_disk']
                cur_data_init_params['save_to_disk'] = argvs['runtime']['save_to_disk']
                cur_data_init_params['name'] = cur_data             
                                  
                if cur_data_init_params['load_from_disk'] == 1:
                    self.data[cur_data] = alm_data.alm_data(cur_data_init_params)
                else:
                    cur_data_df = pd.read_csv(argvs['data'][cur_data]['data_file'], low_memory = False)
                    cur_target_data_df = pd.DataFrame()
                    if 'target_file' in argvs['data'][cur_data]:
                        if os.path.isfile(argvs['data'][cur_data]['target_file']):                    
                            cur_target_data_df = pd.read_csv(argvs['data'][cur_data]['target_file'], low_memory = False)                    
                    
                    cur_data_init_params['input_data_type'] = 'dataframe'
                    cur_data_init_params['target_data_original_df'] = cur_target_data_df
                    cur_data_init_params['train_data_original_df'] = cur_data_df.loc[cur_data_df['extra_data'] == 0,:]
                    cur_data_init_params['test_data_original_df'] = pd.DataFrame()
                    cur_data_init_params['extra_train_data_original_df_lst'] = []
                    cur_data_init_params['extra_train_data_original_df_lst'].append(cur_data_df.loc[cur_data_df['extra_data'] == 1,:])
                    cur_data_init_params['use_extra_train_data'] =  1                
                    self.data[cur_data] = alm_data.alm_data(cur_data_init_params)                
                # in the case that cutomized  data split and slice is needed 
                if cur_data_init_params['test_split_method'] == 2:
                    self.data[cur_data].test_split = varity_obj.test_split
                if cur_data_init_params['cv_split_method'] == 2:
                    self.data[cur_data].cv_split = varity_obj.cv_split
                self.data[cur_data].data_preprocess = varity_obj.data_preprocess   
                # load or create current data set                
                self.data[cur_data].refresh_data()
            else:
                self.data[cur_data] = None
                
            # estimator initiation
        for cur_es in init_estimators:
            if cur_es != 'None':     
                cur_es_init_params = argvs['estimator'][cur_es]                       
                self.estimator[cur_es] = self.construct_estimator(cur_es_init_params)
            else:
                self.estimator[cur_es] = None
            
        for cur_predictor in init_predictors:
            #predictor initiation
            if cur_predictor != '':
                cur_predictor_init_params = argvs['predictor'][cur_predictor]
                cur_predictor_init_params['hyperparameter'] = argvs['hyperparameter']
                cur_predictor_init_params['init_hp_config'] = argvs['runtime']['init_hp_config']
                cur_predictor_init_params['old_system'] = argvs['runtime']['old_system']
                cur_predictor_init_params['shap_train_interaction'] = argvs['runtime']['shap_train_interaction']
                cur_predictor_init_params['shap_test_interaction'] = argvs['runtime']['shap_test_interaction']
                cur_predictor_init_params['session_id'] = session_id
                cur_predictor_init_params['name'] = cur_predictor
                cur_predictor_init_params['data_instance'] = self.data[cur_predictor_init_params['data']]
                cur_predictor_init_params['es_instance'] = self.estimator[cur_predictor_init_params['estimator']]
                self.predictor[cur_predictor] = alm_predictor.alm_predictor(cur_predictor_init_params)
                self.predictor[cur_predictor].filter_test = varity_obj.filter_test
                            
        alm_fun.show_msg(self.log,self.verbose,'Class: [alm_project] [__init__]......done @' + str(datetime.now()))

    def construct_estimator(self, es_init_params):
        algo = []
        algo_names = []
        algo_gs_range = []
        algo_scores = []
        algo_score_directions = []
        algo_importance = []
        algo_type = []
        
        #***************************************************************************************************************************************************************
        # Regression
        #***************************************************************************************************************************************************************
        # None Regressor
        algo.append(None)
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append({})
        algo_names.append("None")
        algo_importance.append('none')
        algo_type.append('regression')   
        
        #Decision Tree Regressor
        algo.append(tree.DecisionTreeRegressor(**{'max_depth':3}))   
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append({'max_depth': np.arange(1, 10, 1)})
        algo_names.append("dct_r")
        algo_importance.append('feature_importances_')
        algo_type.append('regression')    
                     
        # kNN Regressor
        algo.append(knn.KNeighborsRegressor(**{'n_neighbors': 7, 'weights': 'uniform', 'n_jobs':-1}))
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append({'n_neighbors': np.arange(1, 100, 1)})
        algo_names.append("knn_r")
        algo_importance.append('none')
        algo_type.append('regression')
        
        # Bayesian Ridge Regression  
        algo.append(lm.BayesianRidge())
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append({})
        algo_names.append("brr_r")
        algo_importance.append('coef_')   
        algo_type.append('regression')  
    
        # xgb Regressor
        algo.append(xgb.XGBRegressor(**{'subsample': 0.8, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.1, 'n_jobs': 8}))
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append({'learning_rate':np.arange(0.01, 0.11, 0.01), 'max_depth': np.arange(3, 6, 1), 'n_estimators':range(100, 500, 100)})
        algo_names.append("xgb_r")
        algo_importance.append('booster')
        algo_type.append('regression')
        
        
        algo.append(xgb.XGBClassifier())
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append({'learning_rate':np.arange(0.01, 0.1, 0.01), 'max_depth': np.arange(3, 6, 1), 'n_estimators':range(100, 500, 100)})
        algo_names.append("xgb_c")
        algo_importance.append('booster')
        algo_type.append('regression')
        
        # Random Forest Regressor
        algo.append(ensemble.RandomForestRegressor(**{'n_jobs':-1, 'n_estimators': 200, 'max_features': 'auto'}))
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append({'n_estimators':range(100, 500, 100), 'max_features':np.arange(0.1, 1.0, 0.1)})
        algo_names.append("rf_r")
        algo_importance.append('feature_importances_')   
        algo_type.append('regression')
        
        # ElasticNet Regressor
        algo.append(lm.ElasticNet(alpha=0.01, l1_ratio=0.5))
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append({'alpha':np.arange(0, 1, 0.1), 'l1_ratio':np.arange(0, 1, 0.1)})
        algo_names.append("en_r")
        algo_importance.append('coef_')   
        algo_type.append('regression')  
            
        #SVM Regressor
        algo.append(svm.SVR(C=1.0, epsilon=0.1,kernel='linear'))
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append({})
        algo_names.append("svm_r")
        algo_importance.append('coef_')  
        algo_type.append('regression')  
        
        #AdaBoost ElasticNet Regressor
        algo.append(ensemble.AdaBoostRegressor(lm.ElasticNet(alpha=0.1, l1_ratio=0.5),n_estimators=500, random_state=0))
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append({})
        algo_names.append("ada_en_r")
        algo_importance.append('none')    
        algo_type.append('regression')  
        
        #Keras regressor for classification        
        algo.append(None)
        algo_scores.append('rmse')
        algo_score_directions.append(0)
        algo_gs_range.append(None)
        algo_names.append("keras_r")
        algo_importance.append('none')
        algo_type.append('regression')
        
        
        #***************************************************************************************************************************************************************
        # Binary classification 
        #***************************************************************************************************************************************************************        
        # None Classification
        algo.append(None)
        algo_scores.append('auroc')
        algo_score_directions.append(1)
        algo_gs_range.append({})
        algo_names.append("None")
        algo_importance.append('none')
        algo_type.append('classification_binary')  
        
        
        #Decision tree regressor for classification
        algo.append(tree.DecisionTreeRegressor(**{'max_depth':5}))   
        algo_scores.append('auprc')
        algo_score_directions.append(1)
        algo_gs_range.append({'max_depth': np.arange(1, 10, 1)})
        algo_names.append("dct_r_c")
        algo_importance.append('feature_importances_')
        algo_type.append('classification_binary') 
        
        #Decision tree classifier
        algo.append(tree.DecisionTreeClassifier(**{'max_depth':5}))   
        algo_scores.append('auprc')
        algo_score_directions.append(1)
        algo_gs_range.append({'max_depth': np.arange(1, 10, 1)})
        algo_names.append("dct_c")
        algo_importance.append('feature_importances_')
        algo_type.append('classification_binary')  
        
        # Gradient boosted tree regressor for classification
#         algo.append(xgb.XGBRegressor(**{'n_jobs': 8,'subsample': 0.8, 'colsample_bytree': 1, 'max_depth': 3, 'n_estimators': 100, 'learning_rate': 0.02}))
        algo.append(xgb.XGBRegressor(**{'n_jobs': 8}))
        algo_scores.append('auprc')
        algo_score_directions.append(1)
        algo_gs_range.append({'learning_rate':np.arange(0.01, 0.06, 0.01), 'max_depth': np.arange(3, 5, 1), 'n_estimators':range(100, 400, 100)})
        algo_names.append("xgb_r_c")
        algo_importance.append('booster')
        algo_type.append('classification_binary')
        
        # Gradient boosted tree Classifier
        algo.append(xgb.XGBClassifier())
        algo_scores.append('auroc')
        algo_score_directions.append(1)
        algo_gs_range.append({'learning_rate':np.arange(0.01, 0.1, 0.01), 'max_depth': np.arange(3, 6, 1), 'n_estimators':range(100, 500, 100)})
        algo_names.append("xgb_c")
        algo_importance.append('booster')
        algo_type.append('classification_binary')
        
        # Random Forest regressor for classification
#         algo.append(ensemble.RandomForestRegressor(**{'n_jobs':-1, 'n_estimators': 200, 'max_features': 'auto'}))
        algo.append(ensemble.RandomForestRegressor())
        algo_scores.append('auroc')
        algo_score_directions.append(1)
        algo_gs_range.append({'max_features':range(10, 100, 10), 'n_estimators':range(100, 200, 100), 'test_bs_result':['True', 'False']})
        algo_names.append("rf_r_c")
        algo_importance.append('feature_importances_')
        algo_type.append('classification_binary')
        
        # Random Forest Classifier
        algo.append(ensemble.RandomForestClassifier(**{'n_jobs':-1, 'n_estimators': 200, 'max_features': 'auto'}))
        algo_scores.append('auroc')
        algo_score_directions.append(1)
        algo_gs_range.append({'max_features':range(10, 100, 10), 'n_estimators':range(100, 200, 100), 'test_bs_result':['True', 'False']})
        algo_names.append("rf_c")
        algo_importance.append('feature_importances_')
        algo_type.append('classification_binary')   
        
        
        # ElasticNet Regressor for classification
#         algo.append(lm.ElasticNet(alpha=0.01, l1_ratio=0.5))
        algo.append(lm.ElasticNet())
        algo_scores.append('auroc')
        algo_score_directions.append(0)
        algo_gs_range.append({'alpha':np.arange(0, 1, 0.1), 'l1_ratio':np.arange(0, 1, 0.1)})
        algo_names.append("en_r_c")
        algo_importance.append('coef_')   
        algo_type.append('classification_binary')  
        
             
        # Logistic Regression Classifier (binary)
        algo.append(lm.LogisticRegression())
        algo_scores.append('auroc')
        algo_score_directions.append(1)
        algo_gs_range.append({})
        algo_names.append("lgr_c")
        algo_importance.append('coef_')
        algo_type.append('classification_binary')
        
        # KNN Classifier (binary) 
        algo.append(knn.KNeighborsClassifier(**{'n_neighbors': 10, 'weights': 'distance', 'n_jobs':-1}))
        algo_scores.append('auroc')
        algo_score_directions.append(1)
        algo_gs_range.append({})
        algo_names.append("knn_c")
        algo_importance.append('none')
        algo_type.append('classification_binary')
        
        # SVM Regressor for classification 
        algo.append(svm.SVR(C=1.0, epsilon=0.1,kernel='linear'))
        algo_scores.append('auroc')
        algo_score_directions.append(1)
        algo_gs_range.append({})
        algo_names.append("svm_r_c")
        algo_importance.append('coef_')  
        algo_type.append('classification_binary')  
               
        # SVM Classifier    
        algo.append(svm.SVC(**{'C': 1.0}))
        algo_scores.append('auroc')
        algo_score_directions.append(1)
        algo_gs_range.append({})
        algo_names.append("svm_c")
        algo_importance.append('coef_')
        algo_type.append('classification_binary')
        
        #Keras regressor for classification        
        algo.append(None)
        algo_scores.append('auprc')
        algo_score_directions.append(1)
        algo_gs_range.append(None)
        algo_names.append("keras_r_c")
        algo_importance.append('none')
        algo_type.append('classification_binary')
        
        
#         #Tensor flow classifier
#         algo.append(alphame_ml.alm_tf(**{'estimator_name': 'DNNClassifier', 'loss_name': 'cross_entropy', 'hidden_units': [100],
#                                 'activation_fn': tf.nn.sigmoid,'n_classes': 2, 'batch_gd': 1,'batch_size': 0,'num_epochs': 20000, 'learning_rate': 0.0003})    )
#         algo_scores.append('auroc')
#         algo_score_directions.append(1)
#         algo_gs_range.append({})
#         algo_names.append("tf_c")    
#         algo_importance.append('none')
#         algo_type.append('classification_binary')
                
#         #Neural Network (Tensorflow)
#         algo.append(alphame_ml.alm_tf())
#         algo_scores.append('neg_log_loss')
#         algo_score_directions.append(0)
#         algo_gs_range.append({'leraning_rate':[0.01,0.1]})
#         algo_names.append("nn_c")
#         algo_importance.append('none')
#         algo_type.append('classification_multiclass')
        
        #***************************************************************************************************************************************************************
        # multi-class classification 
        #***************************************************************************************************************************************************************        
        # xgb 
        algo.append(xgb.XGBClassifier(**{'subsample': 0.9, 'colsample_bytree': 1, 'max_depth': 5, 'n_estimators': 200, 'learning_rate': 0.05}))
        algo_scores.append('neg_log_loss')
        algo_score_directions.append(0)
        algo_gs_range.append({'learning_rate':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], 'max_depth': [3, 5]})
    #     algo_gs_range.append({ 'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]})
        algo_names.append("xgb_c")
        algo_importance.append('feature_importances_')
        algo_type.append('classification_multiclass')

        # Random Forest Classifier
        algo.append(ensemble.RandomForestClassifier(**{'n_jobs':-1, 'n_estimators': 200, 'max_features': 'auto'}))
        algo_scores.append('neg_log_loss')
        algo_score_directions.append(0)
        algo_gs_range.append({'max_features':range(10, 100, 10), 'n_estimators':range(100, 200, 100), 'test_bs_result':['True', 'False']})
        algo_names.append("rf_c")
        algo_importance.append('feature_importances_')
        algo_type.append('classification_multiclass')

        # Gradient Boost Tree Classifier
        algo.append(ensemble.GradientBoostingClassifier(**{'n_estimators': 200, 'max_features': 'auto', 'max_depth': 3}))
        algo_scores.append('neg_log_loss')
        algo_score_directions.append(0)
        algo_gs_range.append({})
        algo_names.append("gbt_c")   
        algo_importance.append('feature_importances_') 
        algo_type.append('classification_multiclass')
    
        # Logistic Regression Classifier (multi-class)
        algo.append(lm.LogisticRegression())
        algo_scores.append('neg_log_loss')
        algo_score_directions.append(1)
        algo_gs_range.append({})
        algo_names.append("lgr_c")
        algo_importance.append('coef_')
        algo_type.append('classification_multiclass')
               
        # KNN Classifier    (multi-class)
        algo.append(knn.KNeighborsClassifier(**{'n_neighbors': 10, 'weights': 'distance', 'n_jobs':-1}))
        algo_scores.append('neg_log_loss')
        algo_score_directions.append(1)
        algo_gs_range.append({})
        algo_names.append("knn_c")
        algo_importance.append('none')
        algo_type.append('classification_multiclass')
        
        # SVM Classifier    
        algo.append(svm.SVC(**{'C': 1.0, 'kernel': 'linear', 'probability': True}))
        algo_scores.append('neg_log_loss')
        algo_score_directions.append(1)
        algo_gs_range.append({})
        algo_names.append("svm_c")
        algo_importance.append('coef_')
        algo_type.append('classification_multiclass')
        
        i = algo_names.index(es_init_params['algo_name'])
        es_init_params['estimator'] = algo[i]
        es_init_params['gs_range'] = algo_gs_range[i]
        es_init_params['score_name'] = algo_scores[i]
        es_init_params['score_direction'] = algo_score_directions[i]
        es_init_params['feature_importance_name'] = algo_importance[i]
        es_init_params['prediction_transformation'] = None
        estimator = alm_es.alm_es(es_init_params)
        
        return estimator
    