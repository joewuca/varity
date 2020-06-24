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
import xgboost as xgb
from xgboost import plot_tree
# import shap
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import tree
import alm_fun
# from tree import  *
# from summary import *

# import lightgbm as lgb
# import graphviz
class alm_es:

    def __init__(self, es_init_params):
        for key in es_init_params:
            setattr(self, key, es_init_params[key])
        
        #parameters that are not open for configuration yet
        self.weighted_example = 1
        self.single_feature_as_prediction = 1
        self.flip_contamination_train = 0
        self.flip_contamination_test = 0

    def run(self, features, dependent_variable, ml_type, train, test, extra_train= None, validation = None, alm_predictor = None, model_file = None):            
        if alm_predictor is None:
            use_extra_train_data=0
            nofit = 0
            tune_tree_num = 0
            shap_test_interaction = 0
            shap_train_interaction = 0
            shuffle_features = []
            if_feature_engineer = 0
            load_existing_model = 0
        else:
            use_extra_train_data = alm_predictor.use_extra_train_data
            nofit = alm_predictor.nofit
            tune_tree_num = alm_predictor.tune_tree_nums_before_test
            shap_test_interaction = alm_predictor.shap_test_interaction
            shap_train_interaction = alm_predictor.shap_train_interaction
            shuffle_features = alm_predictor.shuffle_features
            if_feature_engineer = alm_predictor.if_feature_engineer
            load_existing_model = alm_predictor.load_existing_model
            eval_obj = alm_predictor.eval_obj

        #####********************************************************************************************
        # Run feature engineer fucntion if necessary
        #####********************************************************************************************             
        if if_feature_engineer:
            [train,test] = self.feature_engineer(train,test)
            
        #####********************************************************************************************
        # If features are nested list, flat the list of list if necessary
        #####********************************************************************************************                                     
        if any(isinstance(i, list) for i in features): 
            features = list(itertools.chain(*features))            
            
        #####********************************************************************************************
        # Shuffle features if necessary , for feature interaction analysis
        #####********************************************************************************************                     
        if len(shuffle_features) > 0 :                        
            for f in shuffle_features:                
                if f  != '':
                    train[f] = np.random.permutation(train[f])
            
        #####********************************************************************************************
        # copy the train,extra_train,test and validation dataset
        #####********************************************************************************************                  
        train = train.copy()
        test = test.copy()
        if (tune_tree_num == 1) & (validation is not None):
            validation = validation.copy()  
        
        if extra_train is not None:
            if extra_train.shape[0] !=0:            
                extra_train = extra_train.copy()

        #####********************************************************************************************
        # Remove extra training examples that weight == 0 
        #####********************************************************************************************                                
        if self.weighted_example == 1:  
            extra_train = extra_train.loc[(extra_train['weight'] != 0),:]

        #####********************************************************************************************
        # Combine train and extra_train dataset to make the final training dataset 
        #####********************************************************************************************
        if use_extra_train_data == 0:  # do not use extra training data
            final_train = train 
            
        if use_extra_train_data == 1:  # only use extra training data
            final_train = extra_train      
            
        if use_extra_train_data == 2:  # use extra training data directly + training data, no prediction
            final_train = pd.concat([extra_train,train])          
#             final_train = final_train.sort_index()  
        #####********************************************************************************************
        # Reorder the traning data and form groups for "ranking" loss function
        #####********************************************************************************************
#         final_train = final_train.sort_values('p_vid')
#         group_counts = final_train['p_vid'].value_counts().sort_index()        
#         group_counts = [5000,5000,len(final_train)-10000]
        group_counts = [len(final_train)]
        query_group = np.array(group_counts)                 
        group_weights = np.ones(len(query_group))
        
        #####********************************************************************************************
        # Separate features , labels or weight of the test and final training set
        #####********************************************************************************************
        
        
        core_train_x = train[features]      
        core_train_y = train[dependent_variable]          
        train_x = final_train[features]
        train_y = final_train[dependent_variable]        
        test_x = test[features]
        test_y = test[dependent_variable]
        test_index = test.index
        if validation is not None:
            validation_x = validation[features]
            validation_y = validation[dependent_variable]            
        
        if 'weight' in final_train.columns:
            weights = final_train['weight']        
            negative_idx = train_y == 1
            positive_idx = train_y == 0            
            negative_weights = weights[negative_idx].sum()
            positive_weights = weights[positive_idx].sum()            
            prior_weight = negative_weights/positive_weights
            weights[positive_idx] = weights[positive_idx] * prior_weight #balance negative and positive weights            
        else:
            weights = [1] * final_train.shape[0]
        
        #####********************************************************************************************
        # Flip the label for training and test set for contamination analysis if necessary
        #####********************************************************************************************                             
        if self.flip_contamination_test == 1:
            test_contamination = test['contamination']         
            test_y = [list(test_y)[i] if list(test_contamination)[i] != 1 else abs(list(test_y)[i] - 1) for i in range(len(test_y))]
            print ("Test contamination " + str((test_contamination == 1).sum()) + " flipped!")
               
        if self.flip_contamination_train == 1: 
            train_contamination = train['contamination']         
            train_y = [list(train_y)[i] if list(train_contamination)[i] != 1 else abs(list(train_y)[i] - 1) for i in range(len(train_y))]
            print ("Train contamination " + str((train_contamination == 1).sum()) + " flipped!")  

        load_model = 0
        if load_existing_model == 1:
            if os.path.isfile(model_file):
                self.estimator.load_model(model_file)                
                load_model = 1
                
        if load_model == 0:                   
            #####********************************************************************************************
            # Reset the estimator for every run 
            #####********************************************************************************************        
            if (self.estimator != None):
                n_estimators = self.estimator.n_estimators
                max_depth = self.estimator.max_depth
                learning_rate = self.estimator.learning_rate
                gamma = self.estimator.gamma
                min_child_weight = self.estimator.min_child_weight
                subsample = self.estimator.subsample
                colsample_bytree = self.estimator.colsample_bytree
                
                if 'regression' in ml_type:
                    self.estimator = xgb.XGBRegressor(**{'max_depth': max_depth, 'n_estimators': n_estimators, 'learning_rate': learning_rate, 'gamma':gamma, 'min_child_weight': min_child_weight, 'subsample': subsample, 'colsample_bytree': colsample_bytree,'n_jobs': -1})
                if 'classification' in ml_type:
                    self.estimator = xgb.XGBClassifier(**{'max_depth': max_depth, 'n_estimators': n_estimators, 'learning_rate': learning_rate, 'gamma':gamma, 'min_child_weight': min_child_weight, 'subsample': subsample, 'colsample_bytree': colsample_bytree,'n_jobs': -1})



            #####********************************************************************************************
            # Fit the model
            #####********************************************************************************************                     
            if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)): # if estimator is None, there is no need to train the model 
                feature_importance = pd.DataFrame(np.zeros(len(features)), index=features).transpose() 
            else:            
                if nofit == 0:
                    if self.weighted_example == 1:
                        if tune_tree_num == 1:
                            self.estimator.n_estimators = 1000
                            if 'rank' in self.estimator.objective:   
                                self.estimator.fit(train_x, train_y, group = query_group, sample_weight = group_weights,verbose = False, eval_set = [(validation_x[features],validation_y)],early_stopping_rounds = 50,eval_metric = eval_obj)
                            else:
                                self.estimator.fit(train_x, train_y, sample_weight = weights,verbose = False, eval_set = [(validation_x[features],validation_y)],early_stopping_rounds = 50,eval_metric = eval_obj)
                        else:
                            if 'rank' in self.estimator.objective:                            
                                self.estimator.fit(train_x,train_y, group = query_group,sample_weight = group_weights)
                            else:
                                print ("Start fit the model : " + str(datetime.now()))
                                print ("Training examples: " + str(train_x.shape[0]) + " Training weights: " + str(weights.sum()) + " # of Trees: " + str(self.estimator.n_estimators))                                                       
                                self.estimator.fit(train_x,train_y,sample_weight = weights)                                 
                                print ("End fit the model : " + str(datetime.now()))                                                                                                                                                                                  
                    else:
                        if 'rank' in self.estimator.objective:                             
                            self.estimator.fit(train_x, train_y,group = query_group)
                        else:
                            self.estimator.fit(train_x, train_y)
        else:
            alm_fun.show_msg (self.log,self.verbose,'Existing model loaded.')
             
        #####********************************************************************************************
        # Record the feature importance
        #####********************************************************************************************   
        if self.feature_importance_name == 'coef_':
            feature_importance = np.squeeze(self.estimator.coef_)
        if self.feature_importance_name == 'feature_importances_' :  
            feature_importance = np.squeeze(self.estimator.feature_importances_)
        if self.feature_importance_name == 'booster' :
            if len(features) == 1:
                feature_importance = np.zeros(len(features))
            else:
                if load_existing_model == 0:
                    feature_importance = []
                    im_dict = self.estimator.get_booster().get_score(importance_type='gain')
                    for feature in features:
                        feature_importance.append(im_dict.get(feature,0))
                else:
                    feature_importance = []
                    im_dict = self.estimator.get_booster().get_score(importance_type='gain')
                    for i in range(len(features)):
                        feature_importance.append(im_dict.get('f' + str(i),0))                    
        if self.feature_importance_name == 'none' :
            feature_importance = np.zeros(len(features))
        
        feature_importance = pd.DataFrame(feature_importance, index=features).transpose()      
  
        #####********************************************************************************************
        # Predict the train and test data 
        #####********************************************************************************************   
        if ml_type == "regression":                   
            if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)):                
                test_y_predicted = np.array(list(np.squeeze(test_x[features])))
            else:
                try:
                    test_y_predicted = self.estimator.predict_proba(test_x[features])[:, 1]   
                except:
                    test_y_predicted = self.estimator.predict(test_x[features])
             
                if self.prediction_transformation is not None:
                    test_y_predicted = self.prediction_transformation(test_y_predicted)     
                          
            test_score_df = pd.DataFrame(np.zeros(2), index=['pcc', 'rmse']).transpose()                               
            rmse = alm_fun.rmse_cal(test_y, test_y_predicted)  
            pcc = alm_fun.pcc_cal(test_y, test_y_predicted)   
            spc = alm_fun.spc_cal(test_y, test_y_predicted)
            test_score_df['rmse'] = rmse
            test_score_df['pcc'] = pcc
            test_score_df['spc'] = spc

            if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
                core_train_y_predicted = np.array(list(np.squeeze(train_x[features])))
            else:
                try:
                    core_train_y_predicted = self.estimator.predict_proba(train_x[features])[:, 1]
                except:
                    core_train_y_predicted = self.estimator.predict(train_x[features]) 
            
                if self.prediction_transformation is not None:
                    core_train_y_predicted = self.prediction_transformation(core_train_y_predicted) 
            
            core_train_score_df = pd.DataFrame(np.zeros(2), index=['pcc', 'rmse']).transpose()                               
            rmse = alm_fun.rmse_cal(core_train_y, core_train_y_predicted)  
            pcc = alm_fun.pcc_cal(core_train_y, core_train_y_predicted)   
            spc = alm_fun.spc_cal(core_train_y, core_train_y_predicted)
            core_train_score_df['rmse'] = rmse
            core_train_score_df['pcc'] = pcc
            core_train_score_df['spc'] = spc
                
        if ml_type == "classification_binary":             
            if shap_test_interaction == 1:
                X = xgb.DMatrix(test_x)
                shap_output_test_interaction = self.estimator.get_booster().predict(X, ntree_limit=-1, pred_interactions=True)  
            else:
                shap_output_test_interaction = None
                                                     
            if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
                test_y_predicted = np.array(list(np.squeeze(test_x[features])))
            else:
                try:
                    test_y_predicted = self.estimator.predict_proba(test_x[features])[:, 1]  
                except:
                    test_y_predicted = self.estimator.predict(test_x[features])             
                if self.prediction_transformation is not None:
                    test_y_predicted = self.prediction_transformation(test_y_predicted)   
                

            test_score_df = pd.DataFrame(np.zeros(10), index=['size', 'prior', 'auroc', 'auprc','aubprc','up_auprc', 'pfr','bpfr','rfp','brfp']).transpose()   
            if  len(np.unique(test_y)) ==  1: 
                test_score_df['size'] = len(test_y)
                test_score_df['auroc'] = np.nan
                test_score_df['auprc'] = np.nan
                test_score_df['aubprc'] = np.nan
                test_score_df['up_auprc'] = np.nan
                test_score_df['prior'] = np.nan
                test_score_df['pfr'] = np.nan
                test_score_df['rfp'] = np.nan
                test_score_df['bpfr'] = np.nan
                test_score_df['brfp'] = np.nan
                test_score_df['logloss'] = np.nan
            else:                
                [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(test_y, test_y_predicted)
                test_score_df['size'] = len(test_y)
                test_score_df['auroc'] = metric['auroc']
                test_score_df['auprc'] = metric['auprc']
                test_score_df['aubprc'] = metric['aubprc']
                test_score_df['up_auprc'] = metric['up_auprc']
                test_score_df['prior'] = metric['prior']
                test_score_df['pfr'] = metric['pfr']
                test_score_df['rfp'] = metric['rfp']
                test_score_df['bpfr'] = metric['bpfr']
                test_score_df['brfp'] = metric['brfp']
                test_score_df['logloss'] = metric['logloss']
             
            #get the shap value for all training data
            if shap_train_interaction == 1:
                X = xgb.DMatrix(train_x)
                shap_output_train_interaction = self.estimator.get_booster().predict(X, ntree_limit=-1, pred_interactions=True)  
            else:
                shap_output_train_interaction = None                    
                                
             
            if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
                core_train_y_predicted = np.array(list(np.squeeze(core_train_x[features])))
            else:
                try:
                    core_train_y_predicted = self.estimator.predict_proba(core_train_x[features])[:, 1]
                except:
                    core_train_y_predicted = self.estimator.predict(core_train_x[features]) 
            
                if self.prediction_transformation is not None:
                    core_train_y_predicted = self.prediction_transformation(core_train_y_predicted) 
                    

            core_train_score_df = pd.DataFrame(np.zeros(10), index=['size', 'prior', 'auroc', 'auprc','aubprc','up_auprc', 'pfr','bpfr','rfp','brfp']).transpose()                                               
            if  len(np.unique(core_train_y)) ==  1: 
                core_train_score_df['size'] = len(core_train_y)
                core_train_score_df['auroc'] = np.nan
                core_train_score_df['auprc'] = np.nan
                core_train_score_df['aubprc'] = np.nan
                core_train_score_df['up_auprc'] = np.nan
                core_train_score_df['prior'] = np.nan
                core_train_score_df['pfr'] = np.nan
                core_train_score_df['rfp'] = np.nan
                core_train_score_df['bpfr'] = np.nan
                core_train_score_df['brfp'] = np.nan
                core_train_score_df['logloss'] = np.nan
            else:   
                [best_y_predicted, metric, multiclass_metrics] = alm_fun.classification_metrics(core_train_y, core_train_y_predicted)
                core_train_score_df['size'] = len(core_train_y)
                core_train_score_df['auroc'] = metric['auroc']
                core_train_score_df['auprc'] = metric['auprc']
                core_train_score_df['aubprc'] = metric['aubprc']
                core_train_score_df['up_auprc'] = metric['up_auprc']
                core_train_score_df['prior'] = metric['prior']
                core_train_score_df['pfr'] = metric['pfr']
                core_train_score_df['rfp'] = metric['rfp']
                core_train_score_df['bpfr'] = metric['bpfr']
                core_train_score_df['brfp'] = metric['brfp']      
                core_train_score_df['logloss'] = metric['logloss']

        if ml_type == "classification_multiclass":
            test_y_predicted_probs = self.estimator.predict_proba(test_x[features])  
            test_y_predicted = self.estimator.predict(test_x[features])
            core_train_y_predicted_probs = self.estimator.predict_proba(core_train_x[features])  
            core_train_y_predicted = self.estimator.predict(core_train_x[features])
            
            if self.prediction_transformation is not None:
                test_y_predicted = self.prediction_transformation(test_y_predicted) 
            
            if self.prediction_transformation is not None:
                core_train_y_predicted = self.prediction_transformation(core_train_y_predicted) 
            
            core_train_score_df = pd.DataFrame(np.zeros(1), index=['neg_log_loss']).transpose()                             
            core_train_score_df['neg_log_loss'] = alm_fun.get_classification_metrics('neg_log_loss', 4, train_y, core_train_y_predicted_probs)
               
            test_score_df = pd.DataFrame(np.zeros(1), index=['neg_log_loss']).transpose()                             
            test_score_df['neg_log_loss'] = alm_fun.get_classification_metrics('neg_log_loss', 4, test_y, test_y_predicted_probs) 
            
  
        core_train_score_df = round(core_train_score_df, self.round_digits)
        test_score_df = round(test_score_df, self.round_digits)
        test_y_predicted = pd.Series(test_y_predicted,index = test_x.index)
        core_train_y_predicted =  pd.Series(core_train_y_predicted,index = core_train_x.index)   

        #####********************************************************************************************
        # Return the result dictionary
        #####********************************************************************************************   
        return_dict = {}
        return_dict ['train_y_predicted'] = core_train_y_predicted
        return_dict ['train_y_truth'] = core_train_y
        return_dict ['train_score_df'] = core_train_score_df
        return_dict ['test_y_predicted'] = test_y_predicted
        return_dict ['test_y_truth'] = test_y
        return_dict ['test_y_index'] = test_index
        return_dict ['test_score_df'] = test_score_df
        return_dict ['feature_importance'] = feature_importance.transpose().sort_values([0])
        return_dict ['shap_output_test_interaction'] = shap_output_test_interaction
        return_dict ['shap_output_train_interaction'] = shap_output_train_interaction
        return_dict ['model'] = self.estimator
        
        if (self.estimator == None) | ((self.single_feature_as_prediction == 1) & (len(features) == 1)):
            return_dict['tuned_tree_num'] = 0
        else:
            if tune_tree_num == 1:
                return_dict['tuned_tree_num'] = len(self.estimator.evals_result()['validation_0'][eval_obj]) - 50
            else:
                return_dict['tuned_tree_num'] = self.estimator.n_estimators
                
        #return the test dataframe in the case some features were engineered        
        if if_feature_engineer:
            predicted_test = test.copy()        
            predicted_test[dependent_variable] = test_y_predicted
        else:
            predicted_test = None
            
        return_dict ['predicted_df'] = predicted_test        
        return (return_dict)        