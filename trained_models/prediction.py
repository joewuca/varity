import xgboost as xgb
import pandas as pd

# Load the VARITY_R and VARITY_ER model
model_R = xgb.XGBClassifier()
model_R.load_model('VARITY_R.model')
model_ER = xgb.XGBClassifier()
model_ER.load_model('VARITY_ER.model')

# Define the features
ids =   ['p_vid','aa_pos','aa_ref','aa_alt']
features = ['provean_score','sift_score','evm_epistatic_score','integrated_fitCons_score','LRT_score','GERP_RS','phyloP30way_mammalian','phastCons30way_mammalian','SiPhy_29way_logOdds','blosum100','in_domain','asa_mean','aa_psipred_E','aa_psipred_H','aa_psipred_C','bsa_max','h_bond_max','salt_bridge_max','disulfide_bond_max','covelent_bond_max','solv_ne_abs_max','mw_delta','pka_delta','pkb_delta','pi_delta','hi_delta','pbr_delta','avbr_delta','vadw_delta','asa_delta','cyclic_delta','charge_delta','positive_delta','negative_delta','hydrophobic_delta','polar_delta','ionizable_delta','aromatic_delta','aliphatic_delta','hbond_delta','sulfur_delta','essential_delta','size_delta']

# Load the target set
target_set = pd.read_csv('BRCA1_features.csv')

# Predict the probability of pathogenicity
model_R_probs = model_R.predict_proba(target_set[features])[:, 1] 
model_ER_probs = model_ER.predict_proba(target_set[features])[:, 1]
results = target_set[ids]
results['VARITY_R'] = model_R_probs
results['VARITY_ER'] = model_ER_probs
results.to_csv('BRCA1_VARITY_predictions.csv', index=False)

print ("OK") 

            