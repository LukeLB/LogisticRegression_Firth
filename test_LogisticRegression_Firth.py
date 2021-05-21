# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:28:16 2021

@author: ll17354
"""
import sys
sys.path.insert(1, r"C:\Users\ll17354\OneDrive - University of Bristol\MyFiles-Migrated\Documents\CheckRisk\Code\firthregression_sklearnestimator")
import LogisticRegression_Firth as FLR
import pandas as pd
from pathlib import Path
import numpy as np

# import and define training data set
file = Path(r"C:\Users\ll17354\OneDrive - University of Bristol\MyFiles-Migrated\Documents\CheckRisk\Code\week3_4\features_df_wRegimeDates.csv")
df = pd.read_csv(file)

y = df.loc[:,'Regime.Bool']
X_LRtrain = df.loc[:,'Intercept':'f5.rolling_skewdist']
X_LRvalid = df.loc[:,'f1.rolling_cov_dist':'f5.rolling_skewdist'] 

# check functionality
FLR = FLR.Firth_LogisticRegression()
FLR_fit = FLR.fit(X_LRtrain, y)
FLR_LRT_pvals = FLR.test_likelihoodratio(X_LRtrain, y)
FLR_wald_pvals = FLR.test_wald()
params = FLR_fit.get_params()
predict = FLR_fit.predict(X_LRvalid)
predict_log = FLR_fit.predict_log_proba(X_LRvalid)
predict_probs = FLR_fit.predict_proba(X_LRvalid)
predict_score = FLR_fit.score(X_LRvalid, y)
