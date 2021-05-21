# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:28:16 2021

@author: ll17354
"""
import sys
# tell the tests what folder LogisticRegressionFirth_sklearnestimator.py is in
sys.path.insert(1, r"C:\Users\ll17354\OneDrive - University of Bristol\MyFiles-Migrated\Documents\Scripts\LogisticRegressionFirth_sklearnestimator")
import LogisticRegression_Firth as FLR
import pandas as pd
from pathlib import Path
import pytest
import numpy as np
import traceback
import logging

# import and define training data set for tests
file = Path(r"test_data.csv")
df = pd.read_csv(file)

y = df.loc[:,'Regime.Bool']
X_LRtrain = df.loc[:,'Intercept':'f5.rolling_skewdist']
X_LRvalid = df.loc[:,'f1.rolling_cov_dist':'f5.rolling_skewdist'] 

# check functionality
FLR = FLR.Firth_LogisticRegression()
FLR_fit = FLR.fit(X_LRtrain, y)

# FLR.fit attributes tests
coefs = FLR_fit.coef_
coefs_expect = np.array([[3.588102105400064801e+00, 
                          -6.161874480153793243e+00,	
                          -8.430385150099156135e+02,	
                          3.670537270425360475e+03,	
                          -1.232422932786114700e+00]])

intercept = FLR_fit.intercept_
intercept_expect = -0.6611830408339714

fitll = FLR_fit.fitll_ 
fitll_expect = -1705.8938627542386

beta = FLR_fit.beta_
beta_expected = np.array([-0.6611830408339714, 
                          3.588102105400065, 
                          -6.161874480153793, 
                          -843.0385150099156, 
                          3670.5372704253605, 
                          -1.2324229327861147])

bse = FLR_fit.bse_
bse_expected = np.array([2.054303594037928671e-01,
                        9.675633854431319181e-01,
                        1.992934124324673872e+00,
                        1.027586521393347766e+03,
                        1.532793562879854790e+03,
                        2.831875382667899954e-01])

@pytest.mark.parametrize("attr, attr_expect", 
                          [(coefs, coefs_expect),
                           (intercept, intercept_expect),
                           (fitll, fitll_expect),
                           (beta, beta_expected),
                           (bse, bse_expected)]
                         )
def test_FLRfit_attributes(attr, attr_expect):
    assert np.allclose(attr, attr_expect)

# check statistical test run
FLR_LRT_pvals = FLR.test_likelihoodratio(X_LRtrain, y)
FLR_LRT_pvals_expect = np.array([0.00021561144352747857, 
                                 0.00020562254050106286, 
                                 0.004604294943848374, 
                                 1, 
                                 1, 
                                 2.9259947464566908e-06])

FLR_wald_pvals = FLR.test_wald()
FLR_wald_pvals_expect = np.array([0.001288510963811218, 
                                  0.00020858147032321028,
                                  0.001989062380010953, 
                                  0.41198449245706725, 
                                  0.016635255558943962, 
                                  1.3492133116077554e-05])

# check internal sklearn functions run with class
@pytest.mark.parametrize("attr, pvals_expect", 
                          [(FLR_LRT_pvals, FLR_LRT_pvals_expect),
                           (FLR_wald_pvals, FLR_wald_pvals_expect)]
                         )
def test_LRT_wald_tests(attr, pvals_expect):
    assert np.allclose(attr, pvals_expect)
    
@pytest.mark.parametrize("FLR, X_LRtrain, X_LRvalid, y",
                         [(FLR, X_LRtrain, X_LRvalid, y)])
def test_sklearnfunctionsrun(FLR, X_LRtrain, X_LRvalid, y):
    try:
        FLR_fit = FLR.fit(X_LRtrain, y)
        FLR_fit.get_params()
        FLR_fit.predict(X_LRvalid)
        FLR_fit.predict_log_proba(X_LRvalid)
        FLR_fit.predict_proba(X_LRvalid)
        FLR_fit.score(X_LRvalid, y)
        assert True
    except Exception as e:
        logging.error(traceback.format_exc())
        assert False
        # Logs the error appropriately. 
    

