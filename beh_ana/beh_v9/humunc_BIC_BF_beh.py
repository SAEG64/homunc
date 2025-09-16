#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:24:21 2023

@author: sergej
"""
import glob
import pandas as pd
import numpy as np
from copy import deepcopy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys
import os
from scipy.io import loadmat

path = os.path.dirname(os.path.realpath(sys.argv[0]))
os.chdir(path)

# =============================================================================
# Compute uncertainty fit difference
# =============================================================================
mdlName = [
    '17_horizon_correct_adjusted',
    '13_gain_magnitude',
    '24_pseudo-optimal_horizon-1',
    '37_binary_energy',
    '6_continuous_energy_trial_start',
    '14_p_foraging_gain',
    '19_wait_when_safe',
    '7_weather_type',
    '25_expected_energy_wO_boundaries',
    '40_expected_energy_w_boundaries',
    '41_expected_energy_change',
    '56_rule & p gain 3 WWS',
    '22_optimal_policy'
]

# .mat file requirements
d_specs = loadmat(path + '/HOMUNC_data_beh_B_pilot_v1.mat')
headers = list(d_specs['header_columns'][0][i][0] for i in range(56))
sbj = [301,302,304,305,306,307,308,310,311,312,313,315,316,317,319,320,321,322,323,324,325,326,327,328,333,334,335]

# Processing loop
bic_all = []  # BICs for all models and subjects
for itr, fle in enumerate(glob.glob(path + "/beh_v9_sub_*.mat")):
    
    # Dealing with .mat file
    data_set = loadmat(path + '/beh_v9_sub_' + str(sbj[itr]) + '.mat')
    rows = data_set['__header__']
    dataVal = data_set['Z']
    dt = pd.concat(list(pd.DataFrame(data_set['Z']['choice'][0][0][0][i]) for i in range(len(data_set['Z']['choice'][0][0][0][:]))))
    dt.columns = headers
    dt['1_id'] = sbj[itr] # override tiing with subject ID
    dt = dt.reset_index(drop=True)

    # Filter data
    dtU = dt[dt['9_button_pressed'].isnull() == False]
    dtU = dtU[dtU['6_continuous_energy_trial_start'] != 0]
    
    dtC = pd.DataFrame({'A': np.arange(len(dtU))})
    bic = []  # BIC per model
    for nme in mdlName:
        
        dtU[nme + '_fit'] = np.nan

        # Prepare data
        # Get model
        m_raw = list(dtU[nme])
        # Get reponses
        respo = list(np.array(dtU["11_choice"]))
        # Correct data for eventual NaNs in model
        respo = [np.nan if np.isnan(m_raw[k]) else respo[k]
                  for k in range(0, len(m_raw))]
        model = [np.nan if np.isnan(respo[k]) else m_raw[k]
                  for k in range(0, len(respo))]
        model = [x for x in model if np.isnan(x) == False]
        respo = [x for x in respo if np.isnan(x) == False]
        respo = np.array(respo)
        model_copy = deepcopy(model)
        # Add constant for intercept
        model = np.array(model)
        model = sm.add_constant(model)

        # Run logit
        mdl = sm.Logit(respo, model)
        # Fit with BFGS to handle singularity in design matrix
        exog = mdl.exog
        u, s, vt = np.linalg.svd(exog, 0)
        result = mdl.fit()
        bic.append(result.bic)

        # Expand data with fits and uncertainties
        if len(result.predict(model)) != len(dtU):
            dtC[nme + '_fit'] = [np.nan] + list(result.predict(model))
        else:
            dtC[nme + '_fit'] = result.predict(model)

    # Concat data and model fits
    dtC.set_index('A', inplace=True)
    dtC = dtC.reindex(np.arange(len(dt)), fill_value = np.nan)
    dt_REV = pd.concat([dt, dtC], axis=1).reset_index(drop=True)
    bic_all.append(bic)
    
    # dt_REV.to_csv(path + "/datall_" + str(itr+1) + ".csv", index = False)

# Compute log-group Bayes factor
# Transpose subject-model to model-subject order
bic_all = np.array(bic_all).T
bcsums = []
for i in range(0, len(bic_all)):
    bcsums.append(sum(bic_all[i]))
# Log group Bayes Factor
bcsums = bcsums-bcsums[-1]
bcsums = pd.DataFrame(bcsums).T
bcsums.columns = mdlName

# Save BICs for PEP computation (with matlab script)
bic_all = [pd.DataFrame(li) for li in bic_all]
bicsRAW = pd.concat([pd.DataFrame(li)
                    for li in bic_all], axis=1).reset_index(drop=True)
bicsRAW.columns = mdlName

# Plotting
mdlName = [
    'remaining time-points',
    'gain magnitude',
    'pseudo optimal values',
    'binary energy state',
    'continuous energy state',
    '$\\mathit{p}$ success',
    'wait when safe',
    'weather type',
    'expected energy',
    'x40_expected_energy_w_boundaries',
    'expected energy change',
    'multi-heuristic policy',
    'optimal policy values'
]
name = mdlName
valu = bcsums.values.tolist()[0]
# Figure Size
fig, ax = plt.subplots(figsize=(16, 8))
# Increase x and y labels
ax.tick_params(axis="x", labelsize=34)
ax.tick_params(axis="y", labelsize=34)
ax.tick_params(bottom=True, left=True, size=5, direction="in")
# Horizontal Bar Plot
ax.barh(name, valu)
# ax.get_yticklabels()[-2].set_color("blue")
# Add Plot Title
ax.set_title('log group Bayes factor (BF)',
              loc='left', size=46)
plt.xlabel("BF (lower is better)", fontsize=40)
# ax.autoscale(enable=True) 