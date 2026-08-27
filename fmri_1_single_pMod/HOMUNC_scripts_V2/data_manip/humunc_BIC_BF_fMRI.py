#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:24:21 2023

@author: sergej
"""
# %%
# =============================================================================
import glob
import pandas as pd
import numpy as np
from copy import deepcopy
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy.contrasts import Treatment
import os
path = os.path.dirname(os.path.abspath(__file__)) + '/'
os.chdir(path)

# =============================================================================
# Compute uncertainty fit difference
# =============================================================================
mdlName = [
    'x17_horizon_correct_adjusted',
    'x13_gain_magnitude',
    'x24_pseudo0x2Doptimal_horizon0x2D1',
    'x37_binary_energy',
    'x6_continuous_energy_trial_start',
    'x14_p_foraging_gain',
    'x19_wait_when_safe',
    'x7_weather_type',
    # 'x25_expected_energy_wO_boundaries',
    # 'x40_expected_energy_w_boundaries',
    # 'x41_expected_energy_change',
    # 'x56_rule0x26PGain3WWS',
    # 'p_delta',
    # 'weather_dynamic',
    # 'p_delta_dynamic',
    'BNW_conditions',
    'x22_optimal_policy'
]

N_trials = []
bic_all = []  # BICs for all models and subjects
for itr, fle in enumerate(glob.glob(path + "/datall_*.csv")):
    
    # Import subject data
    dt = pd.read_csv('datall_' + str(itr+1) + '.csv')
    # Dynamic weather state policy
    dt['weather_dynamic'] = [dt.iloc[i]['x7_weather_type']-1 if dt.iloc[i]['x6_continuous_energy_trial_start'] != 1 else 1 for i in range(len(dt))]
    dt['weather_dynamic'] = [dt.iloc[i]['weather_dynamic'] if dt.iloc[i]['x19_wait_when_safe'] != 1 else 0 for i in range(len(dt))]
    # Get other policies
    delta_p = []
    delta_p_dyn = []
    condi_s = []
    for index, row in dt.iterrows():
        if row['x14_p_foraging_gain'] == row['x59_weather_1_p_gain']:
            delta_p.append(row['x14_p_foraging_gain'] - row['x60_weather_2_p_gain'])
            delta_p_dyn.append(row['x14_p_foraging_gain'] - row['x60_weather_2_p_gain'])
        else:
            delta_p.append(row['x14_p_foraging_gain'] - row['x59_weather_1_p_gain'])
            delta_p_dyn.append(row['x14_p_foraging_gain'] - row['x59_weather_1_p_gain'])
        # Three conditions model
        condi_s.append(0.5)   # 'random'
        if row['x6_continuous_energy_trial_start'] == 1 or row['x19_wait_when_safe'] == 1:
            delta_p_dyn[index] = (-1)*(1-row['x14_p_foraging_gain'])
            condi_s[index] = 0  # wait-when-safe
            if row['x6_continuous_energy_trial_start'] == 1:
                delta_p_dyn[index] = row['x14_p_foraging_gain']
                condi_s[index] = 1  # binary energy state
    dt['p_delta'] = delta_p
    dt['p_delta_dynamic'] = delta_p_dyn
    dt['BNW_conditions'] = condi_s
    
    N_trials.append(len(dt))

    # Data filtering
    dtU = dt[dt['x9_button_pressed'].isnull() == False]
    dtU = dtU[dtU['x6_continuous_energy_trial_start'] != 0]
    
    dtC = pd.DataFrame({'A': np.arange(len(dtU))})
    bic = []  # BIC per model
    for nme in mdlName:
        
        dtU[nme + '_fit'] = np.nan
        
        # m1_name = 'BNW_conditions'
        # m1 = list(dtU[m1_name])
        # m1 = np.array(m1)

        # Prepare data
        # Get model
        m_raw = list(dtU[nme])
        # Get reponses
        respo = list(np.array(dtU["x11_choice"]))
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
        
# =============================================================================
#       Univariate model (numeric)
# =============================================================================
        mod_fin = model
        mod_fin = sm.add_constant(mod_fin)
        
# # =============================================================================
# #       Multivariate model (numeric)
# # =============================================================================
#         mod_fin = np.c_[ m1, model ]
#         mod_fin = sm.add_constant(mod_fin)

# # =============================================================================
# #       Factorial design
# # =============================================================================
#         # Contrast code for m1
#         levels = [1,2,3]
#         contrast = Treatment(reference=0).code_without_intercept(levels)
#         c_mat = contrast.matrix[pd.DataFrame(m1)-1, :]
#         c_mat = np.array(c_mat)
#         mod_fin = np.c_[ c_mat[:,0], model ]              # add a column
        # result = mdl.fit_regularized()
        
        # Run logit
        mdl = sm.Logit(respo, sm.tools.tools.add_constant(mod_fin))
        # Fit with BFGS to handle singularity in design matrix
        exog = mdl.exog
        u, s, vt = np.linalg.svd(exog, 0)
        result = mdl.fit()
        bic.append(result.bic)

        # Expand data with fits and uncertainties
        if len(result.predict()) != len(dtU):
            dtC[nme + '_fit'] = [np.nan] + list(result.predict())
        else:
            dtC[nme + '_fit'] = result.predict()

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
bicsRAW.to_csv(path + '/BICs.csv', index=False)

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
    # 'expected energy',
    # 'x40_expected_energy_w_boundaries',
    # 'expected energy change',
    # 'dynamic $\\mathit{p}$ success',
    # '$\\mathit{p}$ delta',
    # 'dynamic weather state',
    # 'dynamic $\\mathit{p}$ delta',
    'ternary state',
    'optimal policy values'
]
# for i in range(len(mdlName)):
#     mdlName[i] = 'ternary state + '+mdlName[i]
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
ax.set_title('BF (lower is better)',
              loc='left', size=46)
plt.xlabel(" ", fontsize=40)
# ax.autoscale(enable=True) 