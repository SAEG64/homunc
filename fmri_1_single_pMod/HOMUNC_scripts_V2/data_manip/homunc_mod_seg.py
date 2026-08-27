#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:24:21 2023

@author: sergej
"""
# %%
import glob
import pandas as pd
import numpy as np
from copy import deepcopy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
from sympy import *
import sys
import os
# Set path to the directory containing this script
path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)

# Logit derivative for RT regression
x = symbols('x1')
f = 1/(1+math.e**(-x))
fDiff = Derivative(f, x)

# =============================================================================
# Compute uncertainty fit difference
# =============================================================================
beta_mean = True
mdlName = [
    # 'x22_optimal_policy',
    # 'x14_p_foraging_gain',
    'BNW_conditions',
    'p_delta'
]
ucm = pd.DataFrame()
beta0_dist = [[],[]]
beta1_dist = [[],[]]
beta2_dist = [[],[]]
if beta_mean == True:
    mbetas_0 = [0.34512157572919344] # intercept
    mbetas_1 = [0.7483707810898174]  # p delta
    mbetas_2 = [0.5327872624248698]  # BNW conditions
    # mbetas_0 = [-0.17231275444517385, 0.9903778816177154]
    # mbetas_1 = [9.549271134771152, 5.838668194542303]
datall = []
for itr, fle in enumerate(glob.glob(path + "/datall_*.csv")):
    
    # Get data
    d = pd.read_csv('datall_' + str(itr+1) + '.csv')
    dt = d.iloc[:,:60]
    
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
    
    # Filtered dataframe
    dtU = dt[dt['x9_button_pressed'].isnull() == False]
    dtU = dtU[dtU['x6_continuous_energy_trial_start'] != 0]
    
    # Append model and uncertainty fits
    dtC = pd.DataFrame({'A': np.array(dtU.index)})
    i_b = 0
    for nme in mdlName[:-1]:
        
        m1_name = 'p_delta'
        m1 = list(dtU[m1_name])
        model_copy1 = deepcopy(m1)
        m1 = np.array(m1)
        m1 = sm.add_constant(m1)
        
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
        model_copy2 = deepcopy(model)
        # Add constant for intercept
        model = np.array(model)
        model = sm.add_constant(model)

        # Run logit
        # mdl = sm.Logit(respo, m1+model)
        mdl = sm.OLS(dtU['x11_choice'], sm.tools.tools.add_constant(dtU[['p_delta','BNW_conditions']]))
        # # Fit with BFGS to handle singularity in design matrix
        # exog = mdl.exog
        # u, s, vt = np.linalg.svd(exog, 0)
        # result = mdl.fit(method="bfgs", maxiter=100)
        result = mdl.fit()

        # Compute model uncertainties
        # response times
        rt = [np.nan if np.isnan(m_raw[i]) else dtU.iloc[i]['x26_logRT']
              for i in range(0, len(m_raw[:]))]
        rt = [x for x in rt if np.isnan(x) == False]
        rt = np.array(rt)
        
        # Coefs for uncertainty
        if beta_mean == True:
            beta0 = mbetas_0[i_b]
            beta1 = mbetas_1[i_b]
            beta2 = mbetas_2[i_b]
        else:
            beta0 = result.params[0]
            beta1 = result.params[1]
            beta2 = result.params[2]
        # Get subject coefs
        beta0_dist[i_b].append(result.params[0])
        beta1_dist[i_b].append(result.params[1])
        beta2_dist[i_b].append(result.params[2])
        i_b += 1
        
        # Compute model derivative
        dv = [beta0 + beta1*model_copy1[i] + beta2*model_copy2[i]
              for i in range(0, len(model_copy1))]
        uncer = [float(fDiff.doit().subs({x: dv[i]}))
                  for i in range(0, len(dv))]
        # Copy regressor
        uncertainty = deepcopy(uncer)
        # Run linear regression
        uncer = np.array(uncer)
        uncer = sm.add_constant(uncer)
        # Predict RT
        uc_mdl = sm.OLS(rt, uncer).fit()
        uc_result = uc_mdl.predict(uncer)

        # Expand data with fits and uncertainties
        dtC[m1_name+'_'+nme + '_fit'] = result.predict()
        dtC[m1_name+'_'+nme + '_uncertain'] = uncertainty
        dtC[m1_name+'_'+nme + '_uncertain_fit'] = list(uc_result)

    # Concat and save fitted data
    dtC.set_index('A', inplace=True)
    dtC = dtC.reindex(np.arange(len(dt)), fill_value = np.nan)
    dt_REV = pd.concat([dt, dtC], axis=1).reset_index(drop=True)
    dt_REV.to_csv(path + "/datall_" + str(itr+1) + ".csv", index = False)
    
    # Make cat datta file
    datall.append(dt_REV)
    catData = pd.concat(datall, ignore_index = True)
    
    # Dataframe with subject ID and uncertainty model values
    ucm_id = pd.DataFrame()
    ucm_id['id'] = dt_REV['x1_id']
    # ucm_id['OP_uncertain'] = dt_REV[mdlName[0] + '_uncertain']
    # ucm_id['pDel_uncertain'] = dt_REV[mdlName[1] + '_uncertain']
    ucm_id['pDel_uncertain'] = dt_REV[m1_name+'_'+mdlName[0] + '_uncertain']
    ucm = pd.concat([ucm, ucm_id])

# Export joined data
catData.to_csv(path + '/data_group/datall_cat.csv', index=False)  

# Plot uncertainty distribution to evaluate coef sampling effect
import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
# plt.hist(ucm['OP_uncertain']) # optimal policy uncertainty hist
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
plt.hist(ucm['pDel_uncertain']) # p success uncertainty hist
# Compute mean betas
mbeta0 = np.mean(beta0_dist[0])
mbeta1 = np.mean(beta1_dist[0])
mbeta2 = np.mean(beta2_dist[0])
# mbeta0_ps = np.mean(beta0_dist[1])
# mbeta1_ps = np.mean(beta1_dist[1])

# Get column names in matlab format
cols = [{catData.columns[i]} for i in range(len(catData.columns))]
print("=============================")
print("Columns to add in SPM scripts")
print()
print(cols[56:])
print()
print("=============================")

# =============================================================================
# Behavioral analysis of three choice conditions "BES", "Norm" and "WWS"
# =============================================================================
# Get column with 3 choice conditions
# catData['BNW_condition'] = ['delta states' if catData.iloc[i]['x6_continuous_energy_trial_start'] != 1 else 'energy one states' for i in range(len(catData))]
# catData['BNW_condition'] = [catData.iloc[i]['BNW_condition'] if catData.iloc[i]['x19_wait_when_safe'] != 1 else 'wait-when-safe states' for i in range(len(catData))]
# Filtering and correcting pandas' inprecisions
catFilt = catData[catData['p_delta']<0.6] # criteria applies by definition
catFilt['p_delta_dynamic'] = np.round(catFilt['p_delta_dynamic'],1)
catFilt['p_delta'] = np.round(catFilt['p_delta'],1)
catFilt['resp_count'] = 1   # Variable to compute sampling rate
# Check p delta distributions
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
plt.hist(catFilt['p_delta']) # optimal policy uncertainty hist
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
plt.hist(catFilt['p_delta_dynamic']) # optimal policy uncertainty hist

# Aggregated mean data
aggrega = catFilt.groupby(['x1_id','BNW_conditions','p_delta']).mean()
aggrega = aggrega.reset_index()
# Count occurances
agg_sum = catFilt.groupby(['x1_id','BNW_conditions','p_delta'])['resp_count'].sum()
agg_sum = agg_sum.reset_index()
# Get standard deviations and standard errors
agg_BNW = catFilt.groupby(['x1_id','BNW_conditions','p_delta'])['x26_logRT'].std()
agg_BNW = agg_BNW.reset_index()
agg_BNW = agg_BNW.rename(columns = {'x26_logRT': 'x26_logRT_std'})
agg_BNW['logRT_sem'] = agg_BNW['x26_logRT_std']/agg_sum['resp_count']
agg_BNW['RT_sem'] = np.exp(agg_BNW['logRT_sem'])
agg_BNW['choice_sem'] = np.sqrt((aggrega['x11_choice']*(1-aggrega['x11_choice']))/agg_sum['resp_count'])
agg_BNW['resp_count'] = agg_sum['resp_count']
agg_BNW['choice_mean'] = aggrega['x11_choice']
agg_BNW['logRT'] = aggrega['x26_logRT']
agg_BNW['RT'] = aggrega['x10_RT']
# Final averaging over all subjects
agg_fin = agg_BNW.groupby(['BNW_conditions','p_delta']).mean()
agg_fin = agg_fin.reset_index()

# # Debug data filtering issue
# aa = catFilt[catFilt['BNW_conditions']==2]
# aa = aa[aa['p_delta_dynamic']>0.5]
# print('energy points:', aa['x6_continuous_energy_trial_start'],'\n',
#       'curr p success (can not exceed 1):', aa['x14_p_foraging_gain'],'\n',
#       'p delta dynamic value:', aa['p_delta_dynamic'])

# Plot default settings
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
sns.set_style("white")
sns.set_palette("Paired")

# Plot condition splits for p delta with respect to p foraging
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
ci = agg_fin['choice_sem']*1.96 # confidence interval
plot_pFora = sns.regplot(
    x='p_delta', y='choice_mean', data=agg_fin, logistic=True, ci=None,
    ax=ax, label='', scatter_kws={'s':agg_fin['resp_count']*3}, 
    line_kws = {"color": "None"})
ax.errorbar(
    x='p_delta', y='choice_mean', data=agg_fin, yerr = ci, fmt='none', capsize=0, 
    zorder=1, color='C0', label=None)
sns.lineplot(x='p_delta', y='choice_mean', data=agg_fin,
        hue='BNW_conditions', ax=ax)
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
plt.ylabel("Foraging likelihood", fontsize=30)
plt.xlabel("$\\mathit{p}$ delta values", fontsize=30)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=24)

# Plot condition splits for dynamic p delta with respect to RT
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
ci = agg_fin['logRT_sem']*1.96 # confidence interval
plot_pFora = sns.regplot(
    x='p_delta', y='logRT', data=agg_fin, logistic=True, ci=None,
    ax=ax, label='', scatter_kws={'s':agg_fin['resp_count']*3}, 
    line_kws = {"color": "None"})
ax.errorbar(
    x='p_delta', y='logRT', data=agg_fin, yerr = ci, fmt='none', capsize=0, 
    zorder=1, color='C0', label=None)
sns.lineplot(x='p_delta', y='logRT', data=agg_fin,
        hue='BNW_conditions', ax=ax)
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
plt.ylabel("log(RT)", fontsize=30)
plt.xlabel("$\\mathit{p}$ delta values", fontsize=30)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=24)
plt.ylim(ymin=6.3)


# Aggregated mean data
aggrega = catFilt.groupby(['x1_id','BNW_conditions','p_delta_dynamic']).mean()
aggrega = aggrega.reset_index()
# Count occurances
agg_sum = catFilt.groupby(['x1_id','BNW_conditions','p_delta_dynamic'])['resp_count'].sum()
agg_sum = agg_sum.reset_index()
# Get standard deviations and standard errors
agg_BNW = catFilt.groupby(['x1_id','BNW_conditions','p_delta_dynamic'])['x26_logRT'].std()
agg_BNW = agg_BNW.reset_index()
agg_BNW = agg_BNW.rename(columns = {'x26_logRT': 'x26_logRT_std'})
agg_BNW['logRT_sem'] = agg_BNW['x26_logRT_std']/agg_sum['resp_count']
agg_BNW['RT_sem'] = np.exp(agg_BNW['logRT_sem'])
agg_BNW['choice_sem'] = np.sqrt((aggrega['x11_choice']*(1-aggrega['x11_choice']))/agg_sum['resp_count'])
agg_BNW['resp_count'] = agg_sum['resp_count']
agg_BNW['choice_mean'] = aggrega['x11_choice']
agg_BNW['logRT'] = aggrega['x26_logRT']
agg_BNW['RT'] = aggrega['x10_RT']
# Final averaging over all subjects
agg_fin = agg_BNW.groupby(['BNW_conditions','p_delta_dynamic']).mean()
agg_fin = agg_fin.reset_index()


# Plot condition splits for dynamic p delta with respect to p foraging
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
ci = agg_fin['choice_sem']*1.96 # confidence interval
plot_pFora = sns.regplot(
    x='p_delta_dynamic', y='choice_mean', data=agg_fin, logistic=True, ci=None,
    ax=ax, label='', scatter_kws={'s':agg_fin['resp_count']*3}, 
    line_kws = {"color": "None"})
ax.errorbar(
    x='p_delta_dynamic', y='choice_mean', data=agg_fin, yerr = ci, fmt='none', capsize=0, 
    zorder=1, color='C0', label=None)
sns.lineplot(x='p_delta_dynamic', y='choice_mean', data=agg_fin,
        hue='BNW_conditions', ax=ax)
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
plt.ylabel("Foraging likelihood", fontsize=30)
plt.xlabel("$\\mathit{p}$ delta values", fontsize=30)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=24)

# Plot condition splits for dynamic p delta with respect to RT
fig, ax = plt.subplots(figsize=(6, 6),dpi = 600)
ci = agg_fin['logRT_sem']*1.96 # confidence interval
plot_pFora = sns.regplot(
    x='p_delta_dynamic', y='logRT', data=agg_fin, logistic=True, ci=None,
    ax=ax, label='', scatter_kws={'s':agg_fin['resp_count']*3}, 
    line_kws = {"color": "None"})
ax.errorbar(
    x='p_delta_dynamic', y='logRT', data=agg_fin, yerr = ci, fmt='none', capsize=0, 
    zorder=1, color='C0', label=None)
sns.lineplot(x='p_delta_dynamic', y='logRT', data=agg_fin,
        hue='BNW_conditions', ax=ax)
ax.tick_params(bottom=True, left=True, size=5, direction= "in")
plt.ylabel("log(RT)", fontsize=30)
plt.xlabel("$\\mathit{p}$ delta values", fontsize=30)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
ax.tick_params(axis="x", labelsize=24)
ax.tick_params(axis="y", labelsize=24)
plt.ylim(ymin=6.3)