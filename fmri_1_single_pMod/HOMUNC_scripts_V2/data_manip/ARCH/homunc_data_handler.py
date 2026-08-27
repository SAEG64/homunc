#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:21:42 2024

@author: sergej
"""

# =============================================================================
# Requirements
# =============================================================================
import pandas as pd
import numpy as np
import glob
import sys
import os
path = os.path.dirname(os.path.realpath(sys.argv[0]))
os.chdir(path)

# # =============================================================================
# # Concat data
# # =============================================================================
# # Create empty pandas data frame
# d = pd.read_csv(path + "/datall_1.csv")
# catData = pd.DataFrame(columns = list(d.columns.values))

# datall = []
# for itr, fle in enumerate(glob.glob(path + "/datall_*.csv")):
#     dt = pd.read_csv('datall_' + str(itr+1) + '.csv')
#     print('subjects', int(dt.iloc[0]['x1_id']), len(dt))
#     # dt['subject_ID'] = itr+1
#     datall.append(dt)
#     catData = pd.concat(datall, ignore_index = True)

# # # Modify optimal policy
# # catData['optimal_policy_cap'] = [catData.iloc[i]['x22_optimal_policy'] if catData.iloc[i]['x6_continuous_energy_trial_start'] != 1 else np.max(catData['x22_optimal_policy']) for i in range(len(catData))]
# # catData['optimal_policy_cap'] = [catData.iloc[i]['optimal_policy_cap'] if catData.iloc[i]['x19_wait_when_safe'] != 1 else np.min(catData['x22_optimal_policy']) for i in range(len(catData))]

# catData.to_csv(path + '/data_group/datall_cat.csv', index=False)  
# # list(catData.columns.values)

# # check = catData[catData['x19_wait_when_safe'] == 0]


# =============================================================================
# Split data into subjects and sessions mat
# =============================================================================
import scipy.io as sio
d = pd.read_csv(path + "/data_group/datall_cat.csv")
import matplotlib.pyplot as plt
dClean = d[d['p_delta'] < 1]
plt.hist(dClean['p_delta_dynamic']) # optimal policy uncertainty hist
# plt.hist(ucm.iloc[:,2]) # p success uncertainty hist
# d_s = [s_file for sbj, s_file in d.groupby('x1_id')]


# # Split data into subjects and sessions
# sbj_file = []
# for i in range(len(d_s)):
#     d_sess = [sess_file for sbj, sess_file in d_s[i].groupby('x2_session')]
#     sbj_file.append(d_sess)

# # Make all subjects directory
# datall_single = 'data_single'
# if not os.path.exists(datall_single):
#     os.makedirs(datall_single)
    
# for i in range(len(sbj_file)):
    
#     # Make single subjects directory
#     sbj = sbj_file[i][0]['x1_id'].iloc[0]
#     sbj_name = 'subject' + str(sbj)
#     sbj_folder = datall_single + '/' + sbj_name
#     if not os.path.exists(sbj_folder):
#         os.makedirs(sbj_folder)
        
#     for j in range(len(sbj_file[i])):
        
#         # Make session directory
#         sess_name = '_session_' + str(j+1)
#         sbj_sess = sbj_name + sess_name
#         # if not os.path.exists(sbj_folder + '/' + sbj_sess):
#         #     os.makedirs(sbj_folder + '/' + sbj_sess)
        
#         # Export file
#         sio.savemat(os.path.join(sbj_folder + '/', sbj_name + sess_name + '.mat'), sbj_file[i][j])