#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:32:06 2025

@author: sergej
"""
# %% ==========================================================================
# Preprocess data
# =============================================================================
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import os
# Parse data
path = os.path.dirname(__file__)+"/"
os.chdir(path)
data = pd.read_csv("data_beh/datall_cat.csv")

# Data modifications
data['Reward'] = np.where(data['x12_continuous_energy_trial_end'] == 0, -1, 0).astype(np.float32)
data['dead'] = np.where(data['x6_continuous_energy_trial_start'] == 0, 1, 0).astype(np.float32)
# data.loc[data['x14_p_foraging_gain'] > 1, 'x14_p_foraging_gain'] = 0
data['x7_weather_type'] = data['x7_weather_type']-1
data['x7_weather_type'] = data['x7_weather_type'].apply(lambda x: np.random.randint(0, 2) if x == -1 else x)

# Identify categorical and continuous features
categorical_features = ["x7_weather_type", "BNW_conditions"]
continuous_features = ["x17_horizon_correct_adjusted", "x6_continuous_energy_trial_start",
                "x59_weather_1_p_gain", "x60_weather_2_p_gain", 
                "x57_weather_1_gain_magnitude", "x58_weather_2_gain_magnitude",
                "x11_choice", "Reward", "x12_continuous_energy_trial_end"]#,
                # "dead"]

# Apply OneHotEncoder to categorical features
encoder = OneHotEncoder(drop=None, sparse_output=False)
encoded_categorical = encoder.fit_transform(data[categorical_features])
# Convert encoded features to DataFrame
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_categorical, columns=encoded_feature_names)
# Encode absorbing death state
encoded_df.iloc[:,2:].loc[data['x6_continuous_energy_trial_start'] == 0, :] = 0

# Add categorical
processed_data = pd.concat([data[["x1_id", "x4_index_forests", "x2_session"]], 
                            data[continuous_features], encoded_df.iloc[:, :]], axis=1)

# Group data by Participant and Episode
grouped = processed_data.groupby(["x1_id", "x4_index_forests", "x2_session"], sort=False)

# %% ==========================================================================
# Encode trajectories
# =============================================================================
from imitation.data.types import TrajectoryWithRew
# from imitation.data.types import Transitions
from imitation.data import rollout
from typing import List
import torch

trajectories: List[TrajectoryWithRew] = []
Rs = []  # Raw rewards for evaluation
for _, group in grouped:
    group.reset_index(drop=True, inplace=True)

    # Observations
    states = group[["x59_weather_1_p_gain", "x60_weather_2_p_gain"] +
                    list(encoded_df.columns)[:] +
                    ["x17_horizon_correct_adjusted", 
                    "x57_weather_1_gain_magnitude", 
                    "x58_weather_2_gain_magnitude",
                    "x6_continuous_energy_trial_start"]].values#,
                    # "dead"]].values
    # Actions
    actions = group["x11_choice"].values
    
    # Override "days left" with horizon
    repeated_entries = []
    states[:,-4] = states.shape[0] # Set variable

    # Fake transitions for shorter trajectories
    while states.shape[0] < 5:
        # obs
        last_row = states[-1:]  # Select the last row
        repeat_count = 5 - states.shape[0]  # Measure how many copies
        repeated_rows = np.repeat(last_row, repeat_count, axis=0)   # Repeat last row (no change)
        # if group.iloc[-1]["x6_continuous_energy_trial_start"] != 0:
        #     repeated_rows[:,-1] = 0
        # else:
        #     repeated_rows[:,-1] = 1
        repeated_rows[:, np.random.randint(0, 2, size=1)[0]+2] = 1  # Random weather draws
        states = np.vstack((states, repeated_rows))
        # act
        repeated_entries = np.full(repeat_count, np.random.randint(0, 2, size=1)[0])  # Repeat reward until reaching fake horizon
        actions = np.concatenate((actions, repeated_entries))  # Append to original array
        # info
        repeated_entries = [{'success': False} for x in range(repeat_count)]  # Last step of each episode is terminal
    
    # Actions
    actions = torch.tensor(actions, dtype=torch.float32).view(-1, 1)
    actions = np.array(actions, dtype=np.float32)
        
    # Convert horizon to days left
    row_count = states.shape[0]
    ascending_col = np.arange(1, row_count + 1).reshape(-1, 1) - 1
    horizon_adj = np.clip(np.array(states[:,-4]).reshape(-1,1) - ascending_col, 0, 5)
    states[:,-4] = horizon_adj[:,0]

    ## Categorical variables
    categ = {col: 0 for col in encoded_df.columns}
    # Encode random weather last day
    categ[list(categ.keys())[np.random.randint(0, 2, size=1)[0]]] = 1
    if group.iloc[-1]["x12_continuous_energy_trial_end"] != 0:
        # Encode ternary state last day
        bnw = 1
        if group.iloc[-1]["x12_continuous_energy_trial_end"] == 1:
            bnw = 2
        elif group.iloc[-1]["x12_continuous_energy_trial_end"] > 1:
            bnw = 0
        categ[list(categ.keys())[bnw+2]] = 1 
    
    # Add last day/outcome
    last_state = [
        group.iloc[-1]["x59_weather_1_p_gain"], group.iloc[-1]["x60_weather_2_p_gain"]] + list(
        list(categ.values())[:]) + [
        0,
        group.iloc[-1]["x57_weather_1_gain_magnitude"], 
        group.iloc[-1]["x58_weather_2_gain_magnitude"],
        group.iloc[-1]["x12_continuous_energy_trial_end"]#,
        # np.where(group.iloc[-1]["x12_continuous_energy_trial_end"] == 0, 1, 0)
        ]
    states = np.append(states, [last_state], axis=0)
    
    # Rewards
    rewards = np.where(states[:,-1] == 0, -1, 0).astype(np.float32)[:-1]
    if -1 in rewards:   # rewards accounts for next day
        rewards[np.where(rewards == -1)[0][0]-1] = -1

    # terminal = [True if data.iloc[x]['x17_horizon_correct_adjusted'] == 1 else False for x in range(len(data))]  # Last step of each episode is terminal
    done = [False] * (len(actions)-1) + [True]  # Last step of each episode is terminal
    # Additional information
    infos = repeated_entries + [{'success': True} if group.iloc[x]['x17_horizon_correct_adjusted'] == 1 and group.iloc[x]['x12_continuous_energy_trial_end'] != 0 else {'success': False} for x in range(len(group))]
    # Assemble trajectory
    trajectories.append(TrajectoryWithRew(obs=states, acts=actions, rews=rewards, terminal=done, infos=infos))
    
    # Manually extract raw rewards (used for evaluation)
    Rs.append(rewards)

transitions = rollout.flatten_trajectories(trajectories)

# %% ==========================================================================
# Aggregate data for evaluation
# =============================================================================
agg_R = [sum(Rs[i]) for i in range(len(Rs))]
agg = data.groupby(["x1_id", "x4_index_forests", "x2_session"]).sum()
dec = data.groupby(["x1_id", "x4_index_forests", "x2_session"]).agg({'x11_choice': 'mean'})
agg['av_choice'] = dec['x11_choice']
agg['rewards'] = agg_R

summary_stats = agg.groupby("x1_id")["rewards"].agg(["mean", "std", "count"])
summary_stats["SE"] = summary_stats["std"] / np.sqrt(summary_stats["count"])


# agg = agg.groupby(["x1_id"]).mean()
# agg_rew = agg['rewards']
# agg_act = agg['av_choice']

# Rews = [np.sum(np.array(Rs)[i,:]) for i in range(np.array(Rs).shape[0])]


# # Saving the trajectory data as a .npz file
# np.savez("data_beh/trajectories.npz", *[t for t in trajectories])
