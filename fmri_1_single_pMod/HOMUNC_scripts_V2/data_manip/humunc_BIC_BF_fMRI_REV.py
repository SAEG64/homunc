#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model comparison using statsmodels formula API
"""
# %%
import glob
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
path = os.path.dirname(os.path.abspath(__file__)) + '/'
os.chdir(path)

# Define model variables
model_vars = [
    'x17_horizon_correct_adjusted',
    'x13_gain_magnitude',
    'x24_pseudo0x2Doptimal_horizon0x2D1',
    # 'x37_binary_energy',
    'x6_continuous_energy_trial_start',
    'x14_p_foraging_gain',
    # 'x19_wait_when_safe',
    'x7_weather_type',
    'BNW_conditions',
    'x22_optimal_policy'
]

# Define additional models to test
additional_models = [
    'BNW_conditions*x7_weather_type',
    'BNW_conditions*x7_weather_type + x14_p_foraging_gain'
]

# Define model descriptions (for plotting)
model_descriptions = [
    'remaining time-points',
    'gain magnitude',
    'pseudo optimal values',
    # 'binary energy state',
    'continuous energy state',
    '$\\mathit{p}$ success',
    # 'wait when safe',
    'weather type',
    'ternary state',
    'optimal policy values',
    'ternary state * weather type',
    'ternary state * weather type * $\\mathit{p}$ success'
]

# Store BIC and AIC values for all subjects
all_bic = []
all_aic = []
N_trials = []

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
    dtU = dt[dt['x9_button_pressed'].notnull()]
    dtU = dtU[dtU['x6_continuous_energy_trial_start'] != 0]
    
    # Convert response variable to binary
    dtU['choice'] = dtU['x11_choice']
    
    # Create formulas for all models
    formulas = []
    for var in model_vars:
        formulas.append(f"choice ~ {var}")
    
    # Add additional models
    formulas.extend([f"choice ~ {model}" for model in additional_models])
    
    # Store model results for this subject
    bic_subject = []
    aic_subject = []
    
    # Fit each model
    for formula in formulas:
        try:
            # Fit model with formula API
            model = smf.logit(formula=formula, data=dtU)
            result = model.fit(disp=0)  # suppress convergence messages
            
            # Store metrics
            bic_subject.append(result.bic)
            aic_subject.append(result.aic)
            
        except Exception as e:
            print(f"Error fitting model with formula '{formula}': {e}")
            bic_subject.append(np.nan)
            aic_subject.append(np.nan)
    
    all_bic.append(bic_subject)
    all_aic.append(aic_subject)

# Convert to numpy arrays
all_bic = np.array(all_bic).T  # [models, subjects]
all_aic = np.array(all_aic).T

# Calculate sum of BIC/AIC across subjects for each model
bic_sums = np.nansum(all_bic, axis=1)
aic_sums = np.nansum(all_aic, axis=1)

# Calculate delta BIC/AIC (difference from best model)
min_bic_index = np.nanargmin(bic_sums)
min_aic_index = np.nanargmin(aic_sums)

delta_bic = bic_sums - bic_sums[min_bic_index]
delta_aic = aic_sums - aic_sums[min_aic_index]

# Calculate BIC/AIC weights
bic_weights = np.exp(-0.5 * delta_bic)
bic_weights = bic_weights / np.nansum(bic_weights)

aic_weights = np.exp(-0.5 * delta_aic)
aic_weights = aic_weights / np.nansum(aic_weights)

# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    'model': formulas,
    'description': model_descriptions,
    'BIC': bic_sums,
    'delta_BIC': delta_bic,
    'BIC_weight': bic_weights,
    'AIC': aic_sums,
    'delta_AIC': delta_aic,
    'AIC_weight': aic_weights
})

# Sort by BIC (like in the original script)
comparison_df = comparison_df.sort_values('delta_BIC')

# Print model comparison table
print("\n=== Model Comparison Results ===")
print(comparison_df[['description', 'BIC', 'delta_BIC', 'BIC_weight', 'AIC', 'delta_AIC', 'AIC_weight']])

# Save model comparison data
comparison_df.to_csv(path + '/model_comparison.csv', index=False)

# Plot delta BIC with values displayed on bars
fig, ax = plt.subplots(figsize=(20, 8))
bars = ax.barh(comparison_df['description'], comparison_df['delta_BIC'], color='lightgreen')
ax.tick_params(axis="x", labelsize=34)
ax.tick_params(axis="y", labelsize=34)
ax.tick_params(bottom=True, left=True, size=5, direction="in")
ax.set_title('BIC Model Comparison (lower is better)', loc='left', size=46)
ax.set_xlabel('Δ BIC (0 = best model)', fontsize=40)

# Add delta BIC values as text at the end of each bar
for i, bar in enumerate(bars):
    value = comparison_df['delta_BIC'].iloc[i]
    if value > 0:  # Only add text for non-zero values
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}', va='center', fontsize=24)

plt.tight_layout()
plt.savefig(os.path.join(path, 'bic_comparison.png'), dpi=300)
plt.show()

# Add explanation text below the plot
print("\nInterpretation of Δ BIC values:")
print("Δ BIC < 2: Weak evidence against the model")
print("2 < Δ BIC < 6: Positive evidence against the model")
print("6 < Δ BIC < 10: Strong evidence against the model")
print("Δ BIC > 10: Very strong evidence against the model")

# Plot delta AIC with values displayed on bars
fig, ax = plt.subplots(figsize=(20, 8))
bars = ax.barh(comparison_df['description'], comparison_df['delta_AIC'], color='skyblue')
ax.tick_params(axis="x", labelsize=34)
ax.tick_params(axis="y", labelsize=34)
ax.tick_params(bottom=True, left=True, size=5, direction="in")
ax.set_title('AIC Model Comparison (lower is better)', loc='left', size=46)
ax.set_xlabel('Δ AIC (0 = best model)', fontsize=40)

# Add delta AIC values as text at the end of each bar
for i, bar in enumerate(bars):
    value = comparison_df['delta_AIC'].iloc[i]
    if value > 0:  # Only add text for non-zero values
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}', va='center', fontsize=24)

    value = comparison_df['delta_AIC'].iloc[i]
    if value > 0:  # Only add text for non-zero values
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}', va='center', fontsize=24)

    value = comparison_df['delta_AIC'].iloc[i]
    if value > 0:  # Only add text for non-zero values
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}', va='center', fontsize=24)

plt.tight_layout()
plt.savefig(os.path.join(path, 'aic_comparison.png'), dpi=300)
plt.show()

# Add explanation text below the plot
print("\nInterpretation of Δ AIC values:")
print("Δ AIC < 2: Substantial support for the model")
print("4 < Δ AIC < 7: Considerably less support for the model")
print("Δ AIC > 10: Essentially no support for the model")

# Model weights plot
plt.figure(figsize=(19, 10))
x = np.arange(len(comparison_df))
width = 0.35

plt.barh(x - width/2, comparison_df['AIC_weight'], width, label='AIC Weights', color='skyblue')
plt.barh(x + width/2, comparison_df['BIC_weight'], width, label='BIC Weights', color='lightgreen')

plt.yticks(x, comparison_df['description'], fontsize=25)
plt.xlabel('Model Weight', fontsize=40)
plt.title('Model Weights (AIC and BIC)', fontsize=46)
plt.xticks(fontsize=25)
plt.legend(fontsize=30)
plt.tick_params(axis="x", labelsize=34)
plt.tick_params(axis="y", labelsize=34)
plt.tight_layout()
plt.savefig(os.path.join(path, 'model_weights.png'), dpi=300)
plt.show()