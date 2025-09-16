#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homunc Behavioral Analysis
"""
# %% =====================================================================================================
"""=== Model Copmarison ==="""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

path = os.path.dirname(os.path.abspath(__file__)) + '/'
os.chdir(path)

mode_index_order = [10, 9, 8, 4, 6, 7, 5, 3, 2, 1, 0]

d = 'data_beh.csv'
# d = 'data_fmri.csv'
combined_data = pd.read_csv(d)
combined_data['BNW_conditions'] = pd.Categorical(combined_data['x37_binary_energy'] - combined_data['x19_wait_when_safe'])
# combined_data = combined_data[combined_data['x14_p_foraging_gain'] > 0.3]
# combined_data = combined_data[combined_data['x14_p_foraging_gain'] < 0.7]

# Define model variables
model_vars = [
    'x17_horizon_correct_adjusted',
    'x13_gain_magnitude',
    'x24_pseudo0x2Doptimal_horizon0x2D1',
    # 'x37_binary_energy',
    'x6_continuous_energy_trial_start',
    'x41_expected_energy_change',
    # 'x40_expected_energy_w_boundaries',
    'x14_p_foraging_gain',
    'x19_wait_when_safe',
    'x7_weather_type',
    # 'BNW_conditions',
    'x22_optimal_policy',
    'BNW_conditions*x7_weather_type',
    'BNW_conditions*x7_weather_type * x14_p_foraging_gain'
]

# Define model descriptions (for plotting)
model_descriptions = [
    'remaining time-points',
    'gain magnitude',
    'pseudo optimal values',
    # 'binary energy state',
    'continuous energy state',
    'expected energy change',
    # 'expected energy change with boundaries',
    '$\\mathit{p}$ success',
    # 'wait when safe',
    'weather type',
    'ternary state',
    'optimal policy values',
    'ternary state * weather type',
    'ternary state * weather type * $\\mathit{p}$ success'
]

# Create formulas for all models with random intercept by subject
formulas = []
for var in model_vars:
    formulas.append(f"x11_choice ~ {var} + (1|x1_id)")

# Store results
model_results = []
bic_values = []
aic_values = []

# Fit models using logit instead of mixed-effects
for i, formula in enumerate(formulas):
    print(f"Fitting model {i+1}/{len(formulas)}: {formula}")
    
    try:
        # Remove the random effects part for logit model
        fixed_formula = formula.split(" + (1|x1_id)")[0]
        model = smf.logit(formula=fixed_formula, data=combined_data)
        result = model.fit(disp=0)
        
        # Store metrics
        bic_values.append(result.bic)
        aic_values.append(result.aic)
        model_results.append(result)
        
    except Exception as e:
        print(f"Error fitting model with formula '{fixed_formula}': {e}")
        bic_values.append(np.nan)
        aic_values.append(np.nan)
        model_results.append(None)
    
    if result is not None:
        print(result.bic, result.aic)

# Calculate delta BIC/AIC
min_bic_index = np.nanargmin(bic_values)
min_aic_index = np.nanargmin(aic_values)
delta_bic = np.array(bic_values) - bic_values[min_bic_index]
delta_aic = np.array(aic_values) - aic_values[min_aic_index]

# Calculate BIC/AIC weights
bic_weights = np.exp(-0.5 * delta_bic)
bic_weights = bic_weights / np.nansum(bic_weights)

aic_weights = np.exp(-0.5 * delta_aic)
aic_weights = aic_weights / np.nansum(aic_weights)
# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    'model': formulas,
    'description': model_descriptions,
    'BIC': bic_values,
    'delta_BIC': delta_bic,
    'BIC_weight': bic_weights,
    'AIC': aic_values,
    'delta_AIC': delta_aic,
    'AIC_weight': aic_weights
})

# Print model comparison table
print("\n=== Hierarchical Model Comparison Results ===")
print(comparison_df[['description', 'BIC', 'delta_BIC', 'BIC_weight', 'AIC', 'delta_AIC', 'AIC_weight']])

# Save model comparison data
comparison_df.to_csv(path + '/hierarchical_model_comparison.csv', index=False)

# Create a mapping from model index to its order position
model_order_mapping = {idx: order_pos for order_pos, idx in enumerate(mode_index_order)}

# Create a new column in comparison_df for ordering
comparison_df['plot_order'] = comparison_df.index.map(lambda x: model_order_mapping.get(x, len(mode_index_order)))

# # Sort the dataframe by this new ordering
# comparison_df_ordered = comparison_df.sort_values('plot_order')
# comparison_df_ordered = comparison_df.sort_values('delta_BIC')
# order = comparison_df_ordered
# comparison_df_ordered = order
# Save the order from fMRI data results if this is fMRI data
if d == 'data_fmri.csv':
    # Sort by delta_BIC to get the order
    comparison_df_ordered = comparison_df.sort_values('delta_BIC')
    # Save the order (model indices) to a file
    fmri_order = comparison_df_ordered.index.tolist()
    pd.Series(fmri_order).to_csv(path + '/fmri_model_order.csv', index=False, header=['model_index'])
    print("Saved fMRI model order")
else:
    # For behavioral data, try to load the saved fMRI order
    try:
        fmri_order = pd.read_csv(path + '/fmri_model_order.csv')['model_index'].tolist()
        comparison_df_ordered = comparison_df.reindex(fmri_order)
        print("Using saved fMRI model order for behavioral data")
    except FileNotFoundError:
        print("No saved fMRI order found, using behavioral data order")
        comparison_df_ordered = comparison_df.sort_values('delta_BIC')


# Plot delta BIC with values displayed on bars
fig, ax = plt.subplots(figsize=(17, 8))
bars = ax.barh(comparison_df_ordered['description'], comparison_df_ordered['delta_BIC'], color='lightgreen')
ax.tick_params(axis="x", labelsize=34)
ax.tick_params(axis="y", labelsize=34)
ax.tick_params(bottom=True, left=True, size=5, direction="in")
ax.set_title('BIC Model Comparison', loc='left', size=46)
ax.set_xlabel('Δ BIC (0 = best model)', fontsize=40)
if d == 'data_beh.csv':
    ax.set_xlim(0, 1500)  # Set x-axis limit
    ax.set_xlim(0, 8500)  # Set x-axis limit
    # ax.set_xlim(0, 5700)  # Set x-axis limit
elif d == 'data_fmri.csv':
    ax.set_xlim(0, 5500)  # Set x-axis limit

# Add delta BIC values as text at the end of each bar
for i, bar in enumerate(bars):
    value = comparison_df_ordered['delta_BIC'].iloc[i]
    if value > 0:  # Only add text for non-zero values
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}', va='center', fontsize=24)

plt.tight_layout()
plt.savefig(os.path.join(path, 'hierarchical_bic_comparison.png'), dpi=300)
plt.show()

# Plot delta AIC with values displayed on bars
fig, ax = plt.subplots(figsize=(17, 8))
bars = ax.barh(comparison_df_ordered['description'], comparison_df_ordered['delta_AIC'], color='skyblue')
ax.tick_params(axis="x", labelsize=34)
ax.tick_params(axis="y", labelsize=34)
ax.tick_params(bottom=True, left=True, size=5, direction="in")
ax.set_title('AIC Model Comparison', loc='left', size=46)
ax.set_xlabel('Δ AIC (0 = best model)', fontsize=40)
if d == 'data_beh.csv':
    ax.set_xlim(0, 1500)  # Set x-axis limit
    ax.set_xlim(0, 8500)  # Set x-axis limit
    # ax.set_xlim(0, 5700)  # Set x-axis limit
elif d == 'data_fmri.csv':
    ax.set_xlim(0, 5500)  # Set x-axis limit

# Add delta AIC values as text at the end of each bar
for i, bar in enumerate(bars):
    value = comparison_df_ordered['delta_AIC'].iloc[i]
    if value > 0:  # Only add text for non-zero values
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}', va='center', fontsize=24)

plt.tight_layout()
plt.savefig(os.path.join(path, 'hierarchical_aic_comparison.png'), dpi=300)
plt.show()

# Model weights plot
plt.figure(figsize=(15, 8))
x = np.arange(len(comparison_df_ordered))
width = 0.35

plt.barh(x - width/2, comparison_df_ordered['AIC_weight'], width, label='AIC Weights', color='skyblue')
plt.barh(x + width/2, comparison_df_ordered['BIC_weight'], width, label='BIC Weights', color='lightgreen')

plt.yticks(x, comparison_df_ordered['description'], fontsize=25)
plt.xlabel('Model Weight', fontsize=40)
plt.tick_params(bottom=True, left=True, size=5, direction="in")
plt.title('Model Weights', fontsize=46)
plt.xticks(fontsize=25)
plt.legend(fontsize=30)
plt.tick_params(axis="x", labelsize=34)
plt.tick_params(axis="y", labelsize=34)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(path, 'hierarchical_model_weights.png'), dpi=300)
plt.show()

# %% =====================================================================================================
"""=== Test out of sample accuracy of the best model ==="""
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

d = 'data_beh.csv'
dd = 'data_fmri.csv'
data_train = pd.read_csv(d)
data_train['BNW_conditions'] = pd.Categorical(data_train['x37_binary_energy'] - data_train['x19_wait_when_safe'])
data_test = pd.read_csv(dd)
data_test['BNW_conditions'] = pd.Categorical(data_test['x37_binary_energy'] - data_test['x19_wait_when_safe'])

# Get the winning model formula from the first section
best_model_idx = int(comparison_df['delta_BIC'].idxmin())
best_formula = formulas[best_model_idx]
print(f"Best model formula: {best_formula}")

# Remove the random effects part for logit model
fixed_formula = best_formula.split(" + (1|x1_id)")[0]

# Fit the winning model on training data
print("Fitting winning model on training data...")
model_train = smf.logit(formula=fixed_formula, data=data_train)
result_train = model_train.fit(disp=0)
print(result_train.summary())

# Get the model parameters
params = result_train.params
print("Model parameters:", params)

# Apply the model to test data to get predictions
print("Applying model to test data...")
X_test = smf.logit(formula=fixed_formula, data=data_test).exog
y_test = data_test['x11_choice']

# Calculate predictions
pred_probs = 1 / (1 + np.exp(-np.dot(X_test, params)))
pred_binary = (pred_probs >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(pred_binary == y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Calculate AUC (Area Under the ROC Curve)

auc = roc_auc_score(y_test, pred_probs)
print(f"Test AUC: {auc:.4f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, pred_probs)
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, lw=4, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=40)
plt.ylabel('True Positive Rate', fontsize=40)
plt.title('Model Performance', fontsize=46, loc='left')
plt.legend(loc="lower right", fontsize=30)
plt.tick_params(axis="x", labelsize=34)
plt.tick_params(axis="y", labelsize=34)
plt.tick_params(bottom=True, left=True, size=5, direction="in")
plt.tight_layout()
plt.savefig(os.path.join(path, 'model_roc_curve.png'), dpi=300)
plt.show()

# Calculate confusion matrix
cm = confusion_matrix(y_test, pred_binary)
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Wait', 'Forage'])
disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
ax.set_title('Confusion Matrix', fontsize=46, loc='left')
ax.set_xlabel('Predicted label', fontsize=40)
ax.set_ylabel('True label', fontsize=40)
ax.tick_params(axis="x", labelsize=34)
ax.tick_params(axis="y", labelsize=34)
ax.tick_params(bottom=True, left=True, size=5, direction="in")

# Modify text size in the confusion matrix cells
for text in disp.text_.ravel():
    text.set_fontsize(30)

plt.tight_layout()
plt.savefig(os.path.join(path, 'model_confusion_matrix.png'), dpi=300)
plt.show()

# %% =====================================================================================================
"""=== Plot Factor Interactions ==="""
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

# Get the index of the factorial interaction model
factorial_model_idx = model_vars.index('BNW_conditions*x7_weather_type * x14_p_foraging_gain')

# Use the test data from the second section
dd = 'data_fmri.csv'
test_data = pd.read_csv(dd)
test_data['BNW_conditions'] = pd.Categorical(test_data['x37_binary_energy'] - test_data['x19_wait_when_safe'])
test_data = test_data[test_data['x14_p_foraging_gain'] > 0.3]
test_data = test_data[test_data['x14_p_foraging_gain'] < 0.7]

# Define BNW_conditions values and their labels
bnw_values = [1, 0, -1]
bnw_labels = ['Binary Energy', 'Trade-off', 'Wait When Safe']

# Define weather types
weather_values = [1, 2]  # 1 and 2 for weather types
weather_labels = ['Bad Weather', 'Good Weather']

# Create a figure with 3 rows (for BNW conditions) and 2 columns (for weather types)
fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True, sharey=True)

# Loop through each BNW condition and weather type
for i, bnw in enumerate(bnw_values):
    for j, weather in enumerate(weather_values):
        ax = axes[i, j]
        
        # Filter the test data for this specific combination
        subset = test_data[(test_data['BNW_conditions'] == bnw) & 
                          (test_data['x7_weather_type'] == weather)]
        
        # If empty subset, continue
        if subset.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=20)
            continue
        
        # Group by subject and p_gain first to calculate means within each subject
        results = []
        for p_gain in subset['x14_p_foraging_gain'].unique():
            p_gain_data = subset[subset['x14_p_foraging_gain'] == p_gain]
            # Calculate mean choice for each subject at this p_gain
            subject_means = p_gain_data.groupby('x1_id')['x11_choice'].mean()
            # Calculate the grand mean and SEM across subjects
            grand_mean = subject_means.mean()
            sem = subject_means.std() / np.sqrt(len(subject_means)) if len(subject_means) > 1 else 0
            results.append((p_gain, grand_mean, sem))
        
        if not results:  # Skip if no results
            continue
            
        p_gain_values = [r[0] for r in results]
        mean_choice_values = [r[1] for r in results]
        sem_values = [r[2] for r in results]
        
        # Plot the aggregated data points with error bars
        ax.errorbar(p_gain_values, mean_choice_values, yerr=sem_values, 
                   fmt='o', markersize=8, capsize=5, elinewidth=2, 
                   markeredgecolor='black', color='blue', alpha=0.7)
        
        # Add regression line
        if len(p_gain_values) > 1:  # Need at least 2 points for regression
            ax.plot(np.unique(p_gain_values), 
                    np.poly1d(np.polyfit(p_gain_values, mean_choice_values, 1))
                    (np.unique(p_gain_values)), color='blue', lw=3)
        
        ax.set_title(f'{bnw_labels[i]} - {weather_labels[j]}', fontsize=30)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.35, 0.65)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis="x", labelsize=25)
        ax.tick_params(axis="y", labelsize=25)
        ax.tick_params(bottom=True, left=True, size=5, direction="in")
        
        # Add labels
        if i == 2:
            ax.set_xlabel('P(Foraging Gain)', fontsize=30)
        if j == 0:
            ax.set_ylabel('P(Forage)', fontsize=30)
            
        # Add horizontal line at 0.5 probability
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, linewidth=3)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(os.path.join(path, 'factorial_design_plot.png'), dpi=300)
plt.show()

# Create a single plot with all conditions
plt.figure(figsize=(12, 10))

line_styles = ['-', '--']
colors = ['blue', 'red', 'green']
markers = ['o', 'o']

# Create legend elements
legend_elements = []
for i, bnw in enumerate(bnw_labels):
    legend_elements.append(Line2D([0], [0], color=colors[i], lw=4, label=bnw))

for j, weather in enumerate(weather_labels):
    legend_elements.append(Line2D([0], [0], color='black', lw=4, 
                                 linestyle=line_styles[j], label=weather))

# Loop through each BNW condition and weather type
for i, bnw in enumerate(bnw_values):
    for j, weather in enumerate(weather_values):
        # Filter the test data for this specific combination
        subset = test_data[(test_data['BNW_conditions'] == bnw) & 
                          (test_data['x7_weather_type'] == weather)]
        
        # If empty subset, continue
        if subset.empty:
            continue
        
        # Group by p_gain and calculate mean and SEM across subjects
        results = []
        for p_gain in subset['x14_p_foraging_gain'].unique():
            p_gain_data = subset[subset['x14_p_foraging_gain'] == p_gain]
            # Calculate mean choice for each subject at this p_gain
            subject_means = p_gain_data.groupby('x1_id')['x11_choice'].mean()
            # Calculate the grand mean and SEM across subjects
            grand_mean = subject_means.mean()
            sem = subject_means.std() / np.sqrt(len(subject_means)) if len(subject_means) > 1 else 0
            results.append((p_gain, grand_mean, sem))
        
        if not results:  # Skip if no results
            continue
            
        p_gain_values = [r[0] for r in results]
        mean_choice_values = [r[1] for r in results]
        sem_values = [r[2] for r in results]
        
        # Plot the data points with error bars
        plt.errorbar(p_gain_values, mean_choice_values, yerr=sem_values,
                    fmt=markers[j], markersize=8, capsize=5, elinewidth=2,
                    markeredgecolor='black', color=colors[i], alpha=0.7)
        
        # Add regression line if we have enough points
        if len(p_gain_values) > 1:
            plt.plot(np.unique(p_gain_values), 
                    np.poly1d(np.polyfit(p_gain_values, mean_choice_values, 1))
                    (np.unique(p_gain_values)), 
                    linestyle=line_styles[j], color=colors[i], lw=3)

plt.xlabel('P(Foraging Gain)', fontsize=40)
plt.ylabel('P(Forage)', fontsize=40)
plt.title('Interactions', fontsize=46, loc='left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(handles=legend_elements, title='', fontsize=30,
           bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=3)
plt.tick_params(axis="x", labelsize=34)
plt.tick_params(axis="y", labelsize=34)
plt.tick_params(bottom=True, left=True, size=5, direction="in")
# plt.set_xlim(0.35, 0.65)
plt.ylim(-0.05, 1.05)
plt.tight_layout()
plt.subplots_adjust(right=0.8)  # Make room for the legend
plt.savefig(os.path.join(path, 'factorial_interaction_combined.png'), dpi=300, bbox_inches='tight')
plt.show()

# %% =====================================================================================================
"""=== Test Interaction on Data Subset ==="""
d = 'data_beh.csv'
d = 'data_fmri.csv'
combined_data = pd.read_csv(d)
combined_data['BNW_conditions'] = pd.Categorical(combined_data['x37_binary_energy'] - combined_data['x19_wait_when_safe'])

# Filter data
combined_data = combined_data[combined_data['x14_p_foraging_gain'] > 0.3]
combined_data = combined_data[combined_data['x14_p_foraging_gain'] < 0.7]

# Get the winning model formula from the first section
best_model_idx = int(comparison_df['delta_BIC'].idxmin())
best_formula = formulas[best_model_idx]
print(f"Best model formula: {best_formula}")

# Remove the random effects part for logit model
fixed_formula = best_formula.split(" + (1|x1_id)")[0]

# Fit the winning model on training data
print("Fitting winning model on training data...")
model = smf.logit(formula=fixed_formula, data=combined_data)
result = model_train.fit(disp=0)
print(result.summary())

# Check difference trade-off vs. binary energy
from scipy import stats

# Get model results
model_results = result

# Extract coefficients and covariance matrix
params = model_results.params
cov_matrix = model_results.cov_params()

# Define contrast: Difference between Binary Energy and Trade-off interactions with weather
# We want to test if (Binary Energy × Weather) - (Trade-off × Weather) = 0
contrast_weather = np.zeros_like(params)
tradeoff_weather_idx = list(params.index).index("BNW_conditions[T.0]:x7_weather_type")
binary_weather_idx = list(params.index).index("BNW_conditions[T.1]:x7_weather_type")
contrast_weather[binary_weather_idx] = 1
contrast_weather[tradeoff_weather_idx] = -1

# Calculate contrast value and standard error
contrast_value = np.dot(contrast_weather, params)
contrast_se = np.sqrt(np.dot(contrast_weather, np.dot(cov_matrix, contrast_weather)))

# Calculate t-statistic and p-value
t_stat = contrast_value / contrast_se
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=model_results.df_resid))

print("\nDifference between Binary Energy vs Trade-off interaction with weather:")
print(f"Contrast value: {contrast_value:.4f}")
print(f"Standard error: {contrast_se:.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# Same for the three-way interaction with p_success
contrast_three_way = np.zeros_like(params)
tradeoff_three_way_idx = list(params.index).index("BNW_conditions[T.0]:x7_weather_type:x14_p_foraging_gain")
binary_three_way_idx = list(params.index).index("BNW_conditions[T.1]:x7_weather_type:x14_p_foraging_gain")
contrast_three_way[binary_three_way_idx] = 1
contrast_three_way[tradeoff_three_way_idx] = -1

# Calculate contrast value and standard error
contrast_value_3way = np.dot(contrast_three_way, params)
contrast_se_3way = np.sqrt(np.dot(contrast_three_way, np.dot(cov_matrix, contrast_three_way)))

# Calculate t-statistic and p-value
t_stat_3way = contrast_value_3way / contrast_se_3way
p_value_3way = 2 * (1 - stats.t.cdf(abs(t_stat_3way), df=model_results.df_resid))

print("\nDifference between Binary Energy vs Trade-off three-way interaction:")
print(f"Contrast value: {contrast_value_3way:.4f}")
print(f"Standard error: {contrast_se_3way:.4f}")
print(f"t-statistic: {t_stat_3way:.4f}")
print(f"p-value: {p_value_3way:.4f}")

# %% ====================================================================================================
"""=== Plot Weather Interaction ==="""
# Create a plot to compare the interaction effects specifically between Trade-off and Binary Energy
plt.figure(figsize=(10, 8))

# Colors for good and bad weather
weather_colors = ['green', 'magenta']
weather_labels = ['Bad Weather', 'Good Weather']

# Define line styles for different conditions (if needed)
condition_styles = ['-', '--']

plt.figure(figsize=(12, 10))

# Loop through weather types
for j, weather in enumerate([1, 2]):  # 1=Bad, 2=Good
    # Get all data for this weather type
    subset = test_data[test_data['x7_weather_type'] == weather]
    
    # Group by p_gain
    results = []
    for p_gain in subset['x14_p_foraging_gain'].unique():
        p_gain_data = subset[subset['x14_p_foraging_gain'] == p_gain]
        subject_means = p_gain_data.groupby('x1_id')['x11_choice'].mean()
        grand_mean = subject_means.mean()
        sem = subject_means.std() / np.sqrt(len(subject_means)) if len(subject_means) > 1 else 0
        results.append((p_gain, grand_mean, sem))
    
    if results:
        p_gain_values = [r[0] for r in results]
        mean_choice_values = [r[1] for r in results]
        sem_values = [r[2] for r in results]
        
        # Plot with different markers for each weather type
        plt.errorbar(p_gain_values, mean_choice_values, yerr=sem_values,
                    fmt='o', markersize=8, capsize=5, elinewidth=2,
                    color=weather_colors[j-1], 
                    markeredgecolor='black', label=weather_labels[j])
        
        # Add regression line
        if len(p_gain_values) > 1:
            plt.plot(np.unique(p_gain_values), 
                    np.poly1d(np.polyfit(p_gain_values, mean_choice_values, 1))
                    (np.unique(p_gain_values)), 
                    color=weather_colors[j-1], lw=3)

plt.title("Interaction Weather x P(Success)", fontsize=46, loc='left')
plt.xlabel('P(Success)', fontsize=40)
plt.ylabel('P(Forage)', fontsize=40)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=30)
plt.ylim(-0.05, 1.05)
plt.xlim(0.35, 0.65)
plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, linewidth=3)
plt.tick_params(axis="x", labelsize=34)
plt.tick_params(axis="y", labelsize=34)
plt.tick_params(bottom=True, left=True, size=5, direction="in")
plt.tight_layout()
plt.savefig(os.path.join(path, 'weather_comparison.png'), dpi=300)
plt.show()
