# %% ==========================================================================
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp
import pandas as pd
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__)) + "/"
SEED = 42
rng = np.random.default_rng(SEED)

# Data set
d = pd.read_csv(path + 'data_beh/datall_cat.csv')

# Models to compare
mdl_ls = [
  # 'BNW_conditions',
  # 'x17_horizon_correct_adjusted',
  'x6_continuous_energy_trial_start',
  'x24_pseudo0x2Doptimal_horizon0x2D1',
  'x13_gain_magnitude',
  'x7_weather_type',
  'x14_p_foraging_gain',
  'x22_optimal_policy']

# Redefine variables
d['subject'] = d['x1_id']
d['acts'] = d['x11_choice']
d['BNW_fact'] = pd.Categorical(d['BNW_conditions'])
d['weather_fact'] = pd.Categorical(d['x7_weather_type'])

# Filter data
d = d[~d['x9_button_pressed'].isna()]
d = d[d['x6_continuous_energy_trial_start'] != 0]
d = d[d['x14_p_foraging_gain'] > 0.3]
d = d[d['x14_p_foraging_gain'] < 0.7]
d = d.reset_index(drop=True)

# Fit models and estimate ELPD
elbos = []
for mod in mdl_ls[:-1]:
    print("run model:", mod)

    # Get current decision variable
    d['DV_main'] = d[mod]

    # Fit logistic regression using statsmodel variational Bayes mean field
    random = {"subject": '0 + C(subject)'}  # Random intercept for each subject
    log_reg = BinomialBayesMixedGLM.from_formula(
                    'acts ~ DV_main * BNW_fact', random, d)
    result = log_reg.fit_vb()
    
    # Extract Mean (Posterior Mean) and Standard Deviation (Posterior SD)
    posterior_mean = result.params  # Mean of posterior
    # Posterior Standard Deviations (square root of diagonal of posterior covariance)
    posterior_cov = result.cov_params()  # Approximate covariance matrix
    posterior_sd = sp.csr_matrix(np.sqrt(np.diag(posterior_cov))).data  # Standard deviations
    
    # Model evidence using evidence lower bound
    elbo = log_reg.vb_elbo(vb_mean=posterior_mean, vb_sd=posterior_sd)
    elbos.append(elbo)
    print(f"evidence lower bound BNW_fact + {mod} = {elbo:.3f}")

    # Predict actions
    actions_pred = result.predict()
    # Area under the curve
    auc = roc_auc_score(d['acts'], actions_pred)
    print(f"AUC-ROC = {auc:.3f}")

# Test optimal policy and estimate ELBO
for mod in mdl_ls[-1:]:
    print("run model:", mod)

    # Get current decision variable
    d['DV_main'] = d[mod]

    # Fit logistic regression using statsmodel variational Bayes mean field
    random = {"subject": '0 + C(subject)'}  # Random intercept for each subject
    log_reg = BinomialBayesMixedGLM.from_formula(
                    'acts ~ DV_main', random, d)
    result = log_reg.fit_vb()
    
    # Extract Mean (Posterior Mean) and Standard Deviation (Posterior SD)
    posterior_mean = result.params  # Mean of posterior
    # Posterior Standard Deviations (square root of diagonal of posterior covariance)
    posterior_cov = result.cov_params()  # Approximate covariance matrix
    posterior_sd = sp.csr_matrix(np.sqrt(np.diag(posterior_cov))).data  # Standard deviations
    
    # Model evidence using evidence lower bound
    elbo = log_reg.vb_elbo(vb_mean=posterior_mean, vb_sd=posterior_sd)
    elbos.append(elbo)
    print(f"evidence lower bound BNW_fact + {mod} = {elbo:.3f}")

    # Predict actions
    actions_pred = result.predict()
    # Area under the curve
    auc = roc_auc_score(d['acts'], actions_pred)
    print(f"AUC-ROC = {auc:.3f}")

# %%
# Fit and evaluate 3x2 factor model
# =================================
# Filter data
d = d[d['x14_p_foraging_gain'] > 0.3]
d = d[d['x14_p_foraging_gain'] < 0.7]
d = d.reset_index(drop=True)

print("run model: 3x2 factorial")

# Get additional model
mod = 'x14_p_foraging_gain'
d['DV_main'] = d[mod]
print(mod + ' as pMod')

# Fit logistic regression using statsmodel variational Bayes mean field
random = {"subject": '0 + C(subject)'}  # Random intercept for each subject
log_reg = BinomialBayesMixedGLM.from_formula(
                'acts ~ DV_main * BNW_fact * weather_fact', random, d)
result = log_reg.fit_vb()

# Extract Mean (Posterior Mean) and Standard Deviation (Posterior SD)
posterior_mean = result.params  # Mean of posterior
# Posterior Standard Deviations (square root of diagonal of posterior covariance)
posterior_cov = result.cov_params()  # Approximate covariance matrix
posterior_sd = sp.csr_matrix(np.sqrt(np.diag(posterior_cov))).data  # Standard deviations

# Model evidence using evidence lower bound
elbo = log_reg.vb_elbo(vb_mean=posterior_mean, vb_sd=posterior_sd)
elbos.append(elbo)
print(f"evidence lower bound 3x2 factorial = {elbo:.3f}")

# Predict actions
actions_pred = result.predict()
# Area under the curve
auc = roc_auc_score(d['acts'], actions_pred)
print(f"AUC-ROC = {auc:.3f}")