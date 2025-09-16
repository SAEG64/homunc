"""
Convert mat file to csv for behavioral sample
"""

# %% ==========================================================================
# Import mat file
# =============================================================================
from scipy.io import loadmat
import pandas as pd
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__)) + "/"
SEED = 42
rng = np.random.default_rng(SEED)

# Load the .mat file
data = loadmat('HOMUNC_data_beh_A_pilot_v1.mat')

# Print keys to see available variables
print(data.keys())

# Access specific variables
d1 = data['data_cati']
d2 = data['list_cati']
dat = np.concatenate([np.concatenate(d1[:,0]), np.concatenate(d2[:,0])], axis = 1)

# sbjs = ['101', '102', '103', '104', '105', '106', '107', '110', '111', 
#         '112', '113', '114', '115', '116', '117', '118', '119', '120',
#         '122', '123', '124', '125', '126', '127', '128', '129', '130', '131']
headers = [
    'x1_id', 'x2_session', 'x3_trial', 'x4_index_forests', 'x5_order_trials_in_forest', 
    'x6_continuous_energy_trial_start', 'x7_weather_type', 'x8_side_options_on_screen',	
    'x9_button_pressed',	'x10_RT', 'x11_choice', 'x12_continuous_energy_trial_end',	
    'x13_index_forests', 'x14_current_gain_magnitude', 'xempty', 'xempty',
    'x17_weather_1_gain_magnitude', 'x18_weather_2_gain_magnitude', 
    'x19_weather_1_p_gain', 'x20_weather_2_p_gain',
    'x21_dono', 'x22_dono', 'x23_dono', 'x24_dono', 'x25_dono', 'x26_dono', 
    'x27_expected_energy_no_bounds', 'x28_expected_energy_w_boundaries', 
    'x29_dono', 'x30_dono', 'x31_dono']
    # '41_expected_energy_change']
dat_df = pd.DataFrame(dat, columns=headers)
dat_df.to_csv(path + 'HOMUNC_data_beh_A_pilot_v1.csv', index=False)