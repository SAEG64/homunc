# %% 1. =======================================================================
# 1. Get ANN model and requirements
# =============================================================================
import pandas as pd
import numpy as np
import pickle
import torch
import os

SEED = 42
path = os.path.dirname(os.path.abspath(__file__)) + "/"
rng = np.random.default_rng(SEED)

# Get BC model
try:

    # First, try to load the model from pickle file
    with open(os.path.join(path, "bc_model.pkl"), "rb") as f:
        loaded_data = pickle.load(f)
        model = loaded_data['bc_model']
    print("-------------------------------------------")
    print("Successfully loaded model from bc_model.pkl")
    print("-------------------------------------------")

except (FileNotFoundError, KeyError) as e:

    # If loading fails, import from module and compute
    print("-------------------------------------------")
    print(f"Failed to load model from pickle: {e}")
    print("Importing model from model_BC module...")
    print("-------------------------------------------")
    from model_BC import bc_trainer as model

    # Export model results
    try:
        with open(os.path.join(path, "bc_model.pkl"), "wb") as f:
            pickle.dump({
                'bc_model': model
            }, f)
        print("\nSaved model results to bc_model.pkl")
    except Exception as e:
        print(f"\nWarning: Could not save model results: {e}")

# Get test data
from expert_trajectories_REV_V2 import transitions as transitions_test
observations = transitions_test.obs

# Define a function to extract activations from hidden layers
def get_activations_bc(model, observations, layer_names=None):
    """
    Extracts activations from the hidden layers of the BC model.

    Parameters:
    - model: Trained BC model
    - observations: Input observations (numpy array)
    - layer_names: List of layer names to extract (e.g., ['pi', 'vf', 'shared_net', 'action_net'])

    Returns:
    - activations: Dictionary of activations per layer
    """
    activations = {}

    with torch.no_grad():
        obs_tensor = torch.tensor(observations, dtype=torch.float32)

        # Step 1: Extract features from feature extractor
        features = model.policy.features_extractor(obs_tensor)
        activations['features_extractor'] = features.numpy()

        # Step 2: Pass through MLP extractor (policy & value networks)
        pi_features, vf_features = model.policy.mlp_extractor(features)

        # Step 3: Capture activations from policy and value networks
        if layer_names == None or 'pi' in layer_names:
            activations['pi_features'] = pi_features.numpy()

        if layer_names == None or 'vf' in layer_names:
            activations['vf_features'] = vf_features.numpy()

        # # Step 4: Check if shared network exists and extract its activations
        # if hasattr(model.policy.mlp_extractor, "shared_net"):
        #     shared_features = model.policy.mlp_extractor.shared_net(features)
        #     activations['shared_net'] = shared_features.numpy()

        # Step 5: Extract final layer outputs
        if layer_names == 'action_net' in layer_names:
            activations['action_net'] = model.policy.action_net(pi_features).numpy()

        if layer_names == 'value_net' in layer_names:
            activations['value_net'] = model.policy.value_net(vf_features).numpy()

    return activations


# %% 2. =======================================================================
# 2. Filter ANN features and activations
# =============================================================================
def filter_d(action_data, filter_raw = False, filter_strong=False):
    # Create DataFrame using all the keys from action_data
    activations_df = pd.DataFrame()
    col_n = []
    for key, value in action_data.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:  # Check if it's a 2D numpy array 
            multi_d = True
            col_n.append(value.shape[1])
            # Create DataFrames for each column with proper index
            dfs = pd.DataFrame(value, columns=[f"{key}_{i}" for i in range(value.shape[1])])
            # Concat DataFrame with activations_df along column axis
            activations_df = pd.concat([activations_df, dfs], axis=1)
        else:
            # Add each array from action_data as a column
            activations_df[key] = action_data[key]
    
    # Apply filtering if needed
    if len(observations) == len(activations_df):

        # Filter activations_df
        mask = (observations[:, -4] != 0) #|(observations[:, -1] != 0)
        filtered_activations_df = activations_df[mask].copy()
        filtered_activations_df = filtered_activations_df.reset_index(drop=True)

        d = pd.read_csv(path + 'data_beh/datall_cat.csv')

        if filter_raw == False:
            mask1 = (~d['x9_button_pressed'].isna())
            filtered_activations_df = filtered_activations_df[mask1]
            filtered_activations_df = filtered_activations_df.reset_index(drop=True)
            d = d[~d['x9_button_pressed'].isna()]
            d = d.reset_index(drop=True)

            mask2 = (d['x6_continuous_energy_trial_start'] != 0)
            filtered_activations_df = filtered_activations_df[mask2]
            filtered_activations_df = filtered_activations_df.reset_index(drop=True)
            d = d[d['x6_continuous_energy_trial_start'] != 0]
            d = d.reset_index(drop=True)

            if filter_strong == True:
                mask3 = (d['x14_p_foraging_gain'] > 0.3)
                filtered_activations_df = filtered_activations_df[mask3]
                filtered_activations_df = filtered_activations_df.reset_index(drop=True)
                d = d[d['x14_p_foraging_gain'] > 0.3]
                d = d.reset_index(drop=True)
                
                mask4 = (d['x14_p_foraging_gain'] < 0.7)
                filtered_activations_df = filtered_activations_df[mask4]
                filtered_activations_df = filtered_activations_df.reset_index(drop=True)
                d = d[d['x14_p_foraging_gain'] < 0.7]
                d = d.reset_index(drop=True)

        # Convert filtered_activations_df back to dictionary format
        action_data_REV = {}
        if 'multi_d' not in locals():
            for key in action_data.keys():
                action_data_REV[key] = filtered_activations_df[key]
        else:
            for i in range(len(col_n)):
                n = col_n[i]
                key = list(action_data.keys())[i]
                action_data_REV[key] = filtered_activations_df[[f"{key}_{i}" for i in range(n)]].to_numpy()

        return action_data_REV
    
    else:
        print('Data is already filtered')
        return action_data

# Extract activations from key layers
activation_data = get_activations_bc(model, observations)
activ = 'action_net' # Choose the output layer to visualize
activation_acti = get_activations_bc(model, observations, activ)
empir = 'action_emp'
activation_acti[empir] = transitions_test.acts    # Empirical choices

# Apply filter
activation_data = filter_d(activation_data)
activation_acti = filter_d(activation_acti)

# %% 3. =======================================================================
# 3. Extract output layer for GLM
# =============================================================================
# Extract activations for both output layers
activation_all_act = get_activations_bc(model, observations, layer_names='action_net')
activation_all_val = get_activations_bc(model, observations, layer_names='value_net')

# Apply the same filtering as in your analysis
activation_filtered_act = filter_d(activation_all_act, filter_raw=True)
activation_filtered_val = filter_d(activation_all_val, filter_raw=True)

# Load the behavior data
d = pd.read_csv(path + 'data_beh/datall_cat.csv')

# Add action_net activations (could be multiple columns if action_net has multiple units)
action_net = activation_filtered_act['action_net']
for i in range(action_net.shape[1]):
    d[f'action_net'] = action_net[:, i]

# Add value_net activations
value_net = activation_filtered_val['value_net']
for i in range(value_net.shape[1]):
    d[f'value_net'] = value_net[:, i]

# Get all column names
all_columns = list(d.columns)

# Filter for columns >= 57
x57_plus_columns = all_columns[56:]

# Print in the requested format
print("Headers after x56:")
formatted_headers = [f"{{'{col[1:]}'}}" for col in x57_plus_columns]
print("[" + ", ".join(formatted_headers) + "]")

# %% 4. =======================================================================
# 4. Export subject-specific dataframes with neural network outputs
# =============================================================================
import os
import scipy.io
import numpy as np

# Create directory for output files if it doesn't exist
output_dir = os.path.join(path, "beh_v12_agg")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# Get unique subject IDs
subject_ids = d['x1_id'].unique()
print(f"Found {len(subject_ids)} unique subjects")

# Store column names as a separate variable to save with the data
column_names = d.columns.tolist()

# Apply z-score normalization to the pi-net (action_net) values
# Add the z-scored version as a new column
d['pi_net_z'] = (d['action_net'] - d['action_net'].mean()) / d['action_net'].std()

# Loop through each subject and export their data
for subject_id in subject_ids:
    # Filter data for this subject
    subject_df = d[d['x1_id'] == subject_id].copy()
    
    # Reset index
    subject_df = subject_df.reset_index(drop=True)
    
    # Convert DataFrame to numpy array
    subject_matrix = subject_df.values
    
    # Define output filename
    output_filename = f"beh_v12_sub{subject_id}.mat"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save to MAT file with a single matrix named 'x'
    scipy.io.savemat(output_path, {'x': subject_matrix})
    
    print(f"Exported data for subject {subject_id} to {output_filename}")

print(f"All subject data exported to {output_dir}")

# %% 5. =======================================================================
# 5. PCA and visualization
# =============================================================================
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""
Applies PCA to each layer's activations and visualizes the results.

Parameters:
- activation_data (dict): Dictionary containing layer activations as numpy arrays.
"""
pca_res = []
for layer, activations in activation_data.items():
    # Ensure activations are in 2D (flatten if necessary)
    activations = activations.reshape(activations.shape[0], -1)  # (samples, features)

    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced_activations = pca.fit_transform(activations)
    pca_res.append(reduced_activations)
    
    # Scatter plot visualization
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced_activations[:, 0], 
        reduced_activations[:, 1], 
        c=activation_acti[empir],  # Color by value predictions
        cmap='viridis',  # Use 'viridis' for better contrast
        edgecolors='k',
        alpha=0.5
    )

    # Add colorbar to interpret the values
    cbar = plt.colorbar(scatter)
    cbar.set_label("Predicted State Value")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"PCA Projection of {layer}")
    plt.show()

# %% 6. =======================================================================
# 6. Compare AGGREGATED empirical responses with network predictions
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes for better poster visibility
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 25,
    'figure.titlesize': 30
})

# Variable handling
pca_curr = pca_res[1]
pca_x = pca_curr[:,0]
pca_y = pca_curr[:,1]
net_values = activation_acti[activ].T[0]
empirical_labels = activation_acti[empir].T[0]
# Aggregation
feature_str = [str(activation_acti['features_extractor'][i]) for i in range(len(activation_acti['features_extractor']))]
df = pd.DataFrame({'features': feature_str, 'pca_x': pca_x, 'pca_y': pca_y, activ: net_values, empir: empirical_labels})
df_agg = df.groupby('features').agg({'pca_x': 'mean', 'pca_y': 'mean', activ: 'mean', empir: 'mean'}).reset_index()

# Set consistent min/max for both plots
x_min, x_max = df_agg['pca_x'].min(), df_agg['pca_x'].max()
y_min, y_max = df_agg['pca_y'].min(), df_agg['pca_y'].max()
# Add a small buffer for readability
x_buffer = (x_max - x_min) * 0.05
y_buffer = (y_max - y_min) * 0.05
x_limits = [x_min - x_buffer, x_max + x_buffer]
y_limits = [y_min - y_buffer, y_max + y_buffer]

# Create figure with fixed size and matching dimensions
figsize = (10, 8)
marker_size = 100
marker_edge_width = 1

# Create colormaps with the same scale for proper comparison
vmin = min(df_agg[activ].min(), df_agg[empir].min())
vmax = max(df_agg[activ].max(), df_agg[empir].max())

# 1. Scatter plot for predicted values
fig1, ax1 = plt.subplots(figsize=figsize)
sc1 = ax1.scatter(df_agg['pca_x'], df_agg['pca_y'], c=df_agg[activ], cmap='viridis', 
                 edgecolors='k', alpha=0.75, s=marker_size, linewidth=marker_edge_width,
                 vmin=vmin, vmax=vmax)

# Set axis limits for consistency
ax1.set_xlim(x_limits)
ax1.set_ylim(y_limits)

# Add colorbar with consistent formatting
cbar1 = fig1.colorbar(sc1, ax=ax1, pad=0.15, fraction=0.046)
cbar1.set_label("Predicted Action Value")

# Add labels and title
ax1.set_xlabel("Network Component 1")
ax1.set_ylabel("Network Component 2")
ax1.set_title("ANN Predicted Action Values", pad=20)

# Consistent grid and tick formatting
ax1.tick_params(axis='both', which='major', width=2, length=6)
ax1.grid(True, alpha=0.3)

plt.tight_layout(pad=3.0)

# 2. Scatter plot for foraging likelihood
fig2, ax2 = plt.subplots(figsize=figsize)
sc2 = ax2.scatter(df_agg['pca_x'], df_agg['pca_y'], c=df_agg[empir], cmap='viridis', 
                 edgecolors='k', alpha=0.75, s=marker_size, linewidth=marker_edge_width,
                 vmin=vmin, vmax=vmax)

# Add title
ax2.set_title("Aggregated Empirical Actions", pad=20)

# Use the same axis limits for consistency
ax2.set_xlim(x_limits)
ax2.set_ylim(y_limits)

# Add colorbar with the same dimensions and formatting
cbar2 = fig2.colorbar(sc2, ax=ax2, pad=0.15, fraction=0.046)
cbar2.set_label("Action Likelihood")

# Match labels
ax2.set_xlabel("Network Component 1")
ax2.set_ylabel("Network Component 2")

# Matching grid and tick formatting
ax2.tick_params(axis='both', which='major', width=2, length=6)
ax2.grid(True, alpha=0.3)

plt.tight_layout(pad=3.0)

# %% 7. =======================================================================
# 7. Use SHAP explainer to classify the feature importance
# =============================================================================
from sklearn.model_selection import train_test_split
import shap
plt.rcdefaults()

input_features = ['p_bad', 
            'p_good', 
            'p_success', 
            'food_gain',
            # 'weather_bad', 
            # 'weather_good', 
            # 'ternary_wait', 
            # 'ternary_norm', 
            # 'ternary_one',
            'horizon', 
            'gain_bad', 
            'gain_good', 
            'current_energy']

# Optional: Filter to 3x2 model range
activation_data = get_activations_bc(model, observations)
activ = 'action_net' # Choose the output layer to visualize
activation_acti = get_activations_bc(model, observations, activ)
empir = 'action_emp'
activation_acti[empir] = transitions_test.acts    # Empirical choices
activation_data = filter_d(activation_data, filter_strong=True)
activation_acti = filter_d(activation_acti, filter_strong=True)
# Redefine data for efficiency
X = activation_data['features_extractor']
y = activation_acti['action_net']
Y = activation_acti[empir]
# Split the data into training and testing sets
# Using test sample increases efficiency
X_train, X_test, y_train, y_test, actions_train, actions_test = train_test_split(
    X, y, Y, test_size=0.2, random_state=42)

# Model prediction function
def model_predict(observation):
    with torch.no_grad():
        # Convert observation to a PyTorch tensor (ensure it is float32)
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        
        # Get features and predict action
        features = model.policy.features_extractor(observation_tensor)
        pi_features, _ = model.policy.mlp_extractor(features)
        action_logits = model.policy.action_net(pi_features)
        
        # Return probabilities
        return torch.sigmoid(action_logits).detach().numpy().flatten().reshape(-1,1)

# Create a SHAP explainer for the BC model
explainer = shap.Explainer(model_predict, X_train)

# Calculate SHAP values for the test set
shap_values = explainer(X_test)

# Set publication-ready plot settings
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.format': 'tiff',
    'axes.linewidth': 1.5,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Visualize the SHAP values for the PCA-transformed test set
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=input_features, show=False)
plt.title('SHAP Feature Importance Summary', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Feature importance plot (bar chart)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=input_features, show=False)
plt.title('SHAP Feature Importance Ranking', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Mean |SHAP Value|', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Feature interactions
# ====================
# internal energy with food gain
feature1_index = input_features.index("current_energy")
feature2_index = input_features.index("food_gain")
plt.figure(figsize=(10, 8))
shap.dependence_plot(
    feature1_index, 
    shap_values.values, X_test, 
    interaction_index=feature2_index,
    feature_names=input_features,
    show=False)
plt.title('Feature Interaction: Current Energy × Food Gain', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Current Energy', fontsize=16, fontweight='bold')
plt.ylabel('SHAP Value for Current Energy', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# internal energy with p success
feature1_index = input_features.index("current_energy")
feature2_index = input_features.index("p_success")
plt.figure(figsize=(10, 8))
shap.dependence_plot(
    feature1_index, 
    shap_values.values, X_test, 
    interaction_index=feature2_index,
    feature_names=input_features,
    show=False)
plt.title('Feature Interaction: Current Energy × P(Success)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Current Energy', fontsize=16, fontweight='bold')
plt.ylabel('SHAP Value for Current Energy', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# food gain with gain_bad
feature1_index = input_features.index("food_gain")
feature2_index = input_features.index("gain_bad")
plt.figure(figsize=(10, 8))
shap.dependence_plot(
    feature1_index, 
    shap_values.values, X_test, 
    interaction_index=feature2_index,
    feature_names=input_features,
    show=False)
plt.title('Feature Interaction: Food Gain × Gain Bad', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Food Gain', fontsize=16, fontweight='bold')
plt.ylabel('SHAP Value for Food Gain', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# internal energy with horizon
feature1_index = input_features.index("current_energy")
feature2_index = input_features.index("horizon")
plt.figure(figsize=(10, 8))
shap.dependence_plot(
    feature1_index, 
    shap_values.values, X_test, 
    interaction_index=feature2_index,
    feature_names=input_features,
    show=False)
plt.title('Feature Interaction: Current Energy × Horizon', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Current Energy', fontsize=16, fontweight='bold')
plt.ylabel('SHAP Value for Current Energy', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% 8. =======================================================================
# 8. Feature attributions with integrated gradients
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
plt.rcdefaults()

def analyze_layer_attributions(model, layer_name, input_tensor):
    """
    Analyze feature importance by computing mean attributions for each feature using Integrated Gradients.
    
    Args:
        model: The trained BC model
        layer_name: Name of the layer to analyze
        input_tensor: Input features tensor
    """
    from captum.attr import IntegratedGradients, LayerIntegratedGradients
    
    class LayerWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            
            features = self.model.policy.features_extractor(x)
            pi_features, vf_features = self.model.policy.mlp_extractor(features)
            action_net = self.model.policy.action_net(pi_features)
            value_net = self.model.policy.value_net(vf_features)
            
            return {
                'features_extractor': features,
                'pi_features': pi_features,
                'vf_features': vf_features,
                'action_net': action_net,
                'value_net': value_net
            }
    
    wrapped_model = LayerWrapper(model)

    # Get layer output to determine number of neurons
    with torch.no_grad():
        test_output = wrapped_model(input_tensor)
        layer_output = test_output[layer_name]
        n_neurons = layer_output.shape[1]

    # Initialize attributions array
    attributions_all = np.zeros((n_neurons, len(input_tensor), input_tensor.shape[1]))
    
    # Create a specific forward function for each neuron
    def get_neuron_output(i):
        def forward_func(x):
            return wrapped_model(x)[layer_name][:, i]
        return forward_func
    
    # Create baseline (typically zeros)
    baseline = torch.zeros_like(input_tensor)
    
    # Calculate attributions for each neuron using Integrated Gradients
    for i in range(n_neurons):
        # Create an IntegratedGradients instance for this neuron
        ig = IntegratedGradients(get_neuron_output(i))
        
        # Calculate attributions using true Integrated Gradients
        # n_steps controls the precision of the approximation
        # internal_batch_size can be adjusted based on memory constraints
        attributions = ig.attribute(
            input_tensor,
            baselines=baseline,
            n_steps=50,  # Higher values give better approximation
            internal_batch_size=32,
            return_convergence_delta=False
        )
        
        attributions_all[i] = attributions.detach().numpy()
    
    avg_attributions = [np.mean(attributions_all[:,:,i]) for i in range(attributions_all.shape[2])]
    
    # Plot the average attributions as a bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(input_features, avg_attributions)
    plt.xticks(range(len(input_features)), input_features, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Average Attribution')
    plt.title(f'Feature Importance in {layer_name} Layer (True Integrated Gradients)')
    plt.tight_layout()
    plt.show()

    return attributions_all, avg_attributions

layer_names = [
    'pi_features', 
    'vf_features', 
    'action_net', 
    'value_net'
    ]

# Apply the function to analyze feature attributions
print("Analyzing feature attributions for pi_features layer")
neuron_activity_pi, feature_attributions_pi = analyze_layer_attributions(
    model,
    layer_names[0],
    torch.tensor(X_test, dtype=torch.float32)
)
print("Analyzing feature attributions for vf_features layer")
neuron_activity_vf, feature_attributions_vf = analyze_layer_attributions(
    model,
    layer_names[1],
    torch.tensor(X_test, dtype=torch.float32)
)
neuron_activities = [neuron_activity_pi, neuron_activity_vf]

# %% 9. =======================================================================
# 9. Representational dissimilarity matrix (RDM) for trials and neurons
# =============================================================================
from scipy.cluster import hierarchy
plt.rcdefaults()
# # Set global font sizes for better poster visibility
# plt.rcParams.update({
#     'font.size': 20,
#     'axes.titlesize': 30,
#     'axes.labelsize': 20,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20,
#     'legend.fontsize': 30,
#     'figure.titlesize': 25
# })

rsm_results = []
for i, neuron_activity in enumerate(neuron_activities):
    curr_layer = ["pi", "vf"][i]
    print("----------------------------------------------------------")
    print(f"Analyzing neuron activity for layer " + curr_layer)
    # Check the shape of the neuron activity
    print(f"Neuron activity shape: {neuron_activity.shape}")


    # RDM for trials
    # --------------------------------------------------
    # Reshape to get trial patterns (handling the 1-neuron case)
    if neuron_activity.shape[0] == 1:
        # If only 1 neuron, use the data directly
        trial_patterns = neuron_activity[0]  # Shape: (trials, features)
    else:
        # If multiple neurons, average across neurons
        trial_patterns = np.mean(neuron_activity, axis=0)

    print(f"Trial patterns shape: {trial_patterns.shape}")

    # Calculate pairwise distances (1 - correlation by default)
    dissimilarity = pdist(trial_patterns, metric='correlation')
    rdm = squareform(dissimilarity)

    # Plot the RDM using seaborn's heatmap with its built-in colorbar
    plt.figure(figsize=(10, 8))
    sns.heatmap(rdm, cmap='viridis', xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'Dissimilarity (1 - correlation)'})
    plt.title('Representational Dissimilarity Matrix (Trial × Trial)')
    plt.xlabel('Trials')
    plt.ylabel('Trials')
    plt.show()

    # Neuron correlation/dissimilarity
    # -------------------------------------------------
    neuron_activations = np.mean(neuron_activity, axis=1)  # Shape: (32, 1848)
    print(f"Neuron patterns shape: {neuron_activations.shape}")

    # Calculate neuron-to-neuron correlations
    correlation_matrix = np.corrcoef(neuron_activations)

    # Plot the correlation matrix with hierarchical clustering
    plt.figure(figsize=(12, 10))

    # Use hierarchical clustering to organize neurons
    linkage = hierarchy.linkage(correlation_matrix, 'ward')
    dendro = hierarchy.dendrogram(linkage, no_plot=False)
    idx = dendro['leaves']  # Reordering based on clustering

    # Plot the sorted correlation matrix
    sns.heatmap(correlation_matrix[idx, :][:, idx], cmap="RdBu_r", center=0, 
                vmin=-1, vmax=1, square=True, 
                xticklabels=idx, yticklabels=idx)
    plt.title('Neuron-to-Neuron Correlation Matrix')
    plt.xlabel('Neuron Index')
    plt.ylabel('Neuron Index')
    plt.tight_layout()
    plt.show()

# %% 9.2 ====================================================================
# 9.2. Factorial GLM Analysis: Trial Type × Weather Type Effects on ANN Activity
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
plt.rcdefaults()

input_features = ['p_bad', 
            'p_good', 
            'p_success', 
            'food_gain',
            # 'weather_bad', 
            # 'weather_good', 
            # 'ternary_wait', 
            # 'ternary_norm', 
            # 'ternary_one',
            'horizon', 
            'gain_bad', 
            'gain_good', 
            'current_energy']

print("=== Factorial GLM Analysis: ANN Activity by Trial Type and Weather ===")

# 🧠 1. DEFINE YOUR PREDICTORS
# =============================================================================
# Get full dataset (not just X_test)
activation_data = get_activations_bc(model, observations)
activ = 'action_net' 
activation_acti = get_activations_bc(model, observations, activ)
empir = 'action_emp'
activation_acti[empir] = transitions_test.acts
activation_data = filter_d(activation_data, filter_strong=True)
activation_acti = filter_d(activation_acti, filter_strong=True)

# Create comprehensive dataframe with all predictors
X_full = activation_data['features_extractor']
glm_df = pd.DataFrame(X_full, columns=input_features)

# Add behavioral data
d = pd.read_csv(path + 'data_beh/datall_cat.csv')
d = d[~d['x9_button_pressed'].isna()]
d = d[d['x6_continuous_energy_trial_start'] != 0]
d = d[d['x14_p_foraging_gain'] > 0.3]
d = d[d['x14_p_foraging_gain'] < 0.7]
d = d.reset_index(drop=True)

glm_df['subject'] = d['x1_id']
glm_df['empirical_choice'] = activation_acti[empir]

# 📊 2. DEFINE YOUR DEPENDENT VARIABLE (ANN "BOLD" signal analogue)
# =============================================================================
# Extract ANN activations for both layers
pi_activations = activation_data['pi_features']  # Policy network
vf_activations = activation_data['vf_features']  # Value network

# Create different DV options:
# Option 1: Mean activation across all units in policy layer
glm_df['ANN_pi_mean'] = np.mean(pi_activations, axis=1)

# Option 2: Mean activation across all units in value layer  
glm_df['ANN_vf_mean'] = np.mean(vf_activations, axis=1)

# Option 3: Max activation in policy layer (most active unit)
glm_df['ANN_pi_max'] = np.max(pi_activations, axis=1)

# Main Factors: Trial Type (Risk State)
conditions = []
for i in range(len(glm_df)):
    energy = glm_df.iloc[i]['current_energy']
    horizon = glm_df.iloc[i]['horizon']
    
    if energy == 1:
        conditions.append('BES')  # Basic Energy State (Critical)
    elif energy > horizon:
        conditions.append('WWS')  # Well-stocked State (Abundant)
    else:
        conditions.append('Trade-off')  # Trade-off State (Constrained)

glm_df['TrialType'] = pd.Categorical(conditions)

# Weather Type (Environmental Volatility)
weather_conditions = []
for i in range(len(glm_df)):
    p_bad = glm_df.iloc[i]['p_bad']
    p_good = glm_df.iloc[i]['p_good']
    p_current = glm_df.iloc[i]['p_success']
    
    if p_bad == p_current:
        p_other = p_good
        weather_conditions.append('Good')
    elif p_good == p_current:
        p_other = p_bad
        weather_conditions.append('Bad')
    else:
        # Fallback in case neither matches exactly
        weather_conditions.append('Unknown')
    
glm_df['WeatherType'] = pd.Categorical(weather_conditions)

# Parametric Modulator: Continuous probability of success
glm_df['p_current'] = glm_df['p_success']

# Z-score the parametric modulator for better interpretability
glm_df['p_current_z'] = (glm_df['p_current'] - glm_df['p_current'].mean()) / glm_df['p_current'].std()

print(f"Dataset contains {len(glm_df)} trials")
print(f"Trial Type distribution: {glm_df['TrialType'].value_counts()}")
print(f"Weather Type distribution: {glm_df['WeatherType'].value_counts()}")

# 🧪 3. FIT GENERAL LINEAR MODEL (GLM)
# =============================================================================
# Analyze both policy and value networks
layer_names = ['pi_features', 'vf_features']
dv_columns = ['ANN_pi_mean', 'ANN_vf_mean']

glm_results = {}

for layer_idx, (layer_name, dv_col) in enumerate(zip(layer_names, dv_columns)):
    print(f"\n{'='*60}")
    print(f"ANALYZING {layer_name.upper()} LAYER")
    print(f"{'='*60}")
    
    # Fit mixed-effects GLM with subject as random effect
    print("Fitting GLM: ANN_response ~ TrialType * WeatherType + p_current_z")
    
    # Main model with interaction
    formula = f"{dv_col} ~ TrialType * WeatherType + p_current_z"
    glm_model = smf.ols(formula, data=glm_df)
    result = glm_model.fit()
    
    glm_results[layer_name] = result
    
    print(result.summary())
    
    # Compute effect sizes and contrasts
    print(f"\n--- Effect Sizes for {layer_name} ---")
    params = result.params
    for param in params.index:
        if param != 'Intercept':
            cohen_d = params[param] / glm_df[dv_col].std()
            print(f"{param}: β = {params[param]:.4f}, Cohen's d ≈ {cohen_d:.4f}")

# 📉 4. VISUALIZE THE FACTORIAL EFFECTS
# ============================================================================
# Set publication-quality figure settings for manuscript readability
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,             # Increased base font size
    'axes.titlesize': 16,        # Larger title size
    'axes.labelsize': 14,        # Larger axis label size
    'xtick.labelsize': 12,       # Larger tick labels
    'ytick.labelsize': 12,       # Larger tick labels
    'legend.fontsize': 12,       # Larger legend text
    'figure.titlesize': 18,      # Larger figure title
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.format': 'tiff',
    'axes.linewidth': 1.5,       # Slightly thicker axes lines
    'grid.linewidth': 0.8,       # Slightly thicker grid lines
    'lines.linewidth': 2.0,      # Thicker plot lines
    'lines.markersize': 6,       # Larger markers
    'errorbar.capsize': 4        # Larger error bar caps
})

# Create comprehensive visualization with publication settings
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Factorial GLM: Trial Type × Weather Type Effects on ANN Activity', 
             fontsize=40, fontweight='bold')

for layer_idx, (layer_name, dv_col) in enumerate(zip(layer_names, dv_columns)):
    
    # Plot 1: Marginal means (main effects and interaction)
    ax1 = axes[layer_idx, 0]
    
    # Calculate marginal means for visualization
    marginal_means = glm_df.groupby(['TrialType', 'WeatherType'])[dv_col].agg(['mean', 'sem']).reset_index()
    
    # Create grouped bar plot
    trial_types = ['Trade-off', 'BES', 'WWS']
    weather_types = ['Bad', 'Good']
    
    x_pos = np.arange(len(trial_types))
    width = 0.35
    
    bad_means = [marginal_means[(marginal_means['TrialType'] == tt) & 
                               (marginal_means['WeatherType'] == 'Bad')]['mean'].iloc[0] 
                if len(marginal_means[(marginal_means['TrialType'] == tt) & 
                                    (marginal_means['WeatherType'] == 'Bad')]) > 0 else 0 
                for tt in trial_types]
    
    good_means = [marginal_means[(marginal_means['TrialType'] == tt) & 
                                (marginal_means['WeatherType'] == 'Good')]['mean'].iloc[0] 
                 if len(marginal_means[(marginal_means['TrialType'] == tt) & 
                                     (marginal_means['WeatherType'] == 'Good')]) > 0 else 0 
                 for tt in trial_types]
    
    bad_sems = [marginal_means[(marginal_means['TrialType'] == tt) & 
                              (marginal_means['WeatherType'] == 'Bad')]['sem'].iloc[0] 
               if len(marginal_means[(marginal_means['TrialType'] == tt) & 
                                   (marginal_means['WeatherType'] == 'Bad')]) > 0 else 0 
               for tt in trial_types]
    
    good_sems = [marginal_means[(marginal_means['TrialType'] == tt) & 
                               (marginal_means['WeatherType'] == 'Good')]['sem'].iloc[0] 
                if len(marginal_means[(marginal_means['TrialType'] == tt) & 
                                    (marginal_means['WeatherType'] == 'Good')]) > 0 else 0 
                for tt in trial_types]
    
    bars1 = ax1.bar(x_pos - width/2, bad_means, width, yerr=bad_sems, 
                   label='Bad Weather', color='#E64646', alpha=0.8, capsize=3)
    bars2 = ax1.bar(x_pos + width/2, good_means, width, yerr=good_sems,
                   label='Good Weather', color='#5D8AA8', alpha=0.8, capsize=3)
    
    ax1.set_xlabel('Trial Type (Risk State)')
    ax1.set_ylabel(f'Mean ANN Activity ({layer_name})')
    ax1.set_title(f'{layer_name}: Trial Type × Weather Interaction')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(trial_types)
    ax1.legend(frameon=False, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if bar1.get_height() > 0:
            ax1.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + bad_sems[i],
                    f'{bad_means[i]:.3f}', ha='center', va='bottom', fontsize=8)
        if bar2.get_height() > 0:
            ax1.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + good_sems[i],
                    f'{good_means[i]:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Parametric modulator effect (p_current)
    ax2 = axes[layer_idx, 1]
    
    # Scatter plot with separate regression lines for each trial type
    colors = {'Trade-off': '#FF7F50', 'BES': '#32CD32', 'WWS': '#4169E1'}
    linestyles = {'Trade-off': '-', 'BES': '--', 'WWS': '-.'}
    
    # First plot all scatter points without adding them to the legend
    for trial_type in trial_types:
        subset = glm_df[glm_df['TrialType'] == trial_type]
        
        # Skip if no data for this trial type
        if len(subset) == 0:
            continue
            
        # Plot scatter points without adding to legend
        ax2.scatter(subset['p_current'], subset[dv_col], 
                   alpha=0.5, c=colors[trial_type], s=15, edgecolor='none')
    
    # Then plot regression lines and add only those to the legend
    for trial_type in trial_types:
        subset = glm_df[glm_df['TrialType'] == trial_type]
        
        # Skip if no data or insufficient data for this trial type
        if len(subset) <= 1:
            continue
            
        # Fit regression line for this trial type
        z = np.polyfit(subset['p_current'], subset[dv_col], 1)
        p = np.poly1d(z)
        x_range = np.linspace(subset['p_current'].min(), subset['p_current'].max(), 100)
        ax2.plot(x_range, p(x_range), color=colors[trial_type], linestyle=linestyles[trial_type], 
                alpha=0.8, linewidth=2.0, label=f'{trial_type}')
    
    ax2.set_xlabel('P(Success) Current')
    ax2.set_ylabel(f'ANN Activity ({layer_name})')
    ax2.set_title(f'{layer_name}: Parametric Modulation by P(Success)')
    ax2.legend(frameon=False, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Residuals or model fit quality
    ax3 = axes[layer_idx, 2]
    
    # Create predicted vs observed plot
    predicted = glm_results[layer_name].fittedvalues
    observed = glm_df[dv_col]
    
    ax3.scatter(predicted, observed, alpha=0.5, color='#1E3F66', s=15, edgecolor='none')
    
    # Add perfect correlation line
    min_val = min(predicted.min(), observed.min())
    max_val = max(predicted.max(), observed.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1.5)
    
    # Calculate and display R²
    correlation = np.corrcoef(predicted, observed)[0, 1]
    r_squared = correlation ** 2
    ax3.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax3.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.5))
    
    ax3.set_xlabel('Predicted ANN Activity')
    ax3.set_ylabel('Observed ANN Activity')
    ax3.set_title(f'{layer_name}: Model Fit Quality')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure in publication quality
fig.savefig(os.path.join(path, 'factorial_glm_ann_activity.tiff'), 
           format='tiff', dpi=600, bbox_inches='tight')

plt.show()

# Print summary statistics for each condition
print(f"\n{'='*60}")
print("SUMMARY STATISTICS BY CONDITION")
print(f"{'='*60}")

for layer_idx, (layer_name, dv_col) in enumerate(zip(layer_names, dv_columns)):
    print(f"\n--- {layer_name.upper()} LAYER ---")
    summary_stats = glm_df.groupby(['TrialType', 'WeatherType'])[dv_col].agg([
        'count', 'mean', 'std', 'sem'
    ]).round(4)
    print(summary_stats)
    
    print(f"\nParametric Modulator Correlation (p_current vs {dv_col}):")
    # Calculate correlation by trial type
    for trial_type in trial_types:
        subset = glm_df[glm_df['TrialType'] == trial_type]
        if len(subset) > 1:  # Need at least 2 points for correlation
            correlation = subset['p_current'].corr(subset[dv_col])
            print(f"{trial_type}: r = {correlation:.4f}")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")

# %% 9.3 ======================================================================
# 9.3 Analyze interactions between features and neuron activity with 3D plots
# =============================================================================
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
plt.rcdefaults()

# Update font sizes for publication
plt.rcParams.update({
    'font.size': 20,               # Base font size
    'axes.titlesize': 28,          # Title font size
    'axes.labelsize': 20,          # Label font size
    'xtick.labelsize': 16,         # X-tick font size
    'ytick.labelsize': 16,         # Y-tick font size
    'legend.fontsize': 16,         # Legend font size
    'figure.titlesize': 40,        # Figure title size
})

# Get data
activation_data = get_activations_bc(model, observations)
activ = 'action_net'
activation_acti = get_activations_bc(model, observations, activ)
empir = 'action_emp'
activation_acti[empir] = transitions_test.acts
activation_data = filter_d(activation_data, filter_strong=True)
activation_acti = filter_d(activation_acti, filter_strong=True)

# Create DataFrame with all necessary data
df_3d = pd.DataFrame()
df_3d['p_success'] = activation_data['features_extractor'][:, input_features.index('p_success')]
df_3d['current_energy'] = activation_data['features_extractor'][:, input_features.index('current_energy')]
df_3d['horizon'] = activation_data['features_extractor'][:, input_features.index('horizon')]
df_3d['p_bad'] = activation_data['features_extractor'][:, input_features.index('p_bad')]
df_3d['p_good'] = activation_data['features_extractor'][:, input_features.index('p_good')]
df_3d['action_net'] = activation_acti['action_net'].flatten()
df_3d['action_net'] = np.round(df_3d['action_net'], 2)  # Round to 2 decimal places for better readability

# Determine p_other (the alternative probability that's not p_success)
df_3d['p_other'] = df_3d.apply(
    lambda row: row['p_bad'] if row['p_success'] != row['p_bad'] else row['p_good'],
    axis=1
)

# Define trial types
conditions = []
for i in range(len(df_3d)):
    energy = df_3d.iloc[i]['current_energy']
    horizon = df_3d.iloc[i]['horizon']
    
    if energy == 1:
        conditions.append('BES')  # Basic Energy State (Critical)
    elif energy > horizon:
        conditions.append('WWS')  # Well-stocked State (Abundant)
    else:
        conditions.append('Trade-off')  # Trade-off State (Constrained)

df_3d['trial_type'] = conditions

# Create separate 3D plots for each trial type
trial_types = ['BES', 'Trade-off', 'WWS']
fig = plt.figure(figsize=(20, 8))  # Reduced height and width for tighter spacing

for i, trial_type in enumerate(trial_types):
    # Filter data for current trial type
    type_data = df_3d[df_3d['trial_type'] == trial_type]
    
    if len(type_data) == 0:
        print(f"No data for trial type: {trial_type}")
        continue
    
    # Create subplot
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    
    # Create scatter plot with larger points
    scatter = ax.scatter(
        type_data['p_success'], 
        type_data['p_other'],
        type_data['action_net'],
        c=type_data['action_net'],
        cmap='viridis',
        alpha=0.6,
        s=100  # Increased point size
    )
    
    # Create a consistent normalization for all plots
    norm = plt.Normalize(df_3d['action_net'].min(), df_3d['action_net'].max())
    
    # Use the normalized colormap for the scatter
    scatter = ax.scatter(
        type_data['p_success'], 
        type_data['p_other'],
        type_data['action_net'],
        c=type_data['action_net'],
        cmap='viridis',
        alpha=0.6,
        s=100,  # Increased point size
        norm=norm  # Apply the consistent normalization
    )
    
    # Set consistent z-axis limits for all plots
    ax.set_zlim(df_3d['action_net'].min(), df_3d['action_net'].max())
    
    # Don't add individual colorbars here
    
    # Only add the central colorbar after the last plot
    if i == len(trial_types) - 1:
        # Create a single colorbar for all subplots with reduced size
        cbar_ax = fig.add_axes([0.95, 0.25, 0.015, 0.5])  # [left, bottom, width, height] - moved closer
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('ANN Output', size=25, labelpad=15)
        cbar.ax.tick_params(labelsize=20)
        # Format colorbar ticks to show only 1 digit after decimal point
        cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    # Fit a surface to better visualize trends
    if len(type_data) > 10:  # Only try to fit if we have enough data points
        try:
            # Create grid for surface
            x_range = np.linspace(type_data['p_success'].min(), type_data['p_success'].max(), 20)
            y_range = np.linspace(type_data['p_other'].min(), type_data['p_other'].max(), 20)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Fit a 2D polynomial (quadratic)
            Z = griddata(
                (type_data['p_success'], type_data['p_other']), 
                type_data['action_net'], 
                (X, Y), 
                method='cubic',
                fill_value=np.nan
            )
            
            # Plot the surface with the same colormap and normalization as the scatter points
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', norm=norm, alpha=0.5, linewidth=0, antialiased=True)
            
            # Add reference planes to improve readability
            # Get min and max values for axes
            x_min, x_max = type_data['p_success'].min(), type_data['p_success'].max()
            y_min, y_max = type_data['p_other'].min(), type_data['p_other'].max()
            z_min, z_max = type_data['action_net'].min(), type_data['action_net'].max()
            
            # Bottom plane (z_min)
            xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
            zz = np.ones_like(xx) * z_min
            ax.plot_surface(xx, yy, zz, color='gray', alpha=0.2)
            
            # Back plane (y_max)
            xx, zz = np.meshgrid([x_min, x_max], [z_min, z_max])
            yy = np.ones_like(xx) * y_max
            ax.plot_surface(xx, yy, zz, color='gray', alpha=0.2)
            
            # Side plane (x_min)
            yy, zz = np.meshgrid([y_min, y_max], [z_min, z_max])
            xx = np.ones_like(yy) * x_min
            ax.plot_surface(xx, yy, zz, color='gray', alpha=0.2)
            
            # Add grid lines for better orientation
            ax.grid(True, linestyle='--', alpha=0.5, linewidth=2)  # Thicker grid lines
            
            # Format ticks to show only 2 decimal places
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            ax.zaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        except Exception as e:
            print(f"Could not fit surface for {trial_type}: {e}")
    
    # Set labels and title with increased labelpad for leftmost plot
    if i == 0:  # Leftmost plot
        ax.set_xlabel('P(Success)', fontsize=20, labelpad=15)  # Increased padding for visibility
        ax.set_ylabel('P(Alternative)', fontsize=20, labelpad=15)  # Increased padding for visibility
        ax.set_zlabel('ANN Output', fontsize=20, labelpad=15)  # Increased padding for visibility
    else:
        ax.set_xlabel('P(Success)', fontsize=20, labelpad=5)  # Reduced padding for other plots
        ax.set_ylabel('P(Alternative)', fontsize=20, labelpad=5)  # Reduced padding for other plots
        ax.set_zlabel('ANN Output', fontsize=20, labelpad=5)  # Reduced padding for other plots
    
    ax.set_title(f'{trial_type}', fontsize=30)
    
    # Set optimal viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add ticks and tick labels
    ax.set_xticks(np.linspace(type_data['p_success'].min(), type_data['p_success'].max(), 3))
    ax.set_yticks(np.linspace(type_data['p_other'].min(), type_data['p_other'].max(), 3))
    ax.set_zticks(np.linspace(type_data['action_net'].min(), type_data['action_net'].max(), 3))
    
    # Format tick labels with different padding for leftmost plot
    if i == 0:  # Leftmost plot
        ax.tick_params(axis='x', labelsize=16, pad=5)  # More padding for visibility
        ax.tick_params(axis='y', labelsize=16, pad=5)  # More padding for visibility
        ax.tick_params(axis='z', labelsize=16, pad=5)  # More padding for visibility
    else:
        ax.tick_params(axis='x', labelsize=16, pad=2)  # Reduced padding for other plots
        ax.tick_params(axis='y', labelsize=16, pad=2)  # Reduced padding for other plots
        ax.tick_params(axis='z', labelsize=16, pad=2)  # Reduced padding for other plots
    
    # Thicker axes lines for better visibility
    ax.xaxis.line.set_linewidth(2)
    ax.yaxis.line.set_linewidth(2)
    ax.zaxis.line.set_linewidth(2)

plt.tight_layout(pad=1.0)  # Reduced padding between subplots
plt.suptitle('Parametric Modulation by P(Success) and P(Alternative) Across Trial Types', 
            fontsize=40, y=0.98)  # Reduced title size and moved closer to plots
plt.subplots_adjust(wspace=0.15, top=0.9, right=0.9, left=0.05, bottom=0.05)  # Added left margin and slightly increased wspace
plt.show()

# Create 2D contour plots as an alternative visualization
fig, axes = plt.subplots(1, 3, figsize=(20, 8))  # Increased figure size

# Create consistent normalization for all 2D plots
vmin_2d = df_3d['action_net'].min()
vmax_2d = df_3d['action_net'].max()
norm_2d = plt.Normalize(vmin_2d, vmax_2d)

for i, trial_type in enumerate(trial_types):
    # Filter data for current trial type
    type_data = df_3d[df_3d['trial_type'] == trial_type]
    
    if len(type_data) == 0:
        axes[i].text(0.5, 0.5, f"No data for {trial_type}", 
                   horizontalalignment='center', verticalalignment='center',
                   fontsize=30)  # Increased font size for message
        continue
    
    # Create contour plot
    if len(type_data) > 10:
        try:
            # Create grid for contour
            x_range = np.linspace(type_data['p_success'].min(), type_data['p_success'].max(), 20)
            y_range = np.linspace(type_data['p_other'].min(), type_data['p_other'].max(), 20)
            X, Y = np.meshgrid(x_range, y_range)
            
            # Fit a 2D polynomial using griddata
            Z = griddata(
                (type_data['p_success'], type_data['p_other']), 
                type_data['action_net'], 
                (X, Y), 
                method='cubic',
                fill_value=np.nan
            )
            
            # Plot the contour with consistent normalization
            contour = axes[i].contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.8, 
                                     vmin=vmin_2d, vmax=vmax_2d)
            # Add points with larger size and consistent normalization
            axes[i].scatter(type_data['p_success'], type_data['p_other'], 
                          c=type_data['action_net'], cmap='viridis', norm=norm_2d,
                          edgecolors='k', alpha=0.6, s=100)
            
            # Add contour lines with labels
            contour_lines = axes[i].contour(X, Y, Z, levels=5, colors='black', 
                                          alpha=0.7, linewidths=2.0)  # Thicker contour lines
            axes[i].clabel(contour_lines, inline=True, fontsize=18, fmt='%.2f')  # Larger contour labels
            
        except Exception as e:
            print(f"Could not create contour for {trial_type}: {e}")
            # If contour fails, just show scatter with consistent normalization
            axes[i].scatter(type_data['p_success'], type_data['p_other'], 
                          c=type_data['action_net'], cmap='viridis', norm=norm_2d,
                          alpha=0.6, s=100)  # Larger points
    else:
        # If too few points, just show scatter with consistent normalization
        axes[i].scatter(type_data['p_success'], type_data['p_other'], 
                      c=type_data['action_net'], cmap='viridis', norm=norm_2d,
                      alpha=0.6, s=100)  # Larger points
    
    # Set labels and title with increased font sizes
    axes[i].set_xlabel('P(Success)', fontsize=20, labelpad=15)
    axes[i].set_ylabel('P(Alternative)', fontsize=20, labelpad=15)
    axes[i].set_title(f'{trial_type}', fontsize=30)
    
    # Thicker grid lines
    axes[i].grid(True, alpha=0.3, linewidth=1.5)
    
    # Larger tick labels
    axes[i].tick_params(axis='both', which='major', labelsize=20, width=2, length=10, pad=10)
    
    # Thicker frame
    for spine in axes[i].spines.values():
        spine.set_linewidth(2)

# Add a single colorbar for all 2D plots
cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm_2d)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('ANN Output', size=25, labelpad=15)
cbar.ax.tick_params(labelsize=20)
# Format colorbar ticks to show only 1 digit after decimal point
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

plt.tight_layout(pad=5.0)  # Increased padding
plt.suptitle('2D View: Action Network Response to P(Success) and P(Alternative)', 
            fontsize=40, y=1.1)  # Larger title and adjusted position
plt.subplots_adjust(top=0.85, wspace=0.3, right=0.95)  # Adjusted spacing and right margin for colorbar
plt.show()

# %% 10. ======================================================================
# 10. Analyze interactions between features and neuron activity
# =============================================================================
from statsmodels.formula.api import ols
plt.rcdefaults()

layer_names = ["pi_features", "vf_features"]

for layer_idx, layer_name in enumerate(layer_names):
    print(f"\nAnalyzing interactions for {layer_name}")
    
    # Create dataframe with features and mean neuron activity
    interaction_df = pd.DataFrame(X_test, columns=input_features)
    
    # Get mean neuron activity for the current layer from the test set data
    if layer_name == "pi_features":
        # Use neuron_activity_pi which is already based on X_test
        neuron_data = neuron_activities[0]
        # Calculate mean across neurons for each sample
        mean_activity = np.mean(neuron_data, axis=0)
        mean_activity = np.mean(mean_activity, axis=1)
    else:
        # Use neuron_activity_vf which is already based on X_test
        neuron_data = neuron_activities[1]
        # Calculate mean across neurons for each sample
        mean_activity = np.mean(neuron_data, axis=0)
        mean_activity = np.mean(mean_activity, axis=1)
    interaction_df['mean_neuron_response'] = mean_activity

    # Some Feature Engineering
    # For each row, choose the value (p_bad or p_good) that is not equal to p_success
    interaction_df['p_other'] = interaction_df.apply(
        lambda row: row['p_bad'] if row['p_bad'] != row['p_success'] else row['p_good'],
        axis=1
    )
    interaction_df['p_current'] = interaction_df['p_success']  # Use p_success as the current probability gain
    
    # Split data into three cases
    case1_df = interaction_df[interaction_df['current_energy'] == 1].copy()
    case2_df = interaction_df[(interaction_df['current_energy'] > interaction_df['horizon']) & 
                             (interaction_df['current_energy'] != 1)].copy()
    # For case 3, explicitly exclude both case 1 and case 2 conditions
    case3_df = interaction_df[(interaction_df['current_energy'] <= interaction_df['horizon']) & 
                             (interaction_df['current_energy'] != 1)].copy()
    
    cases = [
        ("Case 1: Energy = 1", case1_df),
        ("Case 2: Energy > Horizon (excluding Energy = 1)", case2_df),
        ("Case 3: Energy ≤ Horizon (excluding Energy = 1)", case3_df)
    ]
    
    for case_name, case_df in cases:
        if len(case_df) == 0:
            print(f"{case_name} has no data points. Skipping...")
            continue
            
        print(f"\n=== {case_name} ===")
        print(f"Number of samples: {len(case_df)}")
        
        # Test how neurons respond to interactions between energy and horizon
        interactions = []
        print("-----------------------------------------------------------")
        print("Testing interaction between current energy and horizon")
        try:
            model_interaction = ols('mean_neuron_response ~ current_energy * horizon', data=case_df).fit()
            interactions.append(model_interaction)
            print(model_interaction.summary())
        except Exception as e:
            print(f"Error fitting energy-horizon model: {e}")
            interactions.append(None)
            
        # Test how neurons respond to interactions between p_bad and p_success
        print("-----------------------------------------------------------")
        print("Testing interaction between p_other and p_current")
        try:
            model_interaction = ols('mean_neuron_response ~ p_other * p_current', data=case_df).fit()
            print(model_interaction.summary())    
            interactions.append(model_interaction)
        except Exception as e:
            print(f"Error fitting p_other-p_current model: {e}")
            interactions.append(None)
            
        print("-----------------------------------------------------------")
        print("Testing full interaction")
        try:
            model_interaction = ols('mean_neuron_response ~ current_energy * horizon * p_other * p_current', data=case_df).fit()
            print(model_interaction.summary())    
            interactions.append(model_interaction)
        except Exception as e:
            print(f"Error fitting full interaction model: {e}")
            interactions.append(None)

        # Visualize the interactions if models were successfully fit
        if interactions[0] is not None:
            # Create grid for visualization - energy and horizon
            energy_range = np.linspace(case_df['current_energy'].min(), 
                                      case_df['current_energy'].max(), 20)
            horizon_range = np.linspace(case_df['horizon'].min(), 
                                      case_df['horizon'].max(), 20)
            energy_grid, horizon_grid = np.meshgrid(energy_range, horizon_range)

            # Create new dataframe with all combinations for prediction
            predict_df = pd.DataFrame({
                'current_energy': energy_grid.flatten(),
                'horizon': horizon_grid.flatten()
            })
            
            # Get predictions from the model
            predictions = interactions[0].predict(predict_df)
            
            # Convert predictions to numpy array before reshaping
            predictions_array = predictions.to_numpy()
            
            # 3D surface plot and heatmap for energy-horizon interaction
            fig = plt.figure(figsize=(12, 10))
            
            ax1 = fig.add_subplot(2, 1, 1, projection='3d')
            surf = ax1.plot_surface(energy_grid, horizon_grid, 
                                predictions_array.reshape(energy_grid.shape),
                                cmap='viridis', alpha=0.8)
            ax1.view_init(elev=30, azim=-50)
            ax1.set_xlabel('Current Energy', labelpad=15)
            ax1.set_ylabel('Horizon', labelpad=15)
            ax1.set_zlabel('Predicted Neuron Response', labelpad=15)
            ax1.set_title(f'{case_name}: Energy-Horizon Interaction on {layer_name}')
            fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, pad=0.15)
            
            ax2 = fig.add_subplot(2, 1, 2)
            heatmap = ax2.pcolormesh(energy_grid, horizon_grid, 
                                    predictions_array.reshape(energy_grid.shape), 
                                    cmap='viridis', shading='auto')
            ax2.set_xlabel('Current Energy', labelpad=15)
            ax2.set_ylabel('Horizon', labelpad=15)
            ax2.set_title(f'{case_name}: Heatmap of {layer_name} Response')
            fig.colorbar(heatmap, ax=ax2, pad=0.15)
            
            plt.subplots_adjust(hspace=0.4)
            plt.tight_layout(pad=3.0)
            plt.show()

        if interactions[1] is not None:
            # Create grid for visualization - p_other and p_current
            pOthr_range = np.linspace(case_df['p_other'].min(), 
                                      case_df['p_other'].max(), 20)
            pCurr_range = np.linspace(case_df['p_current'].min(), 
                                      case_df['p_current'].max(), 20)
            pOthr_grid, pCurr_grid = np.meshgrid(pOthr_range, pCurr_range)

            # Create new dataframe with all combinations for prediction
            predict_df = pd.DataFrame({
                'p_other': pOthr_grid.flatten(),
                'p_current': pCurr_grid.flatten()
            })
            
            # Get predictions from the model
            predictions = interactions[1].predict(predict_df)
            predictions_array = predictions.to_numpy()
            
            # 3D surface plot and heatmap for p_other-p_current interaction
            fig = plt.figure(figsize=(12, 10))
            
            ax1 = fig.add_subplot(2, 1, 1, projection='3d')
            surf = ax1.plot_surface(pOthr_grid, pCurr_grid, 
                                predictions_array.reshape(pOthr_grid.shape),
                                cmap='viridis', alpha=0.8)
            ax1.view_init(elev=40, azim=-50)
            ax1.set_xlabel('p other', labelpad=15)
            ax1.set_ylabel('p current', labelpad=15)
            ax1.set_zlabel('Predicted Neuron Response', labelpad=15)
            ax1.set_title(f'{case_name}: p_other-p_current Interaction on {layer_name}')
            fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, pad=0.15)
            
            ax2 = fig.add_subplot(2, 1, 2)
            heatmap = ax2.pcolormesh(pOthr_grid, pCurr_grid, 
                                    predictions_array.reshape(pOthr_grid.shape), 
                                    cmap='viridis', shading='auto')
            ax2.set_xlabel('p other', labelpad=15)
            ax2.set_ylabel('p current', labelpad=15)
            ax2.set_title(f'{case_name}: Heatmap of {layer_name} Response')
            fig.colorbar(heatmap, ax=ax2, pad=0.15)
            
            plt.subplots_adjust(hspace=0.4)
            plt.tight_layout(pad=3.0)
            plt.show()
        
        if interactions[2] is not None:
            # Create 2D visualization of energy vs p_current interaction from the full model
            energy_range = np.linspace(case_df['current_energy'].min(), 
                                      case_df['current_energy'].max(), 20)
            pCurr_range = np.linspace(case_df['p_current'].min(), 
                                     case_df['p_current'].max(), 20)
            energy_grid, pCurr_grid = np.meshgrid(energy_range, pCurr_range)
            
            # Use mean values for the other parameters
            mean_horizon = case_df['horizon'].mean()
            mean_p_other = case_df['p_other'].mean()
            
            # Create new dataframe with all combinations for prediction
            predict_df = pd.DataFrame({
                'current_energy': energy_grid.flatten(),
                'p_current': pCurr_grid.flatten(),
                'horizon': np.full(len(energy_grid.flatten()), mean_horizon),
                'p_other': np.full(len(energy_grid.flatten()), mean_p_other)
            })
            
            # Get predictions from the model
            predictions = interactions[2].predict(predict_df)
            predictions_array = predictions.to_numpy()
            
            # 3D surface plot and heatmap for energy-p_current interaction
            fig = plt.figure(figsize=(12, 10))
            
            ax1 = fig.add_subplot(2, 1, 1, projection='3d')
            surf = ax1.plot_surface(energy_grid, pCurr_grid, 
                                predictions_array.reshape(energy_grid.shape),
                                cmap='viridis', alpha=0.8)
            ax1.view_init(elev=30, azim=-50)
            ax1.set_xlabel('Current Energy', labelpad=15)
            ax1.set_ylabel('P Current', labelpad=15)
            ax1.set_zlabel('Predicted Neuron Response', labelpad=15)
            ax1.set_title(f'{case_name}: Energy-P_current Interaction on {layer_name}\n(horizon={mean_horizon:.2f}, p_other={mean_p_other:.2f})')
            fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, pad=0.15)
            
            ax2 = fig.add_subplot(2, 1, 2)
            heatmap = ax2.pcolormesh(energy_grid, pCurr_grid, 
                                    predictions_array.reshape(energy_grid.shape), 
                                    cmap='viridis', shading='auto')
            ax2.set_xlabel('Current Energy', labelpad=15)
            ax2.set_ylabel('P Current', labelpad=15)
            ax2.set_title(f'{case_name}: Heatmap of {layer_name} Response')
            fig.colorbar(heatmap, ax=ax2, pad=0.15)
            
            plt.subplots_adjust(hspace=0.4)
            plt.tight_layout(pad=3.0)
            plt.show()

# %% 11. ======================================================================
# 11. Analyze value net correlation with log(RT)
# =============================================================================
import statsmodels.formula.api as smf

# Get data
activation_data = get_activations_bc(model, observations)
activ = 'value_net'
activation_acti = get_activations_bc(model, observations, activ)
empir = 'action_emp'
activation_acti[empir] = transitions_test.acts
d = pd.read_csv(path + 'data_beh/datall_cat.csv')
# Filter data
activation_data = filter_d(activation_data)#, filter_strong=True)
activation_acti = filter_d(activation_acti)#, filter_strong=True)
d = d[~d['x9_button_pressed'].isna()]
d = d[d['x6_continuous_energy_trial_start'] != 0]
# d = d[d['x14_p_foraging_gain'] > 0.3]
# d = d[d['x14_p_foraging_gain'] < 0.7]
d = d.reset_index(drop=True)

# Data handling
d['subject'] = d['x1_id']
d['value_net'] = activation_acti['value_net']
d['logits_scaled'] = (d['value_net'] - 
                      d['value_net'].mean()) / (
                          d['value_net'].std())
d['log_rt'] = d['x26_logRT']

# Fit linear regression using statsmodel variational Bayes mean field
md = smf.mixedlm("log_rt ~ logits_scaled", data=d, groups=d["subject"])
mdf = md.fit()

# Print the results
print(mdf.summary())
# Create prediction line
X_plot = np.linspace(d['logits_scaled'].min(), d['logits_scaled'].max(), 100)
y_pred = mdf.params['Intercept'] + mdf.params['logits_scaled'] * X_plot

# Plot the data and regression line with publication-ready formatting
plt.figure(figsize=(12, 8))

# Set publication-ready style parameters
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 25,
    'axes.labelsize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'grid.linewidth': 1.0
})

# Create the scatter plot with larger markers
plt.scatter(d['logits_scaled'], d['log_rt'], alpha=0.4, color='steelblue', 
           s=25, label='Observed data', edgecolors='darkblue', linewidth=0.3)

# Plot regression line with confidence interval if available
plt.plot(X_plot, y_pred, color='darkred', label='Mixed-effects regression', 
         linewidth=3.5, linestyle='-')

# Add regression statistics as text
r_squared = mdf.rsquared if hasattr(mdf, 'rsquared') else 'N/A'
slope = mdf.params['logits_scaled']
intercept = mdf.params['Intercept']
p_value = mdf.pvalues['logits_scaled'] if 'logits_scaled' in mdf.pvalues else 'N/A'

# Format statistics text
stats_text = f'β = {slope:.3f}\np < {p_value:.3f}' if p_value != 'N/A' else f'β = {slope:.3f}'
if r_squared != 'N/A':
    stats_text += f'\nR² = {r_squared:.3f}'

# Add statistics box
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, 
                  edgecolor='gray', linewidth=1),
         verticalalignment='top', fontsize=16, family='monospace')

# Enhanced labels and formatting
plt.xlabel('Standardized Value Network Output', fontweight='bold', labelpad=12)
plt.ylabel('Log Response Time (RT)', labelpad=12)
plt.title('Neural Value Coding Predicts Decision Latency', 
          fontweight='bold', pad=20, fontsize=30)

# Improve legend
plt.legend(frameon=True, fancybox=True, shadow=True, 
          framealpha=0.9, edgecolor='gray', loc='upper right')

# Enhanced grid
plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

# Remove top and right spines for cleaner look
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# Improve tick formatting
plt.tick_params(axis='both', which='major', length=6, width=1.2, 
                direction='out', top=False, right=False)

# Tight layout for better spacing
plt.tight_layout(pad=2.0)

# Add sample size information
n_obs = len(d)
n_subjects = d['subject'].nunique()
plt.figtext(0.02, 0.02, f'N = {n_obs} trials, {n_subjects} subjects', 
           fontsize=16, style='italic', alpha=0.7)

plt.show()

# Compute partial correlation log_rt ~ value_net
import pingouin as pg
partial_corr = pg.partial_corr(data=d, x='logits_scaled', y='log_rt', covar='subject', method='spearman')
print("Partial correlation between log(RT) and value_net output:", partial_corr['r'].values[0])

# =============================================================================
# =============================================================================
# =============================================================================
'''
=== The XAI process is now complete with these steps ===
- Visualized the network's policy net with PCA
- SHAP to interpret the feature importance
- feature attribition and interactions
- RDMs over trials (for RSA) and neurons
- Compared value net with response time
'''
# =============================================================================
# =============================================================================
# =============================================================================
# %% 11. ======================================================================
# 11. Predict ANN output with hierarchical logistic regression
# =============================================================================
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from scipy.interpolate import griddata
from matplotlib import cm
import scipy.sparse as sp

# Get cropped data
activation_data = get_activations_bc(model, observations)
activ = 'action_net'
activation_acti = get_activations_bc(model, observations, activ)
empir = 'action_emp'
activation_acti[empir] = transitions_test.acts
activation_data = filter_d(activation_data, filter_strong=True)
activation_acti = filter_d(activation_acti, filter_strong=True)

# Data set
d = pd.read_csv(path + 'data_beh/datall_cat.csv')
# Filter data
d = d[~d['x9_button_pressed'].isna()]
d = d[d['x6_continuous_energy_trial_start'] != 0]
d = d[d['x14_p_foraging_gain'] > 0.3]
d = d[d['x14_p_foraging_gain'] < 0.7]
d = d.reset_index(drop=True)

# Variable handling
d['subject'] = d['x1_id']
d['acts'] = d['x11_choice']
d['BNW_fact'] = pd.Categorical(d['BNW_conditions'])
d['weather_fact'] = pd.Categorical(d['x7_weather_type'])
d['ANN_output'] = activation_acti['action_net']
d['logits_scaled'] = (d['ANN_output'] - 
                      d['ANN_output'].mean()) / (
                          d['ANN_output'].std())
threshold = d['logits_scaled'].mean()
d['ANN_acts'] = np.where(d['logits_scaled'] > threshold, 1, 0)

print("run factorial model on ANN output")

# Get additional model
mod = 'x14_p_foraging_gain'
d['DV_main'] = d[mod]
# print(mod + ' as pMod')

# Fit logistic regression using statsmodel variational Bayes mean field
random = {"subject": '0 + C(subject)'}  # Random intercept for each subject
log_reg = BinomialBayesMixedGLM.from_formula(
                'ANN_acts ~ BNW_fact * weather_fact * DV_main', random, d)
result = log_reg.fit_vb()

# Extract Mean (Posterior Mean) and Standard Deviation (Posterior SD)
posterior_mean = result.params  # Mean of posterior
# Posterior Standard Deviations (square root of diagonal of posterior covariance)
posterior_cov = result.cov_params()  # Approximate covariance matrix
posterior_sd = sp.csr_matrix(np.sqrt(np.diag(posterior_cov))).data  # Standard deviations

# Model evidence using evidence lower bound
elbo = log_reg.vb_elbo(vb_mean=posterior_mean, vb_sd=posterior_sd)
print(f"evidence lower bound 3x2 factorial = {elbo:.3f}")

# Predict actions
actions_pred = result.predict()
# Area under the curve
auc = roc_auc_score(d['acts'], actions_pred)
print(f"AUC-ROC = {auc:.3f}")

# Accuracy and confusion matrix
preds_hard = np.where(actions_pred > 0.5, 1, 0)
accuracy = accuracy_score(d['ANN_acts'], preds_hard)
cm = confusion_matrix(d['ANN_acts'], preds_hard)
print(f"Mean accuracy: {accuracy}")
print(f"Mean confusion matrix:\n{cm}")

