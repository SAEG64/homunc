#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:32:06 2025

@author: sergej
"""
# %% ==========================================================================
# Set up requirements
# =============================================================================
from imitation.data.wrappers import RolloutInfoWrapper
from gymnasium.envs.registration import register
from imitation.util.util import make_vec_env
from imitation.algorithms import bc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

SEED = 42
path = os.path.dirname(os.path.abspath(__file__)) + "/"
rng = np.random.default_rng(SEED)

# Custom gymnasium environment
# ============================
register(
    id="ForagingEnv-v0",
    entry_point="foraging_env_REV_V2:ForestEnv",  # Update with the correct module path
)

# Create the environment
vec_env = make_vec_env(
    "ForagingEnv-v0",
    rng=rng,
    n_envs=5,                  # Number of parallel environments
    parallel=False,            # Use parallelism (SubprocVecEnv)
    post_wrappers=[
        lambda env, _: RolloutInfoWrapper(env)  # needed for computing rollouts later
    ]
)
    
# %% ==========================================================================
# Train model
# =============================================================================
from expert_trajectories_beh_V2 import transitions#, summary_stats
import scipy.sparse as sp

# Variable batch sizes
batch_sizes = [32, 64, 128]

# Set up the model
bc_trainer = bc.BC(
    observation_space=vec_env.observation_space,
    action_space=vec_env.action_space,
    demonstrations=transitions,
    batch_size=batch_sizes[1],
    rng=rng
)

bc_trainer.train(n_epochs=40)

# Evaluate model performance
# ==========================
if __name__ == "__main__":
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from sklearn.metrics import accuracy_score, confusion_matrix
    # from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    # from scipy.stats import spearmanr
    import torch

    from expert_trajectories_REV_V2 import transitions as transitions_test
    
    print("==================================")
    print("Evaluating BC model performance...")

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

    # Get activations
    activations = get_activations_bc(bc_trainer, 
                                    transitions_test.obs, 
                                    'action_net')
    activations_df = pd.DataFrame(activations['action_net'])
    activations_df['logits'] = activations_df.loc[:,0]
    activations_df['acts'] = transitions_test.acts

    # Add subject ID
    sbj = [ 301, 302, 304, 305, 307, 308, 310, 311, 312, 313, 316, 317, 320,
            321, 322, 323, 324, 325, 326, 327, 328, 334, 335, 306, 315, 319,
            333]
    sbj_trials = np.array([     480, 384, 480, 480, 480, 480, 432, 480, 
                                432, 432, 480, 480, 480, 480, 480, 480,
                                432, 480, 432, 480, 384, 480, 480, 480, 
                                480, 480, 432], dtype=int)/4*5
    sbj_trials = np.array(sbj_trials, dtype=int)
    s_trials = [[sbj[i]] * sbj_trials[i] for i in range(len(sbj))]
    s_trials = [sbj[i] for i in range(len(sbj)) for _ in range(sbj_trials[i])]
    activations_df['subject'] = s_trials    # subbject IDs
    
    # Filter activations_df
    observations = transitions_test.obs
    mask = (observations[:, -4] != 0) #|(observations[:, -1] != 0)
    filtered_activations_df = activations_df[mask].copy()
    filtered_activations_df = filtered_activations_df.reset_index(drop=True)
    
    d = pd.read_csv(path + 'data_beh/datall_cat.csv')
    
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

    # mask3 = (d['x14_p_foraging_gain'] > 0.3)
    # filtered_activations_df = filtered_activations_df[mask3]
    # filtered_activations_df = filtered_activations_df.reset_index(drop=True)
    # d = d[d['x14_p_foraging_gain'] > 0.3]
    # d = d.reset_index(drop=True)
    
    # mask4 = (d['x14_p_foraging_gain'] < 0.7)
    # filtered_activations_df = filtered_activations_df[mask4]
    # filtered_activations_df = filtered_activations_df.reset_index(drop=True)
    # d = d[d['x14_p_foraging_gain'] < 0.7]
    # d = d.reset_index(drop=True)

    # Scale the logits before fitting
    filtered_activations_df['logits_scaled'] = (filtered_activations_df['logits'] - 
                                                filtered_activations_df['logits'].mean()) / filtered_activations_df['logits'].std()

    # Fit logistic regression using statsmodels
    random = {"subject": '0 + C(subject)'}  # Random intercept for each subject
    log_reg = BinomialBayesMixedGLM.from_formula(
                    'acts ~ logits_scaled', random, filtered_activations_df)
    log_reg.fit_vb()
    result = log_reg.fit_vb()
    
    # Extract Mean (Posterior Mean) and Standard Deviation (Posterior SD)
    posterior_mean = result.params  # Mean of posterior
    # Posterior Standard Deviations (square root of diagonal of posterior covariance)
    posterior_cov = result.cov_params()  # Approximate covariance matrix
    posterior_sd = sp.csr_matrix(np.sqrt(np.diag(posterior_cov))).data  # Standard deviations
    
    # Model evidence using evidence lower bound
    elbo = log_reg.vb_elbo(vb_mean=posterior_mean, vb_sd=posterior_sd)
    print(f"evidence lower bound BC DNN = {elbo:.3f}")

    # Predict actions
    actions_pred = result.predict()
    # Area under the curve
    auc = roc_auc_score(filtered_activations_df['acts'], actions_pred)
    print(f"AUC-ROC = {auc:.3f}")
    
    # Accuracy and confusion matrix
    actions_pred = 1 / (1 + np.exp(-filtered_activations_df['logits_scaled']))
    preds_hard = np.where(actions_pred > 0.5, 1, 0)
    accuracy = accuracy_score(filtered_activations_df['acts'], preds_hard)
    cm = confusion_matrix(filtered_activations_df['acts'], preds_hard)
    print(f"Mean accuracy: {accuracy}")
    print(f"Mean confusion matrix:\n{cm}")

    # %% =======================================================================
    # Create validation plot - using homunc_BIC out-of-sample validation settings
    # ========================================================================
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    # ROC Curve - matching homunc_BIC style
    plt.figure(figsize=(10, 10))
    fpr, tpr, _ = roc_curve(filtered_activations_df['acts'], actions_pred)
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
    plt.savefig(path + 'bc_model_roc_curve.png', dpi=300)
    plt.show()

    # Confusion Matrix - matching homunc_BIC style
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Wait', 'Forage'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title('Confusion Matrix', fontsize=46, loc='left')
    ax.set_xlabel('Predicted label', fontsize=40)
    ax.set_ylabel('True label', fontsize=40)
    ax.tick_params(axis="x", labelsize=34)
    ax.tick_params(axis="y", labelsize=34)
    ax.tick_params(bottom=True, left=True, size=5, direction="in")

    # Modify text size in the confusion matrix cells - matching homunc_BIC
    for text in disp.text_.ravel():
        text.set_fontsize(30)

    plt.tight_layout()
    plt.savefig(path + 'bc_model_confusion_matrix.png', dpi=300)
    plt.show()
    
    # Additional validation plots in homunc_BIC style
    # Precision-Recall Curve
    plt.figure(figsize=(10, 10))
    precision, recall, _ = precision_recall_curve(filtered_activations_df['acts'], actions_pred)
    pr_auc = np.trapz(precision, recall)
    plt.plot(recall, precision, lw=4, label=f'PR-AUC = {pr_auc:.3f}')
    plt.axhline(y=filtered_activations_df['acts'].mean(), color='k', linestyle='--', 
                lw=3, label=f'Baseline = {filtered_activations_df["acts"].mean():.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=40)
    plt.ylabel('Precision', fontsize=40)
    plt.title('Precision-Recall Curve', fontsize=46, loc='left')
    plt.legend(loc="lower left", fontsize=30)
    plt.tick_params(axis="x", labelsize=34)
    plt.tick_params(axis="y", labelsize=34)
    plt.tick_params(bottom=True, left=True, size=5, direction="in")
    plt.tight_layout()
    plt.savefig(path + 'bc_model_pr_curve.png', dpi=300)
    plt.show()
    
    # Calibration Plot
    plt.figure(figsize=(10, 10))
    fraction_of_positives, mean_predicted_value = calibration_curve(
        filtered_activations_df['acts'], actions_pred, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", lw=4, 
            markersize=12, label='BC Model')
    plt.plot([0, 1], [0, 1], "k--", lw=3, label='Perfect Calibration')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability', fontsize=40)
    plt.ylabel('Fraction of Positives', fontsize=40)
    plt.title('Calibration Plot', fontsize=46, loc='left')
    plt.legend(fontsize=30)
    plt.tick_params(axis="x", labelsize=34)
    plt.tick_params(axis="y", labelsize=34)
    plt.tick_params(bottom=True, left=True, size=5, direction="in")
    plt.tight_layout()
    plt.savefig(path + 'bc_model_calibration.png', dpi=300)
    plt.show()
    
    # Subject-Level Validation
    plt.figure(figsize=(10, 10))
    subject_performance = filtered_activations_df.groupby('subject').agg({
        'acts': 'mean',
        'logits_scaled': lambda x: (1 / (1 + np.exp(-x))).mean()
    }).reset_index()
    
    plt.scatter(subject_performance['acts'], subject_performance['logits_scaled'], 
               alpha=0.7, s=120, edgecolors='black', linewidth=1)
    # Add perfect prediction line
    min_val = min(subject_performance['acts'].min(), subject_performance['logits_scaled'].min())
    max_val = max(subject_performance['acts'].max(), subject_performance['logits_scaled'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=3)
    
    # Calculate correlation
    subj_corr = np.corrcoef(subject_performance['acts'], subject_performance['logits_scaled'])[0, 1]
    plt.xlabel('Actual Choice Rate (by Subject)', fontsize=40)
    plt.ylabel('Predicted Choice Rate (by Subject)', fontsize=40)
    plt.title(f'Subject-Level Validation (r = {subj_corr:.3f})', fontsize=46, loc='left')
    plt.tick_params(axis="x", labelsize=34)
    plt.tick_params(axis="y", labelsize=34)
    plt.tick_params(bottom=True, left=True, size=5, direction="in")
    plt.tight_layout()
    plt.savefig(path + 'bc_model_subject_validation.png', dpi=300)
    plt.show()
    
    # Prediction Confidence Distribution
    plt.figure(figsize=(10, 10))
    plt.hist(actions_pred[filtered_activations_df['acts'] == 0], bins=20, alpha=0.6, 
            label='Actual Wait', density=True, color='blue', edgecolor='black')
    plt.hist(actions_pred[filtered_activations_df['acts'] == 1], bins=20, alpha=0.6, 
            label='Actual Foraging', density=True, color='red', edgecolor='black')
    plt.axvline(x=0.5, color='k', linestyle='--', lw=3, label='Decision Threshold')
    plt.xlabel('Predicted Probability', fontsize=40)
    plt.ylabel('Density', fontsize=40)
    plt.title('Prediction Confidence Distribution', fontsize=46, loc='left')
    plt.legend(fontsize=30)
    plt.tick_params(axis="x", labelsize=34)
    plt.tick_params(axis="y", labelsize=34)
    plt.tick_params(bottom=True, left=True, size=5, direction="in")
    plt.tight_layout()
    plt.savefig(path + 'bc_model_confidence_distribution.png', dpi=300)
    plt.show()
    
    print("BC model validation plots saved:")
    print("- bc_model_roc_curve.png")
    print("- bc_model_confusion_matrix.png") 
    print("- bc_model_pr_curve.png")
    print("- bc_model_calibration.png")
    print("- bc_model_subject_validation.png")
    print("- bc_model_confidence_distribution.png")

    # ## Spearman's rho
    # # Regress out subject-specific means
    # activations_df['logits_residual'] = activations_df['logits'] - activations_df.groupby('subject')['logits'].transform('mean')
    # activations_df['choices_residual'] = activations_df['acts'] - activations_df.groupby('subject')['acts'].transform('mean')
    # # Compute Spearman’s ρ on residuals
    # rho, pval = spearmanr(activations_df['logits_residual'], activations_df['choices_residual'])
    # print(f"Spearman’s ρ (subject variance removed) = {rho:.3f}, p = {pval:.3f}")

    # # Data split infos
    # sbj = [301, 302, 304, 305, 307, 308, 310, 311, 312, 313, 316, 317, 320,
    #     321, 322, 323, 324, 325, 326, 327, 328, 334, 335, 306, 315, 319,
    #     333]
    # sbj_forests = np.array([    480, 384, 480, 480, 480, 480, 432, 480, 
    #                             432, 432, 480, 480, 480, 480, 480, 480,
    #                             432, 480, 432, 480, 384, 480, 480, 480, 
    #                             480, 480, 432])/4

    # # sbj_forest_chck = np.array([480, 384, 480, 480, 480, 480, 432, 480,
    # #                             432, 432, 480, 480, 480, 480, 480, 480,
    # #                             432, 480, 432, 480, 384, 480, 480, 480,
    # #                             480, 480, 432])

    # # Train and evaluate the model via LOO-CV
    # c = 0
    # acc = []
    # cms = []
    # for i, n in enumerate(sbj_forests):

    #     # Split train and test data
    #     N = n * 5
    #     transitions_train = transitions[int(c+N):]
    #     transitions_test = transitions[int(c):int(c+N)]

    #     # Train model
    #     bc_trainer.demonstrations = transitions_train
    #     bc_trainer.train(n_epochs=5)
        
    #     # Get activations
    #     activations = get_activations_bc(bc_trainer, 
    #                                     transitions_test.obs, 
    #                                     'action_net')
    #     activations_df = pd.DataFrame(activations['action_net'])
    #     activations_df['acts'] = transitions_test.acts
        
    #     # Fit logistic regression
    #     log_reg = LogisticRegression()
    #     log_reg.fit(activations_df.iloc[:, :-1], activations_df['acts'])

    #     # Predict actions, get accuracy and confusion matrix
    #     actions_pred = log_reg.predict(activations_df.iloc[:, :-1])
    #     accuracy = accuracy_score(activations_df['acts'], actions_pred)
    #     cm = confusion_matrix(activations_df['acts'], actions_pred)
    #     acc.append(accuracy)
    #     cms.append(cm)

    #     # Update trainisions counter
    #     c += N
    
    # print(f"Mean accuracy: {np.mean(acc):.2f} ± {np.std(acc):.2f}")
    # print(f"Mean confusion matrix:\n{np.mean(cms, axis=0)}")

# # %% ==========================================================================
# # Evaluate rewards and success rates
# # =============================================================================
# from stable_baselines3.common.evaluation import evaluate_policy
# # import torch

# # obs = torch.tensor(vec_env.reset(), dtype=torch.float32)
# # action, value, features = bc_trainer.policy.forward(obs)  # Get the raw features
# # print("Raw network features (before output function):", features)

# emp_mean_rew = np.mean(summary_stats['mean'])
# emp_std_rew = np.mean(summary_stats['std'])

# ## Rewards
# # Evaluate BC policy on the environment
# reward_mean, reward_std = evaluate_policy(bc_trainer.policy, vec_env, n_eval_episodes=150)
# # Compare BC to export
# print(f"BC model predicted mean reward: {reward_mean:.2f} ± {reward_std:.2f}")
# print(f"empirical aggregated mean reward: {emp_mean_rew:.2f} ± {emp_std_rew:.2f}")

# ## Success rates BC model
# # Get success rate of BC model
# mod_succ = []  # Initialize success list here
# def success_callback(locals_, globals_):
#     """ Custom callback to count successful episodes. """
#     info = locals_['info']  # Get the latest step's info dictionary
#     if "success" in info and info["success"]:
#         globals()["mod_succ"].append(1)  # Track successful episodes
#     else:
#         globals()["mod_succ"].append(0)
# # Evaluate the BC model
# success_rate = evaluate_policy(
#     bc_trainer.policy, vec_env, n_eval_episodes=120,
#     return_episode_rewards=False,  # We only need success info
#     deterministic=True,
#     warn=False,
#     callback=success_callback)
# # Compare BC to export
# # Get success rates from trajectories
# emp_succ = [transitions.infos[i]['success'] for i in range(len(transitions))]
# print(f"BC Model Success Rate: {sum(mod_succ):.2f}%")
# print(f"empirical Success Rate: {np.sum(emp_succ)/(len(transitions)/6)*100:.2f}%")

# # %% ==========================================================================
# # Evaluate actions distributions
# # =============================================================================
# actions_expert = transitions.acts  # Collect actions from the expert
# # Get model actions and trajectories
# actions_bc = []  # Collect actions from the BC model
# traject_bc = []  # Collect states from BC trajectories
# for _ in range(450):
#     obs = vec_env.reset()
#     for day in range(5):
#         traject_bc = traject_bc + [str(list(obs[i])) for i in range(len(obs))]
#         action, _ = bc_trainer.policy.predict(obs, deterministic=True)
#         obs = vec_env.step(action)[0]
#         actions_bc = actions_bc + list(action)

# ## Compare state-action pairs
# # Get expert states
# traject_expert = [str(list(transitions.obs[i])) for i in range(len(transitions.obs))]
# # Aggregate state-actions for expert
# pd_traj_ex = pd.DataFrame([traject_expert, actions_expert]).T
# pd_traj_ex.columns = ['states', 'actions']
# ag_traj_ex = pd_traj_ex.groupby(["states"]).mean()
# # Aggregate state-actions for model
# pd_traj_bc = pd.DataFrame([traject_bc, actions_bc]).T
# pd_traj_bc.columns = ['states', 'actions']
# ag_traj_bc = pd_traj_bc.groupby(["states"]).mean()
# # Compare
# trajs = pd.merge(ag_traj_ex, ag_traj_bc, left_index=True, right_index=True, how='left')
# trajs.columns = ['expert', 'bc_model']
# correlation = trajs[['expert', 'bc_model']].corr()
# print(correlation)

# # # Plot action distributions
# # import matplotlib.pyplot as plt
# # plt.hist(actions_bc, bins=20, alpha=0.5, label="BC Policy")
# # plt.hist(actions_expert, bins=20, alpha=0.5, label="Expert Policy")
# # plt.legend()
# # plt.xlabel("Actions")
# # plt.ylabel("Frequency")
# # plt.title("Action Distributions: BC vs Expert")
# # plt.show()