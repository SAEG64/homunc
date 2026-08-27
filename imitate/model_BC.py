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
# TRAIN BEHAVIORAL-CLONING MODEL
# =============================================================================

from expert_trajectories_beh_V2 import transitions

import os
import numpy as np
import pandas as pd
import torch

from scipy.special import expit

from imitation.algorithms import bc
from stable_baselines3.common.distributions import DiagGaussianDistribution
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    brier_score_loss,
)


# ------------------------------------------------------------------------------
# Reproducibility and training settings
# ------------------------------------------------------------------------------

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

batch_sizes = [32, 64, 128]
selected_batch_size = batch_sizes[1]
n_training_epochs = 40


# ------------------------------------------------------------------------------
# Construct and train BC model on behavioral-sample demonstrations
# ------------------------------------------------------------------------------

bc_trainer = bc.BC(
    observation_space=vec_env.observation_space,
    action_space=vec_env.action_space,
    demonstrations=transitions,
    batch_size=selected_batch_size,
    rng=rng,
)

bc_trainer.train(
    n_epochs=n_training_epochs
)


# %% ==========================================================================
# VALIDATE BC POLICY SCORE IN THE INDEPENDENT fMRI SAMPLE
# =============================================================================
#
# Interpretation
# --------------
# 1. The BC network is trained only on the behavioral sample.
# 2. The trained network generates one BC policy score for every fMRI trial.
# 3. A hierarchical logistic model is fitted in the fMRI sample:
#
#       logit P(forage_it) =
#           beta_0 +
#           beta_BC * standardized_BC_score_it +
#           participant_intercept_i
#
# This tests whether the policy learned in the behavioral sample explains
# trial-level choices in the independent fMRI sample.
#
# Because the final logistic mapping is estimated in the fMRI sample, the
# resulting fitted AUC and accuracy describe the validation mixed model's
# explanatory performance. They are not parameter-free out-of-sample metrics.
#
# ==============================================================================

if __name__ == "__main__":

    from expert_trajectories_REV_V2 import (
        transitions as transitions_test
    )

    print("\n============================================================")
    print("BC CROSS-SAMPLE EXPLANATORY VALIDATION")
    print("============================================================")


    # ==========================================================================
    # 1. VERIFY POLICY PARAMETERIZATION
    # ==========================================================================

    policy = bc_trainer.policy

    print("Training sample: behavioral sample")
    print("Validation sample: fMRI sample")
    print("Action space:", policy.action_space)
    print(
        "Action distribution:",
        type(policy.action_dist).__name__
    )
    print(
        "Action-net output size:",
        policy.action_net.out_features
    )

    if not isinstance(
        policy.action_dist,
        DiagGaussianDistribution
    ):
        raise TypeError(
            "This code assumes a DiagGaussianDistribution for a "
            "one-dimensional continuous Box action space."
        )

    if policy.action_net.out_features != 1:
        raise ValueError(
            "Expected one Gaussian action-mean output per observation, "
            f"but found {policy.action_net.out_features} outputs."
        )


    # ==========================================================================
    # 2. EXTRACT NATIVE GAUSSIAN POLICY PARAMETERS
    # ==========================================================================

    def extract_gaussian_policy_parameters(
        trained_policy,
        observations,
        batch_size=2048,
    ):
        """
        Extract the native mean and standard deviation of the trained
        diagonal-Gaussian policy.

        For this BC model, the action-net output is the conditional Gaussian
        policy mean. It is therefore treated as a continuous BC policy score,
        not as a binary logit.
        """

        observations = np.asarray(
            observations,
            dtype=np.float32
        )

        mean_batches = []
        sd_batches = []

        trained_policy.set_training_mode(False)

        for start in range(
            0,
            len(observations),
            batch_size
        ):
            stop = min(
                start + batch_size,
                len(observations)
            )

            observation_batch = observations[
                start:stop
            ]

            observation_tensor = torch.as_tensor(
                observation_batch,
                dtype=torch.float32,
                device=trained_policy.device,
            )

            with torch.no_grad():

                policy_distribution = (
                    trained_policy.get_distribution(
                        observation_tensor
                    )
                )

                gaussian_distribution = (
                    policy_distribution.distribution
                )

                batch_mean = (
                    gaussian_distribution.mean
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                batch_sd = (
                    gaussian_distribution.stddev
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

            mean_batches.append(batch_mean)
            sd_batches.append(batch_sd)

        policy_mean = np.concatenate(
            mean_batches
        )

        policy_sd = np.concatenate(
            sd_batches
        )

        if len(policy_mean) != len(observations):
            raise RuntimeError(
                "Policy-mean extraction produced an unexpected "
                "number of observations."
            )

        if len(policy_sd) != len(observations):
            raise RuntimeError(
                "Policy-SD extraction produced an unexpected "
                "number of observations."
            )

        if not np.all(
            np.isfinite(policy_mean)
        ):
            raise ValueError(
                "Non-finite BC policy means detected."
            )

        if (
            not np.all(np.isfinite(policy_sd))
            or np.any(policy_sd <= 0)
        ):
            raise ValueError(
                "Invalid BC policy standard deviations detected."
            )

        return policy_mean, policy_sd


    test_observations = np.asarray(
        transitions_test.obs
    )

    observed_actions = np.asarray(
        transitions_test.acts
    ).reshape(-1)

    bc_policy_mean, bc_policy_sd = (
        extract_gaussian_policy_parameters(
            trained_policy=policy,
            observations=test_observations,
        )
    )

    if not (
        len(test_observations)
        == len(observed_actions)
        == len(bc_policy_mean)
        == len(bc_policy_sd)
    ):
        raise ValueError(
            "Observations, actions, policy means, and policy SDs "
            "do not have matching lengths."
        )


    # ==========================================================================
    # 3. RECONSTRUCT PARTICIPANT IDENTIFIERS
    # ==========================================================================
    #
    # This preserves the ordering used in the original transition construction.
    # Carrying participant IDs directly in transition metadata would be safer,
    # but these assertions prevent silent misalignment.
    #
    # ==========================================================================

    subject_ids = np.array(
        [
            301, 302, 304, 305, 307, 308, 310,
            311, 312, 313, 316, 317, 320, 321,
            322, 323, 324, 325, 326, 327, 328,
            334, 335, 306, 315, 319, 333,
        ],
        dtype=int,
    )

    original_trial_counts = np.array(
        [
            480, 384, 480, 480, 480, 480, 432,
            480, 432, 432, 480, 480, 480, 480,
            480, 480, 432, 480, 432, 480, 384,
            480, 480, 480, 480, 480, 432,
        ],
        dtype=int,
    )

    transition_counts = (
        original_trial_counts / 4 * 5
    ).astype(int)

    subject_vector = np.repeat(
        subject_ids,
        transition_counts,
    )

    if len(subject_vector) != len(
        test_observations
    ):
        raise ValueError(
            "Reconstructed participant vector does not match the "
            "number of test transitions.\n"
            f"Participant-vector length: {len(subject_vector)}\n"
            f"Test-transition length: {len(test_observations)}"
        )


    # ==========================================================================
    # 4. BUILD ONE ALIGNED TRIAL-LEVEL DATAFRAME
    # ==========================================================================

    validation_df = pd.DataFrame(
        {
            "acts": observed_actions,
            "subject": subject_vector,
            "bc_policy_mean": bc_policy_mean,
            "bc_policy_sd": bc_policy_sd,
            "observation_filter_variable":
                test_observations[:, -4],
        }
    )

    validation_df["subject"] = (
        validation_df["subject"]
        .astype("category")
    )


    # ==========================================================================
    # 5. APPLY TRANSITION-LEVEL FILTER
    # ==========================================================================

    validation_df = (
        validation_df.loc[
            validation_df[
                "observation_filter_variable"
            ] != 0
        ]
        .reset_index(drop=True)
    )


    # ==========================================================================
    # 6. ALIGN WITH THE ORIGINAL fMRI TRIAL TABLE
    # ==========================================================================
    #
    # IMPORTANT:
    # Confirm that this file contains the fMRI sample rather than the behavioral
    # training sample. The original folder name "data_beh" is potentially
    # misleading.
    #
    # ==========================================================================

    fmri_trial_file = os.path.join(
        path,
        "data_beh",
        "datall_cat.csv",
    )

    trial_data = pd.read_csv(
        fmri_trial_file
    )

    required_trial_columns = [
        "x9_button_pressed",
        "x6_continuous_energy_trial_start",
    ]

    missing_trial_columns = [
        column
        for column in required_trial_columns
        if column not in trial_data.columns
    ]

    if missing_trial_columns:
        raise KeyError(
            "The fMRI trial table is missing required columns: "
            + ", ".join(missing_trial_columns)
        )

    if len(trial_data) != len(validation_df):
        raise ValueError(
            "The fMRI trial table does not align with the BC transitions "
            "after the observation-level filter.\n"
            f"Trial-table rows: {len(trial_data)}\n"
            f"BC-validation rows: {len(validation_df)}\n\n"
            "Verify that datall_cat.csv is the fMRI dataset and that its "
            "trial order matches expert_trajectories_REV_V2."
        )

    trial_data = trial_data.reset_index(
        drop=True
    )

    validation_df = validation_df.reset_index(
        drop=True
    )


    # ==========================================================================
    # 7. APPLY THE SAME EMPIRICAL EXCLUSION CRITERIA
    # ==========================================================================

    empirical_mask = (
        trial_data[
            "x9_button_pressed"
        ].notna()
        &
        (
            trial_data[
                "x6_continuous_energy_trial_start"
            ] != 0
        )
    ).to_numpy()

    validation_df = (
        validation_df.loc[
            empirical_mask
        ]
        .reset_index(drop=True)
    )

    trial_data = (
        trial_data.loc[
            empirical_mask
        ]
        .reset_index(drop=True)
    )

    if len(validation_df) != len(
        trial_data
    ):
        raise RuntimeError(
            "BC validation data and empirical trial data became "
            "misaligned during filtering."
        )


    # ==========================================================================
    # 8. CHECK OUTCOME CODING
    # ==========================================================================

    unique_actions = np.unique(
        validation_df["acts"].dropna()
    )

    if not set(
        unique_actions.tolist()
    ).issubset({0, 1}):
        raise ValueError(
            "The validation outcome must be binary and coded as 0/1. "
            f"Observed values: {unique_actions}"
        )

    validation_df["acts"] = (
        validation_df["acts"]
        .astype(int)
    )

    # %% ==========================================================================
    # LEAVE-ONE-SUBJECT-OUT BC MODEL COMPARISON
    # =============================================================================
    #
    # Models:
    #
    #   Null model:
    #       acts ~ 1 + (1 | subject)
    #
    #   BC model:
    #       acts ~ bc_score_scaled + (1 | subject)
    #
    # Validation:
    #   One complete participant is held out at a time using
    #   leave-one-group-out cross-validation, with subject identifiers
    #   supplied as the grouping variable.
    #
    # Primary comparison:
    #   Joint held-out log predictive density at the participant level.
    #
    # Important:
    #   For each held-out participant, all trials share one latent random
    #   intercept. The likelihood is therefore integrated jointly across all
    #   trials belonging to that participant.
    #
    # Secondary metrics:
    #   Trial-level marginal AUC, accuracy, balanced accuracy, Brier score,
    #   log loss, confusion matrix, precision-recall performance, and calibration.
    #
    # ==============================================================================

    from numpy.polynomial.hermite import hermgauss
    from sklearn.model_selection import LeaveOneGroupOut

    from scipy.special import (
        expit,
        logsumexp,
    )

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        ConfusionMatrixDisplay,
        log_loss,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    from sklearn.calibration import calibration_curve

    from statsmodels.genmod.bayes_mixed_glm import (
        BinomialBayesMixedGLM,
    )


    # ==============================================================================
    # 1. SETTINGS
    # ==============================================================================

    N_QUADRATURE_NODES = 40
    N_BOOTSTRAP = 10000

    PROBABILITY_EPSILON = 1e-12

    bootstrap_rng = np.random.default_rng(
        SEED
    )


    # ==============================================================================
    # 2. BASIC VALIDATION-DATA CHECKS
    # ==============================================================================

    required_validation_columns = [
        "acts",
        "subject",
        "bc_policy_mean",
    ]

    missing_validation_columns = [
        column
        for column in required_validation_columns
        if column not in validation_df.columns
    ]

    if missing_validation_columns:
        raise KeyError(
            "validation_df is missing required columns: "
            + ", ".join(missing_validation_columns)
        )


    validation_df = (
        validation_df[
            required_validation_columns
        ]
        .dropna()
        .copy()
        .reset_index(drop=True)
    )


    validation_df["acts"] = (
        validation_df["acts"]
        .astype(int)
    )

    validation_df["subject"] = (
        validation_df["subject"]
        .astype(str)
    )


    unique_outcomes = set(
        validation_df["acts"].unique().tolist()
    )

    if not unique_outcomes.issubset({0, 1}):
        raise ValueError(
            "acts must be coded as 0/1. "
            f"Observed values: {sorted(unique_outcomes)}"
        )


    n_subjects = validation_df[
        "subject"
    ].nunique()

    if n_subjects < 2:
        raise ValueError(
            "Leave-one-subject-out cross-validation requires "
            "at least two participants."
        )


    if not np.all(
        np.isfinite(
            validation_df[
                "bc_policy_mean"
            ].to_numpy(dtype=float)
        )
    ):
        raise ValueError(
            "Non-finite BC policy scores were found."
        )


    # ==============================================================================
    # 3. MODEL-FITTING HELPER
    # ==============================================================================

    def fit_validation_glmm(
        training_data,
        include_bc_score,
    ):
        """
        Fit a binomial random-intercept mixed model using variational Bayes.

        A non-success optimizer status is treated as a warning when the fitted
        parameter estimates are finite. Statsmodels/SciPy can report precision-loss
        or strict-tolerance failures despite returning usable VB estimates.
        """

        formula = (
            "acts ~ bc_score_scaled"
            if include_bc_score
            else "acts ~ 1"
        )

        variance_component_formulas = {
            "participant_intercept":
                "0 + C(subject)"
        }

        model = BinomialBayesMixedGLM.from_formula(
            formula=formula,
            vc_formulas=variance_component_formulas,
            data=training_data,
        )

        result = model.fit_vb()

        # ----------------------------------------------------------------------
        # Check that the fitted estimates are numerically usable
        # ----------------------------------------------------------------------

        parameter_arrays = [
            np.asarray(result.fe_mean, dtype=float),
            np.asarray(result.fe_sd, dtype=float),
            np.asarray(result.vcp_mean, dtype=float),
            np.asarray(result.vcp_sd, dtype=float),
            np.asarray(result.vc_mean, dtype=float),
            np.asarray(result.vc_sd, dtype=float),
        ]

        if not all(
            np.all(np.isfinite(parameter_array))
            for parameter_array in parameter_arrays
        ):
            raise RuntimeError(
                "The variational-Bayes fit returned non-finite "
                f"parameter estimates for model:\n{formula}"
            )

        if np.any(
            np.asarray(result.fe_sd, dtype=float) <= 0
        ):
            raise RuntimeError(
                "The variational-Bayes fit returned invalid fixed-effect "
                f"posterior standard deviations for model:\n{formula}"
            )

        if np.any(
            np.asarray(result.vcp_sd, dtype=float) <= 0
        ):
            raise RuntimeError(
                "The variational-Bayes fit returned invalid variance-component "
                f"posterior standard deviations for model:\n{formula}"
            )

        # ----------------------------------------------------------------------
        # Inspect optimizer status without automatically rejecting the fit
        # ----------------------------------------------------------------------

        optimization_information = getattr(
            result,
            "optim_retvals",
            None,
        )

        optimization_success = None
        optimization_message = None
        gradient_norm = np.nan

        if optimization_information is not None:

            if isinstance(
                optimization_information,
                dict,
            ):
                optimization_success = (
                    optimization_information.get(
                        "success",
                        None,
                    )
                )

                optimization_message = (
                    optimization_information.get(
                        "message",
                        None,
                    )
                )

                optimizer_gradient = (
                    optimization_information.get(
                        "jac",
                        None,
                    )
                )

            else:
                optimization_success = getattr(
                    optimization_information,
                    "success",
                    None,
                )

                optimization_message = getattr(
                    optimization_information,
                    "message",
                    None,
                )

                optimizer_gradient = getattr(
                    optimization_information,
                    "jac",
                    None,
                )

            if optimizer_gradient is not None:

                optimizer_gradient = np.asarray(
                    optimizer_gradient,
                    dtype=float,
                ).reshape(-1)

                if np.all(
                    np.isfinite(
                        optimizer_gradient
                    )
                ):
                    gradient_norm = float(
                        np.linalg.norm(
                            optimizer_gradient
                        )
                    )

        if optimization_success is False:

            print(
                "\nWARNING: Variational-Bayes optimizer did not report "
                "formal convergence."
            )

            print(
                f"Model: {formula}"
            )

            print(
                f"Optimizer message: {optimization_message}"
            )

            print(
                f"Gradient norm: {gradient_norm:.6g}"
            )

            print(
                "All fitted parameter summaries are finite, so the fit "
                "will be retained. Inspect whether estimates and predictions "
                "remain stable across folds."
            )

        return model, result


    # ==============================================================================
    # 4. EXTRACT RANDOM-INTERCEPT STANDARD DEVIATION
    # ==============================================================================

    def extract_random_intercept_sd(
        fitted_result,
    ):
        """
        Recover the random-intercept SD.

        BinomialBayesMixedGLM parameterizes variance-component parameters
        on the log-standard-deviation scale.
        """

        log_sd_parameters = np.asarray(
            fitted_result.vcp_mean,
            dtype=float,
        ).reshape(-1)

        if len(log_sd_parameters) != 1:
            raise ValueError(
                "Expected exactly one variance component, "
                f"but found {len(log_sd_parameters)}."
            )

        random_intercept_sd = float(
            np.exp(
                log_sd_parameters[0]
            )
        )

        if (
            not np.isfinite(random_intercept_sd)
            or random_intercept_sd < 0
        ):
            raise ValueError(
                "The estimated random-intercept SD is invalid."
            )

        return random_intercept_sd


    # ==============================================================================
    # 5. HELD-OUT FIXED-EFFECT LINEAR PREDICTOR
    # ==============================================================================

    def construct_held_out_eta(
        fitted_result,
        test_data,
        include_bc_score,
    ):
        """
        Apply training-fold fixed effects to held-out participants.
        """

        fixed_effect_means = np.asarray(
            fitted_result.fe_mean,
            dtype=float,
        ).reshape(-1)

        if include_bc_score:

            if len(fixed_effect_means) != 2:
                raise ValueError(
                    "The BC model should contain an intercept "
                    "and one BC-score coefficient."
                )

            intercept = fixed_effect_means[0]
            bc_coefficient = fixed_effect_means[1]

            eta = (
                intercept
                + bc_coefficient
                * test_data[
                    "bc_score_scaled"
                ].to_numpy(dtype=float)
            )

        else:

            if len(fixed_effect_means) != 1:
                raise ValueError(
                    "The null model should contain only an intercept."
                )

            eta = np.full(
                shape=len(test_data),
                fill_value=fixed_effect_means[0],
                dtype=float,
            )

        return np.asarray(
            eta,
            dtype=float,
        ).reshape(-1)


    # ==============================================================================
    # 6. TRIAL-LEVEL MARGINAL PROBABILITIES FOR NEW PARTICIPANTS
    # ==============================================================================

    def marginalize_random_intercept(
        fixed_linear_predictor,
        random_intercept_sd,
        n_nodes=40,
    ):
        """
        Compute trial-level marginal probabilities for unseen participants:

            E_u[sigmoid(eta + u)]

        where:

            u ~ Normal(0, random_intercept_sd^2)

        These probabilities are appropriate for trial-level AUC, accuracy,
        Brier score, calibration, and related predictive diagnostics.
        """

        eta = np.asarray(
            fixed_linear_predictor,
            dtype=float,
        ).reshape(-1)

        random_intercept_sd = float(
            random_intercept_sd
        )

        if random_intercept_sd == 0:
            return expit(eta)

        nodes, weights = hermgauss(
            n_nodes
        )

        random_intercept_values = (
            np.sqrt(2.0)
            * random_intercept_sd
            * nodes
        )

        conditional_probabilities = expit(
            eta[:, None]
            + random_intercept_values[None, :]
        )

        marginal_probabilities = (
            conditional_probabilities
            @ weights
        ) / np.sqrt(np.pi)

        return np.asarray(
            marginal_probabilities,
            dtype=float,
        ).reshape(-1)


    # ==============================================================================
    # 7. JOINT SUBJECT-LEVEL LOG PREDICTIVE DENSITY
    # ==============================================================================

    def subject_log_predictive_density(
        observed,
        fixed_linear_predictor,
        random_intercept_sd,
        n_nodes=40,
    ):
        """
        Calculate the joint log predictive density for all trials belonging
        to one unseen participant.

        The same random intercept is shared by every trial:

            p(y_i | X_i)
            =
            integral [
                product_t p(y_it | X_it, u_i)
            ] p(u_i) du_i

        This is the correct predictive-density unit for participant-level
        cross-validation of a random-intercept model.
        """

        observed = np.asarray(
            observed,
            dtype=int,
        ).reshape(-1)

        eta = np.asarray(
            fixed_linear_predictor,
            dtype=float,
        ).reshape(-1)

        if len(observed) != len(eta):
            raise ValueError(
                "Observed outcomes and linear predictors "
                "must have the same length."
            )

        if len(observed) == 0:
            raise ValueError(
                "Cannot calculate predictive density "
                "for an empty participant."
            )

        nodes, weights = hermgauss(
            n_nodes
        )

        random_intercept_values = (
            np.sqrt(2.0)
            * random_intercept_sd
            * nodes
        )

        # Dimensions:
        # rows    = trials
        # columns = quadrature nodes
        linear_predictor = (
            eta[:, None]
            + random_intercept_values[None, :]
        )

        # Numerically stable Bernoulli log probabilities
        log_probability_one = (
            -np.logaddexp(
                0.0,
                -linear_predictor,
            )
        )

        log_probability_zero = (
            -np.logaddexp(
                0.0,
                linear_predictor,
            )
        )

        trial_log_likelihood = (
            observed[:, None]
            * log_probability_one
            + (1 - observed[:, None])
            * log_probability_zero
        )

        # One conditional joint log likelihood per quadrature node
        conditional_subject_log_likelihood = (
            trial_log_likelihood.sum(
                axis=0
            )
        )

        # Gauss-Hermite integration over the Gaussian random intercept
        subject_log_density = (
            logsumexp(
                conditional_subject_log_likelihood
                + np.log(weights)
            )
            - 0.5
            * np.log(np.pi)
        )

        return float(
            subject_log_density
        )


    # ==============================================================================
    # 8. PREPARE SUBJECT-LEVEL CROSS-VALIDATION
    # ==============================================================================

    cross_validation_data = (
        validation_df[
            [
                "acts",
                "subject",
                "bc_policy_mean",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    group_splitter = LeaveOneGroupOut()

    trial_prediction_records = []
    subject_density_records = []
    fold_summary_records = []


    # ==============================================================================
    # 9. RUN SUBJECT-LEVEL CROSS-VALIDATION
    # ==============================================================================

    for fold_number, (
        training_indices,
        test_indices,
    ) in enumerate(
        group_splitter.split(
            X=cross_validation_data,
            y=cross_validation_data["acts"],
            groups=cross_validation_data["subject"],
        ),
        start=1,
    ):

        training_data = (
            cross_validation_data.iloc[
                training_indices
            ]
            .copy()
            .reset_index(drop=True)
        )

        test_data = (
            cross_validation_data.iloc[
                test_indices
            ]
            .copy()
            .reset_index(drop=True)
        )

        if test_data["subject"].nunique() != 1:
            raise RuntimeError(
                f"LOSO fold {fold_number} must contain exactly "
                "one held-out participant."
            )

        if training_data["subject"].nunique() != n_subjects - 1:
            raise RuntimeError(
                f"LOSO fold {fold_number} contains an unexpected "
                "number of training participants."
            )


        # --------------------------------------------------------------------------
        # Verify that participants do not occur in both sets
        # --------------------------------------------------------------------------

        training_subjects = set(
            training_data["subject"]
        )

        test_subjects = set(
            test_data["subject"]
        )

        if not training_subjects.isdisjoint(
            test_subjects
        ):
            raise RuntimeError(
                f"Participant leakage detected in fold {fold_number}."
            )


        # --------------------------------------------------------------------------
        # Standardize using training-fold statistics only
        # --------------------------------------------------------------------------

        training_score_mean = (
            training_data[
                "bc_policy_mean"
            ].mean()
        )

        training_score_sd = (
            training_data[
                "bc_policy_mean"
            ].std(ddof=1)
        )

        if (
            not np.isfinite(training_score_sd)
            or training_score_sd <= 0
        ):
            raise ValueError(
                f"Invalid BC-score SD in fold {fold_number}."
            )


        training_data[
            "bc_score_scaled"
        ] = (
            (
                training_data[
                    "bc_policy_mean"
                ]
                - training_score_mean
            )
            / training_score_sd
        )


        test_data[
            "bc_score_scaled"
        ] = (
            (
                test_data[
                    "bc_policy_mean"
                ]
                - training_score_mean
            )
            / training_score_sd
        )


        # --------------------------------------------------------------------------
        # Fit null and BC models in the training participants
        # --------------------------------------------------------------------------

        null_model, null_result = (
            fit_validation_glmm(
                training_data=training_data,
                include_bc_score=False,
            )
        )

        bc_model, bc_result = (
            fit_validation_glmm(
                training_data=training_data,
                include_bc_score=True,
            )
        )


        # --------------------------------------------------------------------------
        # Apply fixed effects to held-out participants
        # --------------------------------------------------------------------------

        null_eta = construct_held_out_eta(
            fitted_result=null_result,
            test_data=test_data,
            include_bc_score=False,
        )

        bc_eta = construct_held_out_eta(
            fitted_result=bc_result,
            test_data=test_data,
            include_bc_score=True,
        )


        # --------------------------------------------------------------------------
        # Extract random-intercept distributions
        # --------------------------------------------------------------------------

        null_random_intercept_sd = (
            extract_random_intercept_sd(
                null_result
            )
        )

        bc_random_intercept_sd = (
            extract_random_intercept_sd(
                bc_result
            )
        )


        # --------------------------------------------------------------------------
        # Trial-level marginal predictions for unseen participants
        # --------------------------------------------------------------------------

        null_probability = (
            marginalize_random_intercept(
                fixed_linear_predictor=null_eta,
                random_intercept_sd=(
                    null_random_intercept_sd
                ),
                n_nodes=N_QUADRATURE_NODES,
            )
        )

        bc_probability = (
            marginalize_random_intercept(
                fixed_linear_predictor=bc_eta,
                random_intercept_sd=(
                    bc_random_intercept_sd
                ),
                n_nodes=N_QUADRATURE_NODES,
            )
        )


        null_probability = np.clip(
            null_probability,
            PROBABILITY_EPSILON,
            1 - PROBABILITY_EPSILON,
        )

        bc_probability = np.clip(
            bc_probability,
            PROBABILITY_EPSILON,
            1 - PROBABILITY_EPSILON,
        )


        y_test = (
            test_data[
                "acts"
            ]
            .to_numpy(dtype=int)
        )

        subject_test = (
            test_data[
                "subject"
            ]
            .to_numpy()
        )


        # --------------------------------------------------------------------------
        # Store trial-level out-of-fold predictions
        # --------------------------------------------------------------------------

        trial_prediction_records.append(
            pd.DataFrame(
                {
                    "fold": fold_number,
                    "subject": subject_test,
                    "acts": y_test,
                    "bc_policy_mean":
                        test_data[
                            "bc_policy_mean"
                        ].to_numpy(dtype=float),
                    "bc_score_scaled":
                        test_data[
                            "bc_score_scaled"
                        ].to_numpy(dtype=float),
                    "null_probability":
                        null_probability,
                    "bc_probability":
                        bc_probability,
                }
            )
        )


        # --------------------------------------------------------------------------
        # Calculate joint predictive density separately for each held-out subject
        # --------------------------------------------------------------------------

        fold_subject_density_records = []

        for subject_id in np.unique(
            subject_test
        ):

            subject_mask = (
                subject_test == subject_id
            )

            subject_y = (
                y_test[
                    subject_mask
                ]
            )

            subject_null_eta = (
                null_eta[
                    subject_mask
                ]
            )

            subject_bc_eta = (
                bc_eta[
                    subject_mask
                ]
            )


            null_subject_lpd = (
                subject_log_predictive_density(
                    observed=subject_y,
                    fixed_linear_predictor=(
                        subject_null_eta
                    ),
                    random_intercept_sd=(
                        null_random_intercept_sd
                    ),
                    n_nodes=N_QUADRATURE_NODES,
                )
            )


            bc_subject_lpd = (
                subject_log_predictive_density(
                    observed=subject_y,
                    fixed_linear_predictor=(
                        subject_bc_eta
                    ),
                    random_intercept_sd=(
                        bc_random_intercept_sd
                    ),
                    n_nodes=N_QUADRATURE_NODES,
                )
            )


            subject_record = {
                "fold": fold_number,
                "subject": subject_id,
                "n_trials": int(
                    subject_mask.sum()
                ),
                "null_log_predictive_density":
                    null_subject_lpd,
                "bc_log_predictive_density":
                    bc_subject_lpd,
                "delta_log_predictive_density":
                    bc_subject_lpd
                    - null_subject_lpd,
            }

            fold_subject_density_records.append(
                subject_record
            )

            subject_density_records.append(
                subject_record
            )


        fold_subject_density_df = pd.DataFrame(
            fold_subject_density_records
        )


        fold_summary_records.append(
            {
                "fold": fold_number,
                "n_training_subjects":
                    training_data[
                        "subject"
                    ].nunique(),
                "n_test_subjects":
                    test_data[
                        "subject"
                    ].nunique(),
                "n_test_trials":
                    len(test_data),
                "training_score_mean":
                    training_score_mean,
                "training_score_sd":
                    training_score_sd,
                "null_random_intercept_sd":
                    null_random_intercept_sd,
                "bc_random_intercept_sd":
                    bc_random_intercept_sd,
                "null_subject_elpd":
                    fold_subject_density_df[
                        "null_log_predictive_density"
                    ].sum(),
                "bc_subject_elpd":
                    fold_subject_density_df[
                        "bc_log_predictive_density"
                    ].sum(),
                "delta_subject_elpd":
                    fold_subject_density_df[
                        "delta_log_predictive_density"
                    ].sum(),
            }
        )


        print(
            f"Fold {fold_number}: "
            f"participants={test_data['subject'].nunique()}, "
            f"trials={len(test_data)}, "
            f"Δ ELPD="
            f"{fold_subject_density_df['delta_log_predictive_density'].sum():.3f}"
        )


    # ==============================================================================
    # 10. COMBINE OUT-OF-FOLD RESULTS
    # ==============================================================================

    out_of_fold_predictions = pd.concat(
        trial_prediction_records,
        ignore_index=True,
    )

    subject_predictive_comparison = pd.DataFrame(
        subject_density_records
    )

    fold_summary_table = pd.DataFrame(
        fold_summary_records
    )


    if len(out_of_fold_predictions) != len(
        cross_validation_data
    ):
        raise RuntimeError(
            "Each validation trial must receive exactly "
            "one out-of-fold prediction."
        )


    if (
        out_of_fold_predictions[
            "subject"
        ].nunique()
        != cross_validation_data[
            "subject"
        ].nunique()
    ):
        raise RuntimeError(
            "Each participant must occur in exactly one test fold."
        )


    subject_fold_counts = (
        out_of_fold_predictions
        .groupby(
            "subject",
            observed=True,
        )[
            "fold"
        ]
        .nunique()
    )

    if not np.all(
        subject_fold_counts.to_numpy()
        == 1
    ):
        raise RuntimeError(
            "At least one participant occurs in multiple test folds."
        )


    # ==============================================================================
    # 11. PRIMARY SUBJECT-LEVEL MODEL COMPARISON
    # ==============================================================================

    null_elpd = (
        subject_predictive_comparison[
            "null_log_predictive_density"
        ].sum()
    )

    bc_elpd = (
        subject_predictive_comparison[
            "bc_log_predictive_density"
        ].sum()
    )

    subject_delta_elpd = (
        subject_predictive_comparison[
            "delta_log_predictive_density"
        ].to_numpy(dtype=float)
    )

    delta_elpd = float(
        subject_delta_elpd.sum()
    )


    # Independent-participant cluster estimate of the SE
    delta_elpd_clustered_se = float(
        np.sqrt(
            len(subject_delta_elpd)
            * np.var(
                subject_delta_elpd,
                ddof=1,
            )
        )
    )


    # Participant bootstrap interval for total Δ ELPD
    bootstrap_delta_elpd = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    for bootstrap_index in range(
        N_BOOTSTRAP
    ):

        sampled_subject_contributions = (
            bootstrap_rng.choice(
                subject_delta_elpd,
                size=len(subject_delta_elpd),
                replace=True,
            )
        )

        bootstrap_delta_elpd[
            bootstrap_index
        ] = sampled_subject_contributions.sum()


    delta_elpd_bootstrap_ci = np.quantile(
        bootstrap_delta_elpd,
        [
            0.025,
            0.975,
        ],
    )


    # ==============================================================================
    # 12. SECONDARY TRIAL-LEVEL PREDICTIVE METRICS
    # ==============================================================================

    y_out_of_fold = (
        out_of_fold_predictions[
            "acts"
        ]
        .to_numpy(dtype=int)
    )

    null_out_of_fold_probability = (
        out_of_fold_predictions[
            "null_probability"
        ]
        .to_numpy(dtype=float)
    )

    bc_out_of_fold_probability = (
        out_of_fold_predictions[
            "bc_probability"
        ]
        .to_numpy(dtype=float)
    )


    null_out_of_fold_prediction = (
        null_out_of_fold_probability
        >= 0.5
    ).astype(int)

    bc_out_of_fold_prediction = (
        bc_out_of_fold_probability
        >= 0.5
    ).astype(int)


    def calculate_out_of_fold_metrics(
        observed,
        probability,
        prediction,
        model_label,
    ):
        """
        Calculate predictive metrics from held-out probabilities only.
        """

        probability = np.clip(
            probability,
            PROBABILITY_EPSILON,
            1 - PROBABILITY_EPSILON,
        )

        return {
            "model": model_label,
            "n_trials": len(observed),
            "n_participants":
                out_of_fold_predictions[
                    "subject"
                ].nunique(),
            "auc":
                roc_auc_score(
                    observed,
                    probability,
                ),
            "accuracy":
                accuracy_score(
                    observed,
                    prediction,
                ),
            "balanced_accuracy":
                balanced_accuracy_score(
                    observed,
                    prediction,
                ),
            "brier_score":
                brier_score_loss(
                    observed,
                    probability,
                ),
            "mean_log_loss":
                log_loss(
                    observed,
                    probability,
                    labels=[0, 1],
                ),
            "average_precision":
                average_precision_score(
                    observed,
                    probability,
                ),
        }


    null_metrics = calculate_out_of_fold_metrics(
        observed=y_out_of_fold,
        probability=null_out_of_fold_probability,
        prediction=null_out_of_fold_prediction,
        model_label="Random-intercept null",
    )


    bc_metrics = calculate_out_of_fold_metrics(
        observed=y_out_of_fold,
        probability=bc_out_of_fold_probability,
        prediction=bc_out_of_fold_prediction,
        model_label="BC score + random intercept",
    )


    comparison_table = pd.DataFrame(
        [
            null_metrics,
            bc_metrics,
        ]
    )


    bc_confusion_matrix = confusion_matrix(
        y_out_of_fold,
        bc_out_of_fold_prediction,
        labels=[0, 1],
    )


    # ==============================================================================
    # 13. PRINT CROSS-VALIDATED RESULTS
    # ==============================================================================

    print("\n============================================================")
    print("SUBJECT-LEVEL CROSS-VALIDATED MODEL COMPARISON")
    print("============================================================")

    print(
        comparison_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


    print(
        f"\nNull-model LOSO log predictive density: "
        f"{null_elpd:.3f}"
    )

    print(
        f"BC-model LOSO log predictive density: "
        f"{bc_elpd:.3f}"
    )

    print(
        f"Δ ELPD (BC − null): "
        f"{delta_elpd:.3f}"
    )

    print(
        f"Participant-clustered SE: "
        f"{delta_elpd_clustered_se:.3f}"
    )

    print(
        "Participant-bootstrap 95% interval: "
        f"[{delta_elpd_bootstrap_ci[0]:.3f}, "
        f"{delta_elpd_bootstrap_ci[1]:.3f}]"
    )

    print(
        "\nMean held-out log-loss improvement "
        "(null − BC): "
        f"{null_metrics['mean_log_loss'] - bc_metrics['mean_log_loss']:.5f}"
    )

    print(
        "\nHeld-out BC confusion matrix "
        "[[wait/wait, wait/forage], "
        "[forage/wait, forage/forage]]:"
    )

    print(
        bc_confusion_matrix
    )


    # ==============================================================================
    # 14. SAVE CROSS-VALIDATION OUTPUTS
    # ==============================================================================

    out_of_fold_predictions.to_csv(
        os.path.join(
            path,
            "bc_subject_cv_trial_predictions.csv",
        ),
        index=False,
    )


    subject_predictive_comparison.to_csv(
        os.path.join(
            path,
            "bc_subject_cv_subject_elpd.csv",
        ),
        index=False,
    )


    fold_summary_table.to_csv(
        os.path.join(
            path,
            "bc_subject_cv_fold_summary.csv",
        ),
        index=False,
    )


    comparison_table.to_csv(
        os.path.join(
            path,
            "bc_subject_cv_model_comparison.csv",
        ),
        index=False,
    )


    pd.DataFrame(
        {
            "delta_elpd": [
                delta_elpd
            ],
            "clustered_se": [
                delta_elpd_clustered_se
            ],
            "bootstrap_lower_95": [
                delta_elpd_bootstrap_ci[0]
            ],
            "bootstrap_upper_95": [
                delta_elpd_bootstrap_ci[1]
            ],
        }
    ).to_csv(
        os.path.join(
            path,
            "bc_subject_cv_delta_elpd_summary.csv",
        ),
        index=False,
    )


    pd.DataFrame(
        bc_confusion_matrix,
        index=[
            "observed_wait",
            "observed_forage",
        ],
        columns=[
            "predicted_wait",
            "predicted_forage",
        ],
    ).to_csv(
        os.path.join(
            path,
            "bc_subject_cv_confusion_matrix.csv",
        )
    )


    # %% ==========================================================================
    # FULL-DATA HIERARCHICAL EXPLANATORY MODEL
    # =============================================================================
    #
    # Purpose:
    #   Estimate the group-level BC-score coefficient using all fMRI trials while
    #   accounting for participant-specific baseline tendencies.
    #
    # This full-data model is used for coefficient estimation and uncertainty.
    # Cross-validated results above remain the primary predictive evaluation.
    #
    # ==============================================================================


    # ==============================================================================
    # 15. STANDARDIZE SCORE IN THE COMPLETE fMRI SAMPLE
    # ==============================================================================

    full_score_mean = (
        validation_df[
            "bc_policy_mean"
        ].mean()
    )

    full_score_sd = (
        validation_df[
            "bc_policy_mean"
        ].std(ddof=1)
    )

    if (
        not np.isfinite(full_score_sd)
        or full_score_sd <= 0
    ):
        raise ValueError(
            "The complete-sample BC score has invalid variance."
        )


    validation_df[
        "bc_score_scaled"
    ] = (
        (
            validation_df[
                "bc_policy_mean"
            ]
            - full_score_mean
        )
        / full_score_sd
    )


    # ==============================================================================
    # 16. FIT COMPLETE-SAMPLE HIERARCHICAL MODEL
    # ==============================================================================

    full_bc_model, full_bc_result = (
        fit_validation_glmm(
            training_data=validation_df,
            include_bc_score=True,
        )
    )


    print("\n============================================================")
    print("FULL-DATA HIERARCHICAL BC MODEL")
    print("============================================================")

    print(
        full_bc_result.summary()
    )


    # ==============================================================================
    # 17. EXTRACT FIXED-EFFECT POSTERIOR SUMMARIES
    # ==============================================================================

    fixed_effect_summary = pd.DataFrame(
        {
            "parameter":
                list(
                    full_bc_model.exog_names
                ),
            "posterior_mean":
                np.asarray(
                    full_bc_result.fe_mean
                ),
            "posterior_sd":
                np.asarray(
                    full_bc_result.fe_sd
                ),
        }
    )


    fixed_effect_summary[
        "lower_95_approx"
    ] = (
        fixed_effect_summary[
            "posterior_mean"
        ]
        - 1.96
        * fixed_effect_summary[
            "posterior_sd"
        ]
    )


    fixed_effect_summary[
        "upper_95_approx"
    ] = (
        fixed_effect_summary[
            "posterior_mean"
        ]
        + 1.96
        * fixed_effect_summary[
            "posterior_sd"
        ]
    )


    full_random_intercept_sd = (
        extract_random_intercept_sd(
            full_bc_result
        )
    )


    print(
        "\nApproximate variational posterior summaries:"
    )

    print(
        fixed_effect_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        f"\nParticipant random-intercept SD: "
        f"{full_random_intercept_sd:.4f}"
    )


    # ==============================================================================
    # 18. COMPLETE-SAMPLE FITTED PROBABILITIES
    # ==============================================================================

    fixed_linear_predictor = np.asarray(
        full_bc_model.exog
        @ full_bc_result.fe_mean
    ).reshape(-1)


    random_linear_predictor = np.asarray(
        full_bc_model.exog_vc
        @ full_bc_result.vc_mean
    ).reshape(-1)


    population_fitted_probability = expit(
        fixed_linear_predictor
    )


    conditional_fitted_probability = expit(
        fixed_linear_predictor
        + random_linear_predictor
    )


    validation_df[
        "population_fitted_probability"
    ] = population_fitted_probability


    validation_df[
        "conditional_fitted_probability"
    ] = conditional_fitted_probability


    fixed_effect_summary.to_csv(
        os.path.join(
            path,
            "bc_full_model_fixed_effects.csv",
        ),
        index=False,
    )


    validation_df.to_csv(
        os.path.join(
            path,
            "bc_full_model_trial_predictions.csv",
        ),
        index=False,
    )


    pd.DataFrame(
        {
            "random_intercept_sd": [
                full_random_intercept_sd
            ],
            "bc_score_mean": [
                full_score_mean
            ],
            "bc_score_sd": [
                full_score_sd
            ],
        }
    ).to_csv(
        os.path.join(
            path,
            "bc_full_model_parameters.csv",
        ),
        index=False,
    )


    # %% ==========================================================================
    # HELD-OUT VALIDATION PLOTS
    # =============================================================================
    #
    # Every plot below uses the same subject-level out-of-fold BC probabilities.
    #
    # ==============================================================================


    # ==============================================================================
    # 19. PLOT SETTINGS
    # ==============================================================================

    PLOT_DPI = 600

    PLOT_SIZE_SQUARE = (
        4.8,
        4.8,
    )

    TITLE_SIZE = 24
    AXIS_LABEL_SIZE = 22
    TICK_SIZE = 18
    LEGEND_SIZE = 15
    LINE_WIDTH = 2.5
    FRAME_WIDTH = 0.8


    def format_validation_axis(
        axis,
    ):
        """
        Apply consistent formatting to validation plots.
        """

        axis.tick_params(
            axis="both",
            labelsize=TICK_SIZE,
            width=FRAME_WIDTH,
            length=5,
            direction="in",
        )

        for spine in axis.spines.values():
            spine.set_linewidth(
                FRAME_WIDTH
            )


    # ==============================================================================
    # 20. HELD-OUT ROC CURVE
    # ==============================================================================

    held_out_auc = roc_auc_score(
        y_out_of_fold,
        bc_out_of_fold_probability,
    )

    false_positive_rate, true_positive_rate, _ = (
        roc_curve(
            y_out_of_fold,
            bc_out_of_fold_probability,
        )
    )


    fig, axis = plt.subplots(
        figsize=PLOT_SIZE_SQUARE
    )

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=LINE_WIDTH,
        label=(
            f"BC model\n"
            f"AUC = {held_out_auc:.3f}"
        ),
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
    )

    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        0,
        1.02,
    )

    axis.set_xlabel(
        "False Positive Rate",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_ylabel(
        "True Positive Rate",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_title(
        "Model Performance",
        fontsize=TITLE_SIZE,
        loc="left",
    )

    axis.legend(
        loc="lower right",
        fontsize=LEGEND_SIZE,
        frameon=False,
    )

    format_validation_axis(
        axis
    )

    fig.tight_layout()

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_roc_curve.png",
    #     ),
    #     dpi=PLOT_DPI,
    #     bbox_inches="tight",
    # )

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_roc_curve.pdf",
    #     ),
    #     bbox_inches="tight",
    # )

    # plt.show()


    # ==============================================================================
    # 21. HELD-OUT CONFUSION MATRIX
    # ==============================================================================

    fig, axis = plt.subplots(
        figsize=PLOT_SIZE_SQUARE
    )


    confusion_display = ConfusionMatrixDisplay(
        confusion_matrix=bc_confusion_matrix,
        display_labels=[
            "Wait",
            "Forage",
        ],
    )


    confusion_display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )


    axis.set_title(
        "Confusion Matrix",
        fontsize=TITLE_SIZE,
        loc="left",
    )

    axis.set_xlabel(
        "Predicted Choice",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_ylabel(
        "Observed Choice",
        fontsize=AXIS_LABEL_SIZE,
    )


    for text_element in (
        confusion_display.text_.ravel()
    ):
        text_element.set_fontsize(
            20
        )


    format_validation_axis(
        axis
    )

    fig.tight_layout()

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_confusion_matrix.png",
    #     ),
    #     dpi=PLOT_DPI,
    #     bbox_inches="tight",
    # )

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_confusion_matrix.pdf",
    #     ),
    #     bbox_inches="tight",
    # )

    plt.show()


    # ==============================================================================
    # 22. HELD-OUT PRECISION-RECALL CURVE
    # ==============================================================================

    precision_values, recall_values, _ = (
        precision_recall_curve(
            y_out_of_fold,
            bc_out_of_fold_probability,
        )
    )


    average_precision = (
        average_precision_score(
            y_out_of_fold,
            bc_out_of_fold_probability,
        )
    )


    forage_prevalence = float(
        y_out_of_fold.mean()
    )


    fig, axis = plt.subplots(
        figsize=PLOT_SIZE_SQUARE
    )


    axis.plot(
        recall_values,
        precision_values,
        linewidth=LINE_WIDTH,
        label=(
            f"BC model\n"
            f"AP = {average_precision:.3f}"
        ),
    )


    axis.axhline(
        forage_prevalence,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"Baseline = "
            f"{forage_prevalence:.3f}"
        ),
    )


    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        0,
        1.02,
    )

    axis.set_xlabel(
        "Recall",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_ylabel(
        "Precision",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_title(
        "Held-Out Precision–Recall",
        fontsize=TITLE_SIZE,
        loc="left",
    )

    axis.legend(
        loc="lower left",
        fontsize=LEGEND_SIZE,
        frameon=False,
    )

    format_validation_axis(
        axis
    )

    fig.tight_layout()

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_precision_recall.png",
    #     ),
    #     dpi=PLOT_DPI,
    #     bbox_inches="tight",
    # )

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_precision_recall.pdf",
    #     ),
    #     bbox_inches="tight",
    # )

    plt.show()


    # ==============================================================================
    # 23. HELD-OUT CALIBRATION PLOT
    # ==============================================================================

    fraction_positive, mean_predicted_probability = (
        calibration_curve(
            y_out_of_fold,
            bc_out_of_fold_probability,
            n_bins=10,
            strategy="quantile",
        )
    )


    fig, axis = plt.subplots(
        figsize=PLOT_SIZE_SQUARE
    )


    axis.plot(
        mean_predicted_probability,
        fraction_positive,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=7,
        label="BC model",
    )


    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )


    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        0,
        1.02,
    )

    axis.set_xlabel(
        "Mean Predicted Probability",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_ylabel(
        "Observed Forage Rate",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_title(
        "Held-Out Calibration",
        fontsize=TITLE_SIZE,
        loc="left",
    )

    axis.legend(
        loc="upper left",
        fontsize=LEGEND_SIZE,
        frameon=False,
    )

    format_validation_axis(
        axis
    )

    fig.tight_layout()

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_calibration.png",
    #     ),
    #     dpi=PLOT_DPI,
    #     bbox_inches="tight",
    # )

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_calibration.pdf",
    #     ),
    #     bbox_inches="tight",
    # )

    plt.show()


    # ==============================================================================
    # 24. HELD-OUT SUBJECT-LEVEL VALIDATION
    # ==============================================================================

    subject_performance = (
        out_of_fold_predictions
        .groupby(
            "subject",
            observed=True,
        )
        .agg(
            observed_forage_rate=(
                "acts",
                "mean",
            ),
            predicted_forage_rate=(
                "bc_probability",
                "mean",
            ),
            n_trials=(
                "acts",
                "size",
            ),
        )
        .reset_index()
    )


    subject_correlation = np.corrcoef(
        subject_performance[
            "observed_forage_rate"
        ],
        subject_performance[
            "predicted_forage_rate"
        ],
    )[0, 1]


    fig, axis = plt.subplots(
        figsize=PLOT_SIZE_SQUARE
    )


    axis.scatter(
        subject_performance[
            "observed_forage_rate"
        ],
        subject_performance[
            "predicted_forage_rate"
        ],
        s=75,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.7,
    )


    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
    )


    axis.set_xlim(
        0,
        1,
    )

    axis.set_ylim(
        0,
        1,
    )

    axis.set_xlabel(
        "Observed Forage Rate",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_ylabel(
        "Predicted Forage Rate",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_title(
        (
            "Held-Out Subject-Level Fit\n"
            f"$r$ = {subject_correlation:.3f}"
        ),
        fontsize=TITLE_SIZE,
        loc="left",
    )


    format_validation_axis(
        axis
    )

    fig.tight_layout()

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_subject_validation.png",
    #     ),
    #     dpi=PLOT_DPI,
    #     bbox_inches="tight",
    # )

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_subject_validation.pdf",
    #     ),
    #     bbox_inches="tight",
    # )

    plt.show()


    # ==============================================================================
    # 25. HELD-OUT PREDICTION DISTRIBUTIONS
    # ==============================================================================

    fig, axis = plt.subplots(
        figsize=(
            PLOT_SIZE_SQUARE[0] + 2.0,
            PLOT_SIZE_SQUARE[1],
        )
    )


    axis.hist(
        bc_out_of_fold_probability[
            y_out_of_fold == 0
        ],
        bins=20,
        alpha=0.60,
        density=True,
        edgecolor="black",
        label="Observed Wait",
    )


    axis.hist(
        bc_out_of_fold_probability[
            y_out_of_fold == 1
        ],
        bins=20,
        alpha=0.60,
        density=True,
        edgecolor="black",
        label="Observed Forage",
    )


    axis.axvline(
        0.5,
        linestyle="--",
        linewidth=1.5,
        label="Decision Threshold",
    )


    axis.set_xlim(
        0,
        1,
    )

    axis.set_xlabel(
        "Predictive Checks",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_ylabel(
        "Density",
        fontsize=AXIS_LABEL_SIZE,
    )

    axis.set_title(
        "Held-Out Predictive Checks",
        fontsize=TITLE_SIZE,
        loc="left",
    )

    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        borderaxespad=0,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 1])


    format_validation_axis(
        axis
    )

    fig.tight_layout()

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_prediction_distribution.png",
    #     ),
    #     dpi=PLOT_DPI,
    #     bbox_inches="tight",
    # )

    # fig.savefig(
    #     os.path.join(
    #         path,
    #         "bc_subject_cv_prediction_distribution.pdf",
    #     ),
    #     bbox_inches="tight",
    # )

    plt.show()


    # ==============================================================================
    # 26. FINAL OUTPUT SUMMARY
    # ==============================================================================

    print("\n============================================================")
    print("BC ANALYSIS COMPLETED")
    print("============================================================")

    print("\nCross-validation outputs:")
    print("  bc_subject_cv_trial_predictions.csv")
    print("  bc_subject_cv_subject_elpd.csv")
    print("  bc_subject_cv_fold_summary.csv")
    print("  bc_subject_cv_model_comparison.csv")
    print("  bc_subject_cv_delta_elpd_summary.csv")
    print("  bc_subject_cv_confusion_matrix.csv")

    print("\nFull-data model outputs:")
    print("  bc_full_model_fixed_effects.csv")
    print("  bc_full_model_trial_predictions.csv")
    print("  bc_full_model_parameters.csv")

    print("\nHeld-out figures:")
    print("  bc_subject_cv_roc_curve.png/.pdf")
    print("  bc_subject_cv_confusion_matrix.png/.pdf")
    print("  bc_subject_cv_precision_recall.png/.pdf")
    print("  bc_subject_cv_calibration.png/.pdf")
    print("  bc_subject_cv_subject_validation.png/.pdf")
    print("  bc_subject_cv_prediction_distribution.png/.pdf")

    from scipy.stats import pearsonr, spearmanr

    # Alignment optimal policy-BC model
    optimal_policy_values = trial_data[
        "x22_optimal_policy"
    ].to_numpy(dtype=float)

    bc_values = validation_df[
        "bc_policy_mean"
    ].to_numpy(dtype=float)

    if len(optimal_policy_values) != len(bc_values):
        raise ValueError(
            "BC and optimal-policy regressors have different lengths."
        )

    if not (
        np.all(np.isfinite(optimal_policy_values))
        and np.all(np.isfinite(bc_values))
    ):
        raise ValueError(
            "Non-finite values found in BC or optimal-policy regressor."
        )

    pearson_r, pearson_p = pearsonr(
        optimal_policy_values,
        bc_values,
    )

    spearman_rho, spearman_p = spearmanr(
        optimal_policy_values,
        bc_values,
    )

    print("\n============================================================")
    print("BC–OPTIMAL POLICY OVERLAP")
    print("============================================================")
    print(
        f"Pearson r = {pearson_r:.4f}, "
        f"p = {pearson_p:.4g}"
    )
    print(
        f"Spearman rho = {spearman_rho:.4f}, "
        f"p = {spearman_p:.4g}"
    )

    slope, intercept = np.polyfit(
        optimal_policy_values,
        bc_values,
        deg=1,
    )

    bc_fitted_from_op = (
        intercept
        + slope * optimal_policy_values
    )

    residual_sd = np.std(
        bc_values - bc_fitted_from_op,
        ddof=1,
    )

    print(f"Affine slope = {slope:.4f}")
    print(f"Affine intercept = {intercept:.4f}")
    print(f"Residual SD = {residual_sd:.4f}")

    # %% ==========================================================================
    # RESULTS PRINTOUT
    # =============================================================================
    #
    # Purpose:
    #   Print all behavioral-cloning results needed to update Experiment 1.
    #
    # Notes:
    #   - Predictive metrics are based exclusively on LOSO held-out predictions.
    #   - The full-data coefficient is reported separately as an explanatory
    #     hierarchical-model estimate.
    #   - BC–optimal-policy overlap is printed only if the relevant variables
    #     have already been created.
    #
    # ==============================================================================

    from scipy.stats import pearsonr, spearmanr


    def format_p_value(p_value):
        """
        Format p values for manuscript reporting.
        """

        if not np.isfinite(p_value):
            return "NA"

        if p_value < 0.001:
            return "< .001"

        return f"= {p_value:.3f}".replace("0.", ".")


    # ==============================================================================
    # 1. BASIC SAMPLE INFORMATION
    # ==============================================================================

    n_validation_trials = len(
        out_of_fold_predictions
    )

    n_validation_subjects = (
        out_of_fold_predictions[
            "subject"
        ]
        .nunique()
    )

    subject_trial_summary = (
        out_of_fold_predictions
        .groupby(
            "subject",
            observed=True,
        )
        .size()
    )

    mean_trials_per_subject = float(
        subject_trial_summary.mean()
    )

    sd_trials_per_subject = float(
        subject_trial_summary.std(ddof=1)
    )

    minimum_trials_per_subject = int(
        subject_trial_summary.min()
    )

    maximum_trials_per_subject = int(
        subject_trial_summary.max()
    )


    # ==============================================================================
    # 2. EXTRACT LOSO PREDICTIVE METRICS
    # ==============================================================================

    bc_result_row = (
        comparison_table.loc[
            comparison_table[
                "model"
            ] == "BC score + random intercept"
        ]
        .iloc[0]
    )

    null_result_row = (
        comparison_table.loc[
            comparison_table[
                "model"
            ] == "Random-intercept null"
        ]
        .iloc[0]
    )


    bc_auc = float(
        bc_result_row["auc"]
    )

    bc_accuracy = float(
        bc_result_row["accuracy"]
    )

    bc_balanced_accuracy = float(
        bc_result_row[
            "balanced_accuracy"
        ]
    )

    bc_brier_score = float(
        bc_result_row[
            "brier_score"
        ]
    )

    bc_log_loss = float(
        bc_result_row[
            "mean_log_loss"
        ]
    )

    bc_average_precision = float(
        bc_result_row[
            "average_precision"
        ]
    )


    null_auc = float(
        null_result_row["auc"]
    )

    null_accuracy = float(
        null_result_row["accuracy"]
    )

    null_balanced_accuracy = float(
        null_result_row[
            "balanced_accuracy"
        ]
    )

    null_brier_score = float(
        null_result_row[
            "brier_score"
        ]
    )

    null_log_loss = float(
        null_result_row[
            "mean_log_loss"
        ]
    )

    null_average_precision = float(
        null_result_row[
            "average_precision"
        ]
    )


    log_loss_improvement = (
        null_log_loss
        - bc_log_loss
    )

    brier_improvement = (
        null_brier_score
        - bc_brier_score
    )


    # ==============================================================================
    # 3. CONFUSION-MATRIX DERIVED METRICS
    # ==============================================================================

    true_wait, false_forage, false_wait, true_forage = (
        bc_confusion_matrix.ravel()
    )

    forage_sensitivity = (
        true_forage
        / (
            true_forage
            + false_wait
        )
    )

    wait_specificity = (
        true_wait
        / (
            true_wait
            + false_forage
        )
    )

    positive_predictive_value = (
        true_forage
        / (
            true_forage
            + false_forage
        )
    )

    negative_predictive_value = (
        true_wait
        / (
            true_wait
            + false_wait
        )
    )


    # ==============================================================================
    # 4. FULL-DATA HIERARCHICAL COEFFICIENT
    # ==============================================================================

    bc_fixed_effect_row = (
        fixed_effect_summary.loc[
            fixed_effect_summary[
                "parameter"
            ] == "bc_score_scaled"
        ]
    )

    if len(bc_fixed_effect_row) != 1:
        raise RuntimeError(
            "Could not identify exactly one full-data "
            "BC-score coefficient."
        )

    bc_fixed_effect_row = (
        bc_fixed_effect_row.iloc[0]
    )

    bc_coefficient_mean = float(
        bc_fixed_effect_row[
            "posterior_mean"
        ]
    )

    bc_coefficient_sd = float(
        bc_fixed_effect_row[
            "posterior_sd"
        ]
    )

    bc_coefficient_lower = float(
        bc_fixed_effect_row[
            "lower_95_approx"
        ]
    )

    bc_coefficient_upper = float(
        bc_fixed_effect_row[
            "upper_95_approx"
        ]
    )

    bc_odds_ratio = float(
        np.exp(
            bc_coefficient_mean
        )
    )

    bc_odds_ratio_lower = float(
        np.exp(
            bc_coefficient_lower
        )
    )

    bc_odds_ratio_upper = float(
        np.exp(
            bc_coefficient_upper
        )
    )


    # ==============================================================================
    # 5. SUBJECT-LEVEL PREDICTIVE CORRESPONDENCE
    # ==============================================================================

    subject_rate_pearson_r, subject_rate_pearson_p = (
        pearsonr(
            subject_performance[
                "observed_forage_rate"
            ],
            subject_performance[
                "predicted_forage_rate"
            ],
        )
    )

    subject_rate_spearman_rho, subject_rate_spearman_p = (
        spearmanr(
            subject_performance[
                "observed_forage_rate"
            ],
            subject_performance[
                "predicted_forage_rate"
            ],
        )
    )


    # ==============================================================================
    # 6. PARTICIPANT-LEVEL Δ LOG PREDICTIVE DENSITY
    # ==============================================================================

    n_subjects_favoring_bc = int(
        np.sum(
            subject_delta_elpd > 0
        )
    )

    n_subjects_favoring_null = int(
        np.sum(
            subject_delta_elpd < 0
        )
    )

    n_subjects_tied = int(
        np.sum(
            subject_delta_elpd == 0
        )
    )

    mean_subject_delta_lpd = float(
        np.mean(
            subject_delta_elpd
        )
    )

    median_subject_delta_lpd = float(
        np.median(
            subject_delta_elpd
        )
    )


    # ==============================================================================
    # 7. OPTIONAL BC–OPTIMAL-POLICY OVERLAP
    # ==============================================================================
    #
    # This section will run when the aligned vectors have been created previously:
    #
    #     optimal_policy_values
    #     bc_values
    #
    # Both must refer to the same trials in the same order.
    #
    # ==============================================================================

    has_optimal_policy_overlap = (
        "optimal_policy_values" in globals()
        and "bc_values" in globals()
    )

    if has_optimal_policy_overlap:

        optimal_policy_values_print = np.asarray(
            optimal_policy_values,
            dtype=float,
        ).reshape(-1)

        bc_values_print = np.asarray(
            bc_values,
            dtype=float,
        ).reshape(-1)

        if (
            len(optimal_policy_values_print)
            != len(bc_values_print)
        ):
            raise ValueError(
                "BC and optimal-policy vectors have "
                "different lengths."
            )

        overlap_mask = (
            np.isfinite(
                optimal_policy_values_print
            )
            & np.isfinite(
                bc_values_print
            )
        )

        optimal_policy_values_print = (
            optimal_policy_values_print[
                overlap_mask
            ]
        )

        bc_values_print = (
            bc_values_print[
                overlap_mask
            ]
        )

        bc_op_pearson_r, bc_op_pearson_p = (
            pearsonr(
                optimal_policy_values_print,
                bc_values_print,
            )
        )

        bc_op_spearman_rho, bc_op_spearman_p = (
            spearmanr(
                optimal_policy_values_print,
                bc_values_print,
            )
        )

        bc_op_r_squared = float(
            bc_op_pearson_r ** 2
        )

    else:

        bc_op_pearson_r = np.nan
        bc_op_pearson_p = np.nan
        bc_op_spearman_rho = np.nan
        bc_op_spearman_p = np.nan
        bc_op_r_squared = np.nan


    # ==============================================================================
    # 8. PRINT MANUSCRIPT-READY SUMMARY
    # ==============================================================================

    print("\n")
    print("=" * 76)
    print("EXPERIMENT 1: BEHAVIORAL-CLONING MANUSCRIPT RESULTS")
    print("=" * 76)


    print("\nSAMPLE AND VALIDATION")
    print("-" * 76)

    print(
        f"Validation participants: "
        f"N = {n_validation_subjects}"
    )

    print(
        f"Validation trials: "
        f"n = {n_validation_trials}"
    )

    print(
        "Trials per participant: "
        f"M = {mean_trials_per_subject:.2f}, "
        f"SD = {sd_trials_per_subject:.2f}, "
        f"range = "
        f"[{minimum_trials_per_subject}, "
        f"{maximum_trials_per_subject}]"
    )

    print(
        "Validation procedure: "
        "leave-one-subject-out cross-validation"
    )


    print("\nHELD-OUT BC PERFORMANCE")
    print("-" * 76)

    print(
        f"Accuracy = {bc_accuracy:.3f}"
    )

    print(
        f"Balanced accuracy = "
        f"{bc_balanced_accuracy:.3f}"
    )

    print(
        f"AUC = {bc_auc:.3f}"
    )

    print(
        f"Average precision = "
        f"{bc_average_precision:.3f}"
    )

    print(
        f"Brier score = "
        f"{bc_brier_score:.4f}"
    )

    print(
        f"Mean log loss = "
        f"{bc_log_loss:.4f}"
    )

    print(
        f"Forage sensitivity = "
        f"{forage_sensitivity:.3f}"
    )

    print(
        f"Wait specificity = "
        f"{wait_specificity:.3f}"
    )

    print(
        f"Positive predictive value = "
        f"{positive_predictive_value:.3f}"
    )

    print(
        f"Negative predictive value = "
        f"{negative_predictive_value:.3f}"
    )

    print(
        "Confusion matrix "
        "[[observed wait/predicted wait, "
        "observed wait/predicted forage], "
        "[observed forage/predicted wait, "
        "observed forage/predicted forage]]:"
    )

    print(
        bc_confusion_matrix
    )


    print("\nRANDOM-INTERCEPT NULL PERFORMANCE")
    print("-" * 76)

    print(
        f"Accuracy = {null_accuracy:.3f}"
    )

    print(
        f"Balanced accuracy = "
        f"{null_balanced_accuracy:.3f}"
    )

    print(
        f"AUC = {null_auc:.3f}"
    )

    print(
        f"Average precision = "
        f"{null_average_precision:.3f}"
    )

    print(
        f"Brier score = "
        f"{null_brier_score:.4f}"
    )

    print(
        f"Mean log loss = "
        f"{null_log_loss:.4f}"
    )


    print("\nBC VERSUS NULL MODEL")
    print("-" * 76)

    print(
        "Null-model summed LOSO log "
        f"predictive density = {null_elpd:.3f}"
    )

    print(
        "BC-model summed LOSO log "
        f"predictive density = {bc_elpd:.3f}"
    )

    print(
        f"Δ log predictive density "
        f"(BC − null) = {delta_elpd:.3f}"
    )

    print(
        f"Participant-clustered SE = "
        f"{delta_elpd_clustered_se:.3f}"
    )

    print(
        "Participant-bootstrap 95% interval = "
        f"[{delta_elpd_bootstrap_ci[0]:.3f}, "
        f"{delta_elpd_bootstrap_ci[1]:.3f}]"
    )

    print(
        f"Mean participant Δ log predictive density = "
        f"{mean_subject_delta_lpd:.3f}"
    )

    print(
        f"Median participant Δ log predictive density = "
        f"{median_subject_delta_lpd:.3f}"
    )

    print(
        f"Participants favoring BC = "
        f"{n_subjects_favoring_bc}/"
        f"{n_validation_subjects}"
    )

    print(
        f"Participants favoring null = "
        f"{n_subjects_favoring_null}/"
        f"{n_validation_subjects}"
    )

    print(
        f"Participants tied = "
        f"{n_subjects_tied}/"
        f"{n_validation_subjects}"
    )

    print(
        f"Mean held-out log-loss improvement "
        f"(null − BC) = {log_loss_improvement:.5f}"
    )

    print(
        f"Brier-score improvement "
        f"(null − BC) = {brier_improvement:.5f}"
    )


    print("\nFULL-DATA HIERARCHICAL BC EFFECT")
    print("-" * 76)

    print(
        "Standardized BC-score coefficient: "
        f"posterior mean = {bc_coefficient_mean:.4f}, "
        f"posterior SD = {bc_coefficient_sd:.4f}"
    )

    print(
        "Approximate 95% interval = "
        f"[{bc_coefficient_lower:.4f}, "
        f"{bc_coefficient_upper:.4f}]"
    )

    print(
        f"Odds ratio per 1-SD increase = "
        f"{bc_odds_ratio:.3f}"
    )

    print(
        "Approximate odds-ratio 95% interval = "
        f"[{bc_odds_ratio_lower:.3f}, "
        f"{bc_odds_ratio_upper:.3f}]"
    )

    print(
        f"Participant random-intercept SD = "
        f"{full_random_intercept_sd:.4f}"
    )


    print("\nSUBJECT-LEVEL OBSERVED–PREDICTED CORRESPONDENCE")
    print("-" * 76)

    print(
        "Pearson correlation between observed and "
        "held-out predicted forage rates: "
        f"r = {subject_rate_pearson_r:.3f}, "
        f"p {format_p_value(subject_rate_pearson_p)}"
    )

    print(
        "Spearman correlation between observed and "
        "held-out predicted forage rates: "
        f"rho = {subject_rate_spearman_rho:.3f}, "
        f"p {format_p_value(subject_rate_spearman_p)}"
    )


    print("\nBC–OPTIMAL-POLICY OVERLAP")
    print("-" * 76)

    if has_optimal_policy_overlap:

        print(
            "Pearson correlation: "
            f"r = {bc_op_pearson_r:.3f}, "
            f"p {format_p_value(bc_op_pearson_p)}"
        )

        print(
            "Shared variance: "
            f"R² = {bc_op_r_squared:.3f}"
        )

        print(
            "Spearman correlation: "
            f"rho = {bc_op_spearman_rho:.3f}, "
            f"p {format_p_value(bc_op_spearman_p)}"
        )

    else:

        print(
            "Not calculated: aligned variables "
            "'optimal_policy_values' and 'bc_values' "
            "were not found."
        )


    print("\nMANUSCRIPT SENTENCE TEMPLATE")
    print("-" * 76)

    print(
        "The behavioral-cloning policy score was evaluated in the "
        f"independent fMRI sample (N = {n_validation_subjects}) using "
        "leave-one-subject-out cross-validation. "
        f"The held-out model achieved an accuracy of {bc_accuracy:.2f}, "
        f"a balanced accuracy of {bc_balanced_accuracy:.2f}, and an "
        f"AUC of {bc_auc:.2f}. Relative to a random-intercept null model, "
        "the BC model improved the summed held-out log predictive density "
        f"by {delta_elpd:.2f} "
        f"(participant-bootstrap 95% interval "
        f"[{delta_elpd_bootstrap_ci[0]:.2f}, "
        f"{delta_elpd_bootstrap_ci[1]:.2f}])."
    )

    if has_optimal_policy_overlap:

        print(
            "The BC policy score was strongly associated with the "
            "optimal-policy regressor, "
            f"r = {bc_op_pearson_r:.2f}, "
            f"p {format_p_value(bc_op_pearson_p)}, "
            "indicating substantial shared trial-wise variance."
        )


    # ==============================================================================
    # 9. SAVE ONE MANUSCRIPT SUMMARY TABLE
    # ==============================================================================

    manuscript_summary = pd.DataFrame(
        {
            "result": [
                "n_participants",
                "n_trials",
                "accuracy",
                "balanced_accuracy",
                "auc",
                "average_precision",
                "brier_score",
                "mean_log_loss",
                "forage_sensitivity",
                "wait_specificity",
                "null_accuracy",
                "null_auc",
                "null_brier_score",
                "null_mean_log_loss",
                "null_loso_log_predictive_density",
                "bc_loso_log_predictive_density",
                "delta_log_predictive_density",
                "delta_lpd_clustered_se",
                "delta_lpd_bootstrap_lower_95",
                "delta_lpd_bootstrap_upper_95",
                "subjects_favoring_bc",
                "subjects_favoring_null",
                "full_bc_coefficient_mean",
                "full_bc_coefficient_sd",
                "full_bc_coefficient_lower_95",
                "full_bc_coefficient_upper_95",
                "bc_odds_ratio",
                "random_intercept_sd",
                "subject_rate_pearson_r",
                "subject_rate_pearson_p",
                "bc_op_pearson_r",
                "bc_op_pearson_p",
                "bc_op_r_squared",
                "bc_op_spearman_rho",
                "bc_op_spearman_p",
            ],
            "value": [
                n_validation_subjects,
                n_validation_trials,
                bc_accuracy,
                bc_balanced_accuracy,
                bc_auc,
                bc_average_precision,
                bc_brier_score,
                bc_log_loss,
                forage_sensitivity,
                wait_specificity,
                null_accuracy,
                null_auc,
                null_brier_score,
                null_log_loss,
                null_elpd,
                bc_elpd,
                delta_elpd,
                delta_elpd_clustered_se,
                delta_elpd_bootstrap_ci[0],
                delta_elpd_bootstrap_ci[1],
                n_subjects_favoring_bc,
                n_subjects_favoring_null,
                bc_coefficient_mean,
                bc_coefficient_sd,
                bc_coefficient_lower,
                bc_coefficient_upper,
                bc_odds_ratio,
                full_random_intercept_sd,
                subject_rate_pearson_r,
                subject_rate_pearson_p,
                bc_op_pearson_r,
                bc_op_pearson_p,
                bc_op_r_squared,
                bc_op_spearman_rho,
                bc_op_spearman_p,
            ],
        }
    )

    manuscript_summary.to_csv(
        os.path.join(
            path,
            "bc_experiment1_manuscript_results.csv",
        ),
        index=False,
    )

    print(
        "\nSaved manuscript summary:"
    )

    print(
        "  bc_experiment1_manuscript_results.csv"
    )