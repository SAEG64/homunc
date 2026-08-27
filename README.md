# Feature-Based Policies and Integrated Values in Sequential Foraging Decisions

This repository contains code and behavioral data associated with **Study 1** of the doctoral dissertation:

**Values, Features, and Beliefs in Adaptive Sequential Decision-Making**  
Sergej Golowin  
Heidelberg University, 2026

## Overview

This study investigates how decision-relevant information is represented and integrated during sequential decision-making under outcome uncertainty.

Participants completed a sequential foraging task in which they chose between **foraging** and **waiting** while maintaining an energy reserve across a sequence of trials.

The main analyses tested whether behavior was better characterized by:

- an MDP-derived optimal policy based on integrated long-term state-action values,
- individual decision-relevant task features,
- or a context-dependent multi-feature policy.

A behavioral-cloning model was additionally trained on an independent behavioral sample and evaluated in the fMRI sample as a data-driven estimate of participants' choice policy.

The results support a context-dependent multi-feature representation of behavior while also showing neural signals associated with integrated optimal-policy values.

## Participants

The study included **55 participants** across two independent samples:

- **28 participants** in the behavioral pilot sample,
- **27 participants** in the fMRI sample.

The behavioral sample contained **13,674 analyzed decisions** and the fMRI sample contained **9,236 analyzed decisions**.

Forests contained five decision trials in the behavioral sample and three to five trials in the fMRI sample.

## Main Analyses

The reported analyses include:

- comparison of feature-based and MDP-derived behavioral policies,
- context-dependent multi-feature behavioral modeling,
- out-of-sample behavioral model validation,
- behavioral cloning,
- leave-one-subject-out validation of the behavioral-cloning policy,
- fMRI general linear models,
- analysis of BOLD activity associated with optimal-policy Delta Q,
- analysis of neural interactions between decision-relevant task features,
- region-of-interest analyses.

The main decision variables included:

- foraging success probability,
- gain magnitude,
- participant energy state,
- remaining decision horizon,
- weather type,
- expected energy change,
- ternary energy state,
- MDP-derived Delta Q.

## State-Action Values

The task was formalized as a finite-horizon Markov decision process and solved using backward induction.

The optimal-policy decision variable was defined as:

```text
Delta Q = Q(forage) - Q(wait)
```

The state-action values were **precomputed and saved in the uploaded behavioral data sheets**.

The downstream behavioral and neuroimaging analysis scripts therefore use these stored trial-wise values rather than recomputing the complete MDP solution during each analysis.

## Main Result

Behavior was better captured by a context-dependent multi-feature policy than by the fully integrated optimal-policy value signal alone.

At the neural level, MDP-derived Delta Q was represented across medial frontal, striatal, parietal, and insular regions, while additional analyses identified neural sensitivity to interactions among task features.

Overall, the results suggest that adaptive sequential choice can rely on lower-dimensional, context-sensitive feature representations even when the brain also represents integrated long-term action values.

## Requirements

The analyses are written primarily in Python.

Main packages include:

```text
numpy
pandas
scipy
matplotlib
scikit-learn
statsmodels
torch
gymnasium
stable-baselines3
imitation
```

The fMRI analyses additionally use MATLAB and SPM12.

## Running the Repository

Clone the repository:

```bash
git clone https://github.com/SAEG64/homunc.git
cd homunc
```

Install the required dependencies and run the analysis scripts corresponding to the behavioral, behavioral-cloning, and neuroimaging analyses reported in the dissertation.

The behavioral analysis including model comparisons, imitation learning, and XAI can be done with the directory. The fMRI analysis required the HOMUN_parent directory containing the preprocessed fMRI data not available in this repo.
Golowin, S. (2026).  
**Values, Features, and Beliefs in Adaptive Sequential Decision-Making.**  
Doctoral dissertation, Heidelberg University.
