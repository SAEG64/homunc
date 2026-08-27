#!/usr/bin/env Rscript

# ==============================================================================
# HOMUNC BEHAVIORAL ANALYSIS — HIERARCHICAL GLMM VERSION
# ==============================================================================
#
# Purpose
# -------
# Reimplements the original Python analysis using frequentist binomial
# generalized linear mixed-effects models (GLMMs) fitted with lme4::glmer().
#
# Key change
# ----------
# Every candidate model includes a participant-specific random intercept:
#
#   (1 | x1_id)
#
# Models are fitted by maximum likelihood and compared using AIC and BIC.
#
# Directory behavior
# ------------------
# The script uses the directory containing this R script as its working
# directory, matching the original Python script's behavior.
#
# Required input files in the same directory:
#   - data_beh.csv
#   - data_fmri.csv
#
# Main required packages:
#   lme4, ggplot2, dplyr, tidyr, purrr, readr, broom.mixed,
#   performance, pROC, patchwork
#
# ==============================================================================


# ==============================================================================
# 0. PACKAGE SETUP
# ==============================================================================

required_packages <- c(
  "lme4",
  "ggplot2",
  "dplyr",
  "tidyr",
  "purrr",
  "readr",
  "broom.mixed",
  "performance",
  "pROC",
  "patchwork"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "The following packages are missing:\n  ",
      paste(missing_packages, collapse = ", "),
      "\n\nInstall them with:\n",
      "install.packages(c(",
      paste(sprintf('"%s"', missing_packages), collapse = ", "),
      "))"
    )
  )
}

suppressPackageStartupMessages({
  library(lme4)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(readr)
  library(broom.mixed)
  library(performance)
  library(pROC)
  library(patchwork)
})


# ==============================================================================
# 1. WORKING DIRECTORY
# ==============================================================================

get_script_directory <- function() {
  command_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", command_args, value = TRUE)

  if (length(file_arg) > 0) {
    script_path <- sub("^--file=", "", file_arg[1])
    return(dirname(normalizePath(script_path)))
  }

  # RStudio fallback
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    active_file <- rstudioapi::getActiveDocumentContext()$path

    if (nzchar(active_file)) {
      return(dirname(normalizePath(active_file)))
    }
  }

  # Final fallback: current working directory
  normalizePath(getwd())
}

path <- get_script_directory()
setwd(path)

cat("Working directory:\n", path, "\n\n")


# ==============================================================================
# 2. GLOBAL SETTINGS
# ==============================================================================

theme_set(
  theme_classic(base_size = 18) +
    theme(
      axis.line = element_line(linewidth = 0.8),
      axis.ticks = element_line(linewidth = 0.8),
      plot.title = element_text(
        size = 24,
        face = "bold",
        hjust = 0
      ),
      axis.title = element_text(size = 22),
      axis.text = element_text(size = 18),
      legend.title = element_blank(),
      legend.text = element_text(size = 18)
    )
)

glmer_control <- glmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 200000),
  calc.derivs = TRUE
)


# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

prepare_data <- function(file_name) {

  data <- read_csv(
    file.path(path, file_name),
    show_col_types = FALSE
  )

  required_raw_columns <- c(
    "x11_choice",
    "x1_id",
    "x17_horizon_correct_adjusted",
    "x13_gain_magnitude",
    "x37_binary_energy",
    "x6_continuous_energy_trial_start",
    "x41_expected_energy_change",
    "x14_p_foraging_gain",
    "x19_wait_when_safe",
    "x7_weather_type",
    "x22_optimal_policy"
  )

  missing_columns <- setdiff(required_raw_columns, names(data))

  if (length(missing_columns) > 0) {
    stop(
      paste0(
        "Missing required columns in ",
        file_name,
        ":\n  ",
        paste(missing_columns, collapse = ", ")
      )
    )
  }

  data <- data %>%
    mutate(
      x1_id = factor(x1_id),
      x11_choice = as.integer(x11_choice),
      x7_weather_type = factor(x7_weather_type),
      
      # Success probability centered at .5.
      # Thus, 0 represents a success probability of .5.
      p_success_centered =
        x14_p_foraging_gain - 0.5,
      
      BNW_numeric =
        x37_binary_energy - x19_wait_when_safe,
      
      BNW_conditions = factor(
        BNW_numeric,
        levels = c(0, 1, -1),
        labels = c(
          "Trade-off",
          "Binary Energy",
          "Wait When Safe"
        )
      )
    )
  invalid_choices <- setdiff(unique(na.omit(data$x11_choice)), c(0L, 1L))

  if (length(invalid_choices) > 0) {
    stop(
      paste0(
        "x11_choice must contain only 0 and 1. Found: ",
        paste(invalid_choices, collapse = ", ")
      )
    )
  }

  data
}


fit_glmer_safe <- function(formula_i, data_i, model_name) {

  cat("\n------------------------------------------------------------\n")
  cat("Fitting:", model_name, "\n")
  cat("Formula:", deparse(formula_i), "\n")

  warnings_seen <- character(0)

  fitted_model <- tryCatch(
    withCallingHandlers(
      glmer(
        formula = formula_i,
        data = data_i,
        family = binomial(link = "logit"),
        control = glmer_control,
        nAGQ = 1
      ),
      warning = function(w) {
        warnings_seen <<- c(warnings_seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      message("Model failed: ", conditionMessage(e))
      return(NULL)
    }
  )

  if (is.null(fitted_model)) {
    return(
      list(
        model = NULL,
        warnings = warnings_seen,
        convergence_message = "MODEL FAILED",
        singular = NA
      )
    )
  }

  convergence_message <- fitted_model@optinfo$conv$lme4$messages

  if (is.null(convergence_message)) {
    convergence_message <- "OK"
  } else {
    convergence_message <- paste(convergence_message, collapse = " | ")
  }

  singular_fit <- isSingular(fitted_model, tol = 1e-4)

  cat("AIC:", AIC(fitted_model), "\n")
  cat("BIC:", BIC(fitted_model), "\n")
  cat("N:", nobs(fitted_model), "\n")
  cat("Singular:", singular_fit, "\n")
  cat("Convergence:", convergence_message, "\n")

  if (length(warnings_seen) > 0) {
    cat("Warnings:", paste(unique(warnings_seen), collapse = " | "), "\n")
  }

  list(
    model = fitted_model,
    warnings = warnings_seen,
    convergence_message = convergence_message,
    singular = singular_fit
  )
}


extract_model_metrics <- function(fit_object, model_name, description) {

  model <- fit_object$model

  if (is.null(model)) {
    return(
      tibble(
        model_name = model_name,
        description = description,
        n = NA_integer_,
        logLik = NA_real_,
        parameters = NA_integer_,
        AIC = NA_real_,
        BIC = NA_real_,
        singular = NA,
        convergence = fit_object$convergence_message,
        warnings = paste(unique(fit_object$warnings), collapse = " | ")
      )
    )
  }

  ll <- logLik(model)

  tibble(
    model_name = model_name,
    description = description,
    n = nobs(model),
    logLik = as.numeric(ll),
    parameters = attr(ll, "df"),
    AIC = AIC(model),
    BIC = BIC(model),
    singular = fit_object$singular,
    convergence = fit_object$convergence_message,
    warnings = paste(unique(fit_object$warnings), collapse = " | ")
  )
}


save_plot <- function(plot_object, filename, width, height) {
  ggsave(
    filename = file.path(path, filename),
    plot = plot_object,
    width = width,
    height = height,
    units = "in",
    dpi = 300
  )
}


# ==============================================================================
# 4. MODEL COMPARISON
# ==============================================================================

# Match the original Python script:
#   d = "data_beh.csv"
#   d = "data_fmri.csv"
#
# The second assignment wins, so the primary comparison uses data_fmri.csv.
comparison_file <- "data_beh.csv"

combined_data <- prepare_data(comparison_file)


# ------------------------------------------------------------------------------
# Candidate models
# ------------------------------------------------------------------------------
#
# Important:
# - All models include the same participant random intercept.
# - The factorial "*" operator includes all corresponding lower-order terms.
# - BNW_conditions has Trade-off as its reference category.
#
# The candidate set matches the actual model_vars list in the Python script.

model_formulas <- list(
  
  p_success =
    x11_choice ~
    x14_p_foraging_gain +
    (1 | x1_id),
  
  gain_magnitude =
    x11_choice ~
    x13_gain_magnitude +
    (1 | x1_id),
  
  remaining_time =
    x11_choice ~
    x17_horizon_correct_adjusted +
    (1 | x1_id),
  
  continuous_energy =
    x11_choice ~
    x6_continuous_energy_trial_start +
    (1 | x1_id),
  
  weather_type =
    x11_choice ~
    x7_weather_type +
    (1 | x1_id),
  
  expected_energy_change =
    x11_choice ~
    x41_expected_energy_change +
    (1 | x1_id),
  
  ternary_state =
    x11_choice ~
    BNW_conditions +
    (1 | x1_id),
  
  ternary_weather =
    x11_choice ~
    BNW_conditions *
    x7_weather_type +
    (1 | x1_id),
  
  factorial_policy =
    x11_choice ~
    BNW_conditions *
    x7_weather_type *
    p_success_centered +
    (1 | x1_id),
  
  optimal_policy =
    x11_choice ~
    x22_optimal_policy +
    (1 | x1_id)
)


model_descriptions <- c(
  p_success =
    "italic(p)(success)",
  
  gain_magnitude =
    "gain~magnitude",
  
  remaining_time =
    "remaining~days",
  
  continuous_energy =
    "continuous~energy~state",
  
  weather_type =
    "weather~type",
  
  expected_energy_change =
    "expected~energy~change",
  
  ternary_state =
    "ternary~state",
  
  ternary_weather =
    "ternary~state~'\u00D7'~weather~type",
  
  factorial_policy =
    "'multi-feature model'",
  
  optimal_policy =
    "Delta*italic(Q)~values"
)


# ------------------------------------------------------------------------------
# Ensure all models use exactly the same observations
# ------------------------------------------------------------------------------

all_model_variables <- unique(
  unlist(
    lapply(
      model_formulas,
      function(f) all.vars(f)
    )
  )
)

# Remove response/group columns only if duplicated; complete.cases should include
# them as well, so they remain in all_model_variables.
analysis_data <- combined_data %>%
  filter(
    complete.cases(
      across(all_of(all_model_variables))
    )
  ) %>%
  droplevels()

cat("\n============================================================\n")
cat("PRIMARY MODEL-COMPARISON DATA\n")
cat("============================================================\n")
cat("Input file:", comparison_file, "\n")
cat("Rows before complete-case filtering:", nrow(combined_data), "\n")
cat("Rows used by every model:", nrow(analysis_data), "\n")
cat("Participants:", nlevels(analysis_data$x1_id), "\n")


# ------------------------------------------------------------------------------
# Fit models
# ------------------------------------------------------------------------------

model_fits <- imap(
  model_formulas,
  ~ fit_glmer_safe(
    formula_i = .x,
    data_i = analysis_data,
    model_name = .y
  )
)


# ------------------------------------------------------------------------------
# Build model-comparison table
# ------------------------------------------------------------------------------

comparison_df <- imap_dfr(
  model_fits,
  function(fit_object, model_name) {

    description_i <- model_descriptions[[model_name]]

    extract_model_metrics(
      fit_object = fit_object,
      model_name = model_name,
      description = description_i
    )
  }
)

if (all(is.na(comparison_df$BIC))) {
  stop("All candidate models failed.")
}

comparison_df <- comparison_df %>%
  mutate(
    delta_BIC = BIC - min(BIC, na.rm = TRUE),
    delta_AIC = AIC - min(AIC, na.rm = TRUE),
    BIC_weight_raw = exp(-0.5 * delta_BIC),
    AIC_weight_raw = exp(-0.5 * delta_AIC),
    BIC_weight = BIC_weight_raw / sum(BIC_weight_raw, na.rm = TRUE),
    AIC_weight = AIC_weight_raw / sum(AIC_weight_raw, na.rm = TRUE)
  ) %>%
  arrange(delta_BIC) %>%
  select(
    model_name,
    description,
    n,
    logLik,
    parameters,
    BIC,
    delta_BIC,
    BIC_weight,
    AIC,
    delta_AIC,
    AIC_weight,
    singular,
    convergence,
    warnings
  )

cat("\n============================================================\n")
cat("HIERARCHICAL MODEL-COMPARISON RESULTS\n")
cat("============================================================\n")
print(comparison_df, n = Inf, width = Inf)


# Verify equal observations
unique_n <- unique(na.omit(comparison_df$n))

if (length(unique_n) != 1) {
  stop(
    paste0(
      "Models were fitted to different numbers of observations: ",
      paste(unique_n, collapse = ", ")
    )
  )
}

write_csv(
  comparison_df,
  file.path(path, "hierarchical_model_comparison.csv")
)


# ------------------------------------------------------------------------------
# Save fMRI model order, matching the original directory/output behavior
# ------------------------------------------------------------------------------

if (comparison_file == "data_beh.csv") {

  fmri_order <- comparison_df$model_name

  write_csv(
    tibble(model_name = fmri_order),
    file.path(path, "fmri_model_order.csv")
  )

  cat("\nSaved fMRI model order.\n")
}


# ------------------------------------------------------------------------------
# BIC plot
# ------------------------------------------------------------------------------

bic_plot_data <- comparison_df %>%
  filter(!is.na(delta_BIC)) %>%
  mutate(
    description = factor(
      description,
      levels = rev(description)
    )
  )

bic_plot <- ggplot(
  bic_plot_data,
  aes(
    x = delta_BIC,
    y = description
  )
) +
  geom_col(width = 0.7) +
  geom_text(
    aes(
      label = ifelse(
        delta_BIC > 0,
        sprintf("%.1f", delta_BIC),
        ""
      )
    ),
    hjust = -0.1,
    size = 6
  ) +
  labs(
    title = "BIC Model Comparison",
    x = expression(Delta ~ "BIC (0 = best model)"),
    y = NULL
  ) +
  coord_cartesian(
    xlim = c(
      0,
      max(bic_plot_data$delta_BIC, na.rm = TRUE) * 1.12
    ),
    clip = "off"
  ) +
  theme(
    plot.margin = margin(10, 40, 10, 10)
  )

save_plot(
  bic_plot,
  "hierarchical_bic_comparison.png",
  width = 17,
  height = 8
)


# ------------------------------------------------------------------------------
# AIC plot
# ------------------------------------------------------------------------------

aic_plot_data <- comparison_df %>%
  filter(!is.na(delta_AIC)) %>%
  mutate(
    description = factor(
      description,
      levels = rev(description)
    )
  )

aic_plot <- ggplot(
  aic_plot_data,
  aes(
    x = delta_AIC,
    y = description
  )
) +
  geom_col(width = 0.7) +
  geom_text(
    aes(
      label = ifelse(
        delta_AIC > 0,
        sprintf("%.1f", delta_AIC),
        ""
      )
    ),
    hjust = -0.1,
    size = 6
  ) +
  labs(
    title = "AIC Model Comparison",
    x = expression(Delta ~ "AIC (0 = best model)"),
    y = NULL
  ) +
  coord_cartesian(
    xlim = c(
      0,
      max(aic_plot_data$delta_AIC, na.rm = TRUE) * 1.12
    ),
    clip = "off"
  ) +
  theme(
    plot.margin = margin(10, 40, 10, 10)
  )

save_plot(
  aic_plot,
  "hierarchical_aic_comparison.png",
  width = 17,
  height = 8
)


# ------------------------------------------------------------------------------
# Model-weights plot
# ------------------------------------------------------------------------------

weights_plot_data <- comparison_df %>%
  select(description, AIC_weight, BIC_weight) %>%
  pivot_longer(
    cols = c(AIC_weight, BIC_weight),
    names_to = "criterion",
    values_to = "weight"
  ) %>%
  mutate(
    criterion = recode(
      criterion,
      AIC_weight = "AIC Weights",
      BIC_weight = "BIC Weights"
    ),
    description = factor(
      description,
      levels = rev(comparison_df$description)
    )
  )

weights_plot <- ggplot(
  weights_plot_data,
  aes(
    x = weight,
    y = description,
    fill = criterion
  )
) +
  geom_col(
    position = position_dodge(width = 0.8),
    width = 0.7
  ) +
  labs(
    title = "Model Weights",
    x = "Model Weight",
    y = NULL,
    fill = NULL
  ) +
  coord_cartesian(xlim = c(0, 1)) +
  theme(
    legend.position = "bottom"
  )

save_plot(
  weights_plot,
  "hierarchical_model_weights.png",
  width = 15,
  height = 8
)


# ==============================================================================
# 5. COMMONALITY / COLLINEARITY ANALYSIS
# ==============================================================================

# Match the original in-band definition:
#   0.3 < p(success) < 0.7

combined_data_masked <- combined_data %>%
  filter(
    x14_p_foraging_gain > 0.3,
    x14_p_foraging_gain < 0.7
  ) %>%
  droplevels()

cat("\n============================================================\n")
cat("IN-BAND WEATHER / P(SUCCESS) DIAGNOSTICS\n")
cat("============================================================\n")

cor_test <- cor.test(
  combined_data_masked$x14_p_foraging_gain,
  as.numeric(combined_data_masked$x7_weather_type),
  method = "pearson"
)

cat("Correlation:", unname(cor_test$estimate), "\n")
cat("p-value:", cor_test$p.value, "\n")


# ------------------------------------------------------------------------------
# VIFs
# ------------------------------------------------------------------------------
#
# performance::check_collinearity() works on fitted models and handles factors.
# This is preferable to manually applying standard linear-model VIF formulas
# to dummy-coded columns.

vif_model <- lm(
  x11_choice ~
    x7_weather_type +
    x14_p_foraging_gain,
  data = combined_data_masked
)

vif_results <- performance::check_collinearity(vif_model)

cat("\nVIF diagnostics:\n")
print(vif_results)

write_csv(
  as.data.frame(vif_results),
  file.path(path, "weather_p_success_vif.csv")
)


# ------------------------------------------------------------------------------
# Commonality analysis using linear-model R², matching the original script
# ------------------------------------------------------------------------------

commonality_data <- combined_data_masked %>%
  mutate(
    weather_numeric = as.numeric(x7_weather_type),
    BNW_commonality = BNW_numeric
  ) %>%
  select(
    x11_choice,
    weather_numeric,
    x14_p_foraging_gain,
    BNW_commonality
  ) %>%
  drop_na()

get_r2 <- function(predictors, data_i) {

  formula_i <- reformulate(
    predictors,
    response = "x11_choice"
  )

  summary(
    lm(
      formula = formula_i,
      data = data_i
    )
  )$r.squared
}

R2_full <- get_r2(
  c(
    "weather_numeric",
    "x14_p_foraging_gain",
    "BNW_commonality"
  ),
  commonality_data
)

R2_1 <- get_r2(
  "weather_numeric",
  commonality_data
)

R2_2 <- get_r2(
  "x14_p_foraging_gain",
  commonality_data
)

R2_3 <- get_r2(
  "BNW_commonality",
  commonality_data
)

R2_12 <- get_r2(
  c("weather_numeric", "x14_p_foraging_gain"),
  commonality_data
)

R2_13 <- get_r2(
  c("weather_numeric", "BNW_commonality"),
  commonality_data
)

R2_23 <- get_r2(
  c("x14_p_foraging_gain", "BNW_commonality"),
  commonality_data
)

U1 <- R2_full - R2_23
U2 <- R2_full - R2_13
U3 <- R2_full - R2_12

C12 <- R2_12 - U1 - U2
C13 <- R2_13 - U1 - U3
C23 <- R2_23 - U2 - U3

C123 <- R2_full - (
  U1 + U2 + U3 +
  C12 + C13 + C23
)

commonality_table <- tibble(
  Component = c(
    "U1",
    "U2",
    "U3",
    "C12",
    "C13",
    "C23",
    "C123"
  ),
  Variance = c(
    U1,
    U2,
    U3,
    C12,
    C13,
    C23,
    C123
  )
)

cat("\nCommonality table:\n")
print(commonality_table)

write_csv(
  commonality_table,
  file.path(path, "commonality_analysis.csv")
)


# ==============================================================================
# 6. OUT-OF-SAMPLE PERFORMANCE OF THE WINNING MODEL
# ==============================================================================

data_train <- prepare_data("data_beh.csv")
data_test <- prepare_data("data_fmri.csv")

best_model_name <- comparison_df %>%
  filter(!is.na(delta_BIC)) %>%
  slice_min(delta_BIC, n = 1, with_ties = FALSE) %>%
  pull(model_name)

best_formula <- model_formulas[[best_model_name]]

cat("\n============================================================\n")
cat("OUT-OF-SAMPLE VALIDATION\n")
cat("============================================================\n")
cat("Winning model from primary comparison:", best_model_name, "\n")
cat("Formula:", deparse(best_formula), "\n")


# ------------------------------------------------------------------------------
# Important prediction choice
# ------------------------------------------------------------------------------
#
# The behavioral sample and fMRI sample contain different participants.
# Subject-specific random intercepts estimated in the training set cannot be
# transferred to unseen test participants.
#
# Therefore:
#   re.form = NA
#
# produces population-level predictions using fixed effects only. This is the
# correct out-of-sample prediction for entirely new participants.

best_model_variables <- all.vars(best_formula)

train_complete <- data_train %>%
  filter(
    complete.cases(
      across(all_of(best_model_variables))
    )
  ) %>%
  droplevels()

test_complete <- data_test %>%
  filter(
    complete.cases(
      across(all_of(best_model_variables))
    )
  ) %>%
  droplevels()

winning_train_model <- glmer(
  formula = best_formula,
  data = train_complete,
  family = binomial(link = "logit"),
  control = glmer_control,
  nAGQ = 1
)

pred_probs <- predict(
  winning_train_model,
  newdata = test_complete,
  type = "response",
  re.form = NA,
  allow.new.levels = TRUE
)

pred_binary <- ifelse(pred_probs >= 0.5, 1L, 0L)
y_test <- test_complete$x11_choice

accuracy <- mean(pred_binary == y_test)

roc_object <- pROC::roc(
  response = y_test,
  predictor = pred_probs,
  quiet = TRUE,
  direction = "<"
)

auc_value <- as.numeric(pROC::auc(roc_object))

cat("Test accuracy:", sprintf("%.4f", accuracy), "\n")
cat("Test AUC:", sprintf("%.4f", auc_value), "\n")

write_csv(
  tibble(
    best_model = best_model_name,
    test_n = length(y_test),
    accuracy = accuracy,
    AUC = auc_value
  ),
  file.path(path, "out_of_sample_performance.csv")
)


# ------------------------------------------------------------------------------
# ROC plot
# ------------------------------------------------------------------------------

roc_coordinates <- tibble(
  specificity = roc_object$specificities,
  sensitivity = roc_object$sensitivities,
  false_positive_rate = 1 - specificity
)

roc_plot <- ggplot(
  roc_coordinates,
  aes(
    x = false_positive_rate,
    y = sensitivity
  )
) +
  geom_line(linewidth = 1.4) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    linewidth = 1
  ) +
  coord_equal(
    xlim = c(0, 1),
    ylim = c(0, 1)
  ) +
  labs(
    title = "Model Performance",
    subtitle = paste0("AUC = ", sprintf("%.3f", auc_value)),
    x = "False Positive Rate",
    y = "True Positive Rate"
  )

save_plot(
  roc_plot,
  "model_roc_curve.png",
  width = 10,
  height = 10
)


# ------------------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------------------

confusion_df <- tibble(
  truth = factor(
    y_test,
    levels = c(0, 1),
    labels = c("Wait", "Forage")
  ),
  prediction = factor(
    pred_binary,
    levels = c(0, 1),
    labels = c("Wait", "Forage")
  )
) %>%
  count(truth, prediction, name = "n") %>%
  complete(
    truth,
    prediction,
    fill = list(n = 0)
  )

confusion_plot <- ggplot(
  confusion_df,
  aes(
    x = prediction,
    y = truth,
    fill = n
  )
) +
  geom_tile() +
  geom_text(
    aes(label = n),
    size = 10
  ) +
  labs(
    title = "Confusion Matrix",
    x = "Predicted label",
    y = "True label"
  ) +
  guides(fill = "none") +
  coord_equal()

save_plot(
  confusion_plot,
  "model_confusion_matrix.png",
  width = 8,
  height = 8
)


# ==============================================================================
# 7. FACTORIAL INTERACTION PLOTS
# ==============================================================================

test_data <- prepare_data("data_fmri.csv") %>%
  filter(
    x14_p_foraging_gain > 0.3,
    x14_p_foraging_gain < 0.7
  ) %>%
  droplevels()


# ------------------------------------------------------------------------------
# Subject-first aggregation
# ------------------------------------------------------------------------------

interaction_summary <- test_data %>%
  group_by(
    BNW_conditions,
    x7_weather_type,
    x14_p_foraging_gain,
    x1_id
  ) %>%
  summarise(
    subject_mean_choice = mean(x11_choice),
    .groups = "drop"
  ) %>%
  group_by(
    BNW_conditions,
    x7_weather_type,
    x14_p_foraging_gain
  ) %>%
  summarise(
    mean_choice = mean(subject_mean_choice),
    sem = ifelse(
      n() > 1,
      sd(subject_mean_choice) / sqrt(n()),
      0
    ),
    n_subjects = n(),
    .groups = "drop"
  )


factorial_design_plot <- ggplot(
  interaction_summary,
  aes(
    x = x14_p_foraging_gain,
    y = mean_choice,
    group = 1
  )
) +
  geom_errorbar(
    aes(
      ymin = mean_choice - sem,
      ymax = mean_choice + sem
    ),
    width = 0.01,
    linewidth = 0.7
  ) +
  geom_point(size = 3) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = FALSE,
    linewidth = 1
  ) +
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  facet_grid(
    rows = vars(BNW_conditions),
    cols = vars(x7_weather_type),
    labeller = label_both
  ) +
  coord_cartesian(
    xlim = c(0.35, 0.65),
    ylim = c(-0.05, 1.05)
  ) +
  labs(
    x = expression(italic(p) ~ success),
    y = "P(Forage)"
  )

save_plot(
  factorial_design_plot,
  "factorial_design_plot.png",
  width = 15,
  height = 12
)


factorial_interaction_combined <- ggplot(
  interaction_summary,
  aes(
    x = x14_p_foraging_gain,
    y = mean_choice,
    group = interaction(
      BNW_conditions,
      x7_weather_type
    ),
    linetype = x7_weather_type,
    shape = x7_weather_type
  )
) +
  geom_errorbar(
    aes(
      ymin = mean_choice - sem,
      ymax = mean_choice + sem
    ),
    width = 0.01,
    linewidth = 0.6
  ) +
  geom_point(size = 3) +
  geom_smooth(
    aes(group = interaction(BNW_conditions, x7_weather_type)),
    method = "lm",
    formula = y ~ x,
    se = FALSE,
    linewidth = 1
  ) +
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  facet_wrap(~ BNW_conditions) +
  coord_cartesian(
    ylim = c(-0.05, 1.05)
  ) +
  labs(
    title = "Interactions",
    x = expression(italic(p) ~ success),
    y = "P(Forage)",
    linetype = "Weather",
    shape = "Weather"
  ) +
  theme(
    legend.position = "right"
  )

save_plot(
  factorial_interaction_combined,
  "factorial_interaction_combined.png",
  width = 12,
  height = 10
)


# ==============================================================================
# 8. IN-BAND FACTORIAL GLMM AND PAIRWISE INTERACTION CONTRASTS
# ==============================================================================

inband_model_data <- prepare_data("data_fmri.csv") %>%
  filter(
    x14_p_foraging_gain > 0.3,
    x14_p_foraging_gain < 0.7
  ) %>%
  drop_na(
    x11_choice,
    x1_id,
    BNW_conditions,
    x7_weather_type,
    p_success_centered
  ) %>%
  droplevels()

inband_factorial_model <- glmer(
  x11_choice ~
    BNW_conditions *
    x7_weather_type *
    p_success_centered +
    (1 | x1_id),
  data = inband_model_data,
  family = binomial(link = "logit"),
  control = glmer_control,
  nAGQ = 1
)

cat("\n============================================================\n")
cat("IN-BAND FACTORIAL GLMM\n")
cat("============================================================\n")
print(summary(inband_factorial_model))


# ------------------------------------------------------------------------------
# Contrast helper
# ------------------------------------------------------------------------------
#
# Computes Wald z contrasts between any two fixed-effect coefficients.
# Logistic GLMMs use asymptotic z statistics, not t statistics.

fixed_effect_names <- names(fixef(inband_factorial_model))
fixed_effect_estimates <- fixef(inband_factorial_model)
fixed_effect_covariance <- vcov(inband_factorial_model)

find_coefficient <- function(required_patterns, excluded_patterns = NULL) {

  matches <- fixed_effect_names

  for (pattern_i in required_patterns) {
    matches <- matches[
      grepl(
        pattern_i,
        matches,
        fixed = TRUE
      )
    ]
  }

  if (!is.null(excluded_patterns)) {
    for (pattern_i in excluded_patterns) {
      matches <- matches[
        !grepl(
          pattern_i,
          matches,
          fixed = TRUE
        )
      ]
    }
  }

  if (length(matches) == 1) {
    return(matches)
  }

  if (length(matches) == 0) {
    return(NA_character_)
  }

  stop(
    paste0(
      "Multiple coefficients matched: ",
      paste(matches, collapse = ", ")
    )
  )
}


wald_difference <- function(coef_a, coef_b, label) {

  if (is.na(coef_a) || is.na(coef_b)) {
    return(
      tibble(
        contrast = label,
        coefficient_a = coef_a,
        coefficient_b = coef_b,
        estimate = NA_real_,
        SE = NA_real_,
        z = NA_real_,
        p = NA_real_
      )
    )
  }

  contrast_vector <- rep(
    0,
    length(fixed_effect_estimates)
  )

  names(contrast_vector) <- fixed_effect_names

  contrast_vector[coef_a] <- 1
  contrast_vector[coef_b] <- -1

  estimate <- sum(
    contrast_vector *
    fixed_effect_estimates
  )

  variance <- as.numeric(
    t(contrast_vector) %*%
      fixed_effect_covariance %*%
      contrast_vector
  )

  standard_error <- sqrt(variance)
  z_value <- estimate / standard_error
  p_value <- 2 * pnorm(
    abs(z_value),
    lower.tail = FALSE
  )

  tibble(
    contrast = label,
    coefficient_a = coef_a,
    coefficient_b = coef_b,
    estimate = estimate,
    SE = standard_error,
    z = z_value,
    p = p_value
  )
}


# ------------------------------------------------------------------------------
# Identify factor-specific interaction coefficients
# ------------------------------------------------------------------------------

binary_weather <- find_coefficient(
  required_patterns = c(
    "BNW_conditionsBinary Energy",
    ":x7_weather_type"
  ),
  excluded_patterns = c(
    ":p_success_centered"
  )
)

wws_weather <- find_coefficient(
  required_patterns = c(
    "BNW_conditionsWait When Safe",
    ":x7_weather_type"
  ),
  excluded_patterns = c(
    ":p_success_centered"
  )
)

binary_three_way <- find_coefficient(
  required_patterns = c(
    "BNW_conditionsBinary Energy",
    ":x7_weather_type",
    ":p_success_centered"
  )
)

wws_three_way <- find_coefficient(
  required_patterns = c(
    "BNW_conditionsWait When Safe",
    ":x7_weather_type",
    ":p_success_centered"
  )
)


# Trade-off is the factor reference level. Therefore:
# - Binary vs Trade-off equals the Binary interaction coefficient itself.
# - WWS vs Trade-off equals the WWS interaction coefficient itself.
# - WWS vs Binary requires a coefficient difference.

single_coefficient_test <- function(coef_name, label) {

  if (is.na(coef_name)) {
    return(
      tibble(
        contrast = label,
        coefficient = NA_character_,
        estimate = NA_real_,
        SE = NA_real_,
        z = NA_real_,
        p = NA_real_
      )
    )
  }

  estimate <- fixed_effect_estimates[coef_name]
  standard_error <- sqrt(
    fixed_effect_covariance[
      coef_name,
      coef_name
    ]
  )

  z_value <- estimate / standard_error
  p_value <- 2 * pnorm(
    abs(z_value),
    lower.tail = FALSE
  )

  tibble(
    contrast = label,
    coefficient = coef_name,
    estimate = unname(estimate),
    SE = unname(standard_error),
    z = unname(z_value),
    p = unname(p_value)
  )
}


reference_contrasts <- bind_rows(

  single_coefficient_test(
    binary_weather,
    "Binary Energy vs Trade-off (Weather interaction)"
  ),

  single_coefficient_test(
    binary_three_way,
    "Binary Energy vs Trade-off (Three-way interaction)"
  ),

  single_coefficient_test(
    wws_weather,
    "Wait When Safe vs Trade-off (Weather interaction)"
  ),

  single_coefficient_test(
    wws_three_way,
    "Wait When Safe vs Trade-off (Three-way interaction)"
  )
)

difference_contrasts <- bind_rows(

  wald_difference(
    wws_weather,
    binary_weather,
    "Wait When Safe vs Binary Energy (Weather interaction)"
  ),

  wald_difference(
    wws_three_way,
    binary_three_way,
    "Wait When Safe vs Binary Energy (Three-way interaction)"
  )
)

cat("\nReference-level contrasts:\n")
print(reference_contrasts, n = Inf, width = Inf)

cat("\nDirect coefficient-difference contrasts:\n")
print(difference_contrasts, n = Inf, width = Inf)

write_csv(
  reference_contrasts,
  file.path(path, "factorial_reference_contrasts.csv")
)

write_csv(
  difference_contrasts,
  file.path(path, "factorial_difference_contrasts.csv")
)


# ==============================================================================
# 9. WEATHER × P(SUCCESS) PLOT
# ==============================================================================

weather_summary <- test_data %>%
  group_by(
    x7_weather_type,
    x14_p_foraging_gain,
    x1_id
  ) %>%
  summarise(
    subject_mean_choice = mean(x11_choice),
    .groups = "drop"
  ) %>%
  group_by(
    x7_weather_type,
    x14_p_foraging_gain
  ) %>%
  summarise(
    mean_choice = mean(subject_mean_choice),
    sem = ifelse(
      n() > 1,
      sd(subject_mean_choice) / sqrt(n()),
      0
    ),
    .groups = "drop"
  )

weather_plot <- ggplot(
  weather_summary,
  aes(
    x = x14_p_foraging_gain,
    y = mean_choice,
    group = x7_weather_type,
    linetype = x7_weather_type,
    shape = x7_weather_type
  )
) +
  geom_errorbar(
    aes(
      ymin = mean_choice - sem,
      ymax = mean_choice + sem
    ),
    width = 0.01,
    linewidth = 0.7
  ) +
  geom_point(size = 3) +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = FALSE,
    linewidth = 1.2
  ) +
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  coord_cartesian(
    xlim = c(0.35, 0.65),
    ylim = c(-0.05, 1.05)
  ) +
  labs(
    title = expression(   "Interaction Weather " %*% italic(p)(success) ),
    x = "P(Success)",
    y = "P(Forage)",
    linetype = "Weather",
    shape = "Weather"
  )

save_plot(
  weather_plot,
  "weather_comparison.png",
  width = 12,
  height = 10
)


# ==============================================================================
# 10. OPTIONAL IN-BAND MODEL-COMPARISON SENSITIVITY ANALYSIS
# ==============================================================================

run_inband_model_comparison <- TRUE

if (run_inband_model_comparison) {

  inband_analysis_data <- analysis_data %>%
    filter(
      x14_p_foraging_gain > 0.3,
      x14_p_foraging_gain < 0.7
    ) %>%
    droplevels()

  cat("\n============================================================\n")
  cat("IN-BAND MODEL-COMPARISON SENSITIVITY ANALYSIS\n")
  cat("============================================================\n")
  cat("Rows:", nrow(inband_analysis_data), "\n")
  cat("Participants:", nlevels(inband_analysis_data$x1_id), "\n")

  inband_model_fits <- imap(
    model_formulas,
    ~ fit_glmer_safe(
      formula_i = .x,
      data_i = inband_analysis_data,
      model_name = paste0(.y, "_inband")
    )
  )

  inband_comparison_df <- imap_dfr(
    inband_model_fits,
    function(fit_object, model_name) {

      description_i <- model_descriptions[[model_name]]

      extract_model_metrics(
        fit_object = fit_object,
        model_name = model_name,
        description = description_i
      )
    }
  ) %>%
    mutate(
      delta_BIC = BIC - min(BIC, na.rm = TRUE),
      delta_AIC = AIC - min(AIC, na.rm = TRUE),
      BIC_weight_raw = exp(-0.5 * delta_BIC),
      AIC_weight_raw = exp(-0.5 * delta_AIC),
      BIC_weight = BIC_weight_raw /
        sum(BIC_weight_raw, na.rm = TRUE),
      AIC_weight = AIC_weight_raw /
        sum(AIC_weight_raw, na.rm = TRUE)
    ) %>%
    arrange(delta_BIC) %>%
    select(
      model_name,
      description,
      n,
      logLik,
      parameters,
      BIC,
      delta_BIC,
      BIC_weight,
      AIC,
      delta_AIC,
      AIC_weight,
      singular,
      convergence,
      warnings
    )

  print(inband_comparison_df, n = Inf, width = Inf)

  write_csv(
    inband_comparison_df,
    file.path(
      path,
      "hierarchical_model_comparison_inband.csv"
    )
  )
}


# ==============================================================================
# 11. FINAL MODEL DIAGNOSTICS
# ==============================================================================

diagnostic_table <- imap_dfr(
  model_fits,
  function(fit_object, model_name) {

    model <- fit_object$model

    if (is.null(model)) {
      return(
        tibble(
          model_name = model_name,
          participant_intercept_variance = NA_real_,
          participant_intercept_sd = NA_real_,
          singular = NA,
          convergence = "MODEL FAILED"
        )
      )
    }

    variance_component <- as.data.frame(
      VarCorr(model)
    ) %>%
      filter(
        grp == "x1_id",
        var1 == "(Intercept)"
      )

    tibble(
      model_name = model_name,
      participant_intercept_variance = variance_component$vcov[1],
      participant_intercept_sd = variance_component$sdcor[1],
      singular = isSingular(model, tol = 1e-4),
      convergence = ifelse(
        is.null(model@optinfo$conv$lme4$messages),
        "OK",
        paste(
          model@optinfo$conv$lme4$messages,
          collapse = " | "
        )
      )
    )
  }
)

cat("\n============================================================\n")
cat("RANDOM-INTERCEPT AND CONVERGENCE DIAGNOSTICS\n")
cat("============================================================\n")
print(diagnostic_table, n = Inf, width = Inf)

write_csv(
  diagnostic_table,
  file.path(path, "hierarchical_model_diagnostics.csv")
)

cat("\nAnalysis completed successfully.\n")

# ==============================================================================
# PUBLICATION-READY PLOTTING SECTION
# ==============================================================================
#
# This section creates:
#
#   1. Delta-AIC panel
#   2. Delta-BIC panel
#   3. Model-weights panel
#   4. Combined model-comparison row
#   5. ROC panel
#   6. Confusion-matrix panel
#   7. Combined performance row
#   8. Six-panel factorial-design plot
#   9. Combined factorial-interaction plot
#  10. Weather × p(success) plot
#
# Vector output:
#   PDF via cairo_pdf for importing into Inkscape.
#
# Raster output:
#   PNG at 600 dpi.
#
# ==============================================================================


# ==============================================================================
# 0. PACKAGES
# ==============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(grid)
library(patchwork)
library(pROC)


# ==============================================================================
# 1. CORRECT AND REFRESH MODEL LABELS
# ==============================================================================

# All labels are stored as valid plotmath strings.
# They are parsed only when drawn on the axis.

model_descriptions <- c(
  
  p_success =
    "italic(p)(success)",
  
  gain_magnitude =
    "gain~magnitude",
  
  remaining_time =
    "remaining~days",
  
  continuous_energy =
    "continuous~energy~state",
  
  weather_type =
    "weather~type",
  
  expected_energy_change =
    "expected~energy~change",
  
  ternary_state =
    "ternary~state",
  
  ternary_weather =
    "ternary~state~'\u00D7'~weather~type",
  
  factorial_policy =
    "'multi-feature model'",
  
  optimal_policy =
    "Delta*italic(Q)~values"
)


# Refresh descriptions in case comparison_df was created before labels changed.

comparison_df <- comparison_df %>%
  mutate(
    description = unname(
      model_descriptions[model_name]
    )
  )


# ==============================================================================
# 2. EXPORT SETTINGS
# ==============================================================================

export_dpi <- 600

comparison_panel_width  <- 7.00
weights_panel_width     <- 5.20
comparison_panel_height <- 4.00

performance_panel_width  <- 4.80
performance_panel_height <- 4.80

factorial_panel_width  <- 7.40
factorial_panel_height <- 7.40

interaction_panel_width  <- 6.30
interaction_panel_height <- 4.80

weather_panel_width  <- 5.60
weather_panel_height <- 4.80


# ==============================================================================
# 3. TYPOGRAPHY AND GEOMETRY SETTINGS
# ==============================================================================

publication_font_family <- "sans"

# Model-comparison plots
comparison_base_size   <- 17
comparison_axis_size   <- 20
comparison_tick_size   <- 17
comparison_model_size  <- 17
comparison_value_size  <- 5.0
comparison_legend_size <- 16

# Performance plots: deliberately larger because these panels are later
# reduced substantially in the final Inkscape composite.
performance_base_size   <- 22
performance_axis_size   <- 30
performance_tick_size   <- 24
performance_value_size  <- 7.5
performance_auc_size    <- 7.0

# Interaction plots
interaction_base_size   <- 18
interaction_axis_size   <- 24
interaction_tick_size   <- 20
interaction_strip_size  <- 20
interaction_legend_size <- 18

axis_line_width <- 0.75
tick_line_width <- 0.75


# ==============================================================================
# 4. EXPORT HELPER
# ==============================================================================

save_publication_plot <- function(
    plot_object,
    filename_stub,
    width,
    height
) {
  
  # Editable vector file for Inkscape
  ggsave(
    filename = file.path(
      path,
      paste0(filename_stub, ".pdf")
    ),
    plot = plot_object,
    width = width,
    height = height,
    units = "in",
    device = cairo_pdf,
    bg = "white"
  )
  
  # High-resolution raster version
  ggsave(
    filename = file.path(
      path,
      paste0(filename_stub, ".png")
    ),
    plot = plot_object,
    width = width,
    height = height,
    units = "in",
    dpi = export_dpi,
    bg = "white"
  )
}


# ==============================================================================
# 5. SHARED THEMES
# ==============================================================================

# ------------------------------------------------------------------------------
# Model-comparison theme
# ------------------------------------------------------------------------------

theme_model_comparison <- theme_classic(
  base_size = comparison_base_size,
  base_family = publication_font_family
) +
  theme(
    
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    
    axis.title.x = element_text(
      size = comparison_axis_size,
      face = "plain",
      color = "black",
      margin = margin(t = 8)
    ),
    
    axis.title.y = element_blank(),
    
    axis.text.x = element_text(
      size = comparison_tick_size,
      color = "black",
      margin = margin(t = 3)
    ),
    
    axis.text.y = element_text(
      size = comparison_model_size,
      color = "black",
      margin = margin(r = 5)
    ),
    
    axis.line = element_line(
      linewidth = axis_line_width,
      color = "black"
    ),
    
    axis.ticks = element_line(
      linewidth = tick_line_width,
      color = "black"
    ),
    
    axis.ticks.length = unit(
      0.11,
      "cm"
    ),
    
    panel.grid = element_blank(),
    
    legend.title = element_blank(),
    
    legend.text = element_text(
      size = comparison_legend_size,
      color = "black"
    ),
    
    legend.key.height = unit(
      0.48,
      "cm"
    ),
    
    legend.key.width = unit(
      0.75,
      "cm"
    ),
    
    plot.margin = margin(
      t = 5,
      r = 18,
      b = 5,
      l = 5
    )
  )


# ------------------------------------------------------------------------------
# Performance-panel theme
# ------------------------------------------------------------------------------

theme_performance <- theme_classic(
  base_size = performance_base_size,
  base_family = publication_font_family
) +
  theme(
    
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    
    axis.title.x = element_text(
      size = performance_axis_size,
      face = "plain",
      color = "black",
      margin = margin(t = 10)
    ),
    
    axis.title.y = element_text(
      size = performance_axis_size,
      face = "plain",
      color = "black",
      margin = margin(r = 10)
    ),
    
    axis.text.x = element_text(
      size = performance_tick_size,
      color = "black"
    ),
    
    axis.text.y = element_text(
      size = performance_tick_size,
      color = "black"
    ),
    
    axis.line = element_line(
      linewidth = 0.90,
      color = "black"
    ),
    
    axis.ticks = element_line(
      linewidth = 0.85,
      color = "black"
    ),
    
    axis.ticks.length = unit(
      0.14,
      "cm"
    ),
    
    panel.grid = element_blank(),
    
    plot.margin = margin(
      t = 10,
      r = 12,
      b = 10,
      l = 12
    )
  )


# ------------------------------------------------------------------------------
# Interaction theme
# ------------------------------------------------------------------------------

theme_interaction <- theme_classic(
  base_size = interaction_base_size,
  base_family = publication_font_family
) +
  theme(
    
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    
    axis.title.x = element_text(
      size = interaction_axis_size,
      color = "black",
      margin = margin(t = 8)
    ),
    
    axis.title.y = element_text(
      size = interaction_axis_size,
      color = "black",
      margin = margin(r = 8)
    ),
    
    axis.text.x = element_text(
      size = interaction_tick_size,
      color = "black"
    ),
    
    axis.text.y = element_text(
      size = interaction_tick_size,
      color = "black"
    ),
    
    axis.line = element_line(
      linewidth = axis_line_width,
      color = "black"
    ),
    
    axis.ticks = element_line(
      linewidth = tick_line_width,
      color = "black"
    ),
    
    axis.ticks.length = unit(
      0.11,
      "cm"
    ),
    
    legend.title = element_blank(),
    
    legend.text = element_text(
      size = interaction_legend_size,
      color = "black"
    ),
    
    panel.grid = element_blank(),
    
    plot.margin = margin(
      t = 8,
      r = 10,
      b = 8,
      l = 10
    )
  )


# ==============================================================================
# 6. MODEL-COMPARISON DATA
# ==============================================================================

# Use the same BIC-based model order in AIC, BIC, and model-weight panels.

model_order <- comparison_df %>%
  filter(
    !is.na(delta_BIC),
    !is.na(delta_AIC)
  ) %>%
  arrange(delta_BIC) %>%
  pull(description)

model_levels <- rev(model_order)


aic_plot_data <- comparison_df %>%
  filter(
    !is.na(delta_AIC),
    description %in% model_order
  ) %>%
  mutate(
    description = factor(
      description,
      levels = model_levels
    )
  )


bic_plot_data <- comparison_df %>%
  filter(
    !is.na(delta_BIC),
    description %in% model_order
  ) %>%
  mutate(
    description = factor(
      description,
      levels = model_levels
    )
  )


weights_plot_data <- comparison_df %>%
  filter(
    description %in% model_order
  ) %>%
  select(
    description,
    AIC_weight,
    BIC_weight
  ) %>%
  pivot_longer(
    cols = c(
      AIC_weight,
      BIC_weight
    ),
    names_to = "criterion",
    values_to = "weight"
  ) %>%
  mutate(
    
    criterion = factor(
      criterion,
      levels = c(
        "AIC_weight",
        "BIC_weight"
      ),
      labels = c(
        "AIC Weights",
        "BIC Weights"
      )
    ),
    
    description = factor(
      description,
      levels = model_levels
    )
  )


# ------------------------------------------------------------------------------
# Axis limits
# ------------------------------------------------------------------------------

aic_xmax <- max(
  aic_plot_data$delta_AIC,
  na.rm = TRUE
) * 1.14

bic_xmax <- max(
  bic_plot_data$delta_BIC,
  na.rm = TRUE
) * 1.14

if (!is.finite(aic_xmax) || aic_xmax <= 0) {
  aic_xmax <- 1
}

if (!is.finite(bic_xmax) || bic_xmax <= 0) {
  bic_xmax <- 1
}


# ==============================================================================
# 7. DELTA-AIC PANEL
# ==============================================================================

aic_plot <- ggplot(
  aic_plot_data,
  aes(
    x = delta_AIC,
    y = description
  )
) +
  
  geom_col(
    width = 0.72,
    fill = "skyblue"
  ) +
  
  geom_text(
    aes(
      label = ifelse(
        delta_AIC > 0,
        sprintf("%.1f", delta_AIC),
        ""
      )
    ),
    hjust = -0.08,
    size = comparison_value_size,
    family = publication_font_family
  ) +
  
  scale_x_continuous(
    limits = c(
      0,
      aic_xmax
    ),
    breaks = pretty_breaks(
      n = 6
    ),
    expand = expansion(
      mult = c(0, 0)
    )
  ) +
  
  scale_y_discrete(
    drop = FALSE,
    labels = function(x) {
      parse(text = x)
    }
  ) +
  
  coord_cartesian(
    clip = "off"
  ) +
  
  labs(
    x = expression(
      Delta * "AIC (0 = best model)"
    ),
    y = NULL
  ) +
  
  theme_model_comparison


save_publication_plot(
  plot_object = aic_plot,
  filename_stub = "final_AIC_panel",
  width = comparison_panel_width,
  height = comparison_panel_height
)


# ==============================================================================
# 8. DELTA-BIC PANEL
# ==============================================================================

bic_plot <- ggplot(
  bic_plot_data,
  aes(
    x = delta_BIC,
    y = description
  )
) +
  
  geom_col(
    width = 0.72,
    fill = "lightgreen"
  ) +
  
  geom_text(
    aes(
      label = ifelse(
        delta_BIC > 0,
        sprintf("%.1f", delta_BIC),
        ""
      )
    ),
    hjust = -0.08,
    size = comparison_value_size,
    family = publication_font_family
  ) +
  
  scale_x_continuous(
    limits = c(
      0,
      bic_xmax
    ),
    breaks = pretty_breaks(
      n = 6
    ),
    expand = expansion(
      mult = c(0, 0)
    )
  ) +
  
  scale_y_discrete(
    drop = FALSE,
    labels = function(x) {
      parse(text = x)
    }
  ) +
  
  coord_cartesian(
    clip = "off"
  ) +
  
  labs(
    x = expression(
      Delta * "BIC (0 = best model)"
    ),
    y = NULL
  ) +
  
  theme_model_comparison +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    plot.margin = margin(
      t = 5,
      r = 18,
      b = 5,
      l = 8
    )
  )


save_publication_plot(
  plot_object = bic_plot,
  filename_stub = "final_BIC_panel",
  width = comparison_panel_width,
  height = comparison_panel_height
)


# ==============================================================================
# 9. MODEL-WEIGHTS PANEL
# ==============================================================================

weights_plot <- ggplot(
  weights_plot_data,
  aes(
    x = weight,
    y = description,
    fill = criterion
  )
) +
  
  geom_col(
    position = position_dodge(
      width = 0.72,
      preserve = "single"
    ),
    width = 0.62
  ) +
  
  scale_fill_manual(
    values = c(
      "AIC Weights" = "skyblue",
      "BIC Weights" = "lightgreen"
    )
  ) +
  
  scale_x_continuous(
    limits = c(
      0,
      1
    ),
    breaks = seq(
      0,
      1,
      by = 0.2
    ),
    labels = label_number(
      accuracy = 0.1
    ),
    expand = expansion(
      mult = c(0, 0)
    )
  ) +
  
  scale_y_discrete(
    drop = FALSE
  ) +
  
  labs(
    x = "Model Weight",
    y = NULL,
    fill = NULL
  ) +
  
  theme_model_comparison +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    
    legend.position = c(
      0.72,
      0.87
    ),
    
    legend.direction = "vertical",
    
    legend.background = element_rect(
      fill = "white",
      color = "grey80",
      linewidth = 0.45
    ),
    
    legend.margin = margin(
      t = 4,
      r = 6,
      b = 4,
      l = 6
    ),
    
    plot.margin = margin(
      t = 5,
      r = 8,
      b = 5,
      l = 8
    )
  )


save_publication_plot(
  plot_object = weights_plot,
  filename_stub = "final_model_weights_panel",
  width = weights_panel_width,
  height = comparison_panel_height
)


# ==============================================================================
# PUBLICATION-READY MODEL-COMPARISON PANELS
# Subtle frames and non-overlapping criterion legend
# ==============================================================================


# ==============================================================================
# 1. MODEL LABELS
# ==============================================================================

model_descriptions <- c(
  
  p_success =
    "italic(p)(success)",
  
  gain_magnitude =
    "gain~magnitude",
  
  remaining_time =
    "remaining~days",
  
  continuous_energy =
    "continuous~energy~state",
  
  weather_type =
    "weather~type",
  
  expected_energy_change =
    "expected~energy~change",
  
  ternary_state =
    "ternary~state",
  
  ternary_weather =
    "ternary~state~'\u00D7'~weather~type",
  
  factorial_policy =
    "'multi-feature model'",
  
  optimal_policy =
    "Delta*italic(Q)~values"
)


comparison_df <- comparison_df %>%
  mutate(
    description = unname(
      model_descriptions[model_name]
    )
  )


# ==============================================================================
# 2. EXPORT DIMENSIONS
# ==============================================================================

comparison_panel_width  <- 5.80
comparison_panel_height <- 3.80

weights_panel_width  <- 4.50
weights_panel_height <- 3.80

export_dpi <- 600


# ==============================================================================
# 3. TYPOGRAPHY AND GEOMETRY
# ==============================================================================

publication_font_family <- "sans"

comparison_axis_title_size <- 24
comparison_tick_size       <- 20
comparison_model_size      <- 19
comparison_value_size      <- 5.1
comparison_legend_size     <- 16.5

comparison_tick_width  <- 0.70
comparison_frame_width <- 0.45
comparison_bar_width   <- 0.70


# ==============================================================================
# 4. SHARED THEME
# ==============================================================================

theme_model_comparison_framed <- theme_classic(
  base_size = comparison_tick_size,
  base_family = publication_font_family
) +
  theme(
    
    plot.title = element_blank(),
    plot.subtitle = element_blank(),
    
    axis.title.x = element_text(
      size = comparison_axis_title_size,
      face = "plain",
      color = "black",
      margin = margin(t = 8)
    ),
    
    axis.title.y = element_blank(),
    
    axis.text.x = element_text(
      size = comparison_tick_size,
      color = "black",
      margin = margin(t = 3)
    ),
    
    axis.text.y = element_text(
      size = comparison_model_size,
      color = "black",
      margin = margin(r = 6)
    ),
    
    # Hide the separate classic axes because the panel border
    # provides the enclosing frame.
    axis.line = element_blank(),
    
    axis.ticks = element_line(
      linewidth = comparison_tick_width,
      color = "black"
    ),
    
    axis.ticks.length = unit(
      0.11,
      "cm"
    ),
    
    # Subtle frame rather than a dominant black box.
    panel.border = element_rect(
      fill = NA,
      color = "grey65",
      linewidth = comparison_frame_width
    ),
    
    panel.background = element_rect(
      fill = "white",
      color = NA
    ),
    
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    legend.title = element_blank(),
    
    legend.text = element_text(
      size = comparison_legend_size,
      color = "black"
    ),
    
    legend.key.height = unit(
      0.42,
      "cm"
    ),
    
    legend.key.width = unit(
      0.68,
      "cm"
    ),
    
    plot.margin = margin(
      t = 6,
      r = 12,
      b = 6,
      l = 6
    )
  )


# ==============================================================================
# 5. CONSISTENT MODEL ORDER
# ==============================================================================

model_order <- comparison_df %>%
  filter(
    !is.na(delta_BIC),
    !is.na(delta_AIC)
  ) %>%
  arrange(delta_BIC) %>%
  pull(description)

model_levels <- rev(model_order)


aic_plot_data <- comparison_df %>%
  filter(
    !is.na(delta_AIC),
    description %in% model_order
  ) %>%
  mutate(
    description = factor(
      description,
      levels = model_levels
    )
  )


bic_plot_data <- comparison_df %>%
  filter(
    !is.na(delta_BIC),
    description %in% model_order
  ) %>%
  mutate(
    description = factor(
      description,
      levels = model_levels
    )
  )


weights_plot_data <- comparison_df %>%
  filter(
    description %in% model_order
  ) %>%
  select(
    description,
    AIC_weight,
    BIC_weight
  ) %>%
  pivot_longer(
    cols = c(
      AIC_weight,
      BIC_weight
    ),
    names_to = "criterion",
    values_to = "weight"
  ) %>%
  mutate(
    
    criterion = factor(
      criterion,
      levels = c(
        "AIC_weight",
        "BIC_weight"
      ),
      labels = c(
        "AIC weights",
        "BIC weights"
      )
    ),
    
    description = factor(
      description,
      levels = model_levels
    )
  )


# ==============================================================================
# 6. AXIS LIMITS
# ==============================================================================

aic_xmax <- max(
  aic_plot_data$delta_AIC,
  na.rm = TRUE
) * 1.15

bic_xmax <- max(
  bic_plot_data$delta_BIC,
  na.rm = TRUE
) * 1.15

if (!is.finite(aic_xmax) || aic_xmax <= 0) {
  aic_xmax <- 1
}

if (!is.finite(bic_xmax) || bic_xmax <= 0) {
  bic_xmax <- 1
}


# ==============================================================================
# 7. AIC PANEL
# ==============================================================================

aic_plot <- ggplot(
  aic_plot_data,
  aes(
    x = delta_AIC,
    y = description
  )
) +
  
  geom_col(
    width = comparison_bar_width,
    fill = "#56B4E9"
  ) +
  
  geom_text(
    aes(
      label = ifelse(
        delta_AIC > 0,
        sprintf("%.1f", delta_AIC),
        ""
      )
    ),
    hjust = -0.08,
    size = comparison_value_size,
    family = publication_font_family,
    color = "black"
  ) +
  
  scale_x_continuous(
    limits = c(
      0,
      aic_xmax
    ),
    breaks = scales::pretty_breaks(
      n = 5
    ),
    expand = expansion(
      mult = c(0, 0)
    )
  ) +
  
  scale_y_discrete(
    drop = FALSE,
    labels = function(x) {
      parse(text = x)
    }
  ) +
  
  coord_cartesian(
    clip = "off"
  ) +
  
  labs(
    x = expression(
      Delta * "AIC (0 = best model)"
    ),
    y = NULL
  ) +
  
  theme_model_comparison_framed


# ==============================================================================
# 8. BIC PANEL
# ==============================================================================

bic_plot <- ggplot(
  bic_plot_data,
  aes(
    x = delta_BIC,
    y = description
  )
) +
  
  geom_col(
    width = comparison_bar_width,
    fill = "#009E73"
  ) +
  
  geom_text(
    aes(
      label = ifelse(
        delta_BIC > 0,
        sprintf("%.1f", delta_BIC),
        ""
      )
    ),
    hjust = -0.08,
    size = comparison_value_size,
    family = publication_font_family,
    color = "black"
  ) +
  
  scale_x_continuous(
    limits = c(
      0,
      bic_xmax
    ),
    breaks = scales::pretty_breaks(
      n = 5
    ),
    expand = expansion(
      mult = c(0, 0)
    )
  ) +
  
  scale_y_discrete(
    drop = FALSE
  ) +
  
  coord_cartesian(
    clip = "off"
  ) +
  
  labs(
    x = expression(
      Delta * "BIC (0 = best model)"
    ),
    y = NULL
  ) +
  
  theme_model_comparison_framed +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    
    plot.margin = margin(
      t = 6,
      r = 12,
      b = 6,
      l = 7
    )
  )


# ==============================================================================
# 9. MODEL-WEIGHT PANEL
# ==============================================================================

weights_plot <- ggplot(
  weights_plot_data,
  aes(
    x = weight,
    y = description,
    fill = criterion
  )
) +
  
  geom_col(
    position = position_dodge(
      width = 0.72,
      preserve = "single"
    ),
    width = 0.62
  ) +
  
  scale_fill_manual(
    values = c(
      "AIC weights" = "#56B4E9",
      "BIC weights" = "#009E73"
    )
  ) +
  
  scale_x_continuous(
    limits = c(
      0,
      1
    ),
    breaks = seq(
      0,
      1,
      by = 0.2
    ),
    labels = scales::label_number(
      accuracy = 0.1
    ),
    expand = expansion(
      mult = c(0, 0)
    )
  ) +
  
  scale_y_discrete(
    drop = FALSE
  ) +
  
  labs(
    x = "Model Weight",
    y = NULL,
    fill = NULL
  ) +
  
  theme_model_comparison_framed +
  
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    
    # Legend below the plotting region, so it never covers bars.
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.justification = "center",
    
    legend.background = element_blank(),
    
    legend.margin = margin(
      t = 4,
      r = 0,
      b = 0,
      l = 0
    ),
    
    legend.spacing.x = unit(
      0.20,
      "cm"
    ),
    
    plot.margin = margin(
      t = 6,
      r = 7,
      b = 3,
      l = 7
    )
  ) +
  
  guides(
    fill = guide_legend(
      nrow = 1,
      byrow = TRUE
    )
  )


# ==============================================================================
# 10. EXPORT INDIVIDUAL PANELS
# ==============================================================================

save_publication_plot(
  plot_object = aic_plot,
  filename_stub = "final_AIC_panel_subtle_frame",
  width = comparison_panel_width,
  height = comparison_panel_height
)

save_publication_plot(
  plot_object = bic_plot,
  filename_stub = "final_BIC_panel_subtle_frame",
  width = comparison_panel_width,
  height = comparison_panel_height
)

save_publication_plot(
  plot_object = weights_plot,
  filename_stub = "final_model_weights_panel_subtle_frame",
  width = weights_panel_width,
  height = weights_panel_height
)


# ==============================================================================
# 11. COMBINED MODEL-COMPARISON ROW
# ==============================================================================

comparison_row <- (
  aic_plot +
    bic_plot +
    weights_plot
) +
  patchwork::plot_layout(
    widths = c(
      comparison_panel_width,
      comparison_panel_width,
      weights_panel_width
    )
  )


combined_comparison_width <- (
  comparison_panel_width * 2 +
    weights_panel_width
)


save_publication_plot(
  plot_object = comparison_row,
  filename_stub = "final_model_comparison_row_subtle_frame",
  width = combined_comparison_width,
  height = comparison_panel_height
)


# ==============================================================================
# 12. DISPLAY
# ==============================================================================

print(aic_plot)
print(bic_plot)
print(weights_plot)
print(comparison_row)

# ==============================================================================
# 11. ROC CURVE
# ==============================================================================

roc_object <- pROC::roc(
  response = y_test,
  predictor = pred_probs,
  quiet = TRUE,
  direction = "<"
)

auc_value <- as.numeric(
  pROC::auc(
    roc_object
  )
)


roc_plot_data <- tibble(
  false_positive_rate =
    1 - roc_object$specificities,
  
  true_positive_rate =
    roc_object$sensitivities
) %>%
  arrange(
    false_positive_rate,
    true_positive_rate
  )


roc_plot <- ggplot(
  roc_plot_data,
  aes(
    x = false_positive_rate,
    y = true_positive_rate
  )
) +
  
  geom_line(
    linewidth = 1.35,
    color = "black"
  ) +
  
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    linewidth = 0.90,
    color = "grey50"
  ) +
  
  annotate(
    geom = "text",
    x = 0.97,
    y = 0.055,
    label = sprintf(
      "AUC = %.2f",
      auc_value
    ),
    hjust = 1,
    vjust = 0,
    size = performance_auc_size,
    family = publication_font_family,
    color = "black"
  ) +
  
  scale_x_continuous(
    limits = c(
      0,
      1
    ),
    breaks = seq(
      0,
      1,
      by = 0.2
    ),
    labels = label_number(
      accuracy = 0.1
    ),
    expand = expansion(
      mult = c(0, 0)
    )
  ) +
  
  scale_y_continuous(
    limits = c(
      0,
      1
    ),
    breaks = seq(
      0,
      1,
      by = 0.2
    ),
    labels = label_number(
      accuracy = 0.1
    ),
    expand = expansion(
      mult = c(0, 0)
    )
  ) +
  
  coord_equal(
    ratio = 1,
    clip = "off"
  ) +
  
  labs(
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  
  theme_performance


save_publication_plot(
  plot_object = roc_plot,
  filename_stub = "final_ROC_panel",
  width = performance_panel_width,
  height = performance_panel_height
)


# ==============================================================================
# 12. CONFUSION MATRIX
# ==============================================================================

confusion_plot_data <- tibble(
  
  observed_choice = factor(
    y_test,
    levels = c(
      0,
      1
    ),
    labels = c(
      "Wait",
      "Forage"
    )
  ),
  
  predicted_choice = factor(
    pred_binary,
    levels = c(
      0,
      1
    ),
    labels = c(
      "Wait",
      "Forage"
    )
  )
) %>%
  
  count(
    observed_choice,
    predicted_choice,
    name = "count"
  ) %>%
  
  complete(
    observed_choice,
    predicted_choice,
    fill = list(
      count = 0
    )
  ) %>%
  
  group_by(
    observed_choice
  ) %>%
  
  mutate(
    row_proportion =
      count / sum(count)
  ) %>%
  
  ungroup()


confusion_plot <- ggplot(
  confusion_plot_data,
  aes(
    x = predicted_choice,
    y = observed_choice,
    fill = row_proportion
  )
) +
  
  geom_tile(
    linewidth = 1.0,
    color = "white"
  ) +
  
  geom_text(
    aes(
      label = count
    ),
    size = performance_value_size,
    family = publication_font_family,
    color = "black"
  ) +
  
  scale_fill_gradient(
    low = "white",
    high = "steelblue",
    limits = c(
      0,
      1
    )
  ) +
  
  scale_y_discrete(
    limits = rev
  ) +
  
  coord_equal() +
  
  labs(
    x = "Predicted Choice",
    y = "Observed Choice"
  ) +
  
  theme_performance +
  
  theme(
    legend.position = "none",
    
    axis.ticks = element_blank(),
    axis.line = element_blank(),
    
    panel.border = element_rect(
      fill = NA,
      color = "black",
      linewidth = 0.90
    )
  )


save_publication_plot(
  plot_object = confusion_plot,
  filename_stub = "final_confusion_matrix_panel",
  width = performance_panel_width,
  height = performance_panel_height
)


# ==============================================================================
# 13. COMBINED PERFORMANCE ROW
# ==============================================================================

performance_row <- (
  roc_plot |
    confusion_plot
) +
  plot_layout(
    widths = c(
      1,
      1
    )
  )


save_publication_plot(
  plot_object = performance_row,
  filename_stub = "final_performance_row",
  width = performance_panel_width * 2,
  height = performance_panel_height
)


# ==============================================================================
# 14. PREPARE INTERACTION DATA
# ==============================================================================

interaction_data <- test_data %>%
  
  filter(
    x14_p_foraging_gain > 0.3,
    x14_p_foraging_gain < 0.7
  ) %>%
  
  mutate(
    
    BNW_conditions = factor(
      BNW_conditions,
      levels = c(
        "Binary Energy",
        "Trade-off",
        "Wait When Safe"
      )
    ),
    
    weather_label = case_when(
      
      as.character(x7_weather_type) == "1" ~
        "Bad Weather",
      
      as.character(x7_weather_type) == "2" ~
        "Good Weather",
      
      TRUE ~
        as.character(x7_weather_type)
    ),
    
    weather_label = factor(
      weather_label,
      levels = c(
        "Bad Weather",
        "Good Weather"
      )
    )
  )


interaction_summary <- interaction_data %>%
  
  group_by(
    BNW_conditions,
    weather_label,
    x14_p_foraging_gain,
    x1_id
  ) %>%
  
  summarise(
    subject_mean = mean(
      x11_choice,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  
  group_by(
    BNW_conditions,
    weather_label,
    x14_p_foraging_gain
  ) %>%
  
  summarise(
    
    mean_choice = mean(
      subject_mean,
      na.rm = TRUE
    ),
    
    sem = ifelse(
      n() > 1,
      sd(
        subject_mean,
        na.rm = TRUE
      ) / sqrt(n()),
      0
    ),
    
    n_subjects = n(),
    
    .groups = "drop"
  )


# ==============================================================================
# 15. SIX-PANEL FACTORIAL-DESIGN PLOT
# ==============================================================================

factorial_panel_plot <- ggplot(
  interaction_summary,
  aes(
    x = x14_p_foraging_gain,
    y = mean_choice
  )
) +
  
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed",
    linewidth = 0.70,
    color = "grey50"
  ) +
  
  geom_errorbar(
    aes(
      ymin = pmax(
        mean_choice - sem,
        0
      ),
      ymax = pmin(
        mean_choice + sem,
        1
      )
    ),
    width = 0.008,
    linewidth = 0.70,
    color = "black"
  ) +
  
  geom_point(
    size = 3.1,
    shape = 21,
    fill = "white",
    color = "black",
    stroke = 0.85
  ) +
  
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = FALSE,
    linewidth = 1.05,
    color = "black"
  ) +
  
  facet_grid(
    rows = vars(
      BNW_conditions
    ),
    cols = vars(
      weather_label
    )
  ) +
  
  scale_x_continuous(
    limits = c(
      0.35,
      0.65
    ),
    breaks = c(
      0.4,
      0.5,
      0.6
    ),
    labels = label_number(
      accuracy = 0.1
    ),
    expand = expansion(
      mult = c(
        0.03,
        0.03
      )
    )
  ) +
  
  scale_y_continuous(
    limits = c(
      0,
      1
    ),
    breaks = seq(
      0,
      1,
      by = 0.25
    ),
    labels = label_number(
      accuracy = 0.01
    ),
    expand = expansion(
      mult = c(
        0,
        0
      )
    )
  ) +
  
  labs(
    x = expression(
      italic(p)(success)
    ),
    y = expression(
      italic(P)(forage)
    )
  ) +
  
  theme_bw(
    base_size = interaction_base_size,
    base_family = publication_font_family
  ) +
  
  theme(
    
    axis.title.x = element_text(
      size = interaction_axis_size,
      margin = margin(t = 8),
      color = "black"
    ),
    
    axis.title.y = element_text(
      size = interaction_axis_size,
      margin = margin(r = 8),
      color = "black"
    ),
    
    axis.text = element_text(
      size = interaction_tick_size,
      color = "black"
    ),
    
    axis.ticks = element_line(
      linewidth = tick_line_width,
      color = "black"
    ),
    
    axis.ticks.length = unit(
      0.11,
      "cm"
    ),
    
    strip.background = element_blank(),
    
    strip.text.x = element_text(
      size = interaction_strip_size,
      face = "plain",
      margin = margin(b = 5),
      color = "black"
    ),
    
    strip.text.y = element_text(
      size = interaction_strip_size,
      face = "plain",
      angle = 90,
      margin = margin(l = 5),
      color = "black"
    ),
    
    panel.border = element_rect(
      color = "black",
      fill = NA,
      linewidth = 0.70
    ),
    
    panel.grid.major = element_line(
      color = "grey88",
      linewidth = 0.45,
      linetype = "dashed"
    ),
    
    panel.grid.minor = element_blank(),
    
    panel.spacing.x = unit(
      0.55,
      "cm"
    ),
    
    panel.spacing.y = unit(
      0.45,
      "cm"
    ),
    
    plot.margin = margin(
      t = 8,
      r = 8,
      b = 8,
      l = 8
    )
  )


save_publication_plot(
  plot_object = factorial_panel_plot,
  filename_stub = "final_factorial_six_panel",
  width = factorial_panel_width,
  height = factorial_panel_height
)


# ==============================================================================
# 16. INTERACTION PLOT — SMALL PANEL RELATIVE TO TEXT
# ==============================================================================

# Keep the exported canvas reasonably large.
# The plotting panel itself is reduced using large margins.
interaction_export_width  <- 6.30
interaction_export_height <- 4.80

factorial_combined_plot <- ggplot(
  interaction_summary,
  aes(
    x = x14_p_foraging_gain,
    y = mean_choice,
    color = BNW_conditions,
    linetype = weather_label,
    shape = weather_label,
    group = interaction(
      BNW_conditions,
      weather_label
    )
  )
) +
  
  geom_hline(
    yintercept = 0.5,
    color = "grey55",
    linetype = "dotted",
    linewidth = 0.55
  ) +
  
  geom_errorbar(
    aes(
      ymin = pmax(mean_choice - sem, 0),
      ymax = pmin(mean_choice + sem, 1)
    ),
    width = 0.005,
    linewidth = 0.50,
    alpha = 0.70,
    show.legend = FALSE
  ) +
  
  geom_point(
    size = 2.4,
    stroke = 0.75
  ) +
  
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = FALSE,
    linewidth = 1.05
  ) +
  
  scale_color_manual(
    name = "State",
    values = c(
      "Binary Energy" = "#0072B2",
      "Trade-off" = "#D55E00",
      "Wait When Safe" = "#CC79A7"
    ),
    labels = c(
      "Binary energy",
      "Trade-off",
      "Wait when safe"
    )
  ) +
  
  scale_linetype_manual(
    name = "Weather",
    values = c(
      "Bad Weather" = "solid",
      "Good Weather" = "longdash"
    ),
    labels = c(
      "Bad weather",
      "Good weather"
    )
  ) +
  
  scale_shape_manual(
    name = "Weather",
    values = c(
      "Bad Weather" = 16,
      "Good Weather" = 17
    ),
    labels = c(
      "Bad weather",
      "Good weather"
    )
  ) +
  
  scale_x_continuous(
    limits = c(0.35, 0.65),
    breaks = c(0.4, 0.5, 0.6),
    labels = scales::label_number(accuracy = 0.1),
    expand = expansion(mult = c(0.01, 0.01))
  ) +
  
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.2),
    labels = scales::label_number(accuracy = 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  
  labs(
    x = expression(italic(p)(success)),
    y = expression(italic(P)(forage))
  ) +
  
  theme_classic(
    base_size = 18,
    base_family = publication_font_family
  ) +
  
  theme(
    # Keep text large
    axis.title.x = element_text(
      size = 24,
      margin = margin(t = 10),
      color = "black"
    ),
    
    axis.title.y = element_text(
      size = 24,
      margin = margin(r = 10),
      color = "black"
    ),
    
    axis.text.x = element_text(
      size = 20,
      color = "black"
    ),
    
    axis.text.y = element_text(
      size = 20,
      color = "black"
    ),
    
    axis.line = element_line(
      linewidth = 0.75,
      color = "black"
    ),
    
    axis.ticks = element_line(
      linewidth = 0.70,
      color = "black"
    ),
    
    axis.ticks.length = unit(0.11, "cm"),
    
    # Legend remains large and outside the panel
    legend.position = "right",
    legend.box = "vertical",
    
    legend.title = element_text(
      size = 18,
      face = "plain"
    ),
    
    legend.text = element_text(
      size = 17
    ),
    
    legend.key.width = unit(0.85, "cm"),
    legend.key.height = unit(0.55, "cm"),
    
    # These large margins shrink the actual plotting panel
    # without shrinking the text.
    plot.margin = margin(
      t = 45,
      r = 25,
      b = 45,
      l = 35,
      unit = "pt"
    ),
    
    panel.grid = element_blank()
  ) +
  
  guides(
    color = guide_legend(
      order = 1,
      override.aes = list(
        linetype = "solid",
        shape = 16,
        linewidth = 1.2
      )
    ),
    
    linetype = guide_legend(
      order = 2,
      override.aes = list(
        color = "black",
        linewidth = 1.2
      )
    ),
    
    shape = guide_legend(
      order = 2,
      override.aes = list(
        color = "black",
        size = 3
      )
    )
  )

save_publication_plot(
  plot_object = factorial_combined_plot,
  filename_stub = "final_factorial_combined_small_panel",
  width = interaction_export_width,
  height = interaction_export_height
)

print(factorial_combined_plot)


# ==============================================================================
# 17. WEATHER × p(SUCCESS) PLOT
# ==============================================================================

weather_summary <- interaction_data %>%
  
  group_by(
    weather_label,
    x14_p_foraging_gain,
    x1_id
  ) %>%
  
  summarise(
    subject_mean = mean(
      x11_choice,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  
  group_by(
    weather_label,
    x14_p_foraging_gain
  ) %>%
  
  summarise(
    
    mean_choice = mean(
      subject_mean,
      na.rm = TRUE
    ),
    
    sem = ifelse(
      n() > 1,
      sd(
        subject_mean,
        na.rm = TRUE
      ) / sqrt(n()),
      0
    ),
    
    n_subjects = n(),
    
    .groups = "drop"
  )


weather_interaction_plot <- ggplot(
  weather_summary,
  aes(
    x = x14_p_foraging_gain,
    y = mean_choice,
    color = weather_label,
    linetype = weather_label,
    shape = weather_label,
    group = weather_label
  )
) +
  
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed",
    linewidth = 0.70,
    color = "grey50"
  ) +
  
  geom_errorbar(
    aes(
      ymin = pmax(
        mean_choice - sem,
        0
      ),
      ymax = pmin(
        mean_choice + sem,
        1
      )
    ),
    width = 0.008,
    linewidth = 0.70
  ) +
  
  geom_point(
    size = 3.1,
    fill = "white",
    stroke = 0.85
  ) +
  
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = FALSE,
    linewidth = 1.10
  ) +
  
  scale_color_manual(
    values = c(
      "Bad Weather" = "#D55E00",
      "Good Weather" = "#0072B2"
    )
  ) +
  
  scale_linetype_manual(
    values = c(
      "Bad Weather" = "solid",
      "Good Weather" = "dashed"
    )
  ) +
  
  scale_shape_manual(
    values = c(
      "Bad Weather" = 21,
      "Good Weather" = 24
    )
  ) +
  
  scale_x_continuous(
    limits = c(
      0.35,
      0.65
    ),
    breaks = c(
      0.4,
      0.5,
      0.6
    ),
    labels = label_number(
      accuracy = 0.1
    )
  ) +
  
  scale_y_continuous(
    limits = c(
      0,
      1
    ),
    breaks = seq(
      0,
      1,
      by = 0.2
    ),
    labels = label_number(
      accuracy = 0.1
    ),
    expand = expansion(
      mult = c(
        0,
        0
      )
    )
  ) +
  
  labs(
    x = expression(
      italic(p) ~ success
    ),
    y = expression(
      italic(P)(forage)
    ),
    color = NULL,
    linetype = NULL,
    shape = NULL
  ) +
  
  theme_interaction +
  
  theme(
    legend.position = "right"
  )


save_publication_plot(
  plot_object = weather_interaction_plot,
  filename_stub = "final_weather_p_success_panel",
  width = weather_panel_width,
  height = weather_panel_height
)


# ==============================================================================
# 18. DISPLAY ALL PLOTS
# ==============================================================================

print(aic_plot)
print(bic_plot)
print(weights_plot)
print(comparison_row)

print(roc_plot)
print(confusion_plot)
print(performance_row)

print(factorial_panel_plot)
print(factorial_combined_plot)
print(weather_interaction_plot)


# ==============================================================================
# 19. COMPLETION MESSAGE
# ==============================================================================

cat(
  "\nPublication-ready plots saved successfully:\n",
  "\nModel comparison:\n",
  "  final_AIC_panel.pdf/.png\n",
  "  final_BIC_panel.pdf/.png\n",
  "  final_model_weights_panel.pdf/.png\n",
  "  final_model_comparison_row.pdf/.png\n",
  "\nPerformance:\n",
  "  final_ROC_panel.pdf/.png\n",
  "  final_confusion_matrix_panel.pdf/.png\n",
  "  final_performance_row.pdf/.png\n",
  "\nInteractions:\n",
  "  final_factorial_six_panel.pdf/.png\n",
  "  final_factorial_combined_panel.pdf/.png\n",
  "  final_weather_p_success_panel.pdf/.png\n"
)