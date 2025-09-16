# Load necessary libraries
# library(lme4)
library(brms)
library(dplyr)
library('readr')
library(ROCR)
library(ggplot2)
# Set working directory
set.seed(123)
filepath <- paste(dirname(rstudioapi::getSourceEditorContext()$path), "/", sep = "")
setwd(filepath)
# Data import and filtering
dat <- read_csv("datall_cat.csv")
dat <- subset(dat, !is.na(x9_button_pressed)) # exclude none responses
dat <- filter(dat, x6_continuous_energy_trial_start > 0)
# Filter data to equal range across weather
dat <- dat[dat$x14_p_foraging_gain < 0.7,]
dat <- dat[dat$x14_p_foraging_gain > 0.3,]

# Models to compare
mdl <- c(
  # 'BNW_conditions',
  # 'x17_horizon_correct_adjusted',
  # 'x6_continuous_energy_trial_start',
  # 'x24_pseudo0x2Doptimal_horizon0x2D1',
  # 'x13_gain_magnitude',
  # 'x7_weather_type',
  'x14_p_foraging_gain')#,
  # 'p_delta',
  # 'x22_optimal_policy')

# Raw model components
resp <- dat$x11_choice
mod1 <- dat$BNW_conditions
mod1 <- ifelse(mod1 > 0.5, 3, mod1)
mod1 <- ifelse(mod1 < 0.5, 1, mod1)
mod1 <- ifelse(mod1 == 0.5, 2, mod1)
mod2 <- dat$x7_weather_type

# For saving results of LOO-CV
all_results = list()
acs <- c()
# Loop through models
for (j in 1:length(mdl)) {
  
  cat("Current DV: ", mdl[j], "\n")
  
  # Decision variables
  x1 = mod1             # Fixed predictor
  # Ensure dummy coding for factors
  if (length(unique(x1)) < 4) {
    x1 = factor(x1)
  }
  x2 = mod2             # Fixed predictor
  # Ensure dummy coding for factors
  if (length(unique(x2)) < 4) {
    x2 = factor(x2)
  }
  x3 = c(dat[mdl[j]])[[1]]   # Variable predictor
  # # Ensure dummy coding for factors
  # if (length(unique(x3)) < 4) {
  #   x3 = factor(x3)
  # }
  
  # Define data
  data <- data.frame(
    y = resp,   # Binary outcome variable
    x1 = x1,    # Fixed predictor
    x2 = x2,    # Fixed predictor
    x3 = x3,    # Variable predictor
    group = factor(dat$x1_id)  # Random effect grouping variable
  )
  colnames(data) <- c("y", "x1", "x2","x3", "group")
  
  # Name for plot saving
  file_name = paste("threeVar_ternary_weather_pSucc.png", sep="")
  
  # Run the LOO-CV
  # Initialize requirements
  groups = unique(dat$x1_id)
  n <- length(groups)
  preds <- c()
  coefs <- c()
  
  # Loop through each subject
  for (i in 1:n) {
    # Create training data by excluding the ith observation
    train_data <- data[data$group != groups[i],]
    test_data <- data[data$group == groups[i], ]
    # Remove the 'subject' column from the test data
    test_data <- test_data[, !names(test_data) %in% "group", drop = FALSE]
    
    ##############################################################################
    # Fit the logistic mixed effects model (logit link)
    ##############################################################################
    # Fit a Bayesian logistic regression model with random intercepts
    model <- brm(
      formula = y ~ (
        x1 *
          x2 +
          x3 +
          (1 | group)
      ),
      family = bernoulli(),
      prior = c(
        prior(normal(0, 5), class = "b"), # Normal prior for fixed effects
        prior(normal(0, 5), class = "sd")  # Normal prior for random effects
      ),
      data = data,
      cores = 4,
      chains = 4, 
      iter = 4000,
    warmup = 2000
    )
    
    # Make predictions on the test set, excluding random effects (use only fixed effects)
    # preds <- c(preds, 
    #                  predict(model, newdata = test_data, 
    #                          type = "response", re.form = NA))
    # Posterior predictions of the response variable
    post_preds <- predict(model, summary = TRUE)
    # Accessing the mean (point estimate) and intervals
    preds <- post_preds[, "Estimate"] # means
    # preds_lower <- post_preds[, "Q2.5"]
    # preds_upper <- post_preds[, "Q97.5"]    
    
    # Extract coefficients and store them
    fixed_effects <- fixef(model)  # Fixed-effect coefficients
    coefs <- c(coefs, list(as.numeric(fixed_effects)))
  }
  
  # Compute accuracy and area under the curve (ac)
  randomized_vector <- runif(length(preds), min = 0, max = 1)
  pred_class <- ifelse(runif(length(randomized_vector)) <= preds, 1, 0)
  true_class <- data$y
  # # Accuracy
  # accuracy <- mean(pred_class == true_class)
  # AUC-ROC
  pred <- prediction(pred_class, true_class)
  ac <- performance(pred, "auc")@y.values[[1]]
  
  # Dave final output
  res <- list(accuracy_metric = ac, coefficients = coefs, predictions = preds, posterior_predictions = post_preds)
  all_results[[j]] = res
  acs <- c(acs, res[1])
  
  # Fail save
  save.image(file = "my_data.RData")
  
  # Output the accuracy
  cat(mdl[j], "\n", "Leave-One-Out Cross-Validation accuracy metric: ", toString(res[1]), "\n")
  print("=================================")
}

cat('\ntri-variable model:  ternary state * weather * ', mdl[order(unlist(acs), decreasing=TRUE)[1]],
    " accuracy metric: ", toString(acs[order(unlist(acs), decreasing=TRUE)[1]]),'\n')
print("=================================")


## Diagnostics
library(posterior)
library(bayesplot)
draws <- as_draws_array(model)  # Convert samples to an array
# Extract parameter names, excluding `lp__` if present
parameter_names <- variables(draws)
parameter_names <- parameter_names[!grepl("^lp__", parameter_names)]  # Exclude lp__

# Loop through parameters and generate rank plots
for (param in parameter_names) {
  # Plot rank plot for the current parameter using 'pars'
  plot <- mcmc_rank_overlay(draws, pars = param) +
    ggtitle(paste("Rank Plot for Parameter:", param))
  
  # Display the plot
  print(plot)
}

# Extract posterior samples and param names
posterior <- as_draws_df(model)
names(posterior)
# Check interaction of x1 and x2
mcmc_areas(as_draws_df(model), pars = c("b_x12:x22", "b_x13:x22"))

# Set x3 to its mean (or another value you're interested in)
data$pred <- post_preds[,1]

# Plot using ggplot2
ggplot(data, aes(x = x1, y = pred, color = x2)) +
  geom_line() +
  labs(x = "x1", y = "Predicted Probability", color = "x2") +
  facet_grid(x2 ~ ., scales = "free_y") + # Facet by columns
  theme_minimal() +
  theme(legend.position = "top")

# Test effects via ANOVA
anova_result <- aov(pred ~ x1 * x2, data = data)
summary(anova_result)

