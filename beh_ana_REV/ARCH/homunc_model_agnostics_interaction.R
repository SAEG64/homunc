## Requirements
library('readr')
library(ROCR)
library(dplyr)
library(ggplot2)
# Set working directory
set.seed(123)
filepath <- paste(dirname(rstudioapi::getSourceEditorContext()$path), "/", sep = "")
setwd(filepath)
# # Data import and filtering
# dat <- read_csv("datall_cat.csv")
# dat <- subset(dat, !is.na(x9_button_pressed)) # exclude none responses
# dat <- filter(dat, x6_continuous_energy_trial_start > 0)
load("homunc_Bayes_fit_res_threeVar_filtered_full_interaction.RData")
# # Test optimal policy instead of p success
# data$x3 <- dat$x22_optimal_policy

# Function to extract midpoints from the interval string
get_midpoint <- function(interval) {
  # Extract the lower and upper bounds using regular expressions
  bounds <- as.numeric(unlist(regmatches(interval, gregexpr("[0-9.]+", interval))))
  
  # Calculate the midpoint
  midpoint <- mean(bounds)
  return(midpoint)
}

## Plot weather type effect on p success
# Rename factor levels
# dat$BNW_conditions[dat$x22_optimal_policy == 0] <- "indifferent"
dat$BNW_conditions[dat$BNW_conditions == 1] <- "binary energy"
dat$BNW_conditions[dat$BNW_conditions == 0.5] <- "trade-off"
dat$BNW_conditions[dat$BNW_conditions == 0] <- "wait when safe"
data$x1 <- dat$BNW_conditions

dat$x7_weather_type[dat$x7_weather_type == 1] <- "bad weather"
dat$x7_weather_type[dat$x7_weather_type == 2] <- "good weather"
data$x2 <- dat$x7_weather_type
# # Filter data to equal range across weather
# data <- data[data$x3 < 0.7,]
# data <- data[data$x3 > 0.3,]
# Prep data fro visualization
plot_data <- data.frame(
  GroundTruth = as.numeric(data$y),
  Predicted = as.numeric(preds),
  x1 = data$x1,
  x2 = data$x2,
  x3 = data$x3,
  group = data$group
)
if (is.factor(plot_data$x3) == FALSE) {
  plot_data$bin <- plot_data$x3  # Adjust number of bins as needed
}else {
  plot_data$bin <- plot_data$x3  # Adjust number of bins as needed
}
# Aggregate the count of occurrences per bin and calculate average predicted probabilities for each bin
bin_data_sbj <- plot_data %>%
  group_by(bin, x1, x2, group) %>%
  summarize(
    count = n(),  # Count the occurrences in each bin
    avg_pred_prob = mean(Predicted),
    avg_prob = mean(GroundTruth),
    #.groups = 'drop'
  )
bin_data <- bin_data_sbj %>%
  group_by(bin, x1, x2) %>%
  summarize(
    count = mean(count),  # Average counts in each bin
    avg_pred_prob = mean(avg_pred_prob),
    avg_prob = mean(avg_prob),
    .groups = 'drop'
  )
# Convert bin intervals to midpoints
midpoints <- sapply(bin_data$bin, get_midpoint)
bin_data$bin <- midpoints
# Create the plot
p1 <- ggplot(bin_data, aes(x = bin, y = avg_pred_prob,
                           group = interaction(x1, x2), color = x1, linetype = x2)) + # Plot the predicted logistic curve with different lines for each x1 and x2 combination
  # Plot the predicted logistic curve
  geom_line(size = 1) +  # No linetype specified; lines will be solid by default
  geom_point(size = 2, shape = 5) +

  # # Plot the actual observed responses as bubbles, scaled by the count in each bin
  # geom_point(data = bin_data, aes(x = bin, y = avg_prob, size = as.numeric(count),
  #                                 group = x2, color = x1),
             # alpha = 0.7) +
  # Customize the plot
  ylim(0, 1) +  # Set y-axis limits from 0 to 1
  labs(x = ~italic("p")~" success", y = "Foraging likelihood", title = "Interaction",# ~ italic("p") ~ "success",
       color = "Ternary state", linetype = "Weather type") +
  scale_size_continuous(name = "Empirical sampling", range = c(3, 10)) +  # Adjust size range
  theme_minimal()
# theme(legend.position = "none")
p1 + theme(
  axis.title.x = element_text(size = 30),   # Increase x-axis title text size
  axis.title.y = element_text(size = 30),   # Increase y-axis title text size
  axis.text.x = element_text(size = 22),    # Increase x-axis tick label text size
  axis.text.y = element_text(size = 22),    # Increase y-axis tick label text size
  plot.title = element_text(size = 32),     # Increase plot title text size
  legend.title = element_text(size = 24),   # Increase legend title text size (if applicable)
  legend.text = element_text(size = 22)     # Increase legend text size (if applicable)
)
