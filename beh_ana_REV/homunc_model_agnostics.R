## Requirements
library('readr')
library(ROCR)
library(dplyr)
library(ggplot2)
# Set working directory
set.seed(123)
filepath <- paste(dirname(rstudioapi::getSourceEditorContext()$path), "/", sep = "")
setwd(filepath)
# Function to extract midpoints from the interval string
get_midpoint <- function(interval) {
  # Extract the lower and upper bounds using regular expressions
  bounds <- as.numeric(unlist(regmatches(interval, gregexpr("[0-9.]+", interval))))
  
  # Calculate the midpoint
  midpoint <- mean(bounds)
  return(midpoint)
}
# Data import
load("homunc_Bayes_fit_res_threeVar.RData")
# Add optimal policy
data$x4 <- dat$x21_optimal_policy_binary
data$x4[data$x4 == 2] <- 0.5
data$x5 = post_preds[,1]

## Plot weather type effect on pMod
# Data preprocessing
dat$x7_weather_type[dat$x7_weather_type == 1] <- "bad weather"
dat$x7_weather_type[dat$x7_weather_type == 2] <- "good weather"
data$x2 <- dat$x7_weather_type
# Prep data fro visualization
plot_data <- data.frame(
  GroundTruth = as.numeric(true_class),
  Predicted = as.numeric(preds),
  x1 = data$x1,
  x2 = data$x2,
  x3 = data$x3,
  policy = data$x4,
  #policy = data$x5,
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
    avg_pred_prob = mean(policy),
    avg_prob = mean(GroundTruth),
    #.groups = 'drop'
  )
bin_data <- bin_data_sbj %>%
  group_by(bin, x2) %>%
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
                            group = x2, color = x2, , linetype = x2)) +
  # Plot the predicted logistic curve
  geom_line(size = 2,  linetype = "dashed") +
#  geom_point(size = 2, shape = 5) +

#  # Plot the actual observed responses as bubbles, scaled by the count in each bin
#  geom_point(data = bin_data, aes(x = bin, y = avg_prob, size = as.numeric(count)),
#             alpha = 0.7) +
  # Customize the plot
  ylim(0, 1) +  # Set y-axis limits from 0 to 1
  labs(x = ~italic("p")~" success", y = "Foraging likelihood", title = "Weather type effect",# ~ italic("p") ~ "success",
       color = "Weather type") +
  scale_size_continuous(name = "Empirical sampling", range = c(3, 10)) +  # Adjust size range
  theme_minimal()
# theme(legend.position = "none")
p1 + theme(
  axis.title.x = element_text(size = 36),   # Increase x-axis title text size
  axis.title.y = element_text(size = 36),   # Increase y-axis title text size
  axis.text.x = element_text(size = 28),    # Increase x-axis tick label text size
  axis.text.y = element_text(size = 28),    # Increase y-axis tick label text size
  plot.title = element_text(size = 40),     # Increase plot title text size
  legend.title = element_text(size = 30),   # Increase legend title text size
  legend.text = element_text(size = 28),    # Increase legend text size
  strip.text = element_text(size = 26),     # Increase facet strip text size
  panel.grid.major = element_line(size = 0.8),  # Increase major grid line thickness
  panel.grid.minor = element_line(size = 0.4),   # Increase minor grid line thickness
  aspect.ratio = 1                       # Decrease plot height by setting aspect ratio
)
ggsave("weather_type_effect_plot.png", p1 + theme(
  axis.title.x = element_text(size = 36),   # Increase x-axis title text size
  axis.title.y = element_text(size = 36),   # Increase y-axis title text size
  axis.text.x = element_text(size = 28),    # Increase x-axis tick label text size
  axis.text.y = element_text(size = 28),    # Increase y-axis tick label text size
  plot.title = element_text(size = 40),     # Increase plot title text size
  legend.title = element_text(size = 30),   # Increase legend title text size
  legend.text = element_text(size = 28),    # Increase legend text size
  strip.text = element_text(size = 26),     # Increase facet strip text size
  panel.grid.major = element_line(size = 0.8),  # Increase major grid line thickness
  panel.grid.minor = element_line(size = 0.4),   # Increase minor grid line thickness
  aspect.ratio = 1                       # Decrease plot height by setting aspect ratio
), width = 10, height = 10, dpi = 300)



## Plot ternary state effect on p success after splitting off "true indifferent" states
# Rename and separate WWS from "true indifferent" states
# dat$BNW_conditions[dat$x22_optimal_policy == 0] <- "indifferent"
dat$BNW_conditions[dat$BNW_conditions == 1] <- "binary energy"
dat$BNW_conditions[dat$BNW_conditions == 0.5] <- "trade-off"
dat$BNW_conditions[dat$BNW_conditions == 0] <- "wait when save"
# dat$BNW_conditions <- factor(dat$BNW_conditions)
data$x1 <- dat$BNW_conditions
# Prep data fro visualization
plot_data <- data.frame(
  GroundTruth = as.numeric(true_class),
  Predicted = as.numeric(preds),
  x1 = data$x1,
  x2 = data$x2,
  x3 = data$x3,
  #policy = data$x4,
  policy = data$x5,
  group = data$group
)
if (is.factor(plot_data$x3) == FALSE) {
  plot_data$bin <- plot_data$x3
}else {
  plot_data$bin <- plot_data$x3  # Adjust number of bins as needed
}
# Aggregate the count of occurrences per bin and calculate average predicted probabilities for each bin
bin_data_sbj <- plot_data %>%
  group_by(bin, x1, x2, group) %>%
  summarize(
    count = n(),  # Count the occurrences in each bin
    avg_pred_prob = mean(policy),
    avg_prob = mean(GroundTruth),
    #.groups = 'drop'
  )
bin_data <- bin_data_sbj %>%
  group_by(bin, x1) %>%
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
p2 <- ggplot(bin_data, aes(x = bin, y = avg_pred_prob,
                          group = x1, color = x1, , linetype = x1)) +
  # Plot the predicted logistic curve
  geom_line(size = 2,  linetype = "dashed") +
#  geom_point(size = 2, shape = 5) +
  
  # Plot the actual observed responses as bubbles, scaled by the count in each bin
#  geom_point(data = bin_data, aes(x = bin, y = avg_prob, size = as.numeric(count)),
#             alpha = 0.7) +
  # Customize the plot
  ylim(0, 1) +  # Set y-axis limits from 0 to 1
  labs(x = ~italic("p")~" success", y = "Foraging likelihood", title = "Ternary state effect",# ~ italic("p") ~ "success",
       color = "Ternary state") +
  scale_size_continuous(name = "Empirical sampling", range = c(3, 10)) +  # Adjust size range
  theme_minimal()
# theme(legend.position = "none")
p2 + theme(
  axis.title.x = element_text(size = 36),   # Increase x-axis title text size
  axis.title.y = element_text(size = 36),   # Increase y-axis title text size
  axis.text.x = element_text(size = 28),    # Increase x-axis tick label text size
  axis.text.y = element_text(size = 28),    # Increase y-axis tick label text size
  plot.title = element_text(size = 40),     # Increase plot title text size
  legend.title = element_text(size = 30),   # Increase legend title text size
  legend.text = element_text(size = 28),    # Increase legend text size
  strip.text = element_text(size = 26),     # Increase facet strip text size
  panel.grid.major = element_line(size = 0.8),  # Increase major grid line thickness
  panel.grid.minor = element_line(size = 0.4),   # Increase minor grid line thickness
  aspect.ratio = 1                       # Decrease plot height by setting aspect ratio
)
ggsave("ternary_x_weather_plot.png", p2 + theme(
  axis.title.x = element_text(size = 36),   # Increase x-axis title text size
  axis.title.y = element_text(size = 36),   # Increase y-axis title text size
  axis.text.x = element_text(size = 28),    # Increase x-axis tick label text size
  axis.text.y = element_text(size = 28),    # Increase y-axis tick label text size
  plot.title = element_text(size = 40),     # Increase plot title text size
  legend.title = element_text(size = 30),   # Increase legend title text size
  legend.text = element_text(size = 28),    # Increase legend text size
  strip.text = element_text(size = 26),     # Increase facet strip text size
  panel.grid.major = element_line(size = 0.8),  # Increase major grid line thickness
  panel.grid.minor = element_line(size = 0.4),   # Increase minor grid line thickness
  aspect.ratio = 1                       # Decrease plot height by setting aspect ratio
), width = 10, height = 10, dpi = 300)

