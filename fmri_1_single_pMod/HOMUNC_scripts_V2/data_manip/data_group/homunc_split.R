# Requirements
if (TRUE) {
  library('readr')
  library('depmixS4')
  library(magrittr)
  library(gridExtra)
  library(ggplot2)
  library(plyr)
  library(tidyr)
  library(reshape)
  library(dplyr)
  # Set working directory
  set.seed(123)
  filepath <- paste(dirname(rstudioapi::getSourceEditorContext()$path), "/", sep = "")
  setwd(filepath)
  # Data import
  dat <- read_csv("datall_cat.csv")
}


# # Aggregate data
# if(TRUE) {
#   # Aggregate
#   dat_agg <- aggregate(x11_choice ~ 
#                          x59_weather_1_p_gain +
#                          x60_weather_2_p_gain +
#                          x7_weather_type +
#                          x5_order_trials_in_forest_0x28not_days0x29 +
#                          x6_continuous_energy_trial_start +
#                          x13_gain_magnitude +
#                          x1_id, 
#                        data = dat,
#                        mean)
#   dat_agg <- aggregate(x11_choice ~ 
#                          x59_weather_1_p_gain +
#                          x60_weather_2_p_gain +
#                          x7_weather_type +
#                          x5_order_trials_in_forest_0x28not_days0x29 +
#                          x6_continuous_energy_trial_start +
#                          x13_gain_magnitude, 
#                        data = dat_agg,
#                        mean)
#   names(dat_agg)[names(dat_agg) == 'x11_choice'] <- 'choice_mean'
#   
#   dat_agg2 <- aggregate(x26_logRT ~ 
#                          x59_weather_1_p_gain +
#                          x60_weather_2_p_gain +
#                          x7_weather_type +
#                          x5_order_trials_in_forest_0x28not_days0x29 +
#                          x6_continuous_energy_trial_start +
#                          x13_gain_magnitude +
#                          x1_id, 
#                        data = dat,
#                        mean)
#   dat_agg2 <- aggregate(x26_logRT ~ 
#                          x59_weather_1_p_gain +
#                          x60_weather_2_p_gain +
#                          x7_weather_type +
#                          x5_order_trials_in_forest_0x28not_days0x29 +
#                          x6_continuous_energy_trial_start +
#                          x13_gain_magnitude, 
#                        data = dat_agg2,
#                        mean)
#   names(dat_agg2)[names(dat_agg2) == 'x26_logRT'] <- 'logRT_mean'
#   # Get merge axis
#   dut <- data.frame(dat$x59_weather_1_p_gain,
#                     dat$x60_weather_2_p_gain,
#                     dat$x7_weather_type,
#                     dat$x5_order_trials_in_forest_0x28not_days0x29,
#                     dat$x6_continuous_energy_trial_start,
#                     dat$x13_gain_magnitude,
#                     dat$x11_choice)
#   df1 <- unite(dut, col='agg_col',
#                c('dat.x59_weather_1_p_gain',
#                  'dat.x60_weather_2_p_gain',
#                  'dat.x7_weather_type',
#                  'dat.x5_order_trials_in_forest_0x28not_days0x29',
#                  'dat.x6_continuous_energy_trial_start',
#                  'dat.x13_gain_magnitude'), sep='-')
#   df1$id  <- 1:nrow(df1)
#   df2 <- unite(dat_agg, col='agg_col',
#                c('x59_weather_1_p_gain',
#                  'x60_weather_2_p_gain',
#                  'x7_weather_type',
#                  'x5_order_trials_in_forest_0x28not_days0x29',
#                  'x6_continuous_energy_trial_start',
#                  'x13_gain_magnitude'), sep='-')
#   df2$id  <- 1:nrow(df2)
#   
#   df3 <- unite(dat_agg2, col='agg_col',
#                c('x59_weather_1_p_gain',
#                  'x60_weather_2_p_gain',
#                  'x7_weather_type',
#                  'x5_order_trials_in_forest_0x28not_days0x29',
#                  'x6_continuous_energy_trial_start',
#                  'x13_gain_magnitude'), sep='-')
#   df3$id  <- 1:nrow(df3)
#   # Merge and order
#   df <- merge(df1, df2, 
#               by='agg_col')
#   df <- df[order(df$id.x), ]
#   df_REV <- merge(df, df3, 
#               by='agg_col')
#   df_REV <- df_REV[order(df_REV$id.x), ]
#   # Join with original data
#   dat$choice_mean <- df$choice_mean
#   dat$logRT_mean <- df_REV$logRT_mean
# }
# 
# # Plot aggregated time-series
# if (TRUE) {
#   dat_crop <- dat[1:60,]
#   g0 <- (ggplot(dat_crop, aes(x = roll, y = choice_mean)) + geom_line() +
#            theme(axis.ticks = element_blank(), axis.title.y = element_blank()))
#   g0
# }
# max(dat$choice_mean)
# min(dat$choice_mean)
# median(dat$choice_mean)
# unique(dat$choice_mean)
# as.data.frame(table(dat$choice_mean))

# # Preprocess final dataframe
# write.csv(dat, paste(filepath,"datall_cat_agg.csv",sep=""), row.names=FALSE)

# Resplit the data
library(R.matlab)
library(pharmr)

headers <- colnames(dat)
dfs <- split(dat, f = list(dat$x1_id))
setwd('..')
setwd('..')
pre_path <- '/beh_v12_agg/'
for (i in 1:length(dfs)){
  l <- lapply(dfs, "[[", 'x1_id')[i]
  f <- capture.output(cat(getwd(),pre_path,"beh_v12_sub",l[[1]][1],".mat", sep = ""))
  d <- data.matrix(data.frame(dfs[i]))
  names(d) <- NULL
  rownames(d) <- NULL
  writeMat(con = f, x = d)
}

