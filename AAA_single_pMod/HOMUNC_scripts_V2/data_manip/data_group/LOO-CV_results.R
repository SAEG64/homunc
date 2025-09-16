library('readr')
# library(htmltools)
library(purrr)
set.seed(123)
filepath = "/media/sergej/Extreme SSD/HOMUNC_parent/AAA_pMod_main/HOMUNC_scripts_V2/data_manip/data_group"
setwd(filepath)

ls <- list()
# Two variable models
load("/media/sergej/Extreme SSD/HOMUNC_parent/AAA_pMod_main/HOMUNC_scripts_V2/data_manip/data_group/homunc_Bayes_fit_res.RData")
for (i in 2:length(mdl)-1) {
  # Get model names
  mod = gsub("[x]", "", mdl[order(unlist(acs), decreasing=TRUE)[i]])
  mod = gsub("[0-9]", "", mod)
  mod = gsub("[_]", " ", mod)
  mod = gsub("[D]", " ", mod)
  if (mod == " p foraging gain") {
    mod = " <i>p<i> foraging gain"
  }
  if (mod == " continuous energy trial start") {
    mod =  " continuous energy state"
  }
  # Delete space in beginning
  mod = substr(mod, 2, nchar(mod))
  # Add ternary to model name
  mod = paste("ternary state +", mod)
  # print(mod)
  
  # Create list of model accuracies
  ls[[i]] <- list(Model =  mod, Accuracy = toString(acs[order(unlist(acs), decreasing=TRUE)[i]]))
}

# Three variable model
load("/media/sergej/Extreme SSD/HOMUNC_parent/AAA_pMod_main/HOMUNC_scripts_V2/data_manip/data_group/homunc_Bayes_fit_res_threeVar.RData")
for (i in 1:length(mdl)) {
  # Get model names
  mod = gsub("[x]", "", mdl[order(unlist(acs), decreasing=TRUE)[i]])
  mod = gsub("[0-9]", "", mod)
  mod = gsub("[_]", " ", mod)
  mod = gsub("[D]", " ", mod)
  if (mod == " p foraging gain") {
    mod = " <i>p<i> foraging gain"
  }
  if (mod == " continuous energy trial start") {
    mod =  " continuous energy state"
  }
  # Delete space in beginning
  mod = substr(mod, 2, nchar(mod))
  # Add ternary to model name
  mod = paste("ternary state + weather type +", mod)
  # print(mod)
  
  # Create list of model accuracies
  ls[[i]] <- prepend(ls, list(Model =  mod, Accuracy = toString(acs[order(unlist(acs), decreasing=TRUE)[i]])))
}

# Delete unnecessary model
ls_REV <- ls[1:(length(ls) - 1)]
# Create table format
accuracies <- data.frame(Model = character(0), Accuracy = character(0))
for (i in seq_along(ls)) {
  # Extract values and bind them as rows to the result_df
  accuracies <- rbind(accuracies, c(ls[[i]]$Model, ls[[i]]$Accuracy))
}
colnames(accuracies) <- c("Model", "Accuracy")
print(accuracies)
