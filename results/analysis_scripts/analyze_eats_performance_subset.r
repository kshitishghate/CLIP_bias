library(mlmRev)
library(dplyr)
library(caret)
library(arm)
library(stringr)
library(ggplot2)
library(lme4)
library(forcats)
library(rstanarm)
library(readr)
library(effsize)
library(tidyr)
library(brms)
library(MuMIn)
library(r2mlm)
library(lmeresampler)
options(mc.cores = parallel::detectCores())




# Read in data
data = read_csv('results/data/unimodal_data_for_modeling.csv')
data$total_compute = data$macs / 1e9 * data$samples_seen # GMACs * samples_seen
data$total_params_trained = data$params / 1e9 * data$samples_seen # Gparams * samples_seen
data$total_acts = data$acts / 1e6 * data$samples_seen # macts * samples_seen
data$type_of_stimuli = paste(data$word_category, data$stimuli_type, sep='/')




subset = data %>%
  filter(model_source =='cherti')
subset$centered_samples_seen = log10(subset$samples_seen) - min(log10(subset$samples_seen))
subset$centered_total_compute = log10(subset$total_compute) - min(log10(subset$total_compute))
subset$centered_total_params = log10(subset$total_params_trained) - min(log10(subset$total_params_trained))
subset$centered_total_acts = log10(subset$total_acts) - min(log10(subset$total_acts))
subset$centered_text_macs = log10(subset$text_macs) - min(log10(subset$text_macs))
subset$centered_text_acts = log10(subset$acts) - min(log10(subset$text_acts))
subset$centered_dataset_size = log10(subset$dataset_size) - min(log10(subset$dataset_size))
subset$centered_params = log10(subset$params) - min(log10(subset$params))
subset$centered_macs = log10(subset$macs) - min(log10(subset$macs))
subset$centered_image_macs = log10(subset$image_macs) - min(log10(subset$image_macs))
subset$centered_image_acts = log10(subset$image_acts) - min(log10(subset$image_acts))
subset$centered_acts = log10(subset$acts) - min(log10(subset$acts))


c("model", "architecture", "model_family", "samples_seen", "model_name", "fine_tuned",
                   "epochs", "samples_per_epoch",
                    "image_size", "image_width", "text_width", "embed_dim",
                    "params",
                   "image_params", "text_params",
                   "macs", "image_macs", "text_macs",
                   "acts", "image_acts", "text_acts")



# Filter out any columns with na values
subset <- subset[,colSums(is.na(subset))==0]


best_mod = lmer(effect_size ~ 1 + `acc5vtab+` + (`acc5vtab+` | Test) + (1 | dataset), data=subset)

# Departure from normality
ggplot(data.frame(y=residuals(best_mod)), aes(sample=y)) +
  stat_qq() +
  stat_qq_line() +
  xlab('Theoretical Quantile') +
  ylab('Sample Quantile')+
  theme_bw() +
  theme(text=element_text(size=20))



# Fit bootstrap model
my_coefs <- function(.) {
  coefs = coef(.)
  intercepts = coefs$Test[[1]]
  names(intercepts) = paste(row.names(coefs$Test), '_intercept_coef',sep='')

  performances = coefs$Test[[2]]
  names(performances) = paste(row.names(coefs$Test), '_acc5vtab+_coef',sep='')

  datasets = coefs$dataset[[1]]
  names(datasets) = paste(row.names(coefs$dataset), '_intercept_coef',sep='')

  ranefs = ranef(.)
  ran_intercepts = ranefs$Test[[1]]
  names(ran_intercepts) = paste(row.names(ranefs$Test), '_intercept_ran',sep='')

  ran_performances = ranefs$Test[[2]]
  names(ran_performances) = paste(row.names(ranefs$Test), '_acc5vtab+_ran',sep='')

  ran_datasets = ranefs$dataset[[1]]
  names(ran_datasets) = paste(row.names(ranefs$dataset), '_intercept_ran',sep='')

  between_test_stds = attr(VarCorr(.)[[1]], 'stddev')
  names(between_test_stds) = paste(names(between_test_stds), '_std_test',sep='')

  between_dataset_stds = attr(VarCorr(.)[[2]], 'stddev')
  names(between_dataset_stds) = paste(names(between_dataset_stds), '_std_dataset',sep='')


  corrs = attr(VarCorr(.)[[1]],'correlation')[[2]]
  names(corrs) = c('intercept_ctc_corr')

  return(
  c(fixef(.), intercepts, performances, datasets, ran_intercepts,
    ran_performances, ran_datasets,
    between_test_stds, between_dataset_stds, corrs)
  )


}
