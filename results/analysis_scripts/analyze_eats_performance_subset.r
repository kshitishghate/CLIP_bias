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



data$scaled_total_compute = log10(data$total_compute) - min(log10(data$total_compute))

data$centered_samples_seen = log10(data$samples_seen) - min(log10(data$samples_seen))
data$centered_total_compute = log10(data$total_compute) - min(log10(data$total_compute))
data$centered_total_params = log10(data$total_params_trained) - min(log10(data$total_params_trained))
data$centered_total_acts = log10(data$total_acts) - min(log10(data$total_acts))
data$centered_text_macs = log10(data$text_macs) - min(log10(data$text_macs))
data$centered_text_acts = log10(data$acts) - min(log10(data$text_acts))
data$centered_dataset_size = log10(data$dataset_size) - min(log10(data$dataset_size))
data$centered_params = log10(data$params) - min(log10(data$params))
data$centered_macs = log10(data$macs) - min(log10(data$macs))
data$centered_image_macs = log10(data$image_macs) - min(log10(data$image_macs))
data$centered_image_acts = log10(data$image_acts) - min(log10(data$image_acts))
data$centered_acts = log10(data$acts) - min(log10(data$acts))
data$centered_vtab = (data$vtab - mean(data$vtab)) / sd(data$vtab)

data = left_join(data, model_data)

data_ranked_ctc = rank(data$centered_total_compute)
data$efficiency = data$vtab / log10(data$total_compute)
data$centered_efficiency = log10(data$efficiency) - min(log10(data$efficiency))




c("model", "architecture", "model_family", "samples_seen", "model_name", "fine_tuned",
                   "epochs", "samples_per_epoch",
                    "image_size", "image_width", "text_width", "embed_dim",
                    "params",
                   "image_params", "text_params",
                   "macs", "image_macs", "text_macs",
                   "acts", "image_acts", "text_acts")



# Filter out any columns with na values
data <- data[,colSums(is.na(data))==0]

without_vtab = lmer(effect_size ~ 1 + centered_total_params * centered_samples_seen  + (centered_total_params * centered_samples_seen | Test) + (1 | dataset), data=data)

BIC(lmer(effect_size ~ 1  + (centered_total_compute | Test) + (1 | dataset), data=data))

best_mod = lmer(effect_size ~ 1  + `vtab+` + (`vtab+` | Test) + (1 | dataset), data=data)

corr_plot_data = coef(best_mod)$Test
corr_plot_data$Test = row.names(corr_plot_data)
corr_plot_data = corr_plot_data %>%
  tibble
bias_tests = data %>%
  distinct(Test, test_category, overall_test_category,
           modality, stimuli_type, word_category, type_of_stimuli)

data_together = left_join(corr_plot_data, bias_tests)

ctc_es_cor = list()
vtab_es_cor = list()
for (i in 1:nrow(data_together)) {
  a = data %>%
  filter(Test==data_together$Test[[i]]) %>%
  dplyr::select(c(centered_total_compute, effect_size)) %>% cor()
    ctc_es_cor[[i]] = a[2]
    b = data %>%
    filter(Test==data_together$Test[[i]]) %>% dplyr::select(c(`vtab+`, effect_size)) %>% cor()
    vtab_es_cor[[i]] = b[2]
}
data_together = data_together %>%
  mutate(ctc_es_cor = unlist(ctc_es_cor),
         vtab_es_cor = unlist(vtab_es_cor))

ggplot(data_together, aes(x=`(Intercept)`, y=vtab,
                          # color=type_of_stimuli
)) +
  xlab(TeX('$\\hat{\\alpha} + \\hat{\\beta}_{j[i]}$'))+
  ylab(TeX('$\\hat{\\gamma}_{j[i]}$')) +
  geom_point()


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
  names(performances) = paste(row.names(coefs$Test), '_vtab_coef',sep='')

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

bootstrap_models = bootstrap(best_mod, type='residual', B = 10000, .f=my_coefs)
# saveRDS(bootstrap_models, 'results/plots/eats_vtab_bootstrap.RDS')
bootstrap_models = readRDS('results/plots/eats_vtab_bootstrap.RDS')
