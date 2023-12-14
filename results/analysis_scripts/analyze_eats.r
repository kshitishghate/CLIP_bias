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
library(sjPlot)
library(lmerTest)
library(multilevelTools)
library("lme4")
library("ggplot2")
library("HLMdiag")
library("DHARMa")
library("car")
library("Matrix")
library(latex2exp)


options(mc.cores = parallel::detectCores())


# https://github.com/marklhc/bootmlm


# Read in data
data = read_csv('results/data/unimodal_data_for_modeling.csv')
data$centered_params = log10(data$params) - min(log10(data$params))
data$centered_macs = log10(data$macs) - min(log10(data$macs))
data$centered_image_macs = log10(data$image_macs) - min(log10(data$image_macs))
data$centered_image_acts = log10(data$image_acts) - min(log10(data$image_acts))
data$centered_acts = log10(data$acts) - min(log10(data$acts))
data$total_compute = data$macs / 1e9 * data$samples_seen # GMACs * samples_seen
data$centered_total_compute = log10(data$total_compute) - min(log10(data$total_compute))
data$total_params_trained = data$params / 1e9 * data$samples_seen # Gparams * samples_seen
data$total_acts = data$acts / 1e6 * data$samples_seen # macts * samples_seen
data$centered_dataset_size = log10(data$dataset_size) - min(log10(data$dataset_size))
data$type_of_stimuli = paste(data$word_category, data$stimuli_type, sep='/')

# Rename dataset
data$dataset = case_when(
  data$dataset == 'laion80m' ~ "LAION 80M",
  data$dataset == 'laion400m' ~ "LAION 400M",
  data$dataset == 'laion5b' ~ "LAION 5B",
  data$dataset == 'OpenAI WebImageText' ~ "OpenAI WIT",
  data$dataset == 'yfcc15m' ~ "YFCC 15M",
  data$dataset == 'laion_aesthetic' ~ "LAION Aesthetic",
  data$dataset == 'laion2b' ~ "LAION 2B",
  data$dataset == 'mscoco_finetuned_laion2b' ~ "LAION 2B (MSCOCO Finetuned)",
  data$dataset == 'cc12m' ~ "CC 12M",
  .default = as.character(data$dataset)
)

# Filter out any columns with na values
data <- data[,colSums(is.na(data))==0]


# Base model
BIC(lm(effect_size ~ Test + model, data=data)) # Fixed
BIC(lmer(effect_size ~ 1 + (1 | Test) + (1 | model), data=data)) # Random

# With total compute seems to be a useful contextual predictor
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test) + (1 | model), data=data)) # With total compute as rs
BIC(lmer(effect_size ~ 1 + centered_total_compute + (1 | Test) + (1 | model), data=data)) # With total compute as shared slope

# Num params and num macs perform worse than total compute
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | model), data=data)) # With centered_params as rs
BIC(lmer(effect_size ~ 1 + centered_params + (1 | Test) + (1 | model), data=data)) # With centered_params as shared slope
BIC(lmer(effect_size ~ 1 + centered_macs + (centered_macs | Test) + (1 | model), data=data)) # With centered_macs as rs
BIC(lmer(effect_size ~ 1 + centered_macs + (1 | Test) + (1 | model), data=data)) # With centered_macs as shared slope


# Dataset/dataset size does seem to be a better predictor than model, but not dataset family
# (Chooisng not to model dataset size due to correlation with total compute)
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test) + (1 | dataset/model), data=data)) # Using dataset as a third level
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test) + (1 | dataset), data=data)) # Using dataset instead of model
BIC(lmer(effect_size ~ 1 + centered_total_compute + dataset_size + (centered_total_compute | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + dataset_family + (centered_total_compute | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test) + (1 | dataset_family/dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test) + (1 | dataset_family), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test) + (1 | dataset_family/model), data=data))


# Modeling architecture and model_family
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test) + (1 | architecture), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test) + (1 | architecture) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test) + (1 | model_family), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test) + (1 | model_family) + (1 | dataset), data=data))




# Modeling test category
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | test_category/Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + test_category + (centered_total_compute | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | test_category) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | overall_test_category/Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + overall_test_category + (centered_total_compute | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | overall_test_category) + (1 | dataset), data=data))

# Modeling types of stimuli
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | word_category/Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | stimuli_type/Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | type_of_stimuli/Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | modality/Test) + (1 | dataset), data=data))

BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | word_category) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | stimuli_type) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | type_of_stimuli) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | modality) + (1 | dataset), data=data))

BIC(lmer(effect_size ~ 1 + centered_total_compute + word_category + (centered_total_compute | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + stimuli_type + (centered_total_compute | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + type_of_stimuli + (centered_total_compute | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_total_compute + modality +  (centered_total_compute | Test) + (1 | dataset), data=data))

BIC(lmer(effect_size ~ 1 + centered_total_compute + I(centered_total_compute**2)  + (centered_total_compute + I(centered_total_compute**2) | Test) + (1 | dataset), data=data))

# Fit best model found
best_mod = lmer(effect_size ~ 1 + centered_total_compute  + (centered_total_compute | Test) + (1 | dataset), data=data)
r.squaredGLMM(best_mod)

# Plot random effects
plt = plot_model(best_mod, type = "re", ci.lvl = 0.95)
plt[[2]] + ylab(TeX('$\\hat{\\delta}_{k}$')) + ggtitle('') +ylim(-0.35,0.35)
ggsave('results/plots/re.pdf', width=4.5, height=2.8)
ranef(best_mod)$dataset - se.ranef(best_mod)$dataset * qnorm(0.975)
ranef(best_mod)$dataset + se.ranef(best_mod)$dataset * qnorm(0.975)


# Plot correlations
corr_plot_data = coef(best_mod)$Test
corr_plot_data$Test = row.names(corr_plot_data)
corr_plot_data = corr_plot_data %>%
  tibble
bias_tests = data %>%
  distinct(Test, test_category, overall_test_category,
           modality, stimuli_type, word_category, type_of_stimuli)

data_together = left_join(corr_plot_data, bias_tests)

ggplot(data_together, aes(x=`(Intercept)`, y=centered_total_compute,
                          # color=type_of_stimuli
)) +
  xlab(TeX('$\\hat{\\alpha} + \\hat{\\beta}_{j[i]}$'))+
  ylab(TeX('$\\hat{\\gamma}_{j[i]}$')) +
  geom_point()

ggsave('results/plots/correlation_plot.pdf', width=4, height=2.8)


## Get bootstrap confidence intervals
# set.seed(3962435)
# boot_ci = confint(best_mod, method='boot', nsim=10000,oldNames=FALSE)
# saveRDS(best_ci, 'results/plots/eats_boot_ci.RDS')
boot_ci = readRDS('results/plots/eats_boot_ci.RDS')


# DIAGNOSTICS
# Full set
plot_model(best_mod, type = "diag")

# Linearity, homoscedasticity
ggplot(data.frame(y_hat=predict(best_mod),pearson=residuals(best_mod,type="pearson")),
       aes(x=y_hat,y=pearson)) +
  geom_point(size=2) +
  xlab('Fitted Value') +
  ylab('Residual')+
  theme_bw() +
  theme(text=element_text(size=20))

# Linearity
ggplot(data.frame(centered_total_compute=data$centered_total_compute,pearson=residuals(best_mod,type="pearson")),
      aes(x=centered_total_compute,y=pearson)) +
  geom_point(size=2) +
  xlab('Centered Total Compute') +
  ylab('Residsual')+
  theme_bw() +
  theme(text=element_text(size=20))

# Departure from normality
ggplot(data.frame(y=residuals(best_mod)), aes(sample=y)) +
  stat_qq() +
  stat_qq_line() +
  xlab('Theoretical Quantile') +
  ylab('Sample Quantile')+
  theme_bw() +
  theme(text=element_text(size=20))
ggsave('results/plots/full_model_qq.pdf',width=6,height=5)


# Fit bootstrap model
my_coefs <- function(.) {
  coefs = coef(.)
  intercepts = coefs$Test[[1]]
  names(intercepts) = paste(row.names(coefs$Test), '_intercept_coef',sep='')

  centered_total_computes = coefs$Test[[2]]
  names(centered_total_computes) = paste(row.names(coefs$Test), '_centered_total_compute_coef',sep='')

  datasets = coefs$dataset[[1]]
  names(datasets) = paste(row.names(coefs$dataset), '_intercept_coef',sep='')

  ranefs = ranef(.)
  ran_intercepts = ranefs$Test[[1]]
  names(ran_intercepts) = paste(row.names(ranefs$Test), '_intercept_ran',sep='')

  ran_centered_total_computes = ranefs$Test[[2]]
  names(ran_centered_total_computes) = paste(row.names(ranefs$Test), '_centered_total_compute_ran',sep='')

  ran_datasets = ranefs$dataset[[1]]
  names(ran_datasets) = paste(row.names(ranefs$dataset), '_intercept_ran',sep='')

  between_test_stds = attr(VarCorr(.)[[1]], 'stddev')
  names(between_test_stds) = paste(names(between_test_stds), '_std_test',sep='')

  between_dataset_stds = attr(VarCorr(.)[[2]], 'stddev')
  names(between_dataset_stds) = paste(names(between_dataset_stds), '_std_dataset',sep='')


  corrs = attr(VarCorr(.)[[1]],'correlation')[[2]]
  names(corrs) = c('intercept_ctc_corr')

  return(
  c(fixef(.), intercepts, centered_total_computes, datasets, ran_intercepts,
    ran_centered_total_computes, ran_datasets,
    between_test_stds, between_dataset_stds, corrs)
  )


}

# https://cran.r-project.org/web/packages/lmeresampler/vignettes/lmeresampler-vignette.html

library(lmeresampler)
# bootstrap_models = bootstrap(best_mod, type='residual', B = 10000, .f=my_coefs)
# saveRDS(bootstrap_models, 'results/plots/eats_bootstrap.RDS')
bootstrap_models = readRDS('results/plots/eats_bootstrap.RDS')


# Put bootstrap samples into dataframe
lowers = bootstrap_models$replicates %>%
  summarize_all(.funs = list(val=function(x) { mean(x) - qnorm(0.975) * sd(x) }))
uppers = bootstrap_models$replicates %>%
  summarize_all(.funs = list(val=function(x) { mean(x) + qnorm(0.975) * sd(x) }))
means = bootstrap_models$replicates %>%
  summarize_all(.funs = list(val=function(x) { mean(x) }))
ses = bootstrap_models$replicates %>%
  summarize_all(.funs = list(val=function(x) { sd(x) }))

df = bind_rows(lowers, means, uppers, ses)
df = as_tibble(cbind(nms = names(df), t(df)))
names(df)= c('item','lower','mean','upper','se')

# Differences with RB model: intercept variance component
original = attr(VarCorr(best_mod)$Test, 'stddev')[1]
comp = df %>% filter(item=='(Intercept)_std_test_val')
comp = as.double(comp['mean'][1])
(comp - original)/ original

# Differences with RB model: ctc variance component
original = attr(VarCorr(best_mod)$Test, 'stddev')[2]
comp = df %>% filter(item=='centered_total_compute_std_test_val')
comp = as.double(comp['mean'][1])
(comp - original)/ original

# Differences with RB model: intercept-ctc correlation
original = attr(VarCorr(best_mod)$Test, 'corr')[1,2]
comp = df %>% filter(item=='intercept_ctc_corr_val')
comp = as.double(comp['mean'][1])
(comp - original)/ original

# Differences with RB model: intercept variance component for dataset
original = attr(VarCorr(best_mod)$dataset, 'stddev')[1]
comp = df %>% filter(item=='(Intercept)_std_dataset_val')
comp = as.double(comp['mean'][1])
(comp - original)/ original


# Differences with RB model: Fixed effects
(as.double(df$mean[1:length(fixef(best_mod))]) - fixef(best_mod)) / fixef(best_mod)






