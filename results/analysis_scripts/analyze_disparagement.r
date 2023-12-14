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
options(mc.cores = parallel::detectCores())
library(DHARMa)
library(lmeresampler)

# Read in data
data = read_csv('results/data/complete_slur_distance_results.csv')
data$relative_distance = data$similarity_with_5_most_common - data$similarity_with_controls
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

data$emotion = case_when(
  str_sub(data$image_fp, -5, -5) == 'N' ~ 'Neutral',
  str_sub(data$image_fp, -5, -5) == 'A' ~ 'Anger',
  str_sub(data$image_fp, -5, -5) == 'S' ~ 'Surprise',
  str_sub(data$image_fp, -5, -5) == 'F' ~ 'Fear',
  str_sub(data$image_fp, -6, -5) == 'HO' ~ 'Happiness (Open Mouth)',
  str_sub(data$image_fp, -6, -5) == 'HC' ~ 'Happiness (Closed Mouth)',
)

data <- data[, colSums(is.na(data)) == 0]


BIC(lmer(relative_distance ~ (1 | dataset), data = data))


BIC(lmer(relative_distance ~ centered_total_compute + (1 | dataset), data = data))

BIC(lmer(relative_distance ~ centered_total_compute +
  (1 | image_fp) +
  (1 | dataset), data = data))

BIC(lmer(relative_distance ~ centered_total_compute +
  race +
  (1 | image_fp) +
  (1 | dataset), data = data))
BIC(lmer(relative_distance ~ centered_total_compute +
  gender +
  (1 | image_fp) +
  (1 | dataset), data = data))
BIC(lmer(relative_distance ~ centered_total_compute +
  emotion +
  (1 | image_fp) +
  (1 | dataset), data = data))
BIC(lmer(relative_distance ~ centered_total_compute +
  race +
  gender +
  (1 | image_fp) +
  (1 | dataset), data = data))
BIC(lmer(relative_distance ~ centered_total_compute +
  race +
  emotion +
  (1 | image_fp) +
  (1 | dataset), data = data))
BIC(lmer(relative_distance ~ centered_total_compute +
  emotion +
  gender +
  (1 | image_fp) +
  (1 | dataset), data = data))
BIC(lmer(relative_distance ~ centered_total_compute +
  race +
  gender +
  emotion +
  (1 | image_fp) +
  (1 | dataset), data = data))

BIC(lmer(relative_distance ~ 1 +
  centered_total_compute * race +
  gender +
  emotion +
  (1 | image_fp) +
  (1 | dataset), data = data))
BIC(lmer(relative_distance ~ 1 +
  centered_total_compute * race +
  gender +
  emotion +
  (1 | image_fp) +
  (race | dataset), data = data))


best_mod = lmer(relative_distance ~ 1 +
  centered_total_compute * race +
  gender +
  emotion +
  (1 | image_fp) +
  (race | dataset), data = data)

# Save bootstrap CI
set.seed(3934140)
# boot_ci = confint(best_mod, method = 'boot', nsim = 10000, oldNames = FALSE)
# saveRDS(boot_ci, 'results/plots/disparagement_boot_ci.RDS')
boot_ci = readRDS('results/plots/disparagement_boot_ci.RDS')


# DIAGNOSTICS
# Linearity, homoscedasticity
ggplot(data.frame(y_hat = predict(best_mod), pearson = residuals(best_mod, type = "pearson")),
       aes(x = y_hat, y = pearson)) +
  geom_point(size = 2) +
  xlab('Fitted Value') +
  ylab('Residual') +
  theme_bw() +
  theme(text = element_text(size = 20))


# Linearity
ggplot(data.frame(centered_total_compute = data$centered_total_compute, pearson = residuals(best_mod, type = "pearson")),
       aes(x = centered_total_compute, y = pearson)) +
  geom_point(size = 2) +
  xlab('Centered Total Compute') +
  ylab('Residsual') +
  theme_bw() +
  theme(text = element_text(size = 20))

# Departure from normality
ggplot(data.frame(y = residuals(best_mod)), aes(sample = y)) +
  stat_qq() +
  stat_qq_line() +
  xlab('Theoretical Quantile') +
  ylab('Sample Quantile') +
  theme_bw() +
  theme(text = element_text(size = 20))


my_coefs <- function(.) {
  coefs = coef(.)
  intercepts = coefs$dataset[[1]]
  names(intercepts) = paste(row.names(coefs$dataset), '_intercept_coef', sep = '')

  white_centered_total_compute = fixef(.)['centered_total_compute'] + fixef(.)['centered_total_compute:raceWhite']
  names(white_centered_total_compute) = 'white_centered_total_compute'

  races = coefs$dataset[[3]]
  names(races) = paste(row.names(coefs$dataset), '_race_coef', sep = '')

  ranefs = ranef(.)
  ran_intercepts = ranefs$dataset[[1]]
  names(ran_intercepts) = paste(row.names(ranefs$dataset), '_intercept_ranef', sep = '')

  ran_races = ranefs$dataset[[2]]
  names(ran_races) = paste(row.names(ranefs$dataset), '_race_ranef', sep = '')

  between_image_stds = attr(VarCorr(.)[[1]], 'stddev')
  names(between_image_stds) = paste(names(between_image_stds), '_std_image', sep = '')

  between_dataset_stds = attr(VarCorr(.)[[2]], 'stddev')
  names(between_dataset_stds) = paste(names(between_dataset_stds), '_std_dataset', sep = '')


  return(
    c(fixef(.),
      intercepts, white_centered_total_compute, races,
      ran_intercepts, ran_races,
      between_image_stds, between_dataset_stds)
  )

}

set.seed(902865)
boostrap_models = bootstrap(best_mod, type = 'residual', B = 10000, .f = my_coefs)
saveRDS(boostrap_models, 'results/plots/disparagement_bootstrap.RDS')
boostrap_models = readRDS('results/plots/disparagement_bootstrap.RDS')

lowers = boostrap_models$replicates %>%
  summarize_all(.funs = list(val = function(x) { mean(x) - qnorm(0.975) * sd(x) }))
uppers = boostrap_models$replicates %>%
  summarize_all(.funs = list(val = function(x) { mean(x) + qnorm(0.975) * sd(x) }))
means = boostrap_models$replicates %>%
  summarize_all(.funs = list(val = function(x) { mean(x) }))
ses = boostrap_models$replicates %>%
  summarize_all(.funs = list(val = function(x) { sd(x) }))

df = bind_rows(lowers, means, uppers, ses)
df = as_tibble(cbind(nms = names(df), t(df)))
names(df) = c('item', 'lower', 'mean', 'upper', 'se')


# RB comparison: variance components of image sum contrasts
original = attr(VarCorr(best_mod)$image_fp, 'stddev')[1]
comp = df %>% filter(item == '(Intercept)_std_image_val')
comp = as.double(comp['mean'][1])
(comp - original) / original

# RB comparison: variance component of dataset sum contrasts
original = attr(VarCorr(best_mod)$dataset, 'stddev')[1]
comp = df %>% filter(item == '(Intercept)_std_dataset_val')
comp = as.double(comp['mean'][1])
(comp - original) / original

# RB comparison: variance component of race variable
original = attr(VarCorr(best_mod)$dataset, 'stddev')[2]
comp = df %>% filter(item == 'raceWhite_std_dataset_val')
comp = as.double(comp['mean'][1])
(comp - original) / original

# Fixed effects
(as.double(df$mean[1:length(fixef(best_mod))]) - fixef(best_mod)) / fixef(best_mod)


plt = plot_model(best_mod)
plt + ylim(-0.01, 0.01)