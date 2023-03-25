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



# Read in data
data = read_csv('results/data/bimodal_data_for_modeling.csv')
data$centered_params = log10(data$params) - min(log10(data$params))
data$centered_image_params = log10(data$image_params) - min(log10(data$image_params))
data$centered_text_params = log10(data$text_params) - min(log10(data$text_params))
data$centered_macs = log10(data$macs) - min(log10(data$macs))
data$centered_gmacs = (data$macs / 1e9) - min((data$macs / 1e9))
data$centered_image_macs = log10(data$image_macs) - min(log10(data$image_macs))
data$centered_image_acts = log10(data$image_acts) - min(log10(data$image_acts))
data$centered_acts = log10(data$acts) - min(log10(data$acts))
data$total_compute = data$macs / 1e9 * data$samples_seen # GMACs * samples_seen
data$total_params_trained = data$params / 1e9 * data$samples_seen # Gparams * samples_seen
data$total_acts = data$acts / 1e6 * data$samples_seen # macts * samples_seen
data$centered_dataset_size = log10(data$dataset_size) - min(log10(data$dataset_size))





# Filter out any columns with na values
data <- data[,colSums(is.na(data))==0]

ggplot(aes(x=image_text_dif), data=data) +
  geom_histogram(color='white', bins=25) +
  xlab('Difference Between Image D and Text D') +
  ylab('Count')



data %>%
  ggplot(aes(x=image_text_dif), data=.) +
  geom_histogram(aes(fill=dataset_family),color='white', binwidth=0.25, position='dodge') +
  xlab('Effect Size') +
  ylab('Count')


data %>%
  ggplot(aes(x=image_text_dif), data=.) +
  geom_histogram(aes(fill=model_family),color='white', binwidth=0.25, position='dodge') +
  xlab('Effect Size') +
  ylab('Count')

ggsave('project_v2/model_family.pdf')


# Base model
BIC(glmer(image_text_dif ~ 1 + (1 | model), family=Gamma(link='log'), data=data))

# test category
BIC(glmer(image_text_dif ~ 1 + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))

# test category
BIC(glmer(image_text_dif ~ 1 + (1 | overall_test_category/Image_Test) + (1 | model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ 1 + overall_test_category+ (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ 1 + (1 | overall_test_category) + (1 | model), family=Gamma(link='log'), data=data))

# Dataset instead of model
BIC(glmer(image_text_dif ~ 1 + (1 | Image_Test) + (1 | dataset), family=Gamma(link='log'), data=data))

# Dataset size
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))

# Num params
BIC(glmer(image_text_dif ~ centered_params + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_params*centered_dataset_size + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_params + centered_dataset_size + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))

# Macs
BIC(glmer(image_text_dif ~ centered_macs + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_macs*centered_dataset_size + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_macs + centered_dataset_size + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))

# macs, varying by test
BIC(glmer(image_text_dif ~ centered_macs + (centered_macs | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_macs*centered_dataset_size + (centered_macs | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_macs + centered_dataset_size + (centered_macs | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))

# params, varying by test
BIC(glmer(image_text_dif ~ centered_params + (centered_params | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_params*centered_dataset_size + (centered_params | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_params + centered_dataset_size + (centered_params | Image_Test) + (1 | model), family=Gamma(link='log'), data=data))


# Architecture/model_family
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | model_family/architecture/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | model_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | model_family/architecture), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | model_family) + (1 | dataset), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | architecture)+ (1 | dataset), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | model_family) + (1 | dataset_family), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | architecture)+ (1 | dataset_family), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | model_family) + (1 | dataset_family/dataset), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | architecture)+ (1 | dataset_family/dataset), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | model_family), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | architecture), family=Gamma(link='log'), data=data))

# Dataset/dataset_family
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | dataset), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | dataset_family), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | dataset_family/dataset), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | dataset/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~ centered_dataset_size + (1 | Image_Test) + (1 | dataset_family/dataset/model), family=Gamma(link='log'), data=data))


# Without dataset size
BIC(glmer(image_text_dif ~  (1 | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  (1 | Image_Test) + (1 | dataset/model), family=Gamma(link='log'), data=data))


# centered_params/macs/acts
BIC(glmer(image_text_dif ~  centered_params + (1 | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_params + (centered_params | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_macs + (1 | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_macs + (centered_macs | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_acts + (1 | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_acts + (centered_acts | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))

# Image/text params
BIC(glmer(image_text_dif ~  centered_image_params + (centered_image_params | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_image_params + (1 | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_text_params + (centered_text_params | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_text_params + (1 | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_text_params + centered_image_params+ (1 | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_text_params + centered_image_params+ (centered_text_params | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_text_params + centered_image_params+ (centered_image_params | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))
BIC(glmer(image_text_dif ~  centered_text_params + centered_image_params+ (centered_text_params + centered_image_params | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data))




best_mod = glmer(image_text_dif ~ centered_macs + (centered_macs | Image_Test) + (1 | dataset_family/model), family=Gamma(link='log'), data=data)
# https://stats.stackexchange.com/questions/45401/how-to-validate-diagnose-a-gamma-glm-in-r


# Residual plot
plot(best_mod)
# Does not indicate non-constant variance

# qqplot
plot_model(best_mod, 'diag')
# Does not indicate departure from L2 normality, (since ci intercepts line)


# Plot fixed effects
plot_model(best_mod, 'eff') +
  scale_x_log10(
    breaks = scales::trans_breaks("log10", function(x) 10^x),
   labels = scales::trans_format("log10", scales::math_format(10^.x))
 )


# CI for fixed effects in best mod
all_ci = confint(best_mod, method='boot', boot.type='norm' )

lb = fixef(best_mod) + se.fixef(best_mod) * qnorm(0.025)
fe = fixef(best_mod)
ub = fixef(best_mod) + se.fixef(best_mod) * qnorm(0.975)

exp(lb)
exp(fe)
exp(ub)


plot_model(best_mod, 're')

# CI for standard deviations/correlations
sjplot(best_mod)


# CI for coefficients in best mod


# Plot of coefficients (with CI) for coefficients
lb = exp(coef(best_mod)$Image_Test + se.ranef(best_mod)$Image_Test * qnorm(0.025))
coef = exp(coef(best_mod)$Image_Test)
ub = exp(coef(best_mod)$Image_Test + se.ranef(best_mod)$Image_Test * qnorm(0.975))

uncenter = function(x) {10 ** (x + log10(min(data$macs)))}

