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
options(mc.cores = parallel::detectCores())



# Read in data
data = read_csv('results/data/data_for_modeling.csv')
data$centered_params = log10(data$params) - min(log10(data$params))
data$centered_macs = log10(data$macs) - min(log10(data$macs))
data$centered_image_macs = log10(data$image_macs) - min(log10(data$image_macs))
data$centered_image_acts = log10(data$image_acts) - min(log10(data$image_acts))
data$centered_acts = log10(data$acts) - min(log10(data$acts))
data$total_compute = data$macs / 1e9 * data$samples_seen # GMACs * samples_seen
data$total_params_trained = data$params / 1e9 * data$samples_seen # Gparams * samples_seen
data$total_acts = data$acts / 1e6 * data$samples_seen # macts * samples_seen
data$centered_dataset_size = log10(data$dataset_size) - min(log10(data$dataset_size))
data$type_of_stimuli = paste(data$word_category, data$stimuli_type, sep='/')





# Filter out any columns with na values
data <- data[,colSums(is.na(data))==0]

ggplot(aes(x=effect_size), data=data) +
  geom_histogram(color='white', binwidth=0.25) +
  xlab('Effect Size') +
  ylab('Count')

ggsave('results/plots/r/overall.pdf')

ggplot(aes(x=effect_size), data=data) +
  geom_histogram(aes(fill=modality),color='white', binwidth=0.25, position='dodge') +
  xlab('Effect Size') +
  ylab('Count')

ggsave('project_v2/modality.pdf')

data %>%
  ggplot(aes(x=effect_size), data=.) +
  geom_histogram(aes(fill=overall_test_category),color='white', binwidth=0.25, position='dodge') +
  xlab('Effect Size') +
  ylab('Count')

ggsave('project_v2/bias_category.pdf')


data %>%
  ggplot(aes(x=effect_size), data=.) +
  geom_histogram(aes(fill=model_family),color='white', binwidth=0.25, position='dodge') +
  xlab('Effect Size') +
  ylab('Count')

ggsave('project_v2/model_family.pdf')



# Base model
BIC(lmer(effect_size ~ 1 + (1 | Test) + (1 | model), data=data))

# With random slopes for num params
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | model), data=data))

# With fixed slope for num params
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | model), data=data))

# Using dataset as a third level
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | dataset/model), data=data))

# Using dataset instead of model
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | dataset), data=data))

# Using assumption of uncorrelated slopes/intercepts
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params || Test) + (1 | dataset), data=data))

# Modeling dataset size
BIC(lmer(effect_size ~ 1 + centered_params + dataset_size + (centered_params | Test) + (1 | dataset), data=data))

# Modeling dataset family
BIC(lmer(effect_size ~ 1 + centered_params + dataset_family + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | dataset_family/dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | dataset_family), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | dataset_family/model), data=data))


# Modeling macs instead of num params
BIC(lmer(effect_size ~ 1 + centered_acts + (centered_acts | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_acts*centered_params + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_acts + centered_params + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_acts*centered_params + (centered_acts*centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_acts + centered_params + (centered_acts + centered_params | Test) + (1 | dataset), data=data))

# Modeling acts instead of num params
BIC(lmer(effect_size ~ 1 + centered_macs + (centered_macs | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_macs*centered_params + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_macs + centered_params + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_macs*centered_params + (centered_macs*centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_macs + centered_params + (centered_macs + centered_params | Test) + (1 | dataset), data=data))


# Modeling architecture and model_family
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | architecture), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | architecture) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | model_family), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | model_family) + (1 | dataset), data=data))

# Modeling test category
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | test_category/Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + test_category + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | test_category) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | overall_test_category/Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + overall_test_category + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | overall_test_category) + (1 | dataset), data=data))


# Modeling types of stimuli
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | word_category/Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | stimuli_type/Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | type_of_stimuli/Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | modality/Test) + (1 | dataset), data=data))

BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | word_category) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | stimuli_type) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | type_of_stimuli) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | modality) + (1 | dataset), data=data))

BIC(lmer(effect_size ~ 1 + centered_params + word_category + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + stimuli_type + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + type_of_stimuli + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params + modality +  (centered_params | Test) + (1 | dataset), data=data))

BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params*stimuli_type + (centered_params | Test) + (1 | dataset), data=data))
BIC(lmer(effect_size ~ 1 + centered_params*type_of_stimuli + (centered_params*type_of_stimuli | Test) + (1 | dataset), data=data))


# Modeling fine tuned
best_mod = lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | dataset), data=data)
var(data$effect_size)
summary(best_mod)
r.squaredGLMM(best_mod)

