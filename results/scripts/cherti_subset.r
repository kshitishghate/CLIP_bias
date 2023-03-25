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




subset = data %>%
  filter(model_source =='cherti')
subset$centered_samples_seen = log10(subset$samples_seen) - min(log10(subset$samples_seen))
subset$centered_total_compute = log10(subset$total_compute) - min(log10(subset$total_compute))
subset$centered_total_params = log10(subset$total_params_trained) - min(log10(subset$total_params_trained))
subset$centered_total_acts = log10(subset$total_acts) - min(log10(subset$total_acts))
subset$centered_text_macs = log10(subset$text_macs) - min(log10(subset$text_macs))
subset$centered_text_acts = log10(subset$acts) - min(log10(subset$text_acts))


c("model", "architecture", "model_family", "samples_seen", "model_name", "fine_tuned",
                   "epochs", "samples_per_epoch",
                    "image_size", "image_width", "text_width", "embed_dim",
                    "params",
                   "image_params", "text_params",
                   "macs", "image_macs", "text_macs",
                   "acts", "image_acts", "text_acts")



# Filter out any columns with na values
subset <- subset[,colSums(is.na(subset))==0]

ggplot(aes(x=effect_size), data=subset) +
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


# Create list of performance variables
performance_metrics = list()
for (nam in names(subset)){
  if (grepl('acc', nam, fixed = TRUE) ){
    performance_metrics = append(performance_metrics, nam)
  } else if (grepl('recall', nam, fixed = TRUE) ){
    performance_metrics = append(performance_metrics, nam)
  } else if (grepl('f1', nam, fixed = TRUE) ){
    performance_metrics = append(performance_metrics, nam)
  } else if (grepl('precision', nam, fixed = TRUE) ){
    performance_metrics = append(performance_metrics, nam)
}}

meaningful_metrics = c(
  "acc1vtab+","acc1vtab","acc1imagenetv2","acc1imagenet1k", "acc1fer2013","acc1imagenet-a",
 "acc1imagenet_robustness","acc1vtab/cifar100",
 "acc5vtab+","acc5vtab","acc5imagenetv2","acc5imagenet1k", "acc5fer2013","acc5imagenet-a",
 "acc5imagenet_robustness","acc5vtab/cifar100",
 "mean_per_class_recallvtab+","mean_per_class_recallvtab","mean_per_class_recallimagenetv2",
 "mean_per_class_recallimagenet1k", "mean_per_class_recallfer2013",
 "mean_per_class_recallimagenet-a",
 "mean_per_class_recallimagenet_robustness","mean_per_class_recallvtab/cifar100",
 "image_retrieval_recall@5flickr30k", "image_retrieval_recall@5mscoco_captions",
 "text_retrieval_recall@5flickr30k","text_retrieval_recall@5mscoco_captions"
)


# Performance
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft + (acc5fgvc_aircraft | Test), data=subset)) # Best for ANY single performance metric
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + (acc5fgvc_aircraft | Test), data=subset)) # Best for ANY two performance metrics
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | Test), data=subset)) # Best for a single meaningful performance metric (i.e. one that doesn't involve only cars/airplaces/etc.)
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + `acc1vtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | Test) , data=subset)) # Best for two meaningful performance metrics

# Model features, instead of performance
model_columns = c("model_family", "model")

model_features = c("architecture", "fine_tuned",
                   "epochs", "samples_per_epoch",
                    "image_size", "image_width", "text_width", "embed_dim",
                   "centered_params", "centered_macs", "centered_image_macs",
                   "centered_text_macs", "centered_image_acts",
                   "centered_text_acts", "centered_acts",
                   'centered_total_compute', 'centered_total_params',
                   "centered_total_acts", "centered_dataset_size" )

BIC(lmer(effect_size ~ 1 + centered_macs + (centered_macs | Test) + (1 | dataset), data=subset))
BIC(lmer(effect_size ~ 1 + centered_params + (centered_params | Test) + (1 | dataset), data=subset))

BIC(lmer(effect_size ~ 1 + centered_total_compute + (centered_total_compute | Test), data=subset))
BIC(lmer(effect_size ~ 1 + centered_total_params + (centered_total_params | Test), data=subset))


# Model as random intercept
r.squaredGLMM(lmer(effect_size ~ 1 + centered_total_params + (centered_total_params | Test) + (1 | model), data=subset))
r.squaredGLMM(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | Test)+ (1 | model), data=subset))
r.squaredGLMM(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + `acc1vtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | Test)+ (1 | model) , data=subset)) # Ignoring, as model does not converge
r.squaredGLMM(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + (acc5fgvc_aircraft | Test)+ (1 | model), data=subset)) # Best for ANY two performance metrics


# Using modality as third level
BIC(lmer(effect_size ~ 1 + centered_total_params + (centered_total_params | modality/Test) + (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | modality/Test)+ (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + (acc5fgvc_aircraft | modality/Test)+ (1 | model), data=subset))


# Using modality as fixed intercept
BIC(lmer(effect_size ~ 1 + centered_total_params + modality+ (centered_total_params | Test) + (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + modality+ (`mean_per_class_recallvtab/cifar100` | Test)+ (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + modality+(acc5fgvc_aircraft | Test)+ (1 | model), data=subset))


# Using stimuli type as third level
BIC(lmer(effect_size ~ 1 + centered_total_params + (centered_total_params | stimuli_type/Test) + (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | stimuli_type/Test)+ (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + (acc5fgvc_aircraft | stimuli_type/Test)+ (1 | model), data=subset))

# Using stimuli_type as fixed intercept
BIC(lmer(effect_size ~ 1 + centered_total_params + stimuli_type+ (centered_total_params | Test) + (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + stimuli_type+ (`mean_per_class_recallvtab/cifar100` | Test)+ (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + stimuli_type+(acc5fgvc_aircraft | Test)+ (1 | model), data=subset))



# Using type_of_stimuli as third level
BIC(lmer(effect_size ~ 1 + centered_total_params + (centered_total_params | type_of_stimuli/Test) + (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | type_of_stimuli/Test)+ (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + (acc5fgvc_aircraft | type_of_stimuli/Test)+ (1 | model), data=subset))


# Using overall_test_category as third level
BIC(lmer(effect_size ~ 1 + centered_total_params + (centered_total_params | overall_test_category/Test) + (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | overall_test_category/Test)+ (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + (acc5fgvc_aircraft | overall_test_category/Test)+ (1 | model), data=subset)) # Best for ANY two performance metrics


# Using test_category as third level
BIC(lmer(effect_size ~ 1 + centered_total_params + (centered_total_params | test_category/Test) + (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | test_category/Test)+ (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + (acc5fgvc_aircraft | test_category/Test)+ (1 | model), data=subset)) # Best for ANY two performance metrics


# Using dataset as third level
BIC(lmer(effect_size ~ 1 + centered_total_params + (centered_total_params | Test) + (1 | dataset/model), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | Test)+ (1 | dataset/model), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + (acc5fgvc_aircraft | Test)+ (1 | dataset/model), data=subset)) # Best for ANY two performance metrics



# Using dataset instead of model
BIC(lmer(effect_size ~ 1 + centered_total_params + (centered_total_params | Test) + (1 | dataset), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + (`mean_per_class_recallvtab/cifar100` | Test)+ (1 | dataset), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + (acc5fgvc_aircraft | Test)+ (1 | dataset), data=subset)) # Best for ANY two performance metrics


# Using dataset size instead of model
BIC(lmer(effect_size ~ 1 + centered_total_params + centered_dataset_size + (centered_total_params | Test), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + centered_dataset_size + (`mean_per_class_recallvtab/cifar100` | Test), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + centered_dataset_size + (acc5fgvc_aircraft | Test), data=subset)) # Best for ANY two performance metrics


# Using dataset size with model
BIC(lmer(effect_size ~ 1 + centered_total_params + centered_dataset_size + (centered_total_params | Test) + (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + `mean_per_class_recallvtab/cifar100` + centered_dataset_size + (`mean_per_class_recallvtab/cifar100` | Test) + (1 | model), data=subset))
BIC(lmer(effect_size ~ 1 + acc5fgvc_aircraft*acc5stl10 + centered_dataset_size + (acc5fgvc_aircraft | Test) + (1 | model), data=subset)) # Best for ANY two performance metrics
