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
data = read_csv('results/data/bimodal_data_for_modeling.csv')
data$centered_params = log10(data$params) - min(log10(data$params))
data$centered_image_params = log10(data$image_params) - min(log10(data$image_params))
data$centered_text_params = log10(data$text_params) - min(log10(data$text_params))
data$centered_macs = log10(data$macs) - min(log10(data$macs))
data$centered_image_macs = log10(data$image_macs) - min(log10(data$image_macs))
data$centered_image_acts = log10(data$image_acts) - min(log10(data$image_acts))
data$centered_acts = log10(data$acts) - min(log10(data$acts))
data$total_compute = data$macs / 1e9 * data$samples_seen # GMACs * samples_seen
data$total_params_trained = data$params / 1e9 * data$samples_seen # Gparams * samples_seen
data$total_acts = data$acts / 1e6 * data$samples_seen # macts * samples_seen
data$centered_dataset_size = log10(data$dataset_size) - min(log10(data$dataset_size))



subset = data %>%
  filter(model_source =='cherti')
subset$centered_samples_seen = log10(subset$samples_seen) - min(log10(subset$samples_seen))
subset$centered_total_compute = log10(subset$total_compute) - min(log10(subset$total_compute))
subset$centered_total_params = log10(subset$total_params_trained) - min(log10(subset$total_params_trained))
subset$centered_total_acts = log10(subset$total_acts) - min(log10(subset$total_acts))
subset$centered_text_macs = log10(subset$text_macs) - min(log10(subset$text_macs))
subset$centered_text_acts = log10(subset$acts) - min(log10(subset$text_acts))



# Filter out any columns with na values
subset <- subset[,colSums(is.na(data))==0]

ggplot(aes(x=image_text_dif), data=subset) +
  geom_histogram(color='white', binwidth=0.1) +
  xlab('Effect Size') +
  ylab('Count')



subset %>%
  ggplot(aes(x=image_text_dif), data=.) +
  geom_histogram(aes(fill=overall_test_category),color='white', binwidth=0.25, position='dodge') +
  xlab('Effect Size') +
  ylab('Count')


data %>%
  ggplot(aes(x=image_text_dif), data=.) +
  geom_histogram(aes(fill=model_family),color='white', binwidth=0.25, position='dodge') +
  xlab('Effect Size') +
  ylab('Count')



# Base model
BIC(glmer(image_text_dif ~ 1 + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))

# Does the training dataset matter? -> not here
BIC(glmer(image_text_dif ~ 1 + (1 | Image_Test) + (1 | dataset/model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ 1 + centered_dataset_size + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ 1 + centered_dataset_size + (1 | Image_Test) + (1 | dataset), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ 1 + (1 | Image_Test) + (1 | dataset), family=Gamma(link='log'), data=subset))

# Does the model size matter? -> no
BIC(glmer(image_text_dif ~ centered_macs + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ centered_macs + (centered_macs | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ centered_params + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ centered_params + (centered_params | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))

# Does model architecture have an effect? -> no
BIC(glmer(image_text_dif ~ 1 + (1 | Image_Test) + (1 | architecture/model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ architecture + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))

# Does model performance have an effect? -> no
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

best_bic = BIC(glmer(image_text_dif ~ 1 + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
for (metr in meaningful_metrics) {
  fmla1 = paste0("image_text_dif ~ `", metr, "` + (1 | Image_Test) + (1 | model)")
  bic = BIC(glmer(as.formula(fmla1), family=Gamma(link='log'), data=subset))
  if (bic < best_bic) {
    print(paste0("BIC: ", bic, " for ", metr, " (shared slope)"))
  }

  fmla2 = paste0("image_text_dif ~ `", metr, "` + (`", metr, "` | Image_Test) + (1 | model)")
  bic = BIC(glmer(as.formula(fmla2), family=Gamma(link='log'), data=subset))
  if (bic < best_bic) {
    print(paste0("BIC: ", bic, " for ", metr, " (fixed slope)"))
  }

}

# Does total compute have an effect? -> Also, no
BIC(glmer(image_text_dif ~ centered_total_compute + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ centered_total_compute + (centered_total_compute | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ centered_total_params + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ centered_total_params + (centered_total_params | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ centered_total_acts + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
BIC(glmer(image_text_dif ~ centered_total_acts + (centered_total_acts | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))


best_mod = (glmer(image_text_dif ~ 1 + (1 | Image_Test) + (1 | model), family=Gamma(link='log'), data=subset))
marginal_effects(best_mod, "Image_Test")







# Plot of