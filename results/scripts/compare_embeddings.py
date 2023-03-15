import os

from CLIP import clip
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cross_decomposition import CCA

import pandas as pd
import numpy as np

from results.scripts.plot_cross_modal import load_cross_modal_results
from eats.extract_clip import extract_text, extract_images
import statsmodels.formula.api as smf


def load_complete_embeddings():
    """Load all embeddings from the raw_embeddings folder."""
    all_models = clip.available_models()

    # load embeddings
    embeddings = []
    metadata = []
    for model_name in all_models:
        model_name = model_name.replace('/','')

        embd = pd.read_csv(os.path.join('results','data','raw_embeddings', model_name + '.csv'), sep='\t', header=None)
        embeddings.append(embd)

        mtd = pd.read_csv(os.path.join('results', 'data', 'raw_embeddings', model_name + '_metadata.csv'), sep='\t')
        metadata.append(mtd)
    embeddings = pd.concat(embeddings)
    metadata = pd.concat(metadata)
    embeddings = pd.concat([metadata, embeddings], axis=1)
    return embeddings


def get_cosine_sims(embeddings):
    """Calculate mean cosine similarity between image and text embeddings."""

    # calculate distances between image and text embeddings
    full_info = []
    for model_name in all_models:
        model, preprocess = clip.load(model_name, device='cpu')

        model_name = model_name.replace('/','')
        model_embeddings = embeddings[embeddings['model'] == model_name]


        for (image_test, text_test), test_embd in model_embeddings.groupby(['image_test','text_test']):
            concept_names = test_embd['stimuli_category'].unique().tolist()

            concept_embd = pd.DataFrame(
                extract_text(model, preprocess, concept_names, device='cpu',model_name=model_name),
                index = concept_names
            )

            image_embeddings = test_embd[test_embd['modality'] == 'image']
            text_embeddings = test_embd[test_embd['modality'] == 'text']

            # drop nan columns
            image_embeddings = image_embeddings.dropna(axis=1)
            text_embeddings = text_embeddings.dropna(axis=1)

            # Calculate distance to concept embedding
            embd_dim = concept_embd.shape[1]
            mean_image_sim_to_concept = cosine_similarity(
                image_embeddings.merge(concept_embd, left_on='stimuli_category',right_index=True)[[str(i) + '_x' for i in range(embd_dim)]],
                image_embeddings.merge(concept_embd, left_on='stimuli_category', right_index=True)[[str(i) + '_y' for i in range(embd_dim)]]
            ).trace() / len(image_embeddings)
            mean_text_sim_to_concept = cosine_similarity(
                text_embeddings.merge(concept_embd, left_on='stimuli_category', right_index=True)[
                    [str(i) + '_x' for i in range(embd_dim)]],
                text_embeddings.merge(concept_embd, left_on='stimuli_category', right_index=True)[
                    [str(i) + '_y' for i in range(embd_dim)]]
            ).trace() / len(text_embeddings)

            # Calculate mean pairwise cosine sim within each category
            all_internal_image_sims = []
            all_internal_text_sims = []
            for concept in concept_names:
                image_subset = image_embeddings[image_embeddings['stimuli_category'] == concept].drop(columns=['image_test', 'text_test', 'model', 'modality', 'stimuli_category', 'stimuli_name'])
                image_sims = cosine_similarity(image_subset, image_subset)
                all_internal_image_sims.append(image_sims[np.triu_indices(image_sims.shape[0], k=1)].mean())

                text_subset = text_embeddings[text_embeddings['stimuli_category'] == concept].drop(columns=['image_test', 'text_test', 'model', 'modality', 'stimuli_category', 'stimuli_name'])
                text_sims = cosine_similarity(text_subset, text_subset)
                all_internal_text_sims.append(text_sims[np.triu_indices(text_sims.shape[0], k=1)].mean())

            mean_internal_image_sim = np.mean(all_internal_image_sims)
            mean_internal_image_target_sims = np.mean(all_internal_image_sims[:2])
            mean_internal_image_attr_sims = np.mean(all_internal_image_sims[2:])
            mean_internal_text_sim = np.mean(all_internal_text_sims)
            mean_internal_text_target_sims = np.mean(all_internal_text_sims[:2])
            mean_internal_text_attr_sims = np.mean(all_internal_text_sims[2:])



            # calculate cosine similarity
            mean_sim = cosine_similarity(
                image_embeddings.drop(columns=['image_test', 'text_test', 'model', 'modality', 'stimuli_category', 'stimuli_name']),
                text_embeddings.drop(columns=['image_test', 'text_test', 'model', 'modality', 'stimuli_category', 'stimuli_name'])
            ).mean()

            # calculate mean image and text embeddings
            image_mns = image_embeddings.groupby('stimuli_category').mean()
            text_mns = text_embeddings.groupby('stimuli_category').mean()
            CCA(n_components=1).fit_transform(image_mns, text_mns)


            # calculate mean magnitude of image and text embeddings
            mean_image_mag = np.linalg.norm(image_embeddings.drop(columns=['image_test', 'text_test', 'model', 'modality', 'stimuli_category', 'stimuli_name']), axis=1).mean()
            mean_text_mag = np.linalg.norm(text_embeddings.drop(columns=['image_test', 'text_test', 'model', 'modality', 'stimuli_category', 'stimuli_name']), axis=1).mean()
            mean_mag = np.linalg.norm(test_embd.dropna(axis=1).drop(columns=['image_test', 'text_test', 'model', 'modality', 'stimuli_category', 'stimuli_name']), axis=1).mean()


            # Calculate

            full_info.append([model_name, image_test, text_test, mean_sim, mean_image_mag, mean_text_mag, mean_mag,
                              mean_image_sim_to_concept, mean_text_sim_to_concept,
                              mean_internal_image_sim, mean_internal_image_target_sims, mean_internal_image_attr_sims,
                              mean_internal_text_sim, mean_internal_text_target_sims, mean_internal_text_attr_sims,
                              all_internal_image_sims[0], all_internal_image_sims[1], all_internal_image_sims[2], all_internal_image_sims[3],
                              all_internal_text_sims[0], all_internal_text_sims[1], all_internal_text_sims[2], all_internal_text_sims[3]])

    return pd.DataFrame(full_info, columns=['model', 'image_test', 'text_test', 'mean_sim',
                                            'mean_image_mag', 'mean_text_mag', 'mean_mag',
                                            'mean_image_sim_to_concept','mean_text_sim_to_concept',
                                            'mean_internal_image_sim', 'mean_internal_image_target_sims', 'mean_internal_image_attr_sims',
                                            'mean_internal_text_sim', 'mean_internal_text_target_sims', 'mean_internal_text_attr_sims',
                                            'target_1_image_sim', 'target_2_image_sim', 'attr_1_image_sim', 'attr_2_image_sim',
                                            'target_1_text_sim', 'target_2_text_sim', 'attr_1_text_sim', 'attr_2_text_sim'])


def get_mean_differences(cosine_sims):
    """Get mean differences between text and image bias tests"""
    averaged_results, all_results = load_cross_modal_results()

    # Remove '/' from model names
    all_results = all_results.replace({'model': {'/': ''}}, regex=True)

    joined = cosine_sims.merge(all_results, left_on=['model', 'image_test', 'text_test'],
                               right_on=['model', 'Image_Test', 'Text_Test'], how='outer')

    model_info = pd.DataFrame({
        'model': ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B32", "ViT-B16", "ViT-L14", "ViT-L14@336px"],
        'number_of_parameters': [102007137, 119688033, 178300601, 290979217, 623258305, 151277313, 149620737, 427616513,
                                 427944193],
        'model_family': ["ResNet", "ResNet", "ResNet", "ResNet", "ResNet", "ViT", "ViT", "ViT", "ViT"],
        'embedding_dim': [1024, 512, 640, 768, 1024, 512, 512, 768, 768],
        'input_resolution': [224, 224, 288, 384, 448, 224, 224, 224, 336],
        'vision_transformer_width': [2048, 2048, 2560, 3072, 4096, 768, 768, 1024, 1024],
        'text_transformer_width': [512, 512, 640, 768, 1024, 512, 512, 768, 768],
        'text_transformer_heads': [8, 8, 10, 12, 16, 8, 8, 12, 12],
        'fer2014_performance': [0.642, 0.652, 0.681, 0.687, 0.713, 0.692, 0.695, 0.722, 0.729],
        'ucf101_performance':[0.816, 0.840, 0.857, 0.880, 0.895, 0.855, 0.884, 0.915, 0.920],
        'clevr_counts_performance':[0.536, 0.503, 0.525, 0.538, 0.550, 0.521, 0.571, 0.578, 0.603],
        'hateful_memes_performance':[0.657, 0.682, 0.680, 0.711, 0.750, 0.667, 0.703, 0.762, 0.773],
    })
    model_info['log10_num_parms'] = np.log10(model_info['number_of_parameters'])
    model_info['log2_num_parms'] = np.log2(model_info['number_of_parameters'])
    model_info['centeredlog2_num_parms'] = model_info['log2_num_parms'] - model_info['log2_num_parms'].min()

    joined[['image_test','text_test']].drop_duplicates()

    joined = joined.merge(model_info, on='model')

    joined['all_image_absolute_value'] = np.abs(joined['all_image'])
    joined['all_text_absolute_value'] = np.abs(joined['all_text'])

    return joined




if __name__ == '__main__':
    if os.path.exists(os.path.join('results', 'data', 'bias_and_covariates.csv')):
        print('Skipping data generation')
        mean_dif = pd.read_csv(os.path.join('results', 'data', 'bias_and_covariates.csv'))
    else:
        print('Generating data')
        all_models = clip.available_models()

        # load embeddings
        embeddings = load_complete_embeddings()

        # calculate cosine similarity
        cosine_sims = get_cosine_sims(embeddings)

        # load cross modal results
        mean_dif = get_mean_differences(cosine_sims)

        mean_dif.to_csv(os.path.join('results', 'data', 'bias_and_covariates.csv'), index=False)


    mean_dif['log10_num_parms'] = (mean_dif['log10_num_parms'] - mean_dif['log10_num_parms'].min()) / (
                mean_dif['log10_num_parms'].max() - mean_dif['log10_num_parms'].min())
    mean_dif['mean_text_sim_to_concept'] = (mean_dif['mean_text_sim_to_concept'] - mean_dif[
        'mean_text_sim_to_concept'].min()) / (mean_dif['mean_text_sim_to_concept'].max() - mean_dif[
        'mean_text_sim_to_concept'].min())

    mean_dif['mean_internal_text_target_sims'] = (mean_dif['mean_internal_text_target_sims'] - mean_dif[
        'mean_internal_text_target_sims'].min()) / (mean_dif['mean_internal_text_target_sims'].max() - mean_dif[
        'mean_internal_text_target_sims'].min())

    mean_dif['target_2_text_sim'] = (mean_dif['target_2_text_sim'] - mean_dif[
        'target_2_text_sim'].min()) / (mean_dif['target_2_text_sim'].max() - mean_dif[
        'target_2_text_sim'].min())

    mean_dif['mean_internal_text_sim'] = (mean_dif['mean_internal_text_sim'] - mean_dif[
        'mean_internal_text_sim'].min()) / (mean_dif['mean_internal_text_sim'].max() - mean_dif[
        'mean_internal_text_sim'].min())


    mean_dif['mean_internal_image_target_sims'] = (mean_dif['mean_internal_image_target_sims'] - mean_dif[
        'mean_internal_image_target_sims'].min()) / (mean_dif['mean_internal_image_target_sims'].max() - mean_dif[
        'mean_internal_image_target_sims'].min())

    mean_dif['target_2_image_sim'] = (mean_dif['target_2_image_sim'] - mean_dif[
        'target_2_image_sim'].min()) / (mean_dif['target_2_image_sim'].max() - mean_dif[
        'target_2_image_sim'].min())

    mean_dif['mean_internal_image_sim'] = (mean_dif['mean_internal_image_sim'] - mean_dif[
        'mean_internal_image_sim'].min()) / (mean_dif['mean_internal_image_sim'].max() - mean_dif[
        'mean_internal_image_sim'].min())


    mod = smf.ols('all_text ~ 0 + image_test ', data=mean_dif).fit()
    print(mod.bic)

    mod = smf.ols('all_text ~ 0 + image_test * log10_num_parms ', data=mean_dif).fit()
    print(mod.bic)

    mod = smf.ols('all_text ~ 0 + image_test * mean_internal_text_sim ', data=mean_dif).fit()
    print(mod.bic)

    mod = smf.ols('all_text ~ 0 + image_test * mean_internal_text_target_sims ', data=mean_dif).fit()
    print(mod.bic)

    mod = smf.ols('all_text ~ 0 + image_test * target_2_text_sim ', data=mean_dif).fit()
    print(mod.bic)



    image_data = mean_dif[['image_test','all_image_absolute_value', 'all_images', 'log10_num_parms', 'mean_internal_image_target_sims','target_2_image_sim', 'mean_internal_image_sim']].drop_duplicates()
    imod = smf.ols('all_images ~ 0 + image_test', data=image_data).fit()
    print(imod.bic)

    imod = smf.ols('all_images ~ 0 +image_test +  log10_num_parms ', data=image_data).fit()
    print(imod.bic)


    imod = smf.ols('all_images ~ 0 + image_test * log10_num_parms ', data=image_data).fit()
    print(imod.bic)



    cmod = smf.ols('image_text_dif ~ 0 + image_test', data=mean_dif).fit()
    print(cmod.bic)

    cmod = smf.ols('image_text_dif ~ 0 + image_test + log10_num_parms', data=mean_dif).fit()
    print(cmod.bic)

    cmod = smf.ols('image_text_dif ~ 0 + image_test * log10_num_parms', data=mean_dif).fit()
    print(cmod.bic)

    cmod = smf.ols('image_text_dif ~ 0 + image_test + model + all_images', data=mean_dif).fit()
    print(cmod.bic)

    cmod = smf.ols('image_text_dif ~ 0 + image_test + model + all_text', data=mean_dif).fit()
    print(cmod.bic)

    cmod = smf.ols('image_text_dif ~ 0 + image_test + model', data=mean_dif).fit()
    print(cmod.bic)
