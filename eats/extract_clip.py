import os
import numpy as np

import pandas as pd
import torch
from PIL import Image

from CLIP import clip
from pathlib import Path

global image_embedding_dict
image_embedding_dict = {}
global text_embedding_dict
text_embedding_dict = {}
global image_dict
image_dict = {}


def clear_dict(dict, model_name):
    keys = list(dict.keys())
    for k in keys:
        if k[0] != model_name:
            del dict[k]
def load_words(test, category, nwords=None):
    if category == 'Pleasant':
        all_words = pd.read_csv(os.path.join('ieat', 'data', 'bgb_pleasant-words.csv'))
        words = all_words.sort_values(['pleasantness']).tail(nwords)['word'].tolist()
    elif category == 'Unpleasant':
        all_words = pd.read_csv(os.path.join('ieat', 'data', 'bgb_pleasant-words.csv'))
        words = all_words.sort_values(['pleasantness']).head(nwords)['word'].tolist()
    return words

def load_words_greenwald(category):
    if category == 'pleasant':
        words = pd.read_csv(os.path.join('data', 'greenwald_words', 'pleasant.csv'), header=None)[0].tolist()
    elif category == 'unpleasant':
        words = pd.read_csv(os.path.join('data', 'greenwald_words', 'unpleasant.csv'), header=None)[0].tolist()
    return words


def load_images(test, category, dataset='ieat'):
    if (test, category, dataset) in image_dict.keys():
        return image_dict[(test, category, dataset)]
    if dataset == 'ieat':
        image_dir = os.path.join('ieat', 'data', 'experiments', test.lower(), category.lower())
        image_paths = [os.path.join(image_dir, n) for n in os.listdir(image_dir)]
    elif dataset=='cfd':
        codebook = pd.read_excel(os.path.join('data','CFD Version 3.0','CFD 3.0 Norming Data and Codebook.xlsx'),
                                 sheet_name = 'CFD U.S. Norming Data',
                                 header=7,
                                 engine='openpyxl')


        if test == 'Gender':
            sample_size = codebook.groupby(['EthnicitySelf','GenderSelf'])['Model'].count().min()
            relevant_models = codebook.groupby(['EthnicitySelf','GenderSelf']).sample(sample_size)
            gender_map = {'Male':'M','Female':'F'}
            relevant_models = relevant_models[relevant_models['GenderSelf'] == gender_map[category]]['Model'].tolist()

        elif test in ['Asian','Race']:
            ethnicity_map = {'European-American': 'W', 'African-American': 'B', 'Asian-American': 'A'}
            both_ethnicities_map = {'Asian': ['W', 'A'], 'Race': ['B','W']}

            both_ethnicities = codebook[codebook['EthnicitySelf'].isin(both_ethnicities_map[test])]
            sample_size = both_ethnicities.groupby(['EthnicitySelf','GenderSelf'])['Model'].count().min()

            relevant_models = codebook[codebook['EthnicitySelf'] == ethnicity_map[category]]
            np.random.seed(6471043)
            relevant_models = relevant_models.groupby('GenderSelf').sample(sample_size)['Model'].tolist()

        image_dirs = [os.path.join('data','CFD Version 3.0','Images','CFD', m) for m in relevant_models]
        image_paths = []
        for dir in image_dirs:
            # Only get neutral images
            possible_images = [p for p in os.listdir(dir)if p.replace('.jpg','').split('-')[-1] == 'N']
            if len(possible_images) != 1:
                raise ValueError
            image_paths.append(os.path.join(dir,possible_images[0]))

    elif dataset == 'fairface':
        codebook = pd.read_csv(Path('data') / 'fairface_label_train.csv')

        if test == 'Gender':
            sample_size = int(min(codebook.groupby(['race','gender'])['file'].count().min(),60 * 10 / len(codebook[['race']].drop_duplicates()))) + 1
            relevant_models = codebook.groupby(['race','gender']).sample(sample_size)
            gender_map = {'Male':'Male','Female':'Female'}
            relevant_models = relevant_models[relevant_models['gender'] == gender_map[category]]['file'].tolist()

        elif test in ['Asian','Race']:
            ethnicity_map = {'European-American': ['White'],
                            'African-American': ['Black'],
                            'Asian-American': ['East Asian', 'Southeast Asian']}
            both_ethnicities_map = {'Asian': ['White', 'East Asian', 'Southeast Asian'],'Race':['Black','White']}

            both_ethnicities = codebook[codebook['race'].isin(both_ethnicities_map[test])]
            sample_size = int(min(both_ethnicities.groupby(['race','gender'])['file'].count().min(), 60 * 10 / 2))

            relevant_models = codebook[codebook['race'].isin(ethnicity_map[category])]
            np.random.seed(6471043)
            relevant_models = relevant_models.groupby('gender').sample(sample_size)['file'].tolist()

        image_paths = [os.path.join('data','fairface-img-margin025-trainval',m) for m in relevant_models]

    image_dict[(test, category, dataset)] = image_paths
    return image_paths


def extract_images(model, preprocess, image_paths, device, model_name):
    image_features = []
    if (model_name, tuple(image_paths)) not in image_embedding_dict.keys():
        for i in image_paths:
            image = preprocess(Image.open(i)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features.append(model.encode_image(image))
        image_features = torch.stack(image_features).squeeze().cpu().detach().numpy()
        image_embedding_dict[(model_name, tuple(image_paths))] = image_features
        clear_dict(image_embedding_dict, model_name)
    else:
        image_features = image_embedding_dict[(model_name, tuple(image_paths))]
    if len(image_features.shape) == 1:
        image_features = image_features.reshape(1, -1)
    return image_features


def extract_text(model, preprocess, text, device, model_name):
    try:
        text = np.sort(text)
    except np.AxisError:
        pass
    if (model_name, tuple(text)) not in text_embedding_dict.keys():
        processed_text = clip.tokenize(text).to(device)
        with torch.no_grad():
           text_features = model.encode_text(processed_text).cpu().detach().numpy()
        text_embedding_dict[(model_name, tuple(text))] = text_features
        clear_dict(text_embedding_dict, model_name)
    else:
        text_features = text_embedding_dict[(model_name, tuple(text))]
    return text_features