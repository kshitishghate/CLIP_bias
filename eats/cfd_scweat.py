import os

import xml.etree.ElementTree as ET

import torch
import nltk
import numpy as np
import pandas as pd

from scipy.special import comb
from CLIP import clip
from tqdm import tqdm
from nltk.corpus import wordnet
from pattern.text.en import pluralize

from eats.extract_clip import extract_images, extract_text, load_words_greenwald
from eats.sc_weat import SCWEAT
from eats.result_saving import save_test_results


def test_already_run(test, file_name):
    if not os.path.exists(file_name):
        return False
    previous_results = pd.read_csv(file_name)
    relevant_results = previous_results[
        (previous_results['model'] == test['model'])
        & (previous_results['image_fp'] == test['image_fp'])
    ]
    return len(relevant_results) > 0


def get_all_image_paths():
    all_images = []
    for root, dir, files in os.walk(os.path.join('data','CFD Version 3.0', 'Images')):
        for f in files:
            if f.split('.')[-1] == 'jpg':
                all_images.append(os.path.join(root, f))
    return all_images


def perform_test(device):

    npermutations = 10000

    results_fp = os.path.join('results', 'data', 'cfd_scweat.csv')
    models = clip.available_models()
    all_image_paths = get_all_image_paths()

    # models.reverse()
    total = len(models) * len(all_image_paths)

    if os.path.exists(results_fp):
        completed = pd.read_csv(results_fp)
        completed = [completed['model'].str.contains(a) for a in models]
        completed = pd.concat(completed, axis=1).any(axis=1).sum()
    else:
        completed = 0

    remaining = total - completed

    with tqdm(total=remaining) as pbar:
        for model_name in models:
            model, preprocess = clip.load(model_name, device)

            for image_fp in all_image_paths:
                test = {'image_name': os.path.basename(image_fp),
                        'image_fp':image_fp,
                        'model': model_name}

                if not test_already_run(test, results_fp):
                    test['na'] = len(load_words_greenwald('pleasant'))

                    np.random.seed(5718980)

                    stimuli = {
                        'w': [image_fp],
                        'A': load_words_greenwald('pleasant'),
                        'B': load_words_greenwald('unpleasant'),
                    }

                    embeddings = {
                        'w': extract_images(model, preprocess, stimuli['w'], device, model_name),
                        'A': extract_text(model, preprocess, stimuli['A'], device, model_name),
                        'B': extract_text(model, preprocess, stimuli['B'], device, model_name),
                    }


                    test_result = SCWEAT(embeddings['w'], embeddings['A'], embeddings['B'])
                    test_result = pd.Series({
                        'association_score':test_result.association_score()[0],
                        'dist_to_pleasantness': test_result.mean_similarity('A')[0],
                        'dist_to_unpleasantness': test_result.mean_similarity('B')[0],
                        'dist_std': test_result.AB_std(),
                        'npermutations':npermutations,
                    })

                    save_test_results(pd.concat([pd.Series(test), test_result]), results_fp)
                    pbar.update()


if __name__ == '__main__':
    should_continue=True
    while should_continue:
        if os.path.exists(os.path.join('results','data','cfd_scweat.csv')):
            num_results_so_far = len(pd.read_csv(os.path.join('results','data','cfd_scweat.csv')))
        else:
            num_results_so_far = 0
        try:
            perform_test('cuda' if torch.cuda.is_available() else 'cpu')
        except:
            perform_test('cpu')
        if num_results_so_far == len(pd.read_csv(os.path.join('results','data','cfd_scweat.csv'))):
            should_continue = False
