import os

import xml.etree.ElementTree as ET

import torch
import nltk
import numpy as np
import pandas as pd
import json

from scipy.special import comb
import open_clip
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

def load_seat(test: str):
    with open(os.path.join('data', 'tests', test), 'r') as f:
        stimuli = json.load(f)

    stimuli = {
        'names': [stimuli['targ1']['category'], stimuli['targ2']['category'],
                  stimuli['attr1']['category'], stimuli['attr2']['category']],
        'X': stimuli['targ1']['examples'],
        'Y': stimuli['targ2']['examples'],
        'A': stimuli['attr1']['examples'],
        'B': stimuli['attr2']['examples']
    }
    assert(len(stimuli['X']) == len(stimuli['Y']))
    return stimuli


def perform_test(device):

    npermutations = 10000

    results_fp = os.path.join('results', 'data', 'cfd_7b_scseat.csv')
    models = open_clip.list_pretrained()
    # Not using convnext_xxlarge because it is not supported by timm 0.6.12
    models = [m for m in models if m[0] != 'convnext_xxlarge']

    all_image_paths = get_all_image_paths()

    # models.reverse()
    total = len(models) * len(all_image_paths)

    if os.path.exists(results_fp):
        model_name_strs = ['_'.join(m).replace('/', '') for m in models]
        completed = pd.read_csv(results_fp)
        completed = completed['model'].isin(model_name_strs).astype(int)
        completed = completed.sum()
    else:
        completed = 0

    remaining = total - completed

    a_stimuli = load_seat('sent-weat7b.jsonl')['X']
    b_stimuli = load_seat('sent-weat7b.jsonl')['Y']
    with tqdm(total=remaining) as pbar:
        for model_name in models:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1],
                                                                         device=device)
            tokenizer = open_clip.get_tokenizer(model_name[0])

            for image_fp in all_image_paths:
                test = {'image_name': os.path.basename(image_fp),
                        'image_fp':image_fp,
                        'model': '_'.join(model_name).replace('/', '')}

                if not test_already_run(test, results_fp):
                    test['na'] = len(a_stimuli)

                    np.random.seed(5718980)

                    stimuli = {
                        'w': [image_fp],
                        'A': a_stimuli,
                        'B': b_stimuli,
                    }

                    embeddings = {
                        'w': extract_images(model, preprocess, stimuli['w'], device, model_name),
                        'A': extract_text(model, tokenizer, stimuli['A'], device, model_name),
                        'B': extract_text(model, tokenizer, stimuli['B'], device, model_name),
                    }


                    test_result = SCWEAT(embeddings['w'], embeddings['A'], embeddings['B'])
                    test_result = pd.Series({
                        'association_score':test_result.association_score()[0],
                        'dist_to_math': test_result.mean_similarity('A')[0],
                        'dist_to_arts': test_result.mean_similarity('B')[0],
                        'dist_std': test_result.AB_std(),
                        'npermutations':npermutations,
                    })

                    save_test_results(pd.concat([pd.Series(test), test_result]), results_fp)
                    pbar.update()


if __name__ == '__main__':
    should_continue=True
    while should_continue:
        if os.path.exists(os.path.join('results','data','cfd_7b_scseat.csv')):
            num_results_so_far = len(pd.read_csv(os.path.join('results','data','cfd_7b_scseat.csv')))
        else:
            num_results_so_far = 0
        try:
            perform_test('cuda' if torch.cuda.is_available() else 'cpu')
        except:
            perform_test('cpu')
        if num_results_so_far == len(pd.read_csv(os.path.join('results','data','cfd_7b_scseat.csv'))):
            should_continue = False
