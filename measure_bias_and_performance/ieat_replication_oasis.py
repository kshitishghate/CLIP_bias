import os
import random
import json
import torch
import numpy as np
import pandas as pd
import open_clip
from tqdm import tqdm
from scipy.special import comb

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measure_bias_and_performance.extract_clip import load_images, extract_images, extract_text
from measure_bias_and_performance.sc_weat import WEAT
from measure_bias_and_performance.utils import test_already_run, save_test_results, cherti_et_al_ckpts, cherti_et_al_models

def load_seat(test: str):
    with open(os.path.join('data', 'tests', test), 'r') as f:
        stimuli = json.load(f)
    
    return {
        'names': [stimuli['targ1']['category'], stimuli['targ2']['category'],
                  stimuli['attr1']['category'], stimuli['attr2']['category']],
        'X': stimuli['targ1']['examples'],
        'Y': stimuli['targ2']['examples'],
        'A': stimuli['attr1']['examples'],
        'B': stimuli['attr2']['examples']
    }

def perform_test():
    all_tests = pd.read_csv(os.path.join('data', 'cross_modal_controlled.csv'))
    all_tests = all_tests.sample(frac=1, replace=False).reset_index(drop=True)

    models = open_clip.list_pretrained()
    models = [m for m in models if m[0] != 'convnext_xxlarge']
    models = random.sample(models, k=len(models))
    total = len(models) * len(all_tests)

    results_fp = os.path.join('results', 'data', 'cross_modal_tests_results_v0.csv')
    if os.path.exists(results_fp):
        model_name_strs = ['_'.join(m).replace('/', '') for m in models]
        completed = pd.read_csv(results_fp)
        completed = completed['model'].isin(model_name_strs).astype(int)
        completed = completed.sum()
    else:
        completed = 0
    remaining = total - completed

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tqdm(total=remaining, smoothing=0) as pbar:
        for model_name in models:
            model, preprocess, tokenizer = None, None, None

            for i, test in all_tests.iterrows():
                if not test_already_run('_'.join(model_name).replace('/', ''), test, results_fp):
                    if model is None:
                        model, _, preprocess = open_clip.create_model_and_transforms(model_name[0],
                                                                                     pretrained=model_name[1],
                                                                                     device=device)
                        tokenizer = open_clip.get_tokenizer(model_name[0])

                    np.random.seed(82804230)

                    attr_folder = test['Attribute'] if test['Attribute'] == 'Valence' else test['Target']

                    # Load both image and text stimuli
                    stimuli_image = {
                        'X': load_images(test['Target'], test['X']),
                        'Y': load_images(test['Target'], test['Y']),
                        'A': load_images(attr_folder, test['A']),
                        'B': load_images(attr_folder, test['B'])
                    }

                    stimuli_text = load_seat(test['test_name'])

                    # Extract embeddings for both image and text stimuli
                    embeddings = {
                        'X_image': extract_images(model, preprocess, stimuli_image['X'], device, model_name),
                        'Y_image': extract_images(model, preprocess, stimuli_image['Y'], device, model_name),
                        'A_image': extract_images(model, preprocess, stimuli_image['A'], device, model_name),
                        'B_image': extract_images(model, preprocess, stimuli_image['B'], device, model_name),
                        'X_text': extract_text(model, tokenizer, stimuli_text['X'], device, model_name),
                        'Y_text': extract_text(model, tokenizer, stimuli_text['Y'], device, model_name),
                        'A_text': extract_text(model, tokenizer, stimuli_text['A'], device, model_name),
                        'B_text': extract_text(model, tokenizer, stimuli_text['B'], device, model_name)
                    }

                    npermutations = min(
                        1000000,
                        int(comb(len(embeddings['X_image']) + len(embeddings['Y_image']), len(embeddings['X_image']))),
                    )

                    # Perform 4 different WEATs
                    weat_combinations = [
                        ('all_image', 'X_image', 'Y_image', 'A_image', 'B_image'),
                        ('all_text', 'X_text', 'Y_text', 'A_text', 'B_text'),
                        ('image_target_text_attr', 'X_image', 'Y_image', 'A_text', 'B_text'),
                        ('text_target_image_attr', 'X_text', 'Y_text', 'A_image', 'B_image')
                    ]

                    for weat_type, x, y, a, b in weat_combinations:
                        test_result = WEAT(embeddings[x], embeddings[y], embeddings[a], embeddings[b])
                        test_result = pd.Series({
                            'model': '_'.join(model_name).replace('/', ''),
                            'weat_type': weat_type,
                            'pvalue': test_result.p(npermutations),
                            'effect_size': test_result.effect_size(),
                            'npermutations': npermutations,
                        })

                        save_test_results(pd.concat([test, test_result]), results_fp)
                    pbar.update()

            test_already_run('_'.join(model_name).replace('/', ''), test, results_fp, hard_reload=True)

if __name__ == '__main__':
    perform_test()
