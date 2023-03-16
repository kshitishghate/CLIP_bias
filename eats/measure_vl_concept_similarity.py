import os
import json

import torch
import nltk
import numpy as np
import pandas as pd

from scipy.special import comb
import open_clip
from tqdm import tqdm



from eats.extract_clip import load_images, extract_images, extract_text
from ieat.weat.weat.test import Test
from eats.result_saving import test_already_run, save_test_results


def load_seat(test: str, name: str):
    with open(os.path.join('data', 'tests', test), 'r') as f:
        stimuli = json.load(f)

    accessible_stimuli = {
        stimuli['targ1']['category']: stimuli['targ1']['examples'],
        stimuli['targ2']['category']: stimuli['targ2']['examples'],
        stimuli['attr1']['category']: stimuli['attr1']['examples'],
        stimuli['attr2']['category']: stimuli['attr2']['examples']
    }
    assert(len(accessible_stimuli[stimuli['targ1']['category']]) == len(accessible_stimuli[stimuli['targ2']['category']]))
    return accessible_stimuli[name]

def perform_test():
    all_tests = pd.read_csv(os.path.join('data', 'cross_modal_tests.csv'))

    models = open_clip.list_pretrained()
    # Not using convnext_xxlarge yet because it is not supported by timm 0.6.12
    models = [m for m in models if m[0] != 'convnext_xxlarge']

    total = len(models) * len(all_tests) * 2
    results_fp = os.path.join('results', 'data', 'vl_concept_similarity.csv')
    if os.path.exists(results_fp):
        model_name_strs = ['_'.join(m).replace('/', '') for m in models]
        completed = pd.read_csv(results_fp)
        completed = completed['model'].isin(model_name_strs).astype(int)
        completed = completed.sum()
    else:
        completed = 0
    remaining = total - completed

    with tqdm(total=remaining) as pbar:
        for model_name in models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # device =  "mps"
            model, _, preprocess = open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1],
                                                                         device=device)
            tokenizer = open_clip.get_tokenizer(model_name[0])

            for i, test in all_tests.iterrows():
                for category in ['target','attribute']:
                    test['category'] = category
                    if not test_already_run('_'.join(model_name).replace('/', ''), test, results_fp):
                        np.random.seed(82804230)

                        # Note that we can't use text as the target, because not all text attribute pairs have the
                        # same number of examples. The results should be roughly equivalent, though.
                        stimuli = {
                            'X': load_images(test[f'image_{category}_dir'], test[f'image_{category}_1']),
                            'Y': load_images(test[f'image_{category}_dir'], test[f'image_{category}_2']),
                            'A': load_seat(test['text_file'], test[f'text_{category}_1']),
                            'B': load_seat(test['text_file'], test[f'text_{category}_2'])
                        }
                        assert(len(stimuli['X']) == len(stimuli['Y']))
                        test['nt'] = len(stimuli['X'])
                        test['naa'] = len(stimuli['A'])
                        test['nab'] = len(stimuli['B'])

                        embeddings = {
                            'X': extract_images(model, preprocess, stimuli['X'], device, model_name),
                            'Y': extract_images(model, preprocess, stimuli['Y'], device, model_name),
                            'A': extract_text(model, tokenizer, stimuli['A'], device, model_name),
                            'B': extract_text(model, tokenizer, stimuli['B'], device, model_name),
                        }

                        npermutations = min(
                            1000000,
                            int(comb(len(embeddings['X']) + len(embeddings['Y']), len(embeddings['X']))),
                        )

                        test_result = Test(embeddings['X'], embeddings['Y'], embeddings['A'], embeddings['B'])
                        test_result = pd.Series({
                            'model':'_'.join(model_name).replace('/', ''),
                            'pvalue':test_result.p(npermutations),
                            'effect_size':test_result.effect_size(),
                            'npermutations':npermutations,
                        })

                        save_test_results(pd.concat([test, test_result]), results_fp)
                        pbar.update()

if __name__ == '__main__':
    perform_test()