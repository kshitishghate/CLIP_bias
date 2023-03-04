import os

import torch
import nltk
import numpy as np
import pandas as pd

from scipy.special import comb
from CLIP import clip
from tqdm import tqdm


from eats.extract_clip import load_images, extract_images, extract_text
from ieat.weat.weat.test import Test
from eats.result_saving import test_already_run, save_test_results



def perform_test():
    all_tests = pd.read_csv(os.path.join('data', 'ieat_tests.csv'))

    total = len(clip.available_models()) * len(all_tests)
    results_fp = os.path.join('results', 'data', 'ieat_replication.csv')
    if os.path.exists(results_fp):
        completed = len(pd.read_csv(results_fp))
    else:
        completed = 0
    remaining = total - completed

    with tqdm(total=remaining) as pbar:
        for model_name in clip.available_models():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # device =  "mps"
            model, preprocess = clip.load(model_name, device)

            for i, test in all_tests.iterrows():
                #test['A'] = context.format(emotion)
                #test['B'] = opposite_context.format(emotion) if comparison_name == 'negation' else context.format('apathy')
                if not test_already_run(model_name, test, results_fp):
                    np.random.seed(82804230)

                    attr_folder = test['Attribute'] if test['Attribute'] == 'Valence' else test['Target']

                    stimuli = {
                        'X': load_images(test['Target'], test['X']),
                        'Y': load_images(test['Target'], test['Y']),
                        'A': load_images(attr_folder, test['A']),
                        'B': load_images(attr_folder, test['B'])
                    }
                    assert len(stimuli['X']) == test['nt'] and len(stimuli['Y']) == test['nt']
                    assert len(stimuli['A']) == test['na'] and len(stimuli['B']) == test['na']

                    embeddings = {
                        'X': extract_images(model, preprocess, stimuli['X'], device, model_name),
                        'Y': extract_images(model, preprocess, stimuli['Y'], device, model_name),
                        'A': extract_images(model, preprocess, stimuli['A'], device, model_name),
                        'B': extract_images(model, preprocess, stimuli['B'], device, model_name),
                    }

                    npermutations = min(
                        1000000,
                        int(comb(len(embeddings['X']) + len(embeddings['Y']), len(embeddings['X']))),

                    )

                    test_result = Test(embeddings['X'], embeddings['Y'], embeddings['A'], embeddings['B'])
                    test_result = pd.Series({
                        'model':model_name,
                        'pvalue':test_result.p(npermutations),
                        'effect_size':test_result.effect_size(),
                        'npermutations':npermutations,
                    })

                    save_test_results(pd.concat([test, test_result]), results_fp)
                    pbar.update()

if __name__ == '__main__':
    perform_test()
