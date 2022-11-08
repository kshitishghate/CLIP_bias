import os

import torch
import pandas as pd
from scipy.special import comb
from tqdm import tqdm

from CLIP import clip
from extract_clip import load_words, load_images, extract_images, extract_text
from ieat.weat.weat.test import Test
from result_saving import test_already_run, save_test_results


if __name__ == '__main__':
    all_tests = pd.read_csv(os.path.join('data','tests.csv'))
    all_tests = all_tests[all_tests['A'] == 'Pleasant']
    all_results = []

    total_to_run = len(clip.available_models() * len(all_tests))
    num_previously_run = len(pd.read_csv(os.path.join('results','data','results.csv')))
    remaining = total_to_run - num_previously_run

    with tqdm(total=remaining) as pbar:
        for model_name in clip.available_models():
            # Load model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(model_name, device)
            for i, test in all_tests.iterrows():
                if not test_already_run(model_name, test, os.path.join('results','data','categorical_results.csv')):

                    stimuli = {
                        'X': load_images(test['Title'], test['X']),
                        'Y': load_images(test['Title'], test['Y']),
                        'A': load_words(test['Title'], test['A'], test['na']),
                        'B': load_words(test['Title'], test['B'], test['na'])
                    }

                    embeddings = {
                        'X': extract_images(model, preprocess, stimuli['X'], device, model_name),
                        'Y': extract_images(model, preprocess, stimuli['Y'], device, model_name),
                        'A': extract_text(model, preprocess, stimuli['A'], device, model_name),
                        'B': extract_text(model, preprocess, stimuli['B'], device, model_name)
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
                        'npermutations':npermutations
                    })
                    save_test_results(pd.concat([test, test_result]),
                                      os.path.join('results','data','valence_results.csv')
                                      )

                    pbar.update(1)



