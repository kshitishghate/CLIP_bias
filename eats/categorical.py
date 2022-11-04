import os

import torch
import numpy as np
import pandas as pd

from scipy.special import comb
from CLIP import clip
from tqdm import tqdm

from eats.extract_clip import load_images, extract_images, extract_text
from ieat.weat.weat.test import Test
from eats.result_saving import test_already_run, save_test_results

def perform_test(hold_out_pos_negative = True):
    nrc = pd.read_csv(os.path.join('data','NRC-Emotion-Lexicon', 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'),
                      sep='\t', header=None, names=['word','category','rating'])
    nrc = nrc.pivot_table(values='rating', columns='category', index='word')
    if hold_out_pos_negative:
        nrc['num_associations'] = (
                nrc['anger']
                + nrc['anticipation']
                + nrc['disgust']
                + nrc['fear']
                + nrc['joy']
                + nrc['sadness']
                + nrc['surprise']
                + nrc['trust']
        )
    else:
        nrc['num_associations'] = (
                nrc['anger']
                + nrc['anticipation']
                + nrc['disgust']
                + nrc['fear']
                + nrc['joy']
                + nrc['sadness']
                + nrc['surprise']
                + nrc['trust']
                + nrc['positive']
                + nrc['negative']
        )

    single_emotion_words = nrc[nrc['num_associations'] == 1]
    no_emotion_words = nrc[nrc['num_associations'] == 0]

    all_tests = pd.read_csv(os.path.join('data','tests.csv'))
    all_tests = all_tests[all_tests['A'] == 'Pleasant']

    total_to_run = len(clip.available_models() * len(all_tests) * (single_emotion_words.shape[1]-1))
    previously_run = pd.read_csv(os.path.join('results','data','categorical_results.csv'))
    num_previously_run = (previously_run['positive_negative_excluded'] == hold_out_pos_negative).sum()
    remaining = total_to_run - num_previously_run


    with tqdm(total=remaining) as pbar:
        for model_name in clip.available_models():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load(model_name, device)

            for emotion in single_emotion_words.columns[:-1]:
                emotion_words = single_emotion_words[single_emotion_words[emotion] == 1]
                emotion_words = emotion_words.index.tolist()
                np.random.seed(35436205)
                no_emotion_words_sample = no_emotion_words.sample(len(emotion_words), replace=False).index.tolist()


                for i, test in all_tests.iterrows():
                    test['A'] = emotion
                    test['B'] = 'no_emotion'
                    test['na'] = len(emotion_words)
                    test.name = None

                    if not test_already_run(model_name, test, os.path.join('results','data','categorical_results.csv')):
                        np.random.seed(82804230)

                        stimuli = {
                            'X': load_images(test['Title'], test['X']),
                            'Y': load_images(test['Title'], test['Y']),
                            emotion: emotion_words,
                            'no_emotion': no_emotion_words_sample
                        }

                        embeddings = {
                            'X': extract_images(model, preprocess, stimuli['X'], device),
                            'Y': extract_images(model, preprocess, stimuli['Y'], device),
                            emotion: extract_text(model, preprocess, stimuli[emotion], device),
                            'no_emotion': extract_text(model, preprocess, stimuli['no_emotion'], device)
                        }

                        npermutations = min(
                            1000000,
                            int(comb(len(embeddings['X']) + len(embeddings['Y']), len(embeddings['X']))),

                        )

                        test_result = Test(embeddings['X'], embeddings['Y'], embeddings[emotion], embeddings['no_emotion'])
                        test_result = pd.Series({
                            'model':model_name,
                            'pvalue':test_result.p(npermutations),
                            'effect_size':test_result.effect_size(),
                            'npermutations':npermutations,
                            'positive_negative_excluded': hold_out_pos_negative
                        })

                        save_test_results(pd.concat([test, test_result]),
                                          os.path.join('results','data','categorical_results.csv')
                                          )
                        pbar.update(1)


if __name__ == '__main__':
    perform_test(True)
