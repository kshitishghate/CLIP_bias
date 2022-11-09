import os

import torch
import nltk
import numpy as np
import pandas as pd

from scipy.special import comb
from CLIP import clip
from tqdm import tqdm
from nltk.corpus import wordnet2021, wordnet

from eats.extract_clip import load_images, extract_images, extract_text
from ieat.weat.weat.test import Test
from eats.result_saving import test_already_run, save_test_results


def get_word_list(word, pos):
    synsets = wordnet.synsets(word)
    synsets = [s for s in synsets if s.pos() == pos]

    words = []
    for syn in synsets:
        lemmas = [l.name() for l in syn.lemmas()]
        words += lemmas
        hypernyms = syn.hyponyms()
        for hyp in hypernyms:
            hypernym_lemmas = [l.name() for l in hyp.lemmas()]
            words += hypernym_lemmas

    words = np.unique(words)
    np.random.seed(36226086)
    np.random.shuffle(words)
    words = [w.replace('_',' ') for w in words]
    return words


def perform_test():
    nouns = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

    all_tests = pd.read_csv(os.path.join('data', 'tests.csv'))
    all_tests = all_tests[all_tests['A'] == 'Pleasant']

    templates = ['{}', 'a person who is feeling {}', 'a person who is conveying {}', 'a person who makes me feel {}']
    negation_templates = ['not {}', 'a person who is not feeling {}', 'a person who is not conveying {}', 'a person who does not make me feel {}']

    total = len(clip.available_models())* len(nouns) * 2 * 2 * len(all_tests)
    results_fp = os.path.join('results', 'data', 'categorical_synonyms_results.csv')
    if os.path.exists(results_fp):
        completed = len(pd.read_csv(results_fp))
    else:
        completed = 0
    remaining = total - completed

    with tqdm(total=remaining) as pbar:
        for model_name in clip.available_models():
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device = 'cpu'
            model, preprocess = clip.load(model_name, device)

            for emotion in nouns:
                for comparison_name in ['negation', 'apathy']:
                    for context, opposite_context in zip(templates, negation_templates):
                        phrases = [context.format(w) for w in get_word_list(emotion, 'n')]
                        if comparison_name == 'negation':
                            opposite_phrases = [opposite_context.format(w) for w in get_word_list(emotion, 'n')]
                        elif comparison_name == 'apathy':
                            opposite_phrases = [context.format(w) for w in get_word_list('apathy', 'n')]

                        phrases = phrases[:min(len(phrases), len(opposite_phrases))]
                        opposite_phrases = opposite_phrases[:min(len(phrases), len(opposite_phrases))]

                        np.random.seed(35436205)

                        for i, test in all_tests.iterrows():
                            test['A'] = context.format(emotion)
                            test['B'] = opposite_context.format(emotion) if comparison_name == 'negation' else context.format('apathy')
                            test['na'] = len(phrases)
                            test.name = None

                            if not test_already_run(model_name, test, results_fp):
                                np.random.seed(82804230)

                                stimuli = {
                                    'X': load_images(test['Title'], test['X']),
                                    'Y': load_images(test['Title'], test['Y']),
                                    emotion: phrases,
                                    comparison_name: opposite_phrases
                                }

                                embeddings = {
                                    'X': extract_images(model, preprocess, stimuli['X'], device, model_name),
                                    'Y': extract_images(model, preprocess, stimuli['Y'], device, model_name),
                                    emotion: extract_text(model, preprocess, stimuli[emotion], device, model_name),
                                    comparison_name: extract_text(model, preprocess, stimuli[comparison_name], device,
                                                                  model_name)
                                }

                                npermutations = min(
                                    1000000,
                                    int(comb(len(embeddings['X']) + len(embeddings['Y']), len(embeddings['X']))),

                                )

                                test_result = Test(embeddings['X'], embeddings['Y'], embeddings[emotion], embeddings[comparison_name])
                                test_result = pd.Series({
                                    'model':model_name,
                                    'pvalue':test_result.p(npermutations),
                                    'effect_size':test_result.effect_size(),
                                    'npermutations':npermutations,
                                })

                                save_test_results(pd.concat([test, test_result]), results_fp)
                                pbar.update()

if __name__ == '__main__':
    get_word_list('trust', 'n')
