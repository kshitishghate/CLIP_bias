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

if __name__ == '__main__':
    all_synonyms = []
    for emotion in ['anger',
         'anticipation',
         'disgust',
         'fear',
         'joy',
         'sadness',
         'surprise',
         'trust']:
        word_list = get_word_list(emotion, 'n')

        association_template = (
            'How much is {word} ' + f'associated with the emotion {emotion}? \n'
            'Enter 1 if {word} ' + f' is not associated with {emotion}, \n'
            'Enter 2 if {word} ' + f' is weakly associated with {emotion}, \n'
            'Enter 3 if {word} ' + f' is moderately associated with {emotion}, or \n'
            'Enter 4 if {word} ' + f' is strongly associated with {emotion}'
        )
        emotion_template = (
            'Is {word} an emotion? (For example: love is an emotion; shark is associated with fear (an emotion),\n'
            ' but shark is not an emotion.)\n'
            ' Enter 1 if {word} is not an emotion, or \n'
            ' Enter 2 if {word} is an emotion\n'
        )




        all_synonyms.append(pd.DataFrame({
            'emotion': emotion,
            'synonym': word_list,
            'q1': [association_template.format(word=w) for w in word_list],
            'r1': np.nan,
            'q2': [emotion_template.format(word=w) for w in word_list],
            'r2': np.nan
        }))

    all_synonyms = pd.concat(all_synonyms).reset_index(drop=True)
    np.random.seed(54283318)
    for i in range(10):
        all_synonyms = all_synonyms.sample(frac=1, replace=False)
        all_synonyms = all_synonyms.reset_index(drop=True)
        os.makedirs(os.path.join('data','synonyms'), exist_ok=True)
        all_synonyms.to_csv(os.path.join('data','synonyms',f'synonym_list_{i}.csv'),index=False)



