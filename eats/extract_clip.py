import os

import pandas as pd
import torch
from PIL import Image

from CLIP import clip


def load_words(test, category, nwords=None):
    if category == 'Pleasant':
        all_words = pd.read_csv(os.path.join('ieat', 'data', 'bgb_pleasant-words.csv'))
        words = all_words.sort_values(['pleasantness']).tail(nwords)['word'].tolist()
    elif category == 'Unpleasant':
        all_words = pd.read_csv(os.path.join('ieat', 'data', 'bgb_pleasant-words.csv'))
        words = all_words.sort_values(['pleasantness']).head(nwords)['word'].tolist()
    return words


def load_images(test, category):
    image_dir = os.path.join('ieat', 'data', 'experiments', test.lower(), category.lower())
    image_paths = [os.path.join(image_dir, n) for n in os.listdir(image_dir)]
    return image_paths


def extract_images(model, preprocess, image_paths, device):
    image_features = []
    for i in image_paths:
        image = preprocess(Image.open(i)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features.append(model.encode_image(image))
    image_features = torch.stack(image_features).squeeze().cpu().detach().numpy()
    return image_features


def extract_text(model, preprocess, text, device):
    processed_text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = model.encode_text(processed_text).cpu().detach().numpy()
    return text_features
