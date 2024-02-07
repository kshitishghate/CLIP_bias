import sys
import os
import random
import pickle
import logging

from typing import Union, Tuple, Callable, List

# taken from references/debias-vision-lang/debias_clip/measuring_bias.py
import torch
import torch.utils.data
from torch.utils.data import DataLoader

import open_clip

sys.path.append('references/debias-vision-lang/debias_clip')
sys.path.append('references/debias-vision-lang')
sys.path.append('references/CLIP')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debias_clip.measuring_bias import gen_prompts, get_labels_img_embeddings, get_prompt_embeddings, eval_ranking
from debias_clip.datasets import IATDataset, FairFace
from debias_clip.model.model import ClipLike, model_loader


from measure_bias_and_performance.utils import cherti_et_al_ckpts, cherti_et_al_models


DEFAULT_OPTS = dict()


def measure_bias(cliplike: ClipLike, img_preproc: Callable, tokenizer: Callable, attribute="gender", opts=DEFAULT_OPTS, device="cpu"):
    # do measurement
    ds = FairFace(mode="val", iat_type=attribute, transforms=img_preproc)
    dl = DataLoader(ds, batch_size=256, num_workers=6)

    prompts: List[str] = gen_prompts()

    evals = "maxskew", "ndkl"

    device = torch.device(device)
    labels_list, image_embeddings = get_labels_img_embeddings(dl, cliplike, device, progress=True)
    prompts_embeddings = get_prompt_embeddings(cliplike, tokenizer, device, prompts)

    result = {}
    for evaluation in evals:
        result[evaluation] = (
            eval_ranking(labels_list, image_embeddings, prompts_embeddings, evaluation,
                         topn=1000 if evaluation == "maxskew" else 1.0)
        )

    return result


if __name__ == "__main__":
    save_dir = 'results/ndkl_maxskew'
    os.makedirs(save_dir, exist_ok=True)
    model_list_path = os.path.join(save_dir, 'models_to_ndkl_maxskew.pkl')

    log_path = os.path.join(save_dir, 'ndkl_maxskew.log')
    logging.basicConfig(filename=log_path, filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    models_on_this_device = open_clip.list_pretrained() + cherti_et_al_models()
    # Not using convnext_xxlarge because it is not supported by timm 0.6.12
    models_on_this_device = [m for m in models_on_this_device if m[0] != 'convnext_xxlarge']

    if os.path.exists(model_list_path):
        with open(model_list_path, 'rb') as f:
            models = pickle.load(f)
    else:
        models = models_on_this_device
        with open(model_list_path, 'wb') as f:
            pickle.dump(models, f)

    models = [m for m in models if m in models_on_this_device]

    models = random.sample(models, k = len(models))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    while len(models) > 0:
        model_name = models.pop()
        model, _, img_preprocess = open_clip.create_model_and_transforms(
            model_name[0],
            pretrained=model_name[1],
            device=device,
            cache_dir='references/scaling-laws-openclip' if '.pt' in model_name[1] else None        )
        tokenizer = open_clip.get_tokenizer(model_name[0])
        model.eval()

        bias_results = measure_bias(
                  model,
                  img_preprocess,
                  tokenizer,
                  attribute="race",
                  device=device)
        # measure bias, lower == less biased
        logging.info([model_name, bias_results])

        # Save progress
        with open(model_list_path, 'rb') as f:
            models = pickle.load(f)
        models = [m for m in models if not (m[0] == model_name[0] and m[1] == model_name[1])]
        with open(model_list_path, 'wb') as f:
            pickle.dump(models, f)

        # Save model results dictionary
        try:
            mname1 = model_name[1]
            if 'references/scaling-laws-openclip' in mname1:
                mname1 = fname.replace('references/scaling-laws-openclip/', '')
            with open(
                    os.path.join(
                        save_dir,
                        f'{model_name[0]}_{mname1}.pkl'), 'wb') as f:
                pickle.dump(bias_results, f)
        except Exception as e:
            print(e)
            print('Could not save results for', model_name)
            continue

