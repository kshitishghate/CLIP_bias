import os
import pandas as pd
import argparse
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._errors import EntryNotFoundError

trained_models_info= pd.read_csv("scaling-laws-openclip/trained_models_info.csv")

# Download epochs
def download_intermediate_ckpts(filename):
    # Make dirs
    hf_dir = 'full_checkpoints/' + filename.replace('.pt', '')
    os.makedirs(os.path.join('scaling-laws-openclip', os.path.normpath(hf_dir)), exist_ok=True)
    epochs_found = 0
    for i in range(300):
        epoch_filename = os.path.normpath(hf_dir + f'/epoch_{i}.pt')
        try:
            hf_hub_download("laion/scaling-laws-openclip", hf_dir + f'/epoch_{i}.pt',
                            cache_dir="scaling-laws-openclip",
                            force_filename=epoch_filename)
            epochs_found += 1
        except EntryNotFoundError:
            print(f'epoch {i} not found for model {filename}')
    print(f'{epochs_found} epochs found for model {filename}')


def download_model(model, samples_seen, dataset):
    res = trained_models_info[
        (trained_models_info.arch==model) & (trained_models_info.samples_seen_pretty==samples_seen) & (trained_models_info.data==dataset)
    ]
    if len(res) == 1:
        filename = res.name.tolist()[0]
        hf_hub_download("laion/scaling-laws-openclip", filename, cache_dir="scaling-laws-openclip", force_filename=filename)
        print(f"'{filename}' downloaded.")
        download_intermediate_ckpts(filename)
        return filename
    else:
        print("The model is not available in the repository")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to download pre-trained models")
    parser.add_argument("--samples_seen", type=str, nargs="+", default=["3B", "13B", "34B"])
    parser.add_argument("--dataset", type=str, nargs="+",default=["80M", "400M", "2B"])
    parser.add_argument("--model", type=str, nargs="+", default=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-H-14", "ViT-g-14"])
    args = parser.parse_args()
    for model in args.model:
        for samples_seen in args.samples_seen:
            for dataset in args.dataset:
                download_model(model, samples_seen, dataset)
