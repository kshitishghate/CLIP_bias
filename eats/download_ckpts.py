import os
import pandas as pd
import argparse
from huggingface_hub import hf_hub_download, list_files_info
from huggingface_hub.utils._errors import EntryNotFoundError

trained_models_info= pd.read_csv("scaling-laws-openclip/trained_models_info.csv")


def download_intermediate_ckpt(epoch_filename):
    hf_dir = os.path.dirname(epoch_filename)
    epoch_num = int(epoch_filename.split('epoch_')[1].split('.pt')[0])
    hf_hub_download("laion/scaling-laws-openclip", hf_dir + f'/epoch_{epoch_num}.pt',
                    cache_dir="scaling-laws-openclip",
                    force_filename=epoch_filename)

def download_intermediate_ckpts(filename):
    # Make dirs
    hf_dir = 'full_checkpoints/' + filename.replace('.pt', '')
    os.makedirs(os.path.join('scaling-laws-openclip', os.path.normpath(hf_dir)), exist_ok=True)
    all_epoch_files_for_model = [f.path for f in list_files_info("laion/scaling-laws-openclip")
                 if f.path.endswith('.pt') and 'full_checkpoints' in f.path
                 and 'epoch_' in f.path and 'latest' not in f.path
                 and filename.replace('.pt', '') in f.path]
    print(f"{len(all_epoch_files_for_model)} epochs found for model {filename}")

    # Download epochs
    for epoch_filename in all_epoch_files_for_model:
        download_intermediate_ckpt(epoch_filename)
    return

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
