# Intrinsic Bias in Vision-Language Encoders

This repository contains the code and supporting data for the paper "Intrinsic Bias is Predicted by Pretraining Data and Correlates with Downstream Performance in Vision-Language Encoders", accepted to NAACL Oral, 2025.

## Repository Structure

- `data/`: Contains supporting text data to perform the EAT experiments
- `measure_bias_and_performance/`: Contains scripts to get bias test results using the `perform_cross_model*.py` scripts
- `experiments/`: Contains scripts to post-process bias data and perform analysis and visualization
- `results/data/`: Contains final data for modeling after running bias tests and post-processing
- `references/`: Contains external data dependencies (requires manual setup)
- `sbatch_jobs/`: Contains SLURM job scripts for running experiments on HPC clusters

## Setup Instructions

### 1. Environment Setup

The repository provides environment configuration files for different platforms:
- `mac_environment.yml`: For macOS
- `ubuntu_environment.yml`: For Ubuntu
- `requirements.txt`: For pip-based installation

Choose the appropriate environment file and create a new conda environment:
```bash
conda env create -f [mac|ubuntu]_environment.yml
conda activate [environment_name]
```

### 2. External Dependencies

#### OASIS and NRC-VAD Data
1. Download OASIS [data](https://osf.io/6pnd7/) and place it in `references/oasis/`
2. Download NRC-VAD lexicon [data](https://saifmohammad.com/WebPages/nrc-vad.html) and place it in `references/nrc_vad/`
3. Run the OASIS preprocessing script to generate the stimuli used in our experiments:
```bash
python references/oasis/oasis.py
```

#### iEAT Repository
1. Clone the iEAT repository:
```bash
git clone https://github.com/ryansteed/ieat.git
cd ieat
git checkout 4d7639bfd0b6c17b127ce163af0ddf985f5a6510
```

2. Replace the experiments directory:
- Copy `ieat/data/experiments` to `ieat/data/experiments_old` and `ieat/data/experiments_oasis`
- Replace the iEAT valence in `ieat/data/experiments_oasis` stimuli with pleasant and unpleasant Oasis stimuli generated in the previous step. 

#### OpenCLIP Repository
1. Clone the OpenCLIP repository:
```bash
git clone https://github.com/LAION-AI/scaling-laws-openclip.git
cd scaling-laws-openclip
git checkout 27d9e6afaeb3cb68334b16f6a649ac3d19879303
```


## Running Experiments

### 1. Bias Measurement
The main bias measurement scripts are in `measure_bias_and_performance/`:
- `perform_cross_modal.py`: Main script for cross-modal bias measurement
- `perform_cross_modal_for_cherti.py`: Script for measuring bias in Cherti models

### 2. Analysis and Visualization
The analysis and visualization scripts are in `experiments/`:
- `data_post_processing.ipynb`: Notebook for post-processing bias data
- `analysis_and_visualization.ipynb`: Notebook for analysis and visualization of results

## Results

The final processed data for modeling is available in `results/data/`. This data is generated after running the bias tests and post-processing steps.

## Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@inproceedings{ghate-etal-2025-intrinsic,
    title = "Intrinsic Bias is Predicted by Pretraining Data and Correlates with Downstream Performance in Vision-Language Encoders",
    author = "Ghate, Kshitish  and
      Slaughter, Isaac  and
      Wilson, Kyra  and
      Diab, Mona T.  and
      Caliskan, Aylin",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.148/",
    pages = "2899--2915",
    ISBN = "979-8-89176-189-6",
    abstract = "While recent work has found that vision-language models trained under the Contrastive Language Image Pre-training (CLIP) framework contain intrinsic social biases, the extent to which different upstream pre-training features of the framework relate to these biases, and hence how intrinsic bias and downstream performance are connected has been unclear. In this work, we present the largest comprehensive analysis to-date of how the upstream pre-training factors and downstream performance of CLIP models relate to their intrinsic biases. Studying 131 unique CLIP models, trained on 26 datasets, using 55 architectures, and in a variety of sizes, we evaluate bias in each model using 26 well-established unimodal and cross-modal principled Embedding Association Tests. We find that the choice of pre-training dataset is the most significant upstream predictor of bias, whereas architectural variations have minimal impact. Additionally, datasets curated using sophisticated filtering techniques aimed at enhancing downstream model performance tend to be associated with higher levels of intrinsic bias. Finally, we observe that intrinsic bias is often significantly correlated with downstream performance ($0.3 \leq r \leq 0.8$), suggesting that models optimized for performance inadvertently learn to amplify representational biases. Comparisons between unimodal and cross-modal association tests reveal that social group bias depends heavily on the modality. Our findings imply that more sophisticated strategies are needed to address intrinsic model bias for vision-language models across the entire model development pipeline."
}
```

## License

MIT License

## Contact

For questions about this repository, please contact the repo owner.