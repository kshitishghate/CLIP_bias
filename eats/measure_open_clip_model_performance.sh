cd ../CLIP_benchmark
python clip_benchmark/cli.py eval --pretrained_model '../data/open_clip_models.csv' \
    --dataset "../data/fer.txt" \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"