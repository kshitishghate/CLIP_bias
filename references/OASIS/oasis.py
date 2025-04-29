# Oasis Image embeddings

import os
import pandas as pd
import pickle
from shutil import copyfile

# Load OASIS dataset
df = pd.read_csv("../references/OASIS/OASIS.csv")

# Create dictionary mapping index to valence mean
oasis_ind_val = dict(zip(df.index, df.Valence_mean))

# Sort indices by valence mean
sorted_indices = sorted(oasis_ind_val, key=oasis_ind_val.get, reverse=True)

# Get top 25 and bottom 25 indices
top_indices = sorted_indices[:25]
bottom_indices = sorted_indices[-25:]


# Define source and destination directories
src_dir = "../references/OASIS/Images"
top_dest_dir = "../references/OASIS/top_25_images"
bottom_dest_dir = "../references/OASIS/bottom_25_images"

# Create destination directories if they don't exist
os.makedirs(top_dest_dir, exist_ok=True)
os.makedirs(bottom_dest_dir, exist_ok=True)

# Copy top 25 images
for idx in top_indices:
    src_file = os.path.join(src_dir, df.loc[idx, 'Theme'] + '.jpg')
    dest_file = os.path.join(top_dest_dir, df.loc[idx, 'Theme'] + '.jpg')
    copyfile(src_file, dest_file)

# Copy bottom 25 images
for idx in bottom_indices:
    src_file = os.path.join(src_dir, df.loc[idx, 'Theme'] + '.jpg')
    dest_file = os.path.join(bottom_dest_dir, df.loc[idx, 'Theme'] + '.jpg')
    copyfile(src_file, dest_file)

print(f"Top 25 images copied to: {top_dest_dir}")
print(f"Bottom 25 images copied to: {bottom_dest_dir}")