import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

def split_dataframe(df, output_dir, n_splits=3):

    output_dir.mkdir(parents=True, exist_ok=True)

    splits = np.array_split(df, n_splits)
    file_paths = []
    for i, split_df in enumerate(splits):
        file_path = output_dir / f"split_{i}.csv"
        split_df.to_csv(file_path, index=False)
        file_paths.append(file_path)
    return file_paths