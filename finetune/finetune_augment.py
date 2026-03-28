from finetune import build_config, setup_model, configure_adapters, train_model, evaluate_model 
import pandas as pd 
from utils import split_dataframe 
from pathlib import Path
import sys 
from augmentation_module import augment_data

PROJECT_ROOT= Path(__file__).resolve().parents[2]
DATA_OUTPUT_DIR = PROJECT_ROOT / "temp_files"
MANIFEST_DIR = DATA_OUTPUT_DIR / "processed" 
SAMPLES_DIR = DATA_OUTPUT_DIR / "sample_data"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest_augment.jsonl"
TRAIN_FILE= PROJECT_ROOT / "data_files" / "train_samples.csv"

for directory in [DATA_OUTPUT_DIR, MANIFEST_DIR, SAMPLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Created directory: {directory}")
    
def prepare_data( logger, DATA_FILE, sample=None):

    logger.info("Loading dataset...")

    df = pd.read_csv(DATA_FILE)

    df["audio_path"] = df["audio_path"].apply(
        lambda path: f"{PROJECT_ROOT}{path}"
    )
    df = df.loc[df["audio_duration_sec"] <= 30].reset_index(drop=True)
    df = df[
        ["audio_path", "audio_duration_sec", "orthographic_text"]
    ].rename(
        columns={
            "audio_path": "audio_filepath",
            "audio_duration_sec": "duration",
            "orthographic_text": "text",
        }
    )

    logger.info(f"Loaded {len(df)} samples")

    if sample:
        logger.info(f"Sampling {sample} samples")
        df = df.sample(sample, random_state=0)

    logger.info("Splitting dataset (95/5)...")


    df.to_json(TRAIN_MANIFEST, orient="records", lines=True)
    logger.info(f"Train samples: {len(df)}")

    return TRAIN_MANIFEST

print("Reading the data File")
df = pd.read_csv(TRAIN_FILE)
print("Spliting the Dataframes")
file_paths = split_dataframe(df, SAMPLES_DIR)
for file_path in file_paths:
    sample_df = pd.read_csv(file_path)
    sample_df = augment_data(sample_df, "temp_files/processed/")
    print(sample_df.shape[0])
    break