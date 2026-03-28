import librosa
import pandas as pd
import numpy as np
import random
import soundfile as sf
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def augment_speed(waveform):
    rate = random.uniform(0.7, 0.9)
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    stretched = stretched / (np.max(np.abs(stretched)) + 1e-9)
    return stretched


def augment_pitch(waveform, sr):
    steps = random.uniform(-2, 2)
    shifted = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=steps)
    shifted = shifted / (np.max(np.abs(shifted)) + 1e-9)
    return shifted


def augment_data(sample: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Applies random augmentations to each sample and stores augmented audio.
    Returns original + augmented dataset shuffled.
    """

    print(f"[INFO] Starting augmentation for {len(sample)} samples...")
    sample["path"] = f"{PROJECT_ROOT}" + sample["audio_path"]

    augmented_data = {
        "utterance_id": [],
        "child_id": [],
        "session_id": [],
        "audio_path": [],
        "audio_duration_sec": [],
        "age_bucket": [],
        "md5_hash": [],
        "filesize_bytes": [],
        "orthographic_text": [],
    }

    total_samples = len(sample)
    for index, row in sample.iterrows():

        audio_path = row["path"]
        waveform, sr = librosa.load(audio_path, sr=None)

        augmentation_choice = random.choice(
            ["speed", "pitch", "speed_pitch"]
        )

        augmented_waveform = waveform

        if augmentation_choice == "speed":
            augmented_waveform = augment_speed(waveform)

        elif augmentation_choice == "pitch":
            augmented_waveform = augment_pitch(waveform, sr)

        elif augmentation_choice == "speed_pitch":
            augmented_waveform = augment_speed(waveform)
            augmented_waveform = augment_pitch(augmented_waveform, sr)

        new_filename = f"{row['utterance_id']}_{augmentation_choice}.flac"
        save_path = output_dir + new_filename
        file_save_path = PROJECT_ROOT / f"{save_path}"
        sf.write(file_save_path, augmented_waveform, sr)

        # append metadata
        augmented_data["utterance_id"].append(row["utterance_id"])
        augmented_data["child_id"].append(row["child_id"])
        augmented_data["session_id"].append(row["session_id"])
        augmented_data["audio_path"].append(str(save_path))
        augmented_data["audio_duration_sec"].append(len(augmented_waveform) / sr)
        augmented_data["age_bucket"].append(row["age_bucket"])
        augmented_data["md5_hash"].append(row["md5_hash"])
        augmented_data["filesize_bytes"].append(os.path.getsize(file_save_path))
        augmented_data["orthographic_text"].append(row["orthographic_text"])

        # print progress every 10 samples
        if (index + 1) % 10 == 0 or (index + 1) == total_samples:
            print(f"[INFO] Augmented {index + 1}/{total_samples} samples")

    print("[INFO] Creating augmented DataFrame...")
    augmented_df = pd.DataFrame(augmented_data)

    final_df = pd.concat([sample, augmented_df], axis=0)
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    print("[INFO] Augmentation complete. Returning final DataFrame.")
    return final_df