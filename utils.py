import json
import pandas as pd 
from sklearn.model_selection import train_test_split
from pathlib import Path

def convert_data_to_csv(talkbank_transcript, datadriven_transcript):
    
    talkbank_df = pd.read_json(talkbank_transcript, lines=True)
    datadriven_df = pd.read_json(datadriven_transcript, lines=True)

    talkbank_df["audio_path"] = talkbank_df["audio_path"].apply(lambda path : f"/data_files/{path}")

    datadriven_df["audio_path"] = datadriven_df["audio_path"].apply(lambda path : f"/data_files/{path}")

    df = pd.concat([talkbank_df, datadriven_df])

    return df

talkbank_transcript = "data_files/train_phon_transcripts_talkbank.jsonl"
datadriven_transcript = "data_files/train_phon_transcripts_dd.jsonl"
df = convert_data_to_csv(talkbank_transcript, datadriven_transcript)

train_samples, validation_samples = train_test_split(df, test_size=0.05, random_state=0)

train_samples.to_csv("data_files/train_samples.csv")
print("Saved Train samples.")

validation_samples.to_csv("data_files/validation_samples.csv")
print("Saved Validation samples.")

