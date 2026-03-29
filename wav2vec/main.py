import sys
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Union, Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Audio, Features, Value, load_from_disk
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
from loguru import logger

from .finetune.score import VALID_IPA_CHARS, score_ipa_cer

# -------------------
# Constants & Paths
# -------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "data_files"
DATA_FILE = DATA_ROOT / "train_samples.csv"
MANIFEST_DIR = PROJECT_ROOT / "processed" / "ortho_dataset"
TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"

PROCESSED_DATASET_DIR = DATA_ROOT / "processed" / "phonetic_dataset"
LOG_FILE = "finetune.log"
SR = 16000  # Audio sampling rate
WAV2VEC2_DOWNSAMPLE = 320

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

# Configure logger
logger.add(LOG_FILE, rotation="10 MB", retention="10 days", level="INFO")


# -------------------
# Data Preparation
# -------------------
def prepare_data() -> Dataset:
    logger.info("Loading dataset from %s", DATA_FILE)
    df = pd.read_csv(DATA_FILE)

    df["audio_path"] = df["audio_path"].apply(lambda path: f"{PROJECT_ROOT}{path}")
    df = df[df["audio_duration_sec"] <= 30].reset_index(drop=True)
    df = df[["audio_path", "audio_duration_sec", "phonetic_text"]].rename(
        columns={
            "audio_path": "audio_filepath",
            "audio_duration_sec": "duration",
            "phonetic_text": "text",
        }
    )

    # Define schema for Dataset
    schema = Features({
        "phonetic_text": Value("string"),
        "audio_path": Value("string"),
    })
    dataset = Dataset.from_pandas(df.reset_index(drop=True), features=schema)
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=SR))
    logger.info("Dataset prepared with %d samples", len(dataset))
    return dataset


# -------------------
# Tokenizer & Processor
# -------------------
def create_processor() -> Wav2Vec2Processor:
    unk_tok = "[UNK]"
    pad_tok = "[PAD]"
    space_tok = "|"

    all_toks = sorted([c for c in VALID_IPA_CHARS if c != " "]) + [unk_tok, pad_tok, space_tok]
    vocab_dict = {char: idx for idx, char in enumerate(all_toks)}

    vocab_path = DATA_ROOT / "vocab" / "phonetic_vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w") as f:
        json.dump(vocab_dict, f)
    logger.info("Vocabulary saved at %s", vocab_path)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token=unk_tok,
        pad_token=pad_tok,
        word_delimiter_token=space_tok,
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SR,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False,
    )

    processor = Wav2Vec2Processor(
        tokenizer=tokenizer, feature_extractor=feature_extractor
    )
    logger.info("Processor created")
    return processor


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


# -------------------
# Preprocessing
# -------------------
def preprocess_dataset(dataset: Dataset, processor: Wav2Vec2Processor) -> Dataset:
    def preprocess_batch(examples):
        input_values = [processor(item["array"], sampling_rate=SR).input_values[0]
                        for item in examples["audio_path"]]
        labels = [processor(text=ex.replace(" ", "|")).input_ids for ex in examples["phonetic_text"]]
        return {"input_values": input_values, "labels": labels}

    if PROCESSED_DATASET_DIR.exists():
        dataset = load_from_disk(str(PROCESSED_DATASET_DIR))
        logger.info("Loaded preprocessed dataset from %s (%d examples)", PROCESSED_DATASET_DIR, len(dataset))
    else:
        dataset = dataset.map(preprocess_batch, batched=True, num_proc=4)
        dataset.save_to_disk(str(PROCESSED_DATASET_DIR))
        logger.info("Preprocessed dataset saved to %s", PROCESSED_DATASET_DIR)

    # Filter out invalid CTC samples
    before_filter = len(dataset)
    dataset = dataset.filter(lambda ex: (len(ex["input_values"]) // WAV2VEC2_DOWNSAMPLE) > len(ex["labels"]) 
                             and len(ex["labels"]) > 0 and len(ex["input_values"]) > 0,
                             num_proc=4)
    logger.info("CTC filter applied: %d -> %d samples", before_filter, len(dataset))
    return dataset


# -------------------
# Model & Training
# -------------------
def create_model(processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_encoder()
    logger.info("Model loaded and feature encoder frozen")
    return model


def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    return {"cer": score_ipa_cer(label_str, pred_str)}


def train_model(model: Wav2Vec2ForCTC, processor: Wav2Vec2Processor, train_dataset: Dataset, eval_dataset: Dataset):
    output_dir = PROJECT_ROOT / "models" / "wav2vec2-phonetic"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        group_by_length=False,
        per_device_train_batch_size=27,
        per_device_eval_batch_size=27,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        num_train_epochs=20,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=100,
        warmup_steps=500,
        lr_scheduler_type="linear",
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        save_total_limit=2,
        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to="none",
    )

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished")
    return trainer


# -------------------
# Main Execution
# -------------------
if __name__ == "__main__":
    dataset = prepare_data()
    processor = create_processor()
    processed_dataset = preprocess_dataset(dataset, processor)

    # Split dataset
    dataset_split = processed_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    model = create_model(processor)
    trainer = train_model(model, processor, train_dataset, eval_dataset)

    # Save model and processor
    save_dir = PROJECT_ROOT / "models" / "wav2vec2-phonetic-final"
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    processor.save_pretrained(str(save_dir))
    processor.feature_extractor.save_pretrained(str(save_dir))
    torch.save(trainer.args, save_dir / "training_args.pt")
    logger.info("Model and processor saved to %s", save_dir)