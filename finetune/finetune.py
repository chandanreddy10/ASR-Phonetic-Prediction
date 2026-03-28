import json
import os
import sys
import logging
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
import os
os.environ["NUMBA_CUDA_DEFAULT_PTX_CC"] = "8.0"
from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from omegaconf import OmegaConf, open_dict
from sklearn.model_selection import train_test_split
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from nemo_adapter import (
    add_global_adapter_cfg,
    patch_transcribe_lhotse,
    update_model_cfg,
    update_model_config_to_support_adapter,
)

from score import english_spelling_normalizer, score_wer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

DATA_FILE = PROJECT_ROOT / "data_files" / "train_samples.csv"
MANIFEST_DIR = PROJECT_ROOT / "processed" / "ortho_dataset"

TRAIN_MANIFEST = MANIFEST_DIR / "train_manifest.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val_manifest.jsonl"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE ="finetune.log"

torch.set_float32_matmul_precision("high")


def setup_logging(log_file=LOG_FILE):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging(log_file=LOG_FILE)


def prepare_data(sample=None):

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

    train_df, val_df = train_test_split(df, test_size=0.05, random_state=0)

    train_df.to_json(TRAIN_MANIFEST, orient="records", lines=True)
    val_df.to_json(VAL_MANIFEST, orient="records", lines=True)

    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")

    return TRAIN_MANIFEST, VAL_MANIFEST


def build_config(train_manifest, val_manifest, sample=None):

    logger.info("Building training configuration...")

    DEVICES = 1
    PRECISION = "bf16-mixed"
    BATCH_SIZE = 4
    NUM_WORKERS = 4

    yaml_path = "asr_adaptation.yaml"
    cfg = OmegaConf.load(yaml_path)

    overrides = OmegaConf.create(
        {
            "model": {
                "pretrained_model": "nvidia/parakeet-tdt-0.6b-v2",
                "adapter": {
                    "adapter_name": "asr_children_orthographic",
                    "adapter_module_name": "encoder",
                    "linear": {"in_features": 1024},
                },
                "train_ds": {
                    "manifest_filepath": str(train_manifest),
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "use_lhotse": False,
                    "channel_selector": "average",
                },
                "validation_ds": {
                    "manifest_filepath": str(val_manifest),
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "use_lhotse": False,
                    "channel_selector": "average",
                },
                "optim": {
                    "lr": 0.001,
                    "weight_decay": 0.0,
                },
            },
            "trainer": {
                "devices": DEVICES,
                "precision": PRECISION,
                "strategy": "auto",
                "max_epochs": 1, #if sample else None,
                "max_steps": -1, #if sample else 10000,
                "val_check_interval": 0.2, # if sample else 500,
                "enable_progress_bar": True,
                "accumulate_grad_batches":4,
            },
            "exp_manager": {
                "exp_dir": str(PROJECT_ROOT / "models" / "orthographic_finetune_nemo"),
            },
        }
    )

    cfg = OmegaConf.merge(cfg, overrides)

    logger.info("Configuration ready")

    return cfg

def setup_model(cfg):

    logger.info("Initializing trainer...")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    logger.info(f"Experiment directory: {exp_log_dir}")

    logger.info("Loading pretrained model...")

    model_cfg = ASRModel.from_pretrained(
        cfg.model.pretrained_model,
        return_config=True,
    )

    update_model_config_to_support_adapter(model_cfg, cfg)

    model = ASRModel.from_pretrained(
        cfg.model.pretrained_model,
        override_config_path=model_cfg,
        trainer=trainer,
    )

    with open_dict(model.cfg):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False

    model.change_decoding_strategy(model.cfg.decoding)

    logger.info("Model loaded successfully")

    return model, trainer, exp_log_dir


def configure_adapters(model, cfg):

    logger.info("Configuring adapters...")

    cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
    model.setup_training_data(cfg.model.train_ds)

    cfg.model.validation_ds = update_model_cfg(
        model.cfg.validation_ds,
        cfg.model.validation_ds,
    )

    model.setup_multiple_validation_data(cfg.model.validation_ds)

    model.setup_optimization(cfg.model.optim)

    with open_dict(cfg.model.adapter):

        adapter_name = cfg.model.adapter.pop("adapter_name")
        adapter_type = cfg.model.adapter.pop("adapter_type")
        adapter_module_name = cfg.model.adapter.pop("adapter_module_name", None)

        adapter_type_cfg = cfg.model.adapter[adapter_type]

        if adapter_module_name and ":" not in adapter_name:
            adapter_name = f"{adapter_module_name}:{adapter_name}"

        adapter_global_cfg = cfg.model.adapter.pop(
            model.adapter_global_cfg_key,
            None,
        )

        if adapter_global_cfg:
            add_global_adapter_cfg(model, adapter_global_cfg)

    model.add_adapter(adapter_name, cfg=adapter_type_cfg)

    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(adapter_name, enabled=True)

    model.freeze()
    model.train()
    model.unfreeze_enabled_adapters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(
        f"Trainable parameters: {trainable_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model

def train_model(model, trainer):

    logger.info("Starting training...")

    trainer.fit(model)

    logger.info("Training finished")


def evaluate_model(exp_log_dir, cfg):

    logger.info("Starting evaluation...")

    nemo_ckpts = sorted((exp_log_dir / "checkpoints").glob("*.nemo"))

    if not nemo_ckpts:
        raise FileNotFoundError("No .nemo checkpoints found")

    best_ckpt = nemo_ckpts[-1]

    logger.info(f"Loading checkpoint: {best_ckpt}")

    eval_model = ASRModel.restore_from(best_ckpt, map_location="cuda")

    with open_dict(eval_model.cfg):
        eval_model.cfg.decoding.greedy.use_cuda_graph_decoder = False

    eval_model.change_decoding_strategy(eval_model.cfg.decoding)

    patch_transcribe_lhotse(eval_model)

    with open(cfg.model.validation_ds.manifest_filepath) as f:
        val_entries = [json.loads(line) for line in f]

    audio_files = [e["audio_filepath"] for e in val_entries]
    references = [e["text"] for e in val_entries]

    logger.info(f"Running inference on {len(audio_files)} files")

    raw = eval_model.transcribe(
        audio_files,
        batch_size=cfg.model.validation_ds.batch_size,
        channel_selector="average",
        verbose=False,
    )

    if isinstance(raw, tuple):
        raw = raw[0]

    predictions = [h.text if hasattr(h, "text") else h for h in raw]

    normalizer = EnglishTextNormalizer(english_spelling_normalizer)

    filtered = [
        (r, p)
        for r, p in zip(references, predictions)
        if normalizer(r) != ""
    ]

    references, predictions = zip(*filtered)

    wer = score_wer(references, predictions)

    logger.info(f"Validation WER: {wer:.4f}")

    logger.info("Sample predictions:")

    for ref, pred in zip(references[:5], predictions[:5]):
        logger.info(f"REF:  {ref}")
        logger.info(f"PRED: {pred}")

#Main
def main():

    logger.info("========== FINETUNING STARTED ==========")

    train_manifest, val_manifest = prepare_data()

    cfg = build_config(train_manifest, val_manifest)

    model, trainer, exp_log_dir = setup_model(cfg)

    model = configure_adapters(model, cfg)

    train_model(model, trainer)

    
    evaluate_model(exp_log_dir, cfg)

    
    logger.info("========== FINETUNING COMPLETE ==========")


if __name__ == "__main__":
    main()