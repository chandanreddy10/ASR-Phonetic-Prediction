from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Dict, List, Union, Optional

from IPython.display import display
from loguru import logger
import numpy as np
import pandas as pd
import tqdm


from matplotlib import ticker
import matplotlib.pyplot as plt

from datasets import Dataset, Audio, Features, Value, load_from_disk
import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)

from .finetune.score import VALID_IPA_CHARS, score_ipa_cer