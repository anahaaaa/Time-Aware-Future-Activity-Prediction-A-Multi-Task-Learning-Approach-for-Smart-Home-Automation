from dataclasses import dataclass
import random
import numpy as np
import torch

@dataclass
class DataConfig:
  """Central configuration for data, model, and training."""

    # -----------------------------
    # Column Names
    # -----------------------------
    TIMESTAMP_COL: str = "Timestamp"
    SENSOR_COL: str = "Sensor"
    VALUE_COL: str = "Value"
    ACTIVITY_COL: str = "Activity_Label"

    # -----------------------------
    # Model Architecture
    # -----------------------------
    EMBED_DIM: int = 16
    HIDDEN: int = 128
    HEADS: int = 4

    TRANSFORMER_LAYERS: int = 4
    TRANSFORMER_FF_DIM: int = 512

    LSTM_HIDDEN: int = 64
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.3

    # -----------------------------
    # Sequence / Window
    # -----------------------------
    WINDOW_SIZE: int = 150
    STRIDE: int = 5
    SEQ_LEN: int = 5   # 🔥 IMPORTANT: used in sequence building

    # -----------------------------
    # Time Prediction
    # -----------------------------
    TIME_BINS: list = (0, 2, 5, 15, 30, 60, 120)  # minutes
    NUM_TIME_BINS: int = len(TIME_BINS) - 1

    TIME_FEATURES: list = ("hour_sin", "hour_cos", "day_sin", "day_cos")

    # -----------------------------
    # Training
    # -----------------------------
    BATCH_SIZE: int = 16
    EPOCHS: int = 30
    LR: float = 1e-3
    SEED: int = 42
