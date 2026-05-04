import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Optional

from configs.config import DataConfig


def preprocess(
    df: pd.DataFrame,
    cfg: DataConfig,
    encoders: Optional[Dict] = None,
    fit: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Preprocess sensor data for activity prediction.

    Steps:
    - Clean sensor & activity labels
    - Encode categorical features
    - Create temporal features
    - Generate sequence-related features

    Args:
        df: Input dataframe
        cfg: Configuration object
        encoders: Pre-fitted encoders (for val/test)
        fit: Whether to fit encoders (True for train)

    Returns:
        df: Processed dataframe
        encoders: Dictionary of fitted encoders
    """

    df = df.copy()

    
    # CLEANING
  
    df[cfg.SENSOR_COL] = df[cfg.SENSOR_COL].astype(str).str.strip().str.lower()

    df[cfg.VALUE_COL] = df[cfg.VALUE_COL].replace({"ON": 1, "OFF": 0})
    df[cfg.VALUE_COL] = pd.to_numeric(df[cfg.VALUE_COL], errors="coerce").fillna(0)
    df[cfg.VALUE_COL] = df[cfg.VALUE_COL].clip(0, 1).astype(np.float32)

    df[cfg.ACTIVITY_COL] = df[cfg.ACTIVITY_COL].fillna("other").astype(str)
    df[cfg.ACTIVITY_COL] = df[cfg.ACTIVITY_COL].str.replace(
        r"^[BIE]-", "", regex=True
    )
    df[cfg.ACTIVITY_COL] = df[cfg.ACTIVITY_COL].str.strip().str.lower()

    
    # TIME PROCESSING
  
    df[cfg.TIMESTAMP_COL] = pd.to_datetime(
        df[cfg.TIMESTAMP_COL], errors="coerce"
    )
    df = df.dropna(subset=[cfg.TIMESTAMP_COL])
    df = df.sort_values(cfg.TIMESTAMP_COL).reset_index(drop=True)

   
    # ENCODERS

    if fit:
        encoders = {
            cfg.SENSOR_COL: LabelEncoder(),
            cfg.ACTIVITY_COL: LabelEncoder()
        }

        encoders[cfg.SENSOR_COL].fit(df[cfg.SENSOR_COL])

        # Add "start" token for previous activity
        encoders[cfg.ACTIVITY_COL].fit(
            list(df[cfg.ACTIVITY_COL].unique()) + ["start"]
        )

    if encoders is None:
        raise ValueError("Encoders must be provided when fit=False")

    
    # ENCODING
  
    df[cfg.SENSOR_COL + "_id"] = encoders[cfg.SENSOR_COL].transform(
        df[cfg.SENSOR_COL]
    )
    df[cfg.ACTIVITY_COL + "_id"] = encoders[cfg.ACTIVITY_COL].transform(
        df[cfg.ACTIVITY_COL]
    )

    
    # PREVIOUS ACTIVITY

    df["prev_activity"] = df[cfg.ACTIVITY_COL].shift(1)
    df["prev_activity"] = df["prev_activity"].fillna("start")

    df["prev_activity_id"] = encoders[cfg.ACTIVITY_COL].transform(
        df["prev_activity"]
    )

 
    # TIME GAPS

    df["delta_t"] = (
        df[cfg.TIMESTAMP_COL].diff().dt.total_seconds() / 60
    )
    df["delta_t"] = df["delta_t"].clip(lower=0).fillna(0)

    df["log_dt"] = np.log1p(df["delta_t"])


    # PREVIOUS ACTIVITY DURATION

    df["activity_change"] = (
        df[cfg.ACTIVITY_COL] != df[cfg.ACTIVITY_COL].shift()
    ).cumsum()

    duration_map = df.groupby("activity_change")[cfg.TIMESTAMP_COL].transform(
        lambda x: (x.max() - x.min()).total_seconds() / 60
    )

    df["prev_act_duration"] = duration_map.shift(1).fillna(0)
    df["log_prev_dur"] = np.log1p(df["prev_act_duration"])

  
    # CYCLICAL TIME FEATURES
    
    df["hour"] = (
        df[cfg.TIMESTAMP_COL].dt.hour +
        df[cfg.TIMESTAMP_COL].dt.minute / 60 +
        df[cfg.TIMESTAMP_COL].dt.second / 3600
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)

    df["day"] = df[cfg.TIMESTAMP_COL].dt.dayofweek

    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7).astype(np.float32)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7).astype(np.float32)

    return df, encoders
