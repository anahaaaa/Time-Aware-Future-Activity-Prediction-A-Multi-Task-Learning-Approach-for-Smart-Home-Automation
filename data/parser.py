import pandas as pd


# -------------------------------------------------
# Load raw CASAS TXT dataset
# -------------------------------------------------
def load_txt(path: str) -> pd.DataFrame:
    """
    Load raw CASAS dataset (TXT format) into a DataFrame.
    """
    rows = []

    with open(path) as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 3:
                continue

            date, time, sensor = parts[:3]
            timestamp = f"{date} {time}"

            value = parts[3] if len(parts) >= 4 else None
            activity = " ".join(parts[4:]) if len(parts) > 4 else None

            rows.append({
                "Sensor": sensor,
                "Value": value,
                "Activity": activity,
                "Timestamp": timestamp
            })

    df = pd.DataFrame(rows)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    return df


# -------------------------------------------------
# Create BIO-style activity labels
# -------------------------------------------------
def create_activity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw activity annotations into BIO format labels.
    """
    current_activity = None
    labels = []

    for act in df["Activity"]:
        if pd.isna(act):
            labels.append(f"I-{current_activity}" if current_activity else "Other")
            continue

        act = act.strip().lower()

        if "begin" in act:
            current_activity = act.replace(" begin", "").strip()
            labels.append(f"B-{current_activity}")

        elif "end" in act:
            labels.append(f"I-{current_activity}" if current_activity else "Other")
            current_activity = None

        else:
            labels.append(f"I-{current_activity}" if current_activity else "Other")

    df["Activity_Label"] = labels
    return df


# -------------------------------------------------
# Create sensor token
# -------------------------------------------------
def create_sensor_token(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine sensor and value into a single token.
    """
    df["Value"] = df["Value"].astype(str)
    df["SensorToken"] = df["Sensor"] + "_" + df["Value"]
    return df


# -------------------------------------------------
# Full pipeline (IMPORTANT)
# -------------------------------------------------
def process_dataset(path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline for raw CASAS dataset.
    """
    df = load_txt(path)
    df = create_activity_labels(df)
    df = create_sensor_token(df)

    df = df[
        ["Sensor", "Value", "Activity", "Timestamp",
         "Activity_Label", "SensorToken"]
    ]

    return df
