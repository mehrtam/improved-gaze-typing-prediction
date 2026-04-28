"""
Improved Gaze-Typing Prediction Pipeline
========================================

Two-stage Bidirectional LSTM pipeline for predicting typed characters from
synchronized eye-gaze and hand-tracking data.

  Stage 1: predicts which finger executed a keypress, using normalized hand
           kinematics, temporal deltas, and wrist velocity.
  Stage 2: per-finger character classifier, using gaze-to-key and finger-to-key
           distances on top of the Stage 1 features.

Evaluated with leave-one-subject-out cross-validation across 5 participants.

Usage
-----
    # 1. (Optional) extract per-participant .rar archives
    python improved_gaze_typing_prediction_pipeline.py --extract /path/to/rars

    # 2. Run the two-stage evaluation
    python improved_gaze_typing_prediction_pipeline.py --data /path/to/extracted
"""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_SEQUENCE_LENGTH = 50
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 5  # for EarlyStopping


# ---------------------------------------------------------------------------
# Data extraction (optional, for .rar archives)
# ---------------------------------------------------------------------------

def extract_rar_files(file_list, output_dir):
    """
    Extract a list of .rar archives into per-participant subdirectories.

    Requires the `rarfile` package and the `unrar` binary on PATH.
    """
    import rarfile

    if not file_list:
        print("No RAR files to extract.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for rar_path in file_list:
        if not os.path.exists(rar_path):
            print(f"Warning: file not found at {rar_path}, skipping.")
            continue

        participant_name = os.path.splitext(os.path.basename(rar_path))[0]
        extract_path = os.path.join(output_dir, participant_name)
        os.makedirs(extract_path, exist_ok=True)

        try:
            with rarfile.RarFile(rar_path) as rf:
                rf.extractall(path=extract_path)
            files = os.listdir(extract_path)
            print(f"Extracted {rar_path} → {extract_path} ({len(files)} files).")
        except rarfile.BadRarFile:
            print(f"Error: {rar_path} is not a valid RAR file.")
        except Exception as exc:  # noqa: BLE001
            print(f"Unexpected error extracting {rar_path}: {exc}")


# ---------------------------------------------------------------------------
# Preprocessing and feature engineering
# ---------------------------------------------------------------------------

def load_and_preprocess_data(folder_path):
    """
    Load CSVs, drop typed-incorrectly rows, interpolate missing values, and
    engineer normalized 3D positions, temporal deltas, wrist velocity, and
    gaze-to-key / finger-to-key distance features.
    """
    print("Step 1: Loading and preprocessing data...")
    all_csv_files = glob.glob(os.path.join(folder_path, "**/*.csv"), recursive=True)
    if not all_csv_files:
        print(f"Error: No CSV files found under {folder_path}.")
        return None

    aggregated_df = pd.concat(
        [pd.read_csv(f) for f in all_csv_files], ignore_index=True
    )
    print(f"Aggregated data loaded: {len(aggregated_df)} rows.")

    # Remove rows where the participant typed the wrong key
    print("Removing rows where PressedLetter != CurrentLetter...")
    aggregated_df = aggregated_df[
        aggregated_df["PressedLetter"] == aggregated_df["CurrentLetter"]
    ].copy()
    print(f"Remaining rows after typo removal: {len(aggregated_df)}")
    if len(aggregated_df) == 0:
        print("No valid rows remaining. Check the dataset.")
        return None

    # Interpolate missing values within each (participant, trial, letter) group
    print("Interpolating missing data...")
    numerical_cols = aggregated_df.select_dtypes(include=np.number).columns
    aggregated_df[numerical_cols] = (
        aggregated_df.groupby(["ParticipantID", "TrialNumber", "LetterIndex"])[
            numerical_cols
        ].transform(lambda x: x.interpolate(limit_direction="both"))
    )
    aggregated_df.dropna(inplace=True)

    df = aggregated_df.copy()
    new_features = {}

    # ---------- Normalize 3D positions to wrist-centered frame ----------
    print("Step 2: Wrist-centered normalization...")
    df["Origin_X"] = (df["Left_Hand_WristRoot_X"] + df["Right_Hand_WristRoot_X"]) / 2
    df["Origin_Y"] = (df["Left_Hand_WristRoot_Y"] + df["Right_Hand_WristRoot_Y"]) / 2
    df["Origin_Z"] = (df["Left_Hand_WristRoot_Z"] + df["Right_Hand_WristRoot_Z"]) / 2

    raw_coord_cols = [
        col for col in df.columns
        if (col.endswith(("_X", "_Y", "_Z")))
        and "GazeRay" not in col
        and not col.startswith("normalized_")
        and not col.startswith("Origin_")
    ]
    key_cols = [c for c in raw_coord_cols if c.startswith("Key_")]

    for col in raw_coord_cols:
        new_features[f"normalized_{col}"] = df[col] - df[f"Origin_{col[-1]}"]

    # ---------- Infer the pressed finger from 3D proximity ----------
    print("Step 3: Inferring pressed finger from 3D proximity...")
    normalized_df = pd.concat(
        [df, pd.DataFrame(new_features, index=df.index)], axis=1
    )

    finger_tip_x_cols = [
        c for c in normalized_df.columns
        if "Tip_X" in c and c.startswith("normalized_")
    ]
    finger_tip_names = [
        c.replace("normalized_", "").replace("_X", "") for c in finger_tip_x_cols
    ]

    available_keys = [
        c.replace("normalized_Key_", "").replace("_X", "")
        for c in normalized_df.columns
        if c.startswith("normalized_Key_") and c.endswith("_X")
    ]
    pressed_letters = normalized_df["PressedLetter"].dropna().unique()
    processed_keys = [k for k in pressed_letters if k in available_keys]

    all_distances = pd.DataFrame(
        index=normalized_df.index, columns=finger_tip_names, dtype=float
    )

    for key in processed_keys:
        mask = normalized_df["PressedLetter"] == key
        if not mask.any():
            continue
        key_coords = normalized_df.loc[
            mask,
            [
                f"normalized_Key_{key}_X",
                f"normalized_Key_{key}_Y",
                f"normalized_Key_{key}_Z",
            ],
        ].values

        for finger in finger_tip_names:
            finger_coords = normalized_df.loc[
                mask,
                [
                    f"normalized_{finger}_X",
                    f"normalized_{finger}_Y",
                    f"normalized_{finger}_Z",
                ],
            ].values
            all_distances.loc[mask, finger] = (
                (finger_coords - key_coords) ** 2
            ).sum(axis=1)

    df["InferredPressedFinger"] = all_distances.idxmin(axis=1, skipna=True)
    df.dropna(subset=["InferredPressedFinger"], inplace=True)

    # ---------- Temporal deltas, wrist velocity, gaze and finger distances ----------
    print("Step 4: Computing temporal deltas...")
    temp_df = pd.concat([df, pd.DataFrame(new_features)], axis=1)
    normalized_cols = [c for c in temp_df.columns if c.startswith("normalized_")]
    for col in normalized_cols:
        new_features[f"delta_{col}"] = (
            temp_df.groupby(["ParticipantID", "TrialNumber", "LetterIndex"])[col]
            .diff()
            .fillna(0)
        )

    print("Step 5: Geometric and kinematic features...")
    new_features["Left_Hand_WristRoot_Velocity"] = (
        temp_df.groupby(["ParticipantID", "TrialNumber", "LetterIndex"])[
            "normalized_Left_Hand_WristRoot_X"
        ].transform(lambda x: x.diff().fillna(0))
    )
    new_features["Right_Hand_WristRoot_Velocity"] = (
        temp_df.groupby(["ParticipantID", "TrialNumber", "LetterIndex"])[
            "normalized_Right_Hand_WristRoot_X"
        ].transform(lambda x: x.diff().fillna(0))
    )

    # Combine left/right gaze hits into a single 3D gaze position per frame
    temp_df["GazeHitPosition_X"] = np.nan
    temp_df["GazeHitPosition_Y"] = np.nan
    temp_df["GazeHitPosition_Z"] = np.nan

    both = (temp_df["LeftGazeHit"] == 1) & (temp_df["RightGazeHit"] == 1)
    left_only = (temp_df["LeftGazeHit"] == 1) & (temp_df["RightGazeHit"] == 0)
    right_only = (temp_df["LeftGazeHit"] == 0) & (temp_df["RightGazeHit"] == 1)

    for axis in ("X", "Y", "Z"):
        temp_df.loc[both, f"GazeHitPosition_{axis}"] = (
            temp_df.loc[both, f"LeftGazeHitPosition_{axis}"]
            + temp_df.loc[both, f"RightGazeHitPosition_{axis}"]
        ) / 2
        temp_df.loc[left_only, f"GazeHitPosition_{axis}"] = temp_df.loc[
            left_only, f"LeftGazeHitPosition_{axis}"
        ]
        temp_df.loc[right_only, f"GazeHitPosition_{axis}"] = temp_df.loc[
            right_only, f"RightGazeHitPosition_{axis}"
        ]

    unique_keys = [c.split("_")[1] for c in key_cols if c.endswith("_X")]
    for key in unique_keys:
        kx, ky, kz = (
            f"normalized_Key_{key}_X",
            f"normalized_Key_{key}_Y",
            f"normalized_Key_{key}_Z",
        )
        if kx in temp_df.columns and ky in temp_df.columns and kz in temp_df.columns:
            new_features[f"gaze_dist_{key}"] = np.sqrt(
                (temp_df["GazeHitPosition_X"] - temp_df[kx]) ** 2
                + (temp_df["GazeHitPosition_Y"] - temp_df[ky]) ** 2
                + (temp_df["GazeHitPosition_Z"] - temp_df[kz]) ** 2
            )

    # Vectorized finger-to-pressed-key distance (no .apply — fast on large frames)
    pressed_key_x = pd.Series(index=temp_df.index, dtype=float)
    pressed_key_y = pd.Series(index=temp_df.index, dtype=float)
    pressed_key_z = pd.Series(index=temp_df.index, dtype=float)
    for key in available_keys:
        mask = temp_df["PressedLetter"] == key
        if mask.any():
            pressed_key_x.loc[mask] = temp_df.loc[mask, f"normalized_Key_{key}_X"]
            pressed_key_y.loc[mask] = temp_df.loc[mask, f"normalized_Key_{key}_Y"]
            pressed_key_z.loc[mask] = temp_df.loc[mask, f"normalized_Key_{key}_Z"]

    for finger in finger_tip_names:
        new_features[f"finger_dist_{finger}"] = np.sqrt(
            (temp_df[f"normalized_{finger}_X"] - pressed_key_x) ** 2
            + (temp_df[f"normalized_{finger}_Y"] - pressed_key_y) ** 2
            + (temp_df[f"normalized_{finger}_Z"] - pressed_key_z) ** 2
        )

    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
    df.dropna(inplace=True)

    print(f"Preprocessing complete. Final dataframe: {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Sequence creation, model, and LOOCV evaluation
# ---------------------------------------------------------------------------

def create_padded_sequences(df, features, target_col):
    """Group rows by (participant, trial, letter) and pad to a fixed length."""
    sequences, labels, participants = [], [], []
    grouped = df.groupby(["ParticipantID", "TrialNumber", "LetterIndex"])

    max_len = min(max(len(g) for _, g in grouped), MAX_SEQUENCE_LENGTH)

    for group_key, group in grouped:
        padded = pad_sequences(
            [group[features].values],
            maxlen=max_len,
            dtype="float32",
            padding="post",
            truncating="post",
            value=0.0,
        )[0]
        sequences.append(padded)
        labels.append(group[target_col].iloc[-1])
        participants.append(group_key[0])

    return np.array(sequences), np.array(labels), np.array(participants)


def build_model(input_shape, num_classes):
    """Bidirectional LSTM with masking for variable-length keystroke sequences."""
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def run_loocv_evaluation(df, features, target_col, stage_name):
    """Leave-one-subject-out CV with per-fold class weights and early stopping."""
    print(f"\n--- Running LOOCV for {stage_name} ---")
    if df.empty:
        print("Skipping: empty dataframe.")
        return None

    X, y, pids = create_padded_sequences(df, features, target_col)
    if len(X) == 0:
        print("Skipping: no sequences generated.")
        return None

    unique_pids = np.unique(pids)
    if len(unique_pids) < 2:
        print("Skipping: need at least 2 participants for LOOCV.")
        return None

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    y_cat = to_categorical(y_enc, num_classes=num_classes)

    print(f"Total samples: {X.shape[0]}. Classes: {list(le.classes_)}")

    accuracies, y_true_all, y_pred_all = [], [], []
    for train_idx, test_idx in LeaveOneOut().split(unique_pids):
        train_pids = unique_pids[train_idx]
        test_pid = unique_pids[test_idx][0]

        train_mask = np.isin(pids, train_pids)
        test_mask = pids == test_pid

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y_cat[train_mask], y_cat[test_mask]
        y_te_labels = np.argmax(y_te, axis=1)

        model = build_model(X_tr.shape[1:], num_classes)
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

        counts = pd.Series(y_tr.argmax(axis=1)).value_counts()
        total = len(y_tr)
        cw = {i: total / (len(counts) * c) for i, c in counts.items()}

        model.fit(
            X_tr,
            y_tr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            validation_data=(X_te, y_te),
            callbacks=[es],
            class_weight=cw,
        )

        y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
        accuracies.append(accuracy_score(y_te_labels, y_pred))
        y_true_all.extend(y_te_labels)
        y_pred_all.extend(y_pred)

    avg_acc = float(np.mean(accuracies)) if accuracies else 0.0
    report = classification_report(
        y_true_all, y_pred_all, target_names=le.classes_, zero_division=0
    )
    return avg_acc, report, le, y_true_all, y_pred_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_stage1_features(df):
    """Stage 1 uses only hand kinematics (no key or gaze features)."""
    feats = [
        c for c in df.columns
        if (c.startswith("normalized_") or c.startswith("delta_"))
        and "Key_" not in c
        and "Gaze" not in c
        and "finger_dist" not in c
    ]
    feats.extend(["Left_Hand_WristRoot_Velocity", "Right_Hand_WristRoot_Velocity"])
    return feats


def select_stage2_features(df):
    """Stage 2 adds gaze-to-key and finger-to-key distances on top of kinematics."""
    feats = [
        c for c in df.columns
        if (
            c.startswith("normalized_")
            or c.startswith("delta_")
            or c.startswith("gaze_dist_")
            or c.startswith("finger_dist_")
        )
    ]
    feats.extend(["Left_Hand_WristRoot_Velocity", "Right_Hand_WristRoot_Velocity"])
    return feats


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--data",
        default="./extracted_participants",
        help="Folder containing per-participant CSVs (recursive).",
    )
    parser.add_argument(
        "--extract",
        default=None,
        help="If set, extract .rar files from this folder into --data first.",
    )
    args = parser.parse_args()

    if args.extract:
        rar_files = sorted(glob.glob(os.path.join(args.extract, "*.rar")))
        extract_rar_files(rar_files, args.data)

    df = load_and_preprocess_data(args.data)
    if df is None:
        return

    # ---- Stage 1: Finger Prediction ----
    stage1_feats = select_stage1_features(df)
    res1 = run_loocv_evaluation(df, stage1_feats, "InferredPressedFinger", "Stage 1")
    if res1 is not None:
        avg, report, _, _, _ = res1
        print("\n" + "=" * 50)
        print(f"Stage 1 (Finger Prediction) — LOOCV avg accuracy: {avg:.4f}")
        print(report)
        print("=" * 50)

    # ---- Stage 2: Per-Finger Character Prediction ----
    print("\nStage 2: Character prediction (one model per inferred finger)")
    finger_results = {}
    for finger_id in sorted(df["InferredPressedFinger"].unique()):
        finger_df = df[df["InferredPressedFinger"] == finger_id].copy()
        stage2_feats = select_stage2_features(finger_df)
        if not stage2_feats:
            print(f"No features available for {finger_id}, skipping.")
            continue

        res = run_loocv_evaluation(
            finger_df, stage2_feats, "PressedLetter", f"Stage 2 — {finger_id}"
        )
        if res is not None:
            finger_results[finger_id] = res
            avg, _, _, _, _ = res
            print(f"  {finger_id}: avg accuracy = {avg:.4f}")

    print("\n" + "=" * 50)
    print("Final Stage 2 Summary")
    for finger_id, (avg, _, le, _, _) in finger_results.items():
        print(f"  {finger_id} (keys: {list(le.classes_)}): {avg:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
