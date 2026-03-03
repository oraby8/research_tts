import pandas as pd
import soundfile as sf
import os
import argparse
import numpy as np
import subprocess
import shutil
import logging


import os
import io
import csv
import torch
import torchaudio
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm


def prepare_uae_dataset(
    output_dir="/mnt/nvme1/Chatterbox-Multilingual-TTS-Fine-Tuning/data_train",
    max_samples=None,
):
    # Setup paths
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")

    # Define datasets to process
    datasets_config = [
        #{"path": "MBZUAI/MIXAT", "split": "train", "text_col": "transcript"},
        {"path": "AhmedBadawy11/UAE_100K", "split": "train", "text_col": "text"},
        # {"path": "oddadmix/da7ee7_sep_cleaned-8k", "split": "train", "text_col": "transcription"},
        # {"path": "rahafvii/EGY2K", "split": "train", "text_col": "text"},
        # {"path": "ArabicSpeech/sawtarabi", "split": "train", "text_col": "text_not_diacritized"},
    ]

    print("Processing datasets...", flush=True)

    # Prepare metadata list
    metadata = []

    # Global counter for unique filenames across datasets
    processed_count = 0

    # Open CSV file for writing immediately to stream results
    with open(metadata_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["file_name", "transcription", "duration_seconds"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for ds_config in datasets_config:
            dialect = None
            if ds_config["path"] == "ArabicSpeech/sawtarabi":
                dialect = "dialect"
            ds_path = ds_config["path"]
            split = ds_config["split"]
            text_col = ds_config["text_col"]

            print(f"Loading dataset '{ds_path}' (split={split})...", flush=True)
            try:
                # Load dataset in streaming mode
                ds = load_dataset(
                    ds_path,
                    split=split,
                    streaming=True,
                    token=os.environ.get("HF_TOKEN"),
                )
                # Avoid problematic decoding (torchcodec)
                ds = ds.cast_column("audio", Audio(decode=False))
                if ds_config["path"] == "ArabicSpeech/sawtarabi":
                    ds = ds.filter(lambda x: x["dialect"] in ["CS_EGY_ENG", "EGY"])
            except Exception as e:
                print(f"Error loading dataset {ds_path}: {e}")
                continue

            print(f"Processing samples from {ds_path}...", flush=True)

            for i, ex in enumerate(tqdm(ds, desc=f"Processing {ds_path}")):
                if max_samples and processed_count >= max_samples:
                    break

                try:
                    # Extract text
                    text = ex.get(text_col, "").strip()
                    if not text:
                        continue

                    # Extract audio
                    audio_data = ex.get("audio")
                    if not audio_data or not audio_data.get("bytes"):
                        continue

                    audio_bytes = audio_data["bytes"]

                    # Load audio using torchaudio (decodes bytes)
                    with io.BytesIO(audio_bytes) as f:
                        waveform, sr = torchaudio.load(f)

                    # Convert to mono if needed
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)

                    # Calculate duration
                    duration = waveform.shape[1] / sr

                    # Define output filename
                    filename = f"utterance_{processed_count + 1:05d}.wav"
                    rel_path = os.path.join("audio", filename)
                    abs_path = os.path.join(audio_dir, filename)

                    # Save audio file (using soundfile for robustness/speed)
                    # Resample to 24kHz if needed? Usually 24k or 44.1k is good for TTS.
                    # Keeping original SR for now unless specified otherwise.
                    # Actually, standardizing to 24kHz is often good for TTS training.
                    # Let's resample to 24kHz to be safe and consistent.
                    TARGET_SR = 24000
                    if sr != TARGET_SR:
                        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                        waveform = resampler(waveform)
                        sr = TARGET_SR

                    # Save
                    torchaudio.save(abs_path, waveform, sr)

                    # Prepare metadata entry
                    entry = {
                        "file_name": rel_path,
                        "transcription": text,
                        "duration_seconds": f"{duration:.2f}",
                    }

                    # Write to CSV
                    writer.writerow(entry)

                    processed_count += 1

                except Exception as e:
                    # print(f"Error processing sample {i} from {ds_path}: {e}")
                    continue

            if max_samples and processed_count >= max_samples:
                break

    print(f"Dataset preparation complete.")
    print(f"Processed {processed_count} samples.")
    print(f"Audio files saved to: {audio_dir}")
    print(f"Metadata saved to: {metadata_path}")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def normalize_text(text):
    """
    Basic text normalization.
    Remove extra spaces, potentially non-Arabic characters if strict mode is on.
    For now, we keep it simple as per the general requirement.
    """
    if not isinstance(text, str):
        return ""
    return text.strip()


def calculate_cps(row):
    """Calculate Characters Per Second."""
    text = row["transcription"]
    duration = row["duration_seconds"]

    if duration <= 0:
        return 0

    # Remove spaces for char count? Paper doesn't specify, but usually yes or no.
    # Assuming raw character count including spaces or excluding?
    # Usually standard CPS includes all characters.
    char_count = len(text)
    return char_count / duration


def filter_by_cps(df, min_cps=4, max_cps=20):
    """
    Filter DataFrame based on CPS thresholds.
    Paper suggests [4, 10] lower bound and [15, 25] upper bound.
    We use defaults 4 and 20.
    """
    df["cps"] = df.apply(calculate_cps, axis=1)

    original_count = len(df)
    df_filtered = df[(df["cps"] >= min_cps) & (df["cps"] <= max_cps)].copy()
    filtered_count = len(df_filtered)

    logging.info(
        f"CPS Filtering: Kept {filtered_count}/{original_count} samples (Removed {original_count - filtered_count})"
    )
    return df_filtered


def is_silent(audio_path, threshold=0.01):
    """
    Check if audio file is silent or near silent.
    """
    try:
        data, samplerate = sf.read(audio_path)
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(data**2))
        return rms < threshold
    except Exception as e:
        logging.error(f"Error reading {audio_path}: {e}")
        return True


def run_source_separation_batch(audio_paths, output_dir, chunk_size=500):
    """
    Run Demucs source separation on multiple audio files in batch (chunked).
    Returns a dict mapping original audio paths to separated vocal paths.
    """
    if shutil.which("demucs") is None:
        logging.warning("Demucs not found. Skipping source separation.")
        return {}

    if not audio_paths:
        return {}

    valid_audio_paths = [p for p in audio_paths if os.path.exists(p)]
    if not valid_audio_paths:
        return {}

    results = {}
    total_chunks = (len(valid_audio_paths) + chunk_size - 1) // chunk_size

    for i in range(0, len(valid_audio_paths), chunk_size):
        chunk = valid_audio_paths[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        logging.info(
            f"Running Demucs batch {chunk_num}/{total_chunks} ({len(chunk)} files)..."
        )

        try:
            cmd = [
                "demucs",
                "-n",
                "htdemucs",
                "-o",
                output_dir,
                "--two-stems",
                "vocals",
            ] + chunk

            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            for audio_path in chunk:
                filename = os.path.splitext(os.path.basename(audio_path))[0]
                separated_path = os.path.join(
                    output_dir, "htdemucs", filename, "vocals.wav"
                )
                results[audio_path] = (
                    separated_path if os.path.exists(separated_path) else None
                )

        except subprocess.CalledProcessError as e:
            logging.error(f"Demucs batch {chunk_num} failed: {e}")
            for p in chunk:
                results[p] = None

    return results


def process_data(
    input_csv, output_csv, audio_dir, output_audio_dir=None, apply_separation=False
):
    """
    Main processing pipeline.
    """
    df = pd.read_csv(input_csv)
    logging.info(f"Loaded {len(df)} samples from {input_csv}")

    # Ensure text is string
    df["transcription"] = df["transcription"].apply(normalize_text)

    # 1. CPS Filtering
    df_filtered = filter_by_cps(df)

    # 2. Source Separation & Silence Filtering
    # If apply_separation is True, we attempt separation.
    # If not, we just check the original file for silence.

    if apply_separation:
        logging.info("Starting Source Separation (Demucs)...")
        if output_audio_dir:
            os.makedirs(output_audio_dir, exist_ok=True)
    else:
        logging.info(
            "Skipping Source Separation. Checking original files for silence..."
        )

    # Collect all audio paths first
    audio_paths = []
    for idx, row in df_filtered.iterrows():
        audio_path = str(row["file_name"])

        # Resolve path
        if not os.path.isabs(audio_path):
            if audio_dir:
                audio_path = os.path.join(audio_dir, audio_path)
            else:
                pass
        audio_path = os.path.abspath(audio_path)

        if os.path.exists(audio_path):
            audio_paths.append((idx, audio_path))

    # Run batch separation if requested
    separation_map = {}
    if apply_separation and audio_paths:
        all_paths = [p for _, p in audio_paths]
        separation_results = run_source_separation_batch(
            all_paths, output_audio_dir if output_audio_dir else "temp_separation"
        )
        for original_path, separated_path in separation_results.items():
            separation_map[original_path] = separated_path

    # Iterate over the filtered dataframe to check audio quality
    valid_indices = []

    for idx, row in df_filtered.iterrows():
        audio_path = str(row["file_name"])

        # Resolve path
        if not os.path.isabs(audio_path):
            if audio_dir:
                audio_path = os.path.join(audio_dir, audio_path)
            else:
                pass
        audio_path = os.path.abspath(audio_path)

        if not os.path.exists(audio_path):
            logging.warning(f"File not found: {audio_path}")
            continue

        file_to_check = audio_path

        # Use separated vocals if available
        if apply_separation and audio_path in separation_map:
            separated_vocals = separation_map[audio_path]
            if separated_vocals:
                file_to_check = separated_vocals

        # Check for silence
        if is_silent(file_to_check):
            logging.info(
                f"Sample {row['file_name']} is silent (or near silent). Removing."
            )
        else:
            valid_indices.append(idx)

    # Filter based on valid indices
    df_final = df_filtered.loc[valid_indices]
    removed_count = len(df_filtered) - len(df_final)
    logging.info(
        f"Silence/Quality Filtering: Kept {len(df_final)}/{len(df_filtered)} samples (Removed {removed_count})"
    )

    # Save processed metadata
    df_final.to_csv(output_csv, index=False)
    logging.info(f"Saved processed metadata to {output_csv}")


if __name__ == "__main__":
    output_dir = "/mnt/nvme1/Chatterbox-Multilingual-TTS-Fine-Tuning/data_train"

    prepare_uae_dataset("/mnt/nvme1/Chatterbox-Multilingual-TTS-Fine-Tuning/data_train")

    input_csv = output_dir + "/metadata.csv"
    output_csv = output_dir + "/processed_metadata.csv"
    audio_dir = output_dir
    output_audio_dir = output_dir + "/separated_audio"
    apply_separation = False

    process_data(
        input_csv,
        output_csv,
        audio_dir=audio_dir,
        output_audio_dir=output_audio_dir,
        apply_separation=apply_separation,
    )
