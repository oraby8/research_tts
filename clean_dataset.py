import os
import pandas as pd
import torch
import shutil
import sys
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Config
DATA_DIR = "/workspace/audio_data"
METADATA_FILE = "processed_metadata.csv"
BACKUP_FILE = "processed_metadata_backup.csv"
LANGUAGE_ID = "ar"
MAX_TEXT_POS_EMB = 2050  # From logs: 2050 (position embedding limit)


def main():
    metadata_path = os.path.join(DATA_DIR, METADATA_FILE)
    backup_path = os.path.join(DATA_DIR, BACKUP_FILE)

    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found.")
        return

    # Backup
    if not os.path.exists(backup_path):
        print(f"Backing up {metadata_path} to {backup_path}...")
        shutil.copy(metadata_path, backup_path)
    else:
        print(f"Backup {backup_path} already exists. Using original file.")

    # Load tokenizer
    print("Loading tokenizer...")
    try:
        # Load model on CPU to avoid GPU usage
        model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device("cpu"))
        tokenizer = model.tokenizer
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load CSV
    print("Loading CSV...")
    df = pd.read_csv(metadata_path)
    print(f"Original entries: {len(df)}")

    if "transcription" not in df.columns:
        print("Error: 'transcription' column not found.")
        return

    # Filter
    valid_rows = []
    removed_count = 0

    print("Filtering dataset...")
    for idx, row in df.iterrows():
        text = row["transcription"]
        if not isinstance(text, str):
            text = str(text) if pd.notna(text) else ""

        try:
            tokens = tokenizer.text_to_tokens(text, language_id=LANGUAGE_ID)
            # tokens is likely a tensor with shape [1, seq_len] or [seq_len]
            # Verify shape
            if isinstance(tokens, torch.Tensor):
                if tokens.dim() == 2:
                    tokens = tokens.squeeze(0)
                length = tokens.numel()
            else:
                # If list or numpy
                length = len(tokens)

            # Check length (adding 2 for special tokens as per train.py)
            if length + 2 <= MAX_TEXT_POS_EMB:
                valid_rows.append(row)
            else:
                removed_count += 1
                if removed_count <= 5:
                    print(f"Removing text (len {length + 2}): {text[:50]}...")
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # If error, maybe remove or keep? Removing is safer for training stability.
            removed_count += 1

    print(f"Removed {removed_count} entries.")
    print(f"Remaining entries: {len(valid_rows)}")

    # Save
    new_df = pd.DataFrame(valid_rows)
    # Ensure columns order is preserved if possible, but DataFrame constructor does decent job
    new_df.to_csv(metadata_path, index=False)
    print(f"Saved cleaned dataset to {metadata_path}")


if __name__ == "__main__":
    main()
