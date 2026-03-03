import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import pyarabic.araby as araby
import re
from transformers import pipeline


def normalize_arabic_text(text):
    if not text:
        return ""
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_english(text):
    """Simple check to throw out chunks that are heavily code-switched (if desired)."""
    return bool(re.search(r"[a-zA-Z]", text))


def transcribe_chunks(
    input_dir,
    output_csv,
    model_name="openai/whisper-large-v3-turbo",
    filter_english=True,
):
    print(f"🎙️ Loading Whisper Model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline("automatic-speech-recognition", model=model_name, device=device)

    # Find all wavs
    wav_files = list(Path(input_dir).glob("*.wav"))
    print(f"🔍 Found {len(wav_files)} chunks in {input_dir}")

    transcriptions = []

    with open(output_csv, "w", encoding="utf-8") as f:
        # Write header
        f.write("file_name,transcription,normalized_transcription\n")

        for wav_path in tqdm(wav_files, desc="Transcribing"):
            try:
                res = pipe(str(wav_path), generate_kwargs={"language": "arabic"})
                text = res["text"].strip()

                # Filtering logic
                if not text:
                    continue
                if filter_english and contains_english(text):
                    continue

                norm_text = normalize_arabic_text(text)

                # Ensure the text is long enough (throw out simple "um" or "ah")
                if len(norm_text.split()) < 2:
                    continue

                # Append to CSV
                file_name = wav_path.name
                f.write(f"{file_name},{text},{norm_text}\n")
                f.flush()

                transcriptions.append(
                    {"file_name": file_name, "text": text, "norm": norm_text}
                )

            except Exception as e:
                print(f"❌ Failed to transcribe {wav_path}: {e}")

    print(f"✅ Transcribed {len(transcriptions)} valid chunks. Saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 2: Transcribe VAD chunks with Whisper"
    )
    parser.add_argument(
        "--chunk_dir",
        type=str,
        required=True,
        help="Directory containing the extracted VAD chunks",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="sft_metadata.csv",
        help="Output CSV with transcripts",
    )
    parser.add_argument(
        "--allow_english",
        action="store_true",
        help="Allow code-switching in transcripts",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Whisper model to use",
    )

    args = parser.parse_args()

    transcribe_chunks(
        args.chunk_dir, args.output_csv, args.model, not args.allow_english
    )
