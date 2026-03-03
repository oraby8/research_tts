import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm

import torch
from datasets import load_dataset, Audio
from transformers import pipeline
import pyarabic.araby as araby
from jiwer import wer as calculate_jiwer_wer


def normalize_arabic_text(text):
    if not text:
        return ""
    # Strip diacritics
    text = araby.strip_tashkeel(text)
    # Strip tatweel
    text = araby.strip_tatweel(text)
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Normalize Alifs
    text = re.sub(r"[أإآ]", "ا", text)
    # Normalize Taa Marbuta
    text = text.replace("ة", "ه")
    # Normalize Yaa
    text = text.replace("ى", "ي")
    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav-dir", type=str, default="test_swivid_outputs")
    parser.add_argument(
        "-d",
        "--dialect",
        type=str,
        default="UAE",
        help="MSA | SAU | UAE | ALG | IRQ | EGY | MAR",
    )
    parser.add_argument(
        "-m", "--model", type=str, default="openai/whisper-large-v3-turbo"
    )
    args = parser.parse_args()

    wav_dir = Path(args.wav_dir)
    if not wav_dir.exists():
        print(f"Directory {wav_dir} not found!")
        return

    print("Loading dataset...")
    ds = load_dataset("SWivid/Habibi", args.dialect, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))

    print(f"Loading ASR model {args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline("automatic-speech-recognition", model=args.model, device=device)

    wer_objs = []
    pr_text_norms = []
    gt_text_norms = []

    print("Calculating WER...")
    for idx, item in enumerate(tqdm(ds), start=1):
        gt_text = item.get("text", "")
        # Find corresponding generated file
        file_name = f"sample_{idx:02d}_generated.wav"
        audio_path = wav_dir / file_name

        if not audio_path.exists():
            continue

        try:
            # Transcribe
            res = pipe(str(audio_path), generate_kwargs={"language": "arabic"})
            pr_text = res["text"]

            # Normalize
            gt_text_norm = normalize_arabic_text(gt_text)
            pr_text_norm = normalize_arabic_text(pr_text)

            wer_value = calculate_jiwer_wer([gt_text_norm], [pr_text_norm])

            wer_obj = {
                "audio_path": str(audio_path),
                "gt_text": gt_text,
                "pr_text": pr_text,
                "gt_text_norm": gt_text_norm,
                "pr_text_norm": pr_text_norm,
                "wer": wer_value,
            }
            wer_objs.append(wer_obj)
            pr_text_norms.append(pr_text_norm)
            gt_text_norms.append(gt_text_norm)

        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")

    if not wer_objs:
        print("No files were processed.")
        return

    wer_result_path = wav_dir / f"_wer_results_{args.dialect}.jsonl"
    global_wer = calculate_jiwer_wer(gt_text_norms, pr_text_norms)

    with open(wer_result_path, "w", encoding="utf-8") as f:
        for wer_obj in wer_objs:
            f.write(json.dumps(wer_obj, ensure_ascii=False) + "\n")
        f.write(f"\nGlobal WER: {global_wer}\n")

    print(f"Processed {len(wer_objs)} files.")
    print(f"Global WER: {global_wer:.4f}")
    print(f"Results saved to {wer_result_path}")


if __name__ == "__main__":
    main()
