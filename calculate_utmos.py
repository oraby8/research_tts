import argparse
import json
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def main():
    parser = argparse.ArgumentParser(description="UTMOS Evaluation")
    parser.add_argument("-w", "--wav-dir", type=str, required=True, help="Audio file path.")
    args = parser.parse_args()

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device)

    audio_extensions = {".wav", ".mp3", ".flac"}
    audio_paths = [p for p in Path(args.wav_dir).rglob("*") if p.suffix in audio_extensions]
    utmos_score = 0

    utmos_result_path = Path(args.wav_dir) / "_utmos_results.jsonl"
    with open(utmos_result_path, "w", encoding="utf-8") as f:
        for audio_path in tqdm(audio_paths, desc="Processing"):
            wav, sr = torchaudio.load(audio_path)
            wav_tensor = wav.to(device)
            score = predictor(wav_tensor, sr)
            line = {}
            line["wav"], line["utmos"] = str(audio_path.stem), score.item()
            utmos_score += score.item()
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        avg_score = utmos_score / len(audio_paths) if len(audio_paths) > 0 else 0
        f.write(f"\nUTMOS: {avg_score:.4f}\n")

    print(f"UTMOS: {avg_score:.4f}")
    print(f"UTMOS results saved to {utmos_result_path}")


if __name__ == "__main__":
    main()
