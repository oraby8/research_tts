import os
import argparse
import subprocess
import torch
import torchaudio
import json
from pathlib import Path

try:
    from silero_vad import (
        load_silero_vad,
        get_speech_timestamps,
        save_audio,
        read_audio,
    )
except ImportError:
    print("WARNING: silero-vad not found. Please run: pip install silero-vad")


def download_youtube_audio(url, output_dir, max_downloads=10):
    """Downloads audio from YouTube using yt-dlp in 24kHz mono."""
    print(f"📥 Downloading audio from {url}...")

    # We download as wav, 24kHz, mono to match Chatterbox config
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--postprocessor-args",
        "-ar 24000 -ac 1",
        "--max-downloads",
        str(max_downloads),  # Limit downloads for testing
        "-o",
        os.path.join(output_dir, "%(id)s_full.wav"),
        url,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Successfully downloaded audio(s) to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading {url}. Code: {e.returncode}")
        # Note: yt-dlp returns an error if max-downloads is reached, so we don't completely fail.
        return True


def split_audio_with_vad(
    audio_path, output_dir, prefix, min_duration=2.0, max_duration=12.0
):
    """Uses Silero VAD to split long audio into speech chunks."""
    print(f"✂️ Splitting audio {audio_path} using VAD...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        model = load_silero_vad(onnx=False)
    except Exception as e:
        print(f"❌ Failed to load Silero VAD: {e}")
        return []

    try:
        wav = read_audio(audio_path, sampling_rate=24000)
    except Exception as e:
        print(f"❌ Failed to read audio {audio_path}: {e}")
        return []

    # Silero VAD supports 16000 Hz, so we resample just for timestamp detection
    wav_16k = torchaudio.transforms.Resample(24000, 16000)(wav)

    speech_timestamps = get_speech_timestamps(
        wav_16k,
        model,
        sampling_rate=16000,
        min_speech_duration_ms=int(min_duration * 1000),
        max_speech_duration_s=max_duration,
        min_silence_duration_ms=500,
    )

    chunk_paths = []

    for i, ts in enumerate(speech_timestamps):
        # Convert 16000 Hz timestamps back to 24000 Hz
        start_sample = int(ts["start"] * (24000 / 16000))
        end_sample = int(ts["end"] * (24000 / 16000))

        chunk_name = f"{prefix}_chunk_{i:04d}.wav"
        chunk_path = os.path.join(output_dir, chunk_name)

        save_audio(chunk_path, wav[start_sample:end_sample], sampling_rate=24000)
        chunk_paths.append(chunk_path)

    print(f"✅ Extracted {len(chunk_paths)} speech chunks.")
    return chunk_paths


def process_youtube_list(urls_file, output_dir, chunk_dir, max_downloads):
    """Process a list of YouTube URLs."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(chunk_dir, exist_ok=True)

    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    all_chunks = []

    for i, url in enumerate(urls):
        print(f"\n--- Processing URL {i + 1}/{len(urls)} ---")
        download_youtube_audio(url, output_dir, max_downloads)

    # Process all downloaded files in the temp directory
    wav_files = list(Path(output_dir).glob("*_full.wav"))
    print(f"\n🔍 Found {len(wav_files)} full audio files to split...")

    for full_audio_path in wav_files:
        vid_id = full_audio_path.name.replace("_full.wav", "")
        chunks = split_audio_with_vad(
            str(full_audio_path),
            chunk_dir,
            prefix=vid_id,
            min_duration=2.0,
            max_duration=12.0,
        )
        all_chunks.extend(chunks)

        try:
            os.remove(full_audio_path)
            print(f"🧹 Cleaned up full audio {full_audio_path}")
        except OSError:
            pass

    metadata = {"chunks": all_chunks}
    with open(os.path.join(chunk_dir, "vad_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n🎉 Stage 1 Complete! Total speech chunks extracted: {len(all_chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: Download YouTube and VAD chunking"
    )
    parser.add_argument(
        "--urls",
        type=str,
        required=True,
        help="Text file with YouTube URLs (one per line)",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default="./dpo_temp_full_audio",
        help="Temp dir for full downloads",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./dpo_raw_chunks",
        help="Output directory for VAD chunks",
    )
    parser.add_argument(
        "--max_downloads", type=int, default=10, help="Max videos to download per URL"
    )

    args = parser.parse_args()
    process_youtube_list(args.urls, args.temp_dir, args.out_dir, args.max_downloads)
