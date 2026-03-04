# Best-of-Breed TTS Data Pipeline

This pipeline combines the best practices from **CosyVoice 3**, **GLM-TTS**, and **MiniMax-Speech** to prepare high-quality TTS datasets.

## Features
1.  **Robust Ingestion:** Resamples to 24kHz, standardizes loudness.
2.  **Pyannote VAD:** Strict speaker diarization/segmentation (discards < 1.5s).
3.  **Consensus Transcription:** Uses **Faster-Whisper** (Primary) and **FunASR** (Secondary) to verify transcripts. Discards ambiguous audio (WER > 15%).
4.  **Acoustic Punctuation:** (TODO) Uses MFA to align text and restore punctuation based on silence.
5.  **Quality Filtering:** Checks Text/Audio ratio and Speaker Consistency (Cam++).

## Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to accept `pyannote/speaker-diarization-3.1` terms on HuggingFace and set `HF_TOKEN`.*

2.  Configure `configs/pipeline_config.yaml`:
    *   Set your `pyannote_token` or use `HF_TOKEN` env var.
    *   Adjust paths.

## Usage
1.  Place raw audio files in `input_audio/`.
2.  Run the pipeline:
    ```bash
    python scripts/run_pipeline.py
    ```
3.  Check `output_dataset/metadata.jsonl` and `output_dataset/wavs/`.
