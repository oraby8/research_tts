#!/bin/bash
# ==========================================
# UAE DPO DATA PIPELINE
# Target: @EmaratTV/shorts
# ==========================================

set -e

# Configuration
URLS_FILE="youtube_urls_uae.txt"
DPO_DATA_DIR="../../audio_data_dpo_uae"
MAX_DOWNLOADS=100  # Set to desired number of videos

TEMP_FULL_AUDIO="$DPO_DATA_DIR/00_temp_full_downloads"
HUMAN_CHUNKS="$DPO_DATA_DIR/01_human_chunks"
SFT_METADATA="$DPO_DATA_DIR/02_sft_metadata.csv"
SYNTH_LOSERS="$DPO_DATA_DIR/03_synth_losers"
FINAL_DPO_CSV="$DPO_DATA_DIR/processed_dpo_metadata.csv"

# Since you probably don't have a UAE SFT checkpoint yet, we use None to use base model
SFT_MODEL_CHECKPOINT=""

mkdir -p "$DPO_DATA_DIR"

echo "=========================================="
echo "🚀 STAGE 1: Downloading & VAD Chunking (UAE)"
echo "=========================================="
python 01_download_and_vad.py \
    --urls "$URLS_FILE" \
    --temp_dir "$TEMP_FULL_AUDIO" \
    --out_dir "$HUMAN_CHUNKS" \
    --max_downloads "$MAX_DOWNLOADS"

echo "=========================================="
echo "🎙️ STAGE 2: Whisper Auto-Transcription (UAE)"
echo "=========================================="
python 02_transcribe_whisper.py \
    --chunk_dir "$HUMAN_CHUNKS" \
    --output_csv "$SFT_METADATA"

echo "=========================================="
echo "🤖 STAGE 3: Synthesizing 'Loser' Audio (UAE)"
echo "=========================================="
if [ -z "$SFT_MODEL_CHECKPOINT" ]; then
    python 03_generate_losers.py \
        --sft_csv "$SFT_METADATA" \
        --human_audio_dir "$HUMAN_CHUNKS" \
        --synth_audio_dir "$SYNTH_LOSERS" \
        --output_csv "$FINAL_DPO_CSV"
else
    python 03_generate_losers.py \
        --sft_csv "$SFT_METADATA" \
        --human_audio_dir "$HUMAN_CHUNKS" \
        --synth_audio_dir "$SYNTH_LOSERS" \
        --output_csv "$FINAL_DPO_CSV" \
        --model_checkpoint "$SFT_MODEL_CHECKPOINT"
fi

echo "=========================================="
echo "🎉 UAE PIPELINE COMPLETE!"
echo "Your DPO dataset is ready at: $FINAL_DPO_CSV"
echo "=========================================="