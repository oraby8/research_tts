#!/bin/bash
# ==========================================
# FULL END-TO-END DPO DATA PIPELINE
# YouTube -> VAD -> Whisper -> TTS Synthesizer -> DPO Metadata
# ==========================================

set -e

# Configuration
URLS_FILE="youtube_urls.txt"
DPO_DATA_DIR="../../audio_data_dpo"

TEMP_FULL_AUDIO="$DPO_DATA_DIR/00_temp_full_downloads"
HUMAN_CHUNKS="$DPO_DATA_DIR/01_human_chunks"
SFT_METADATA="$DPO_DATA_DIR/02_sft_metadata.csv"
SYNTH_LOSERS="$DPO_DATA_DIR/03_synth_losers"
FINAL_DPO_CSV="$DPO_DATA_DIR/processed_dpo_metadata.csv"
SFT_MODEL_CHECKPOINT="../../output_data_egyptian_partial/best_checkpoint/model.safetensors"

mkdir -p "$DPO_DATA_DIR"

echo "=========================================="
echo "🚀 STAGE 1: Downloading & VAD Chunking"
echo "=========================================="
if [ ! -f "$URLS_FILE" ]; then
    echo "⚠️ Warning: $URLS_FILE not found! Creating a dummy file."
    echo "# Add your YouTube URLs here" > "$URLS_FILE"
    echo "Please add URLs to $URLS_FILE and re-run."
    exit 1
fi

python 01_download_and_vad.py \
    --urls "$URLS_FILE" \
    --temp_dir "$TEMP_FULL_AUDIO" \
    --out_dir "$HUMAN_CHUNKS"

echo "=========================================="
echo "🎙️ STAGE 2: Whisper Auto-Transcription"
echo "=========================================="
python 02_transcribe_whisper.py \
    --chunk_dir "$HUMAN_CHUNKS" \
    --output_csv "$SFT_METADATA"

echo "=========================================="
echo "🤖 STAGE 3: Synthesizing 'Loser' Audio for DPO"
echo "=========================================="
# NOTE: If you haven't run SFT yet, remove the --model_checkpoint argument to just use the base model
python 03_generate_losers.py \
    --sft_csv "$SFT_METADATA" \
    --human_audio_dir "$HUMAN_CHUNKS" \
    --synth_audio_dir "$SYNTH_LOSERS" \
    --output_csv "$FINAL_DPO_CSV" \
    --model_checkpoint "$SFT_MODEL_CHECKPOINT"

echo "=========================================="
echo "🎉 PIPELINE COMPLETE!"
echo "Your DPO dataset is ready at: $FINAL_DPO_CSV"
echo "You can now run 'python train_dpo.py' from the main directory."
echo "=========================================="