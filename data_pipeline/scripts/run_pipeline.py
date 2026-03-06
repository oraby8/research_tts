import os
import sys
import yaml
import argparse
import logging
import json
import torch
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.audio_utils import load_audio, save_audio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_jsonl(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def append_jsonl(filepath, data):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def stage_1_vad(config, input_dir, output_dir):
    """
    Stage 1: VAD and Audio Segmentation
    Loads VAD model, cuts audio into chunks.
    Resumable via 01_vad_segments.jsonl
    """
    logging.info("--- Starting Stage 1: VAD & Segmentation ---")
    from modules.vad import VADProcessor
    
    vad_file = os.path.join(output_dir, "01_vad_segments.jsonl")
    processed_files = set()
    if os.path.exists(vad_file):
        existing_data = load_jsonl(vad_file)
        processed_files = set([item['original_file'] for item in existing_data])
        logging.info(f"Resuming Stage 1: Found {len(processed_files)} previously processed files.")
        
    input_files = list(Path(input_dir).rglob("*.wav")) + list(Path(input_dir).rglob("*.mp3"))
    files_to_process = [f for f in input_files if str(f) not in processed_files]
    
    if not files_to_process:
        logging.info("Stage 1 complete! All files already processed.")
        return
        
    logging.info(f"Initializing VADProcessor for {len(files_to_process)} files...")
    vad = VADProcessor(config)
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
    
    for file_path in tqdm(files_to_process, desc="Stage 1 - VAD"):
        filename = Path(file_path).stem
        try:
            waveform, sr = load_audio(file_path, target_sr=config['sample_rate'])
        except Exception as e:
            logging.error(f"Failed to load {file_path}: {e}")
            continue

        try:
            segments = vad.process({"waveform": waveform, "sample_rate": sr})
            segments = vad.filter_segments(segments)
        except Exception as e:
            logging.error(f"VAD failed for {file_path}: {e}")
            continue

        # Check if file had 0 segments to prevent retrying empty files
        if not segments:
            # We add a dummy entry just to mark the original file as processed
            append_jsonl(vad_file, {
                "original_file": str(file_path),
                "audio_filepath": None
            })
            continue

        for i, seg in enumerate(segments):
            start_sample = int(seg['start'] * sr)
            end_sample = int(seg['end'] * sr)
            
            segment_audio = waveform[:, start_sample:end_sample]
            duration = seg['end'] - seg['start']
            
            seg_filename = f"{filename}_{i:04d}.wav"
            seg_path = os.path.join(output_dir, "wavs", seg_filename)
            save_audio(segment_audio, seg_path, sr=sr)
            
            meta = {
                "original_file": str(file_path),
                "audio_filepath": seg_path,
                "duration": duration,
                "speaker": seg['speaker'],
                "start_time": seg['start'],
                "end_time": seg['end']
            }
            append_jsonl(vad_file, meta)
            
    # Cleanup memory
    del vad
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("--- Stage 1 Complete ---")

def stage_2_asr(config, output_dir):
    """
    Stage 2: ASR Transcription and Normalization
    Loads Whisper & OmniASR. Reads from 01_vad_segments.jsonl.
    Resumable via 02_transcriptions.jsonl
    """
    logging.info("--- Starting Stage 2: ASR & Normalization ---")
    
    vad_file = os.path.join(output_dir, "01_vad_segments.jsonl")
    asr_file = os.path.join(output_dir, "02_transcriptions.jsonl")
    
    if not os.path.exists(vad_file):
        logging.error(f"Stage 1 output missing: {vad_file}. Run --stage 1 first.")
        return
        
    all_segments = load_jsonl(vad_file)
    # Filter out the dummy "no segment" marker files
    segments_to_process = [s for s in all_segments if s.get('audio_filepath')]
    
    processed_segments = set()
    if os.path.exists(asr_file):
        existing_asr = load_jsonl(asr_file)
        processed_segments = set([item['audio_filepath'] for item in existing_asr])
        logging.info(f"Resuming Stage 2: Found {len(processed_segments)} previously transcribed segments.")

    pending_segments = [s for s in segments_to_process if s['audio_filepath'] not in processed_segments]
    
    if not pending_segments:
        logging.info("Stage 2 complete! All segments already transcribed.")
        return

    from modules.asr import ASRProcessor
    from modules.normalizer import TextNormalizer
    
    logging.info(f"Initializing ASRProcessor & Normalizer for {len(pending_segments)} segments...")
    asr = ASRProcessor(config)
    normalizer = TextNormalizer(config)
    
    for seg in tqdm(pending_segments, desc="Stage 2 - ASR"):
        seg_path = seg['audio_filepath']
        
        if not os.path.exists(seg_path):
            logging.warning(f"Audio file missing: {seg_path}")
            continue
            
        text, confidence, wer_diff = asr.transcribe(seg_path)
        
        if not text:
            logging.debug(f"Transcription rejected (WER/Empty): {seg_path}")
            # Mark as processed but rejected (so we don't retry)
            seg['text'] = None
            append_jsonl(asr_file, seg)
            
            # Clean up rejected audio to save space
            if os.path.exists(seg_path):
                os.remove(seg_path)
            continue
            
        normalized_text = normalizer.normalize(text)
        
        # Add ASR results to the segment metadata
        seg['raw_text'] = text
        seg['text'] = normalized_text
        seg['confidence'] = confidence
        seg['wer_diff'] = wer_diff
        
        append_jsonl(asr_file, seg)
        
    # Cleanup memory
    del asr
    del normalizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("--- Stage 2 Complete ---")

def stage_3_filter(config, output_dir):
    """
    Stage 3: Quality Filtering
    Loads filter models. Reads from 02_transcriptions.jsonl.
    Resumable via final_metadata.jsonl
    """
    logging.info("--- Starting Stage 3: Quality Filtering ---")
    
    asr_file = os.path.join(output_dir, "02_transcriptions.jsonl")
    final_file = os.path.join(output_dir, "final_metadata.jsonl")
    
    if not os.path.exists(asr_file):
        logging.error(f"Stage 2 output missing: {asr_file}. Run --stage 2 first.")
        return
        
    all_transcriptions = load_jsonl(asr_file)
    # Only process segments that actually have text (weren't rejected in Stage 2)
    transcriptions_to_process = [t for t in all_transcriptions if t.get('text')]
    
    processed_segments = set()
    if os.path.exists(final_file):
        existing_final = load_jsonl(final_file)
        # We track both successful and previously-rejected segments so we don't re-filter
        processed_segments = set([item['audio_filepath'] for item in existing_final])
        logging.info(f"Resuming Stage 3: Found {len(processed_segments)} previously filtered segments.")
        
    # We need a secondary file to track segments that FAILED the filter, so we don't retry them
    rejected_file = os.path.join(output_dir, "03_rejected_filter.jsonl")
    if os.path.exists(rejected_file):
        existing_rejected = load_jsonl(rejected_file)
        processed_segments.update([item['audio_filepath'] for item in existing_rejected])
        
    pending_segments = [t for t in transcriptions_to_process if t['audio_filepath'] not in processed_segments]
    
    if not pending_segments:
        logging.info("Stage 3 complete! All segments already filtered.")
        return

    from modules.filters import QualityFilter
    
    logging.info(f"Initializing QualityFilter for {len(pending_segments)} segments...")
    quality_filter = QualityFilter(config)
    
    for seg in tqdm(pending_segments, desc="Stage 3 - Filtering"):
        seg_path = seg['audio_filepath']
        
        # 1. Ratio Check
        if not quality_filter.check_ratio(seg['text'], seg['duration']):
            logging.debug(f"Ratio check failed: {seg_path}")
            append_jsonl(rejected_file, seg)
            if os.path.exists(seg_path):
                os.remove(seg_path)
            continue
            
        # 2. Speaker Consistency (Optional)
        # embedding = quality_filter.extract_embedding(segment_audio)
        # if not quality_filter.check_speaker_consistency(embedding, speaker_centroid): ...
        
        # Passes all filters! Add to final metadata.
        append_jsonl(final_file, seg)
        
    # Cleanup memory
    del quality_filter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("--- Stage 3 Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Sequential Best-of-Breed TTS Data Pipeline")
    parser.add_argument("--config", default="configs/pipeline_config.yaml", help="Path to config file")
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all", 
                        help="Stage to run (1: VAD, 2: ASR, 3: Filter, all: Run sequentially)")
    args = parser.parse_args()
    
    # Load Config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    config = load_config(config_path)
    
    input_dir = config['paths']['input_dir']
    output_dir = config['paths']['output_dir']
    
    if args.stage in ["1", "all"]:
        stage_1_vad(config, input_dir, output_dir)
        
    if args.stage in ["2", "all"]:
        stage_2_asr(config, output_dir)
        
    if args.stage in ["3", "all"]:
        stage_3_filter(config, output_dir)

if __name__ == "__main__":
    main()
