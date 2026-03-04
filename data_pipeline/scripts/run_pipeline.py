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
from modules.vad import VADProcessor
from modules.asr import ASRProcessor
from modules.filters import QualityFilter
from modules.normalizer import TextNormalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_file(file_path, config, vad, asr, quality_filter, normalizer, output_dir):
    """
    Process a single audio file: VAD -> Cut -> Transcribe -> Filter -> Save
    """
    filename = Path(file_path).stem
    
    # 1. Load Audio
    try:
        waveform, sr = load_audio(file_path, target_sr=config['sample_rate'])
    except Exception as e:
        logging.error(f"Failed to load {file_path}: {e}")
        return []

    # 2. VAD / Segmentation
    try:
        segments = vad.process(file_path)
        segments = vad.filter_segments(segments) # Apply min/max duration
    except Exception as e:
        logging.error(f"VAD failed for {file_path}: {e}")
        return []

    processed_segments = []
    
    for i, seg in enumerate(segments):
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        
        # Cut Audio
        segment_audio = waveform[:, start_sample:end_sample]
        duration = seg['end'] - seg['start']
        
        # Save temporarily for ASR (some ASR libs need file path)
        seg_filename = f"{filename}_{i:04d}.wav"
        seg_path = os.path.join(output_dir, "wavs", seg_filename)
        save_audio(segment_audio, seg_path, sr=sr)
        
        # 3. Transcribe (Consensus)
        text, confidence, wer_diff = asr.transcribe(seg_path)
        
        if not text:
            logging.debug(f"Transcription rejected (WER/Empty): {seg_filename}")
            os.remove(seg_path) # Cleanup
            continue
        
        # 4. Text Normalization
        # Normalize text before filtering and saving
        normalized_text = normalizer.normalize(text)
        
        # 5. Quality Filtering
        # Ratio Check (using Normalized text)
        if not quality_filter.check_ratio(normalized_text, duration):
            logging.debug(f"Ratio check failed: {seg_filename}")
            os.remove(seg_path)
            continue
            
        # Speaker Consistency (TODO: Needs embedding extraction first)
        # embedding = quality_filter.extract_embedding(segment_audio)
        # if not quality_filter.check_speaker_consistency(embedding, speaker_centroid): ...
        
        # 6. Save Metadata
        meta = {
            "audio_filepath": seg_path,
            "text": normalized_text,
            "raw_text": text,
            "duration": duration,
            "speaker": seg['speaker'],
            "original_file": str(file_path),
            "confidence": confidence,
            "wer_diff": wer_diff
        }
        processed_segments.append(meta)
        
    return processed_segments

def main():
    parser = argparse.ArgumentParser(description="Best-of-Breed TTS Data Pipeline")
    parser.add_argument("--config", default="configs/pipeline_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load Config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    config = load_config(config_path)
    
    # Setup Paths
    input_dir = config['paths']['input_dir']
    output_dir = config['paths']['output_dir']
    os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)
    
    # Initialize Modules
    logging.info("Initializing modules...")
    vad = VADProcessor(config)
    asr = ASRProcessor(config)
    quality_filter = QualityFilter(config)
    normalizer = TextNormalizer(config)
    
    # Run Pipeline
    all_metadata = []
    input_files = list(Path(input_dir).rglob("*.wav")) + list(Path(input_dir).rglob("*.mp3"))
    
    logging.info(f"Found {len(input_files)} files in {input_dir}")
    
    for file_path in tqdm(input_files):
        segments = process_file(file_path, config, vad, asr, quality_filter, normalizer, output_dir)
        all_metadata.extend(segments)
        
    # Save Final Metadata
    meta_path = os.path.join(output_dir, "metadata.jsonl")
    with open(meta_path, 'w', encoding='utf-8') as f:
        for item in all_metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    logging.info(f"Pipeline complete. {len(all_metadata)} segments saved to {meta_path}")

if __name__ == "__main__":
    main()
