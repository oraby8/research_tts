import os
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
import logging
from .utils import get_next_device

class VADProcessor:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.device = get_next_device()
        self._init_pipeline()
        
    def _init_pipeline(self):
        token = self.config.get('vad', {}).get('pyannote_token') or os.environ.get("HF_TOKEN")
        if not token:
            logging.warning("Pyannote token not found. VAD might fail if using gated models.")
            
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=token
            )
            device = torch.device(self.device)
            self.pipeline.to(device)
            
            # Set lower batch size to avoid OOM on large files
            batch_size = self.config.get('vad', {}).get('batch_size', 8)
            if hasattr(self.pipeline, 'segmentation_batch_size'):
                self.pipeline.segmentation_batch_size = batch_size
            if hasattr(self.pipeline, 'embedding_batch_size'):
                self.pipeline.embedding_batch_size = batch_size
                
        except Exception as e:
            logging.error(f"Failed to load Pyannote pipeline: {e}")
            raise

    def process(self, audio_input):
        """
        Run diarization on audio file or waveform dict.
        Returns list of segments: [{'start': 0.0, 'end': 1.5, 'speaker': 'SPEAKER_00'}, ...]
        """
        if not self.pipeline:
             raise RuntimeError("VAD Pipeline not initialized")
             
        diarization = self.pipeline(audio_input)
        
        if hasattr(diarization, "speaker_diarization"):
            diarization = diarization.speaker_diarization
            
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
            
        return segments

    def filter_segments(self, segments):
        """
        Filter segments based on min/max duration.
        """
        min_dur = self.config['vad']['min_segment_duration']
        max_dur = self.config['vad']['max_segment_duration']
        
        filtered = []
        for seg in segments:
            duration = seg['end'] - seg['start']
            if min_dur <= duration <= max_dur:
                filtered.append(seg)
                
        return filtered
