import os
import torch
from pyannote.audio import Pipeline
from pyannote.core import Segment
import logging

class VADProcessor:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self._init_pipeline()
        
    def _init_pipeline(self):
        token = self.config.get('vad', {}).get('pyannote_token') or os.environ.get("HF_TOKEN")
        if not token:
            logging.warning("Pyannote token not found. VAD might fail if using gated models.")
            
        try:
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pipeline.to(device)
        except Exception as e:
            logging.error(f"Failed to load Pyannote pipeline: {e}")
            raise

    def process(self, audio_path):
        """
        Run diarization on audio file.
        Returns list of segments: [{'start': 0.0, 'end': 1.5, 'speaker': 'SPEAKER_00'}, ...]
        """
        if not self.pipeline:
             raise RuntimeError("VAD Pipeline not initialized")
             
        diarization = self.pipeline(audio_path)
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
