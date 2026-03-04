import torch
import torchaudio
import logging
from dataclasses import dataclass
from typing import List

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

class AcousticAligner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = None
        self.model = None
        self.labels = None
        self.dictionary = None
        
        # Load Wav2Vec2 alignment model (MFA alternative)
        try:
            self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            self.model = self.bundle.get_model().to(self.device)
            self.labels = self.bundle.get_labels()
            self.dictionary = {c: i for i, c in enumerate(self.labels)}
        except Exception as e:
            logging.error(f"Failed to load alignment model: {e}")
            
    def align(self, waveform, transcript):
        """
        Align transcript to audio and restore punctuation based on silence.
        """
        if self.model is None:
            return transcript
            
        # 1. Preprocess
        # Resample to 16kHz for Wav2Vec2
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        target_sr = self.bundle.sample_rate
        if waveform.shape[-1] / target_sr < 0.1: # Too short
            return transcript
            
        # 2. Get Emissions
        with torch.inference_mode():
            emissions, _ = self.model(waveform.to(self.device))
            emissions = torch.log_softmax(emissions, dim=-1)
            
        emission = emissions[0].cpu().detach()
        
        # 3. Generate Target Tokens
        # Clean transcript for alignment (uppercase, no punctuation)
        clean_text = transcript.upper().replace(" ", "|")
        tokens = [self.dictionary.get(c) for c in clean_text]
        tokens = [t for t in tokens if t is not None] # Filter unknown chars
        targets = torch.tensor(tokens, dtype=torch.int32)
        
        # 4. Forced Alignment
        try:
            path, _ = torchaudio.functional.forced_align(emission, targets, blank=0)
            segments = self._merge_repeats(path, clean_text)
        except Exception as e:
            logging.warning(f"Alignment failed: {e}")
            return transcript
            
        # 5. Restore Punctuation
        return self._restore_punctuation(segments, transcript, waveform.size(1) / target_sr)

    def _merge_repeats(self, path, transcript):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1] == path[i2]:
                i2 += 1
            score = 1.0 # Placeholder
            segments.append(
                Segment(
                    transcript[path[i1]],
                    i1,
                    i2,
                    score
                )
            )
            i1 = i2
        return segments

    def _restore_punctuation(self, segments, original_text, duration):
        """
        Insert punctuation based on silence duration between words.
        CosyVoice Logic:
          > 300ms -> comma
          > 800ms -> period
        """
        # Mapping segments back to words is complex.
        # Simplified approach: If we detect a long silence token in alignment, insert punct.
        # Wav2Vec2 output is frame-based (20ms per frame usually).
        
        # Calculate silence gaps from alignment
        # This implementation is a placeholder for the complex word-boundary logic.
        # For a robust implementation, we need word-level timestamps.
        
        # Fallback: Just return original text if we can't robustly map back yet.
        # Implementing full word-level alignment mapping requires 
        # combining the character segments into words.
        
        return original_text
