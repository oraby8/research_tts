import torch
import logging
import numpy as np
from .utils import get_next_device

class QualityFilter:
    def __init__(self, config):
        self.config = config
        self.speaker_encoder = None
        self.device = get_next_device()
        self._init_encoder()
        
    def _init_encoder(self):
        # Initialize Speaker Encoder (e.g., Cam++ or WavLM)
        # For now, we'll use a placeholder or load if configured
        model_name = self.config['filtering']['speaker_consistency']['model']
        if model_name == "cam++":
             # Load Cam++ via FunASR or ModelScope
             try:
                 from modelscope.pipelines import pipeline
                 from modelscope.utils.constant import Tasks
                 # self.speaker_encoder = pipeline(Tasks.speaker_verification, model='damo/speech_campplus_sv_zh-cn_16k-common-advanced')
                 pass
             except ImportError:
                 logging.warning("ModelScope/Cam++ not found. Speaker consistency check might be skipped.")
    
    def check_ratio(self, text, duration):
        """
        Check if text-to-audio ratio is valid.
        """
        if duration <= 0: return False
        
        # Simple token count (approximation)
        num_tokens = len(text.split())
        if self.config.get('alignment', {}).get('text_norm', {}).get('lang') == 'ar':
             num_tokens = len(text) # Character count for Arabic
             
        ratio = num_tokens / duration
        min_r = self.config['filtering']['ratio']['min_tokens_per_sec']
        max_r = self.config['filtering']['ratio']['max_tokens_per_sec']
        
        if min_r <= ratio <= max_r:
            return True
        logging.debug(f"Ratio check failed: {ratio:.2f} tokens/sec (Tokens: {num_tokens}, Dur: {duration:.2f}s)")
        return False

    def check_speaker_consistency(self, audio_embedding, speaker_centroid):
        """
        Check cosine similarity between current segment and speaker centroid.
        """
        if audio_embedding is None or speaker_centroid is None:
            return True
            
        sim = torch.cosine_similarity(audio_embedding, speaker_centroid, dim=-1)
        threshold = self.config['filtering']['speaker_consistency']['min_similarity']
        
        if sim < threshold:
             logging.info(f"Speaker consistency failed: {sim:.4f} < {threshold}")
             return False
        return True
