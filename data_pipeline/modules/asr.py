import logging
import torch
import jiwer
from faster_whisper import WhisperModel
# from funasr import AutoModel # Uncomment when funasr is installed and verified
# Using a placeholder for FunASR to avoid import errors if not installed immediately
try:
    from funasr import AutoModel
except ImportError:
    AutoModel = None

class ASRProcessor:
    def __init__(self, config):
        self.config = config
        self.whisper = None
        self.funasr = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_models()
        
    def _init_models(self):
        # Initialize Whisper (Model A)
        model_a_conf = self.config['asr']['model_a']
        try:
            logging.info(f"Loading Whisper model: {model_a_conf['name']}")
            self.whisper = WhisperModel(
                model_a_conf['size'],
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
        except Exception as e:
            logging.error(f"Failed to load Whisper: {e}")
            raise

        # Initialize FunASR (Model B) - Optional/Language dependent
        model_b_conf = self.config['asr']['model_b']
        if AutoModel and model_b_conf.get('model_id'):
            try:
                logging.info(f"Loading FunASR model: {model_b_conf['model_id']}")
                self.funasr = AutoModel(
                    model=model_b_conf['model_id'],
                    device=self.device,
                    disable_update=True
                )
            except Exception as e:
                logging.warning(f"Failed to load FunASR: {e}. Secondary checks might be skipped.")
        else:
            logging.warning("FunASR not installed or configured. Skipping secondary model.")

    def transcribe(self, audio_path, lang="auto"):
        """
        Transcribe audio using consensus strategy.
        Returns:
            text (str): The best transcript (usually from Whisper).
            confidence (float): Confidence score.
            wer (float): WER/CER between models (if both ran).
        """
        # 1. Run Whisper
        segments, info = self.whisper.transcribe(audio_path, beam_size=5, language=None if lang=="auto" else lang)
        whisper_text = " ".join([s.text for s in segments]).strip()
        detected_lang = info.language
        
        # 2. Run Secondary (FunASR) if applicable (e.g., Chinese)
        funasr_text = ""
        wer = 0.0
        
        # Logic: If Chinese, use Paraformer as second opinion.
        # If English, we might skip or use another model if configured.
        if self.funasr and detected_lang == "zh":
            try:
                # FunASR inference
                res = self.funasr.generate(input=audio_path)
                # Parse result (structure depends on model, usually a list of dicts)
                if isinstance(res, list) and len(res) > 0:
                    funasr_text = res[0].get('text', '')
                
                # Normalize and Compare
                # Simple CER for Chinese
                wer = jiwer.cer(whisper_text, funasr_text)
                
            except Exception as e:
                logging.warning(f"FunASR inference failed: {e}")
        
        # 3. Consensus Decision
        threshold = self.config['asr']['consensus']['cer_threshold']
        
        # If we ran a second model and the discrepancy is high, flag it
        if funasr_text and wer > threshold:
            logging.info(f"Consensus failed. WER: {wer:.2f} > {threshold}. \nWhisper: {whisper_text}\nFunASR: {funasr_text}")
            return None, 0.0, wer # Reject
            
        # Return Whisper text as primary
        return whisper_text, 1.0, wer # 1.0 is placeholder confidence
