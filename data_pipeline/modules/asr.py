import logging
import torch
import jiwer
from faster_whisper import WhisperModel
from .utils import get_next_device
try:
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
except ImportError:
    ASRInferencePipeline = None

class ASRProcessor:
    def __init__(self, config):
        self.config = config
        self.whisper = None
        self.omni_asr = None
        
        self.whisper_device = get_next_device()
        self.omni_device = get_next_device()
        
        self._init_models()
        
    def _init_models(self):
        # Initialize Whisper (Model A)
        model_a_conf = self.config['asr']['model_a']
        try:
            logging.info(f"Loading Whisper model: {model_a_conf['name']} on {self.whisper_device}")
            
            # Extract device type and index for faster-whisper
            device_type = "cuda" if self.whisper_device.startswith("cuda") else "cpu"
            device_index = int(self.whisper_device.split(":")[1]) if ":" in self.whisper_device else 0
            
            self.whisper = WhisperModel(
                model_a_conf['size'],
                device=device_type,
                device_index=device_index,
                compute_type="float16" if device_type == "cuda" else "int8"
            )
        except Exception as e:
            logging.error(f"Failed to load Whisper: {e}")
            raise

        # Initialize OmniASR-LLM-7B (Model B)
        model_b_conf = self.config['asr'].get('model_b', {})
        if ASRInferencePipeline:
            try:
                model_card = model_b_conf.get('model_card', 'omniASR_LLM_7B')
                logging.info(f"Loading OmniASR model: {model_card} on {self.omni_device}")
                self.omni_asr = ASRInferencePipeline(model_card=model_card, device=self.omni_device)
            except Exception as e:
                logging.warning(f"Failed to load OmniASR: {e}. Secondary checks will be skipped.")
        else:
            logging.warning("omnilingual_asr not installed. Skipping secondary model.")

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
        
        # 2. Run Secondary (OmniASR-LLM-7B) for cross-validation
        omni_text = ""
        wer = 0.0

        # Map faster-whisper language code to BCP-47 script-tagged format expected by OmniASR
        # e.g. "ar" -> "arb_Arab", "en" -> "eng_Latn".  Falls back to passing raw code.
        _LANG_MAP = {
            "ar": "arb_Arab",
            "en": "eng_Latn",
            "de": "deu_Latn",
            "fr": "fra_Latn",
            "zh": "zho_Hans",
            "es": "spa_Latn",
        }
        omni_lang = _LANG_MAP.get(detected_lang, detected_lang)

        if self.omni_asr:
            try:
                results = self.omni_asr.transcribe(
                    [audio_path],
                    lang=[omni_lang],
                    batch_size=1,
                )
                omni_text = results[0].strip() if results else ""

                # Compare with Whisper output using CER
                if omni_text:
                    wer = jiwer.cer(whisper_text, omni_text)

            except Exception as e:
                logging.warning(f"OmniASR inference failed: {e}")
        
        # 3. Consensus Decision
        threshold = self.config['asr']['consensus']['cer_threshold']

        # If both models ran and their outputs diverge too much, reject the segment
        if omni_text and wer > threshold:
            logging.info(
                f"Consensus failed. CER: {wer:.2f} > {threshold}."
                f"\nWhisper:  {whisper_text}\nOmniASR: {omni_text}"
            )
            return None, 0.0, wer  # Reject

        # Return Whisper text as primary
        return omni_text, 1.0, wer  # 1.0 is placeholder confidence
