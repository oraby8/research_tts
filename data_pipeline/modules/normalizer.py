import re
import logging
from num2words import num2words

class TextNormalizer:
    def __init__(self, config):
        self.config = config
        self.lang = config.get('alignment', {}).get('text_norm', {}).get('lang', 'en')
        
    def normalize(self, text):
        """
        Normalize text for TTS: expand numbers, clean punctuation.
        """
        if not text:
            return ""
            
        # Basic cleanup
        text = text.strip()
        
        # Language-specific normalization
        if self.lang == 'ar':
            text = self._normalize_ar(text)
        else:
            text = self._normalize_en(text)
            
        # Remove unwanted characters (brackets, weird symbols)
        text = re.sub(r'[\[\(\)\]\{\}<>]', '', text)
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _normalize_en(self, text):
        """
        English-specific normalization.
        """
        # Convert numbers to words (simple implementation)
        # "123" -> "one hundred twenty-three"
        def replace_num(match):
            try:
                return num2words(match.group(0), lang='en')
            except:
                return match.group(0)
                
        text = re.sub(r'\d+', replace_num, text)
        
        # Expand common abbreviations (can be extended)
        replacements = {
            "Mr.": "Mister",
            "Mrs.": "Missus",
            "Dr.": "Doctor",
            "St.": "Saint",
            "&": "and"
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
            
        return text

    def _normalize_ar(self, text):
        """
        Arabic-specific normalization.
        """
        # Convert numbers to Arabic characters
        def replace_num(match):
            try:
                return num2words(match.group(0), lang='ar')
            except:
                return match.group(0)
        
        text = re.sub(r'\d+', replace_num, text)
        
        # Normalize punctuation (full-width to half-width or vice-versa if needed)
        # Here we just ensure consistency
        text = text.replace('،', ',').replace('۔', '.').replace('؟', '?').replace('!', '!')
        
        return text
