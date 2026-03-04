import torch
import librosa
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3tokenizer import S3_SR
from config import ASRConfig
from dataset import ArabicDataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

def check_lengths():
    print("Loading Model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    config = ASRConfig()
    
    print("Loading Dataset...")
    dataset = ArabicDataset(config, model)
    
    print(f"Checking {len(dataset)} samples for CTC compatibility...")
    
    invalid_count = 0
    total_speech_len = 0
    total_text_len = 0
    
    for i in tqdm(range(min(1000, len(dataset)))): # Check first 1000
        try:
            sample = dataset[i]
            audio = sample["audio"]
            text = sample["text"]
            
            # Audio -> Speech Tokens
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio
            audio_16k = librosa.resample(audio_np, orig_sr=model.sr, target_sr=S3_SR)
            
            with torch.no_grad():
                speech_tokens, speech_len = model.s3gen.tokenizer.forward([audio_16k], max_len=2000)
                
            # Text -> Text Tokens
            text_tokens = model.tokenizer.text_to_tokens(text, language_id=config.language_id)
            
            s_len = speech_tokens.shape[1]
            t_len = text_tokens.shape[1]
            
            # CTC Constraint: Input Length >= 2 * Target Length + 1 (Rule of thumb for convergence)
            # Hard Constraint: Input Length >= Target Length
            
            total_speech_len += s_len
            total_text_len += t_len
            
            if s_len < t_len:
                print(f"❌ INVALID SAMPLE {i}: Speech Len {s_len} < Text Len {t_len}")
                print(f"Text: {text}")
                invalid_count += 1
            elif s_len < t_len * 2:
                 # Warning zone
                 pass
                 
        except Exception as e:
            print(f"Error processing {i}: {e}")
            
    print("\nSTATS:")
    print(f"Invalid CTC Samples: {invalid_count} / 1000")
    print(f"Avg Speech Len: {total_speech_len/1000:.1f}")
    print(f"Avg Text Len: {total_text_len/1000:.1f}")
    print(f"Ratio (Speech/Text): {total_speech_len/total_text_len:.2f}")
    
    if total_speech_len/total_text_len < 2.0:
        print("\n⚠️ WARNING: Speech tokens are very short relative to text!")
        print("S3 Tokenizer might be compressing too much (25Hz?).")
        print("CTC needs redundancy (blanks) to work well.")

if __name__ == "__main__":
    check_lengths()
