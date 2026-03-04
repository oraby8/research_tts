import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import os
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3tokenizer import S3_SR
from config import ASRConfig
from dataset import TestArabicDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redefine to match pretrain_reward_model.py
class Token2TextRewardModel(nn.Module):
    def __init__(self, speech_vocab_size, text_vocab_size, hidden_dim=512):
        super().__init__()
        self.emb = nn.Embedding(speech_vocab_size + 1, hidden_dim, padding_idx=speech_vocab_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.0, # Dropout irrelevant for inference
            batch_first=True,
            norm_first=True # Must match training!
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=18)
        self.proj = nn.Linear(hidden_dim, text_vocab_size)

    def forward(self, speech_tokens, src_key_padding_mask=None):
        x = self.emb(speech_tokens)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.proj(x)

def greedy_decoder(logits, blank_idx=0):
    """Simple CTC Greedy Decoder"""
    # logits: [T, V]
    # collapse repeated, remove blank
    probs = F.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    
    decoded = []
    prev_idx = -1
    for idx in preds:
        idx = idx.item()
        if idx != blank_idx and idx != prev_idx:
            decoded.append(idx)
        prev_idx = idx
    return decoded

def test_reward_model():
    print("Loading Base TTS Model and Tokenizers...")
    config = ASRConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
    model.t3.eval()
    model.s3gen.eval()

    print("\nLoading Reward Model (token2text_reward.pt)...")
    
    # Get vocabs from base model
    speech_vocab = model.t3.hp.speech_tokens_dict_size
    text_vocab = model.t3.hp.text_tokens_dict_size
    
    reward_model = Token2TextRewardModel(speech_vocab, text_vocab, hidden_dim=512).to(device)
    
    try:
        # Load weights
        checkpoint_path = "checkpoints/reward_model/token2text_reward.pt"
        if not os.path.exists(checkpoint_path):
            checkpoint_path = "token2text_reward.pt" # Fallback
            
        print(f"Loading from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Check for transposed weights (legacy handling if loaded from old Linear saving style)
        if "emb.weight" in state_dict:
            w = state_dict["emb.weight"]
            target_shape = reward_model.emb.weight.shape # [8195, 256]
            
            # Case 1: [256, 8194] -> Needs Transpose + Padding
            if w.shape[0] == target_shape[1] and w.shape[1] == target_shape[0] - 1:
                 print("Transposing weights and adding padding...")
                 w = w.t() # [8194, 256]
                 padding_row = torch.zeros(1, w.shape[1]).to(w.device)
                 w = torch.cat([w, padding_row], dim=0) # [8195, 256]
            
            # Case 2: [8194, 256] -> Just needs Padding
            elif w.shape[0] == target_shape[0] - 1:
                 print("Adding padding row...")
                 padding_row = torch.zeros(1, w.shape[1]).to(w.device)
                 w = torch.cat([w, padding_row], dim=0)
            
            state_dict["emb.weight"] = w
            
        reward_model.load_state_dict(state_dict)
        print("✅ Reward Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load Token2TextRewardModel: {e}")
        print("Continuing with random weights (Expect bad results)...")

    reward_model.eval()
    
    print("\nLoading Evaluation Dataset...")
    test_ds = TestArabicDataset(config, model)
    from tqdm import tqdm
    print(f"Dataset Size: {len(test_ds)}")
    
    total_samples = 0
    total_wer = 0.0 # Actually CER (Character Error Rate) since tokens are chars
    
    pbar = tqdm(range(len(test_ds)), desc="Benchmarking (CTC)")
    
    for i in pbar:
        try:
            sample = test_ds[i]
            audio = sample["audio"]
            text_gt = sample["text"] # String
            
            # Audio preprocessing
            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio
            audio_16k = librosa.resample(audio_np, orig_sr=model.sr, target_sr=S3_SR)
            
            # Speech Tokenization
            with torch.no_grad():
                speech_tokens, speech_token_lens = model.s3gen.tokenizer.forward([audio_16k], max_len=1000)
                speech_tokens = speech_tokens.to(device)
            
            # Pad
            sot_speech = model.t3.hp.start_speech_token
            eot_speech = model.t3.hp.stop_speech_token
            speech_tokens = F.pad(speech_tokens, (1, 0), value=sot_speech)
            speech_tokens = F.pad(speech_tokens, (0, 1), value=eot_speech)
            
            # Inference
            with torch.no_grad():
                # [B, T, V]
                logits = reward_model(speech_tokens) 
                
            # Decode (Greedy)
            # Logits shape [1, T, V]
            decoded_indices = greedy_decoder(logits[0])
            
            # Convert indices back to string?
            # Chatterbox tokenizer doesn't easily expose ID->Str?
            # We can compare at ID level if we tokenize GT.
            
            gt_tokens = model.tokenizer.text_to_tokens(text_gt, language_id=config.language_id).squeeze(0).tolist()
            
            # Calculate simple CER (Levenshtein distance)
            import editdistance
            dist = editdistance.eval(decoded_indices, gt_tokens)
            length = max(len(gt_tokens), 1)
            cer = dist / length
            
            total_wer += cer
            total_samples += 1
            
            avg_cer = total_wer / total_samples
            pbar.set_postfix({"CER": f"{avg_cer:.4f}"})
            
            if i < 3: # Print first few examples
                print(f"\nExample {i}:")
                print(f"GT IDs: {gt_tokens[:10]}...")
                print(f"Pred IDs: {decoded_indices[:10]}...")
                print(f"CER: {cer:.2f}")
                
        except Exception as e:
            # print(f"Error skipping sample: {e}")
            pass

    if total_samples > 0:
        print("\n" + "="*50)
        print(f"📊 FINAL CTC BENCHMARK")
        print(f"  - **Average CER**: {total_wer/total_samples:.4f}")
        print("="*50)

if __name__ == "__main__":
    try:
        import editdistance
    except ImportError:
        print("Please install editdistance: pip install editdistance")
        exit(1)
    test_reward_model()
