import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from chatterbox import ChatterboxMultilingualTTS
from chatterbox.models.s3tokenizer import S3_SR
from config import ASRConfig
from dataset import TestArabicDataset

class Token2TextRewardModel(nn.Module):
    def __init__(self, vocab_size=8194, hidden_dim=256, text_vocab_size=2454, num_layers=4):
        super().__init__()
        # Matches Embedding(8194, 256) behavior when bias=False and transposed
        self.emb = nn.Linear(vocab_size, hidden_dim, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_dim, text_vocab_size)
        
    def forward(self, speech_one_hot):
        x = self.emb(speech_one_hot)
        return self.proj(self.transformer(x))


def test_reward_model():
    print("Loading Base TTS Model and Tokenizers...")
    config = ASRConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Needs explicit padding_side="left" to avoid warnings (not functionally critical for this test)
    model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))
    model.t3.eval()
    model.s3gen.eval()

    print("\nLoading Reward Model (token2text_reward.pt)...")
    # Base on error log: text_vocab_size=2454, hidden_dim=256, speech_vocab_size (at training time) = 8194
    actual_speech_vocab = 8194 
    text_vocab_size = 2454 # Predicted keywords
    hidden_dim = 256
    
    reward_model = Token2TextRewardModel(
        vocab_size=actual_speech_vocab, 
        hidden_dim=hidden_dim,
        text_vocab_size=text_vocab_size,
        num_layers=8
    ).to(device)
    
    try:
        state_dict = torch.load("token2text_reward.pt", map_location=device)
        # Transpose emb.weight if from Embedding[8194, 256] -> Linear[256, 8194]
        if "emb.weight" in state_dict:
            if state_dict["emb.weight"].shape == (actual_speech_vocab, hidden_dim):
                state_dict["emb.weight"] = state_dict["emb.weight"].t()
        
        # Ensure we filter out any unwanted bias if it accidentally exists (it shouldn't)
        keys_to_delete = [k for k in state_dict if k == "emb.bias"]
        for k in keys_to_delete:
            del state_dict[k]

        reward_model.load_state_dict(state_dict)
        print("✅ Reward Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load Token2TextRewardModel: {e}")
        return
        
    reward_model.eval()
    
    print("\nLoading Evaluation Dataset...")
    test_ds = TestArabicDataset(config, model)
    from tqdm import tqdm
    print(f"Dataset Size: {len(test_ds)}")
    
    num_samples = len(test_ds)
    all_recalls = []
    all_precisions = []
    all_f1s = []
    all_accuracies = []
    
    # Special tokens to ignore in comparison (Start, End, Space, Language, Null)
    ignored_tokens = {0, 1, 2, 255, 256, 721}  # Adjust based on your vocab
    
    pbar = tqdm(range(num_samples), desc="Benchmarking Reward Model")
    for i in pbar:
        sample = test_ds[i]
        audio = sample["audio"]
        texts = [sample["text"]]
        
        # Audio preprocessing
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
            
        audio_16k = librosa.resample(audio_np, orig_sr=model.sr, target_sr=S3_SR)
        
        # Speech Tokenization
        try:
            with torch.no_grad():
                speech_tokens, speech_token_lens = model.s3gen.tokenizer.forward([audio_16k], max_len=1000)
                speech_tokens = speech_tokens.to(device)
                
            # Add SOT and EOT
            sot_speech = model.t3.hp.start_speech_token
            eot_speech = model.t3.hp.stop_speech_token
            speech_tokens = F.pad(speech_tokens, (1, 0), value=sot_speech)
            speech_tokens = F.pad(speech_tokens, (0, 1), value=eot_speech)
            
            with torch.no_grad():
                # Convert to soft one-hot
                speech_one_hot = torch.zeros(
                    speech_tokens.size(0), 
                    speech_tokens.size(1), 
                    actual_speech_vocab, 
                    device=device
                )
                speech_one_hot.scatter_(2, speech_tokens.unsqueeze(-1), 1.0)
                
                # Forward pass
                predicted_text_logits = reward_model(speech_one_hot)
                pooled_logits = predicted_text_logits.mean(dim=1)
                
                # Get Top-K probability text tokens
                probs = torch.sigmoid(pooled_logits)
                top_k = 20
                top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
                
                # STATISTICAL COMPARISON
                # 1. GT Content Tokens
                gt_tokens = model.tokenizer.text_to_tokens(texts[0], language_id=config.language_id).squeeze(0).tolist()
                filtered_gt = {t for t in set(gt_tokens) if t not in ignored_tokens}
                
                # 2. Predicted Content Tokens
                pred_tokens = top_indices[0].tolist()
                filtered_pred = {t for t in pred_tokens if t not in ignored_tokens}
                
                # 3. Metrics Calculation
                hits = filtered_pred.intersection(filtered_gt)
                
                # Recall: TP / (TP + FN)
                recall = len(hits) / max(len(filtered_gt), 1)
                # Precision: TP / (TP + FP)
                precision = len(hits) / max(len(filtered_pred), 1)
                # F1: 2 * (P * R) / (P + R)
                f1 = 2 * len(hits) / max(len(filtered_pred) + len(filtered_gt), 1)
                # Accuracy (Jaccard): TP / (TP + FP + FN)
                accuracy = len(hits) / max(len(filtered_pred.union(filtered_gt)), 1)
                
                all_recalls.append(recall)
                all_precisions.append(precision)
                all_f1s.append(f1)
                all_accuracies.append(accuracy)
                
                # Update progress bar with moving average
                avg_recall = sum(all_recalls) / len(all_recalls)
                pbar.set_postfix({"recall": f"{avg_recall:.4f}"})

        except Exception as e:
            continue

    if all_recalls:
        print("\n" + "="*50)
        print(f"📊 FINAL BENCHMARK RESULTS (Total Samples: {len(all_recalls)})")
        print(f"  - **Average Precision (Top 20)**: {sum(all_precisions)/len(all_precisions):.4f}")
        print(f"  - **Average Recall (Top 20)**:    {sum(all_recalls)/len(all_recalls):.4f}")
        print(f"  - **Average F1-Score**:          {sum(all_f1s)/len(all_f1s):.4f}")
        print(f"  - **Average Accuracy (Jaccard)**: {sum(all_accuracies)/len(all_accuracies):.4f}")
        print("="*50)
    else:
        print("No samples were successfully processed.")

    print("Benchmarking Complete.")

    print("Testing Complete.")

if __name__ == "__main__":
    test_reward_model()
