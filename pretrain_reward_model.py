import os
import torch
import torch.nn as nn
import logging
import librosa
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from accelerate import Accelerator

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3tokenizer import S3_SR
from config import ASRConfig
from dataset import ArabicDataset, collate_fn, TestArabicDataset
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Token2TextRewardModel(nn.Module):
    """The ASR Reward Model for DiffRO."""
    def __init__(self, speech_vocab_size, text_vocab_size, hidden_dim=512, dropout=0.25):
        super().__init__()
        # In this pre-training step, we take standard integer tokens, so we use an Embedding.
        # In opt1_true_diffro, it takes soft one-hot vectors, so it uses a Linear layer.
        # They are mathematically equivalent if the Linear layer has no bias.
        self.emb = nn.Embedding(speech_vocab_size + 1, hidden_dim, padding_idx=speech_vocab_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout, 
            batch_first=True,
            norm_first=True # Better stability for deeper models
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=18)  # Scaled up to 18 layers
        self.proj = nn.Linear(hidden_dim, text_vocab_size)

    def apply_spec_augment(self, x, mask_prob=0.1, mask_length=10):
        """Simple SpecAugment-like Time Masking on Embeddings"""
        if not self.training: 
            return x
            
        B, T, D = x.shape
        # Create a mask for time steps
        # We want to mask `mask_prob` percent of time steps in blocks of `mask_length`
        
        # Vectorized implementation of random masking
        # Simple version: Randomly zero out timesteps
        mask = torch.rand(B, T, device=x.device) < mask_prob
        
        # Apply mask (broadcasting over D)
        # x[mask] = 0.0 # This zeros out; for embeddings, maybe better to replace with mean or zero
        x = x * (~mask.unsqueeze(-1)).float()
        
        return x

    def forward(self, speech_tokens, src_key_padding_mask=None):
        # speech_tokens: [Batch, SeqLen] integer labels
        x = self.emb(speech_tokens)
        
        # Apply Augmentation (Training Only)
        x = self.apply_spec_augment(x, mask_prob=0.15, mask_length=5)
        
        # PyTorch TransformerEncoder expects a mask where True means "ignore this position"
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.proj(x)


def prepare_batch(model, batch, config, max_speech_vocab):
    """Extracts ground truth text and speech tokens from audio"""
    device = config.device
    audios = batch["audio"]
    texts = batch["text"]

    batch_size = len(texts)

    # 1. Process Audio to get Speech Tokens
    audio_16k_list = []
    for audio in audios:
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = audio
        audio_16k = librosa.resample(audio_np, orig_sr=model.sr, target_sr=S3_SR)
        audio_16k_list.append(audio_16k)

    s3_tokenizer = model.s3gen.tokenizer

    try:
        with torch.no_grad():
            speech_tokens, speech_token_lens = s3_tokenizer.forward(audio_16k_list[:batch_size], max_len=1000)
            speech_tokens = speech_tokens.to(device)
    except Exception as e:
        logger.error(f"Speech Tokenization failed: {e}")
        return None

    if isinstance(speech_token_lens, list):
        speech_token_lens = torch.tensor(speech_token_lens, device=device)
    else:
        speech_token_lens = speech_token_lens.to(device)
        
    sot_speech = model.t3.hp.start_speech_token
    eot_speech = model.t3.hp.stop_speech_token
    speech_tokens = F.pad(speech_tokens, (1, 0), value=sot_speech)
    speech_tokens = F.pad(speech_tokens, (0, 1), value=eot_speech)
    speech_token_lens = speech_token_lens + 2


    # 2. Process Text to get Text Tokens (The Targets)
    text_tokens_list = []
    for text in texts:
        try:
            tokens = model.tokenizer.text_to_tokens(text, language_id=config.language_id)
            tokens = tokens.squeeze(0)
            text_tokens_list.append(tokens)
        except Exception as e:
            logger.error(f"Text tokenization failed for '{text}': {e}")
            return None

    max_text_len = max(t.shape[0] for t in text_tokens_list)
    text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long, device=device)
    text_token_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i, t in enumerate(text_tokens_list):
        text_tokens[i, : t.shape[0]] = t.to(device)
        text_token_lens[i] = t.shape[0]

    sot_text = model.t3.hp.start_text_token
    eot_text = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot_text)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot_text)
    text_token_lens = text_token_lens + 2

    return {
        "text_tokens": text_tokens,
        "text_token_lens": text_token_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_token_lens,
    }


def train_reward_model(config: ASRConfig):
    """Trains the ASR Token2Text Reward Model on ground truth dataset"""
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="fp16"
    )
    config.device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project="diffro-reward-model", name="asr_reward_pretrain")
        logger.info("Initializing base model to extract tokens...")

    # Load frozen base model just for token extraction
    base_model = ChatterboxMultilingualTTS.from_pretrained(device=accelerator.device)
    # Manually set components to eval and freeze them
    base_model.t3.eval()
    base_model.s3gen.eval()
    base_model.ve.eval()
    for m in [base_model.t3, base_model.s3gen, base_model.ve]:
        for p in m.parameters():
            p.requires_grad = False
    
    max_speech_vocab = base_model.t3.hp.speech_tokens_dict_size
    max_text_vocab = base_model.t3.hp.text_tokens_dict_size

    if accelerator.is_main_process:
        logger.info(f"Initializing Token2Text Model...")
        logger.info(f"Speech Vocab: {max_speech_vocab} | Text Vocab: {max_text_vocab}")

    # Initialize the actual Reward Model we will train
    # Increase capacity: 18 layers
    reward_model = Token2TextRewardModel(max_speech_vocab, max_text_vocab, hidden_dim=512, dropout=0.25).to(accelerator.device)
    # Note: Token2TextRewardModel definition must also be updated to accept hidden_dim or defaults changed
    
    # Use CTC Loss for Sequential Modeling
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    
    dataset = ArabicDataset(config, base_model)
    val_dataset = TestArabicDataset(config, base_model)

    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    eval_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Standard ASR training setup
    # Boost LR for scratch training (1e-5 is too slow, use 5e-4)
    lr = 5e-4 
    if accelerator.is_main_process:
        logger.info(f"🚀 Boosting Learning Rate to {lr} for scratch training")
        
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=lr, weight_decay=0.01)  # Added weight decay
    total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps) # Increased warmup

    reward_model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        reward_model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    global_step = 0
    IGNORE_ID = -100
    
    # Early Stopping Variables
    best_eval_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = getattr(config, 'early_stopping_patience', 3)
    
    for epoch in range(config.num_epochs):
        reward_model.train()
        epoch_loss = 0
        pbar = tqdm(train_dataloader, disable=not accelerator.is_main_process)
        
        for batch in pbar:
            with accelerator.accumulate(reward_model):
                prepared = prepare_batch(base_model, batch, config, max_speech_vocab)
                if prepared is None: continue

                speech_tokens = prepared["speech_tokens"]
                text_tokens = prepared["text_tokens"]
                speech_token_lens = prepared["speech_token_lens"]
                text_token_lens = prepared["text_token_lens"]

                # We mask padding so Transformer ignores it
                src_key_padding_mask = (torch.arange(speech_tokens.shape[1], device=accelerator.device)[None, :] >= speech_token_lens[:, None])
                
                # Replace padding with max_vocab index for the Embedding
                speech_tokens[src_key_padding_mask] = max_speech_vocab

                # Forward Pass
                logits = reward_model(speech_tokens, src_key_padding_mask=src_key_padding_mask) # [B, T, V_text]

                # --- CTC Loss Implementation ---
                # 1. Log Softmax: [B, T, V] -> [T, B, V] (Required by CTCLoss)
                log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                
                # 2. Prepare Lengths (CPU usually preferred for CuDNN CTC, but Torch handles device)
                # Adjust input len by removing special tokens if needed, but here we keep full sequence
                input_lengths = speech_token_lens 
                target_lengths = text_token_lens
                
                # 3. Compute CTC
                loss = ctc_loss_fn(log_probs, text_tokens, input_lengths, target_lengths)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(reward_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                if accelerator.is_main_process:
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    wandb.log({"train/asr_loss": loss.item()}, step=global_step)

                if global_step % 1000 == 0:
                    reward_model.eval()
                    eval_loss = 0
                    eval_steps = 0
                    
                    with torch.no_grad():
                        for eval_batch in eval_dataloader:
                            prepared_eval = prepare_batch(base_model, eval_batch, config, max_speech_vocab)
                            if prepared_eval is None: continue

                            speech_tokens = prepared_eval["speech_tokens"]
                            text_tokens = prepared_eval["text_tokens"]
                            speech_token_lens = prepared_eval["speech_token_lens"]
                            text_token_lens = prepared_eval["text_token_lens"]

                            src_key_padding_mask = (torch.arange(speech_tokens.shape[1], device=accelerator.device)[None, :] >= speech_token_lens[:, None])
                            speech_tokens[src_key_padding_mask] = max_speech_vocab

                            logits = reward_model(speech_tokens, src_key_padding_mask=src_key_padding_mask)
                            
                            # Eval CTC
                            log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                            e_loss = ctc_loss_fn(log_probs, text_tokens, speech_token_lens, text_token_lens)
                            
                            eval_loss += e_loss.item()
                            eval_steps += 1

                    if eval_steps > 0:
                        avg_eval_loss = torch.tensor(eval_loss / eval_steps, device=accelerator.device)
                        avg_eval_loss = accelerator.reduce(avg_eval_loss, reduction="mean")
                        avg_eval_loss_val = avg_eval_loss.item()
                        
                        if accelerator.is_main_process:
                            logger.info(f"Step {global_step} - Eval Loss: {avg_eval_loss_val:.4f}")
                            wandb.log({"eval/asr_loss": avg_eval_loss_val}, step=global_step)
                            
                        # Early Stopping Check
                        if avg_eval_loss_val < best_eval_loss:
                            best_eval_loss = avg_eval_loss_val
                            patience_counter = 0
                            if accelerator.is_main_process:
                                logger.info(f"🏆 New Best Model! Saving...")
                                # Save best model immediately
                                unwrapped_model = accelerator.unwrap_model(reward_model)
                                linear_equiv_weights = unwrapped_model.state_dict()
                                linear_equiv_weights["emb.weight"] = unwrapped_model.emb.weight[:-1].t()
                                os.makedirs("./checkpoints/reward_model", exist_ok=True)
                                torch.save(linear_equiv_weights, "./checkpoints/reward_model/token2text_reward.pt")
                        else:
                            patience_counter += 1
                            if accelerator.is_main_process:
                                logger.info(f"⚠️ No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                            
                            if patience_counter >= early_stopping_patience:
                                if accelerator.is_main_process:
                                    logger.info(f"🛑 Early Stopping Triggered at step {global_step}!")
                                break
                    
                    reward_model.train()
            
            # Break epoch loop if early stopping triggered
            if patience_counter >= early_stopping_patience:
                break

        if patience_counter >= early_stopping_patience:
            break

        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} Complete. Training Loss: {epoch_loss/len(train_dataloader):.4f}")
            # We already saved best model during eval loop


    if accelerator.is_main_process: wandb.finish()


if __name__ == "__main__":
    config = ASRConfig()
    config.num_epochs = 50 # Increase from 3 to 50 for proper CTC convergence
    train_reward_model(config)
