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
    def __init__(self, speech_vocab_size, text_vocab_size, hidden_dim=256):
        super().__init__()
        # In this pre-training step, we take standard integer tokens, so we use an Embedding.
        # In opt1_true_diffro, it takes soft one-hot vectors, so it uses a Linear layer.
        # They are mathematically equivalent if the Linear layer has no bias.
        self.emb = nn.Embedding(speech_vocab_size + 1, hidden_dim, padding_idx=speech_vocab_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)  # Increased layers for better accuracy
        self.proj = nn.Linear(hidden_dim, text_vocab_size)

    def forward(self, speech_tokens, src_key_padding_mask=None):
        # speech_tokens: [Batch, SeqLen] integer labels
        x = self.emb(speech_tokens)
        
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
    reward_model = Token2TextRewardModel(max_speech_vocab, max_text_vocab).to(accelerator.device)
    
    dataset = ArabicDataset(config, base_model)
    val_dataset = TestArabicDataset(config, base_model)

    train_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    eval_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Standard ASR training setup
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=config.learning_rate) # e.g. 5e-4
    total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps)

    reward_model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        reward_model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    global_step = 0
    IGNORE_ID = -100
    
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
                logits = reward_model(speech_tokens, src_key_padding_mask=src_key_padding_mask)

                # Since this is ASR, speech sequence is longer than text sequence. 
                # DiffRO aligns them differently or pools them. Here we use an extremely simplified 
                # Sequence-to-Sequence pooling (mean over time) to predict just the bag-of-words presence,
                # or a pooled representation for the full text.
                
                # Note: True ASR uses CTC loss or cross-attention decoders. 
                # For this simplest proxy reward, we average the speech encodings over time to predict 
                # the next text tokens in an autoregressive way (Requires AR Decoder).
                # To keep this architecture simple and matching opt1 (an Encoder-only architecture), 
                # we will just sum predictions along the time axis.
                
                logits = logits.mean(dim=1) # Shape: [Batch, Text_Vocab]
                
                # We reward the model for predicting any token that appears in the target text
                target_one_hot = torch.zeros(logits.size(0), max_text_vocab, device=accelerator.device)
                for i in range(logits.size(0)):
                    valid_len = text_token_lens[i]
                    valid_text_tokens = text_tokens[i, :valid_len]
                    target_one_hot[i].scatter_(0, valid_text_tokens, 1.0)
                
                # Multi-label Soft Margin Loss matches Bag-of-Words prediction 
                # It evaluates if the right text is generally present in the audio
                loss = F.multilabel_soft_margin_loss(logits, target_one_hot)

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
                            logits = logits.mean(dim=1)
                            
                            target_one_hot = torch.zeros(logits.size(0), max_text_vocab, device=accelerator.device)
                            for i in range(logits.size(0)):
                                valid_len = text_token_lens[i]
                                valid_text_tokens = text_tokens[i, :valid_len]
                                target_one_hot[i].scatter_(0, valid_text_tokens, 1.0)
                            
                            e_loss = F.multilabel_soft_margin_loss(logits, target_one_hot)
                            eval_loss += e_loss.item()
                            eval_steps += 1

                    if eval_steps > 0:
                        avg_eval_loss = torch.tensor(eval_loss / eval_steps, device=accelerator.device)
                        avg_eval_loss = accelerator.reduce(avg_eval_loss, reduction="mean")
                        if accelerator.is_main_process:
                            logger.info(f"Step {global_step} - Eval Loss: {avg_eval_loss.item():.4f}")
                            wandb.log({"eval/asr_loss": avg_eval_loss.item()}, step=global_step)
                    
                    reward_model.train()

        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch} Complete. Training Loss: {epoch_loss/len(train_dataloader):.4f}")
            unwrapped_model = accelerator.unwrap_model(reward_model)
            
            # Linear weights are equivalent to Embedding lookup table! We can save the weights directly 
            # to be compatible with opt1_true_diffro.py's Linear layer
            linear_equiv_weights = unwrapped_model.state_dict()
            linear_equiv_weights["emb.weight"] = unwrapped_model.emb.weight[:-1].t() # drop padding token and transpose for linear load
            
            torch.save(linear_equiv_weights, "token2text_reward.pt")
            logger.info(f"Saved reward model weights to token2text_reward.pt")

    if accelerator.is_main_process: wandb.finish()


if __name__ == "__main__":
    config = ASRConfig()
    config.num_epochs = 3 # Reward models converge quickly on Bag-of-Words!
    train_reward_model(config)
