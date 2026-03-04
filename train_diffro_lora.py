import os
import torch
import torch.nn.functional as F
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import save_file, load_file
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import wandb

# Reuse existing training components
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from config import ASRConfig
from dataset import ArabicDataset, collate_fn, TestArabicDataset
from train_lora import prepare_batch, apply_lora # Reuse batch preparation logic

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Redefine Token2TextRewardModel to ensure self-contained execution and match pretrain_reward_model.py
class Token2TextRewardModel(torch.nn.Module):
    """
    The ASR Reward Model. 
    Matches architecture of pretrain_reward_model.py so we can load weights.
    """
    def __init__(self, speech_vocab_size, text_vocab_size, hidden_dim=512):
        super().__init__()
        # Standard embedding for loading weights, but we will bypass it for soft inputs
        self.emb = torch.nn.Embedding(speech_vocab_size + 1, hidden_dim, padding_idx=speech_vocab_size)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.0, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=18)
        self.proj = torch.nn.Linear(hidden_dim, text_vocab_size)

    def forward(self, speech_tokens, src_key_padding_mask=None):
        # Normal forward for hard tokens (not used in DiffRO training usually)
        x = self.emb(speech_tokens)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.proj(x)

def gumbel_softmax_sampling(logits, temperature=1.0, hard=False):
    """
    Sample from logits using Gumbel-Softmax reparameterization trick.
    This allows gradients to flow back through the sampling process!
    """
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim=-1)

    if hard:
        # Straight-through estimator
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Just use the soft distribution for the reward model
        ret = y_soft
    return ret

def train_diffro(config: ASRConfig):
    """
    DiffRO (Differentiable Reward Optimization) Training Loop with LoRA SFT Loading.
    """
    
    # 1. Setup Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision="fp16"
    )
    config.device = accelerator.device
    
    # Set output dir to distinct folder
    config.output_dir = "output_diffro"
    
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        wandb.init(project="chatterbox-diffro", name=f"diffro_lora_{datetime.now().strftime('%Y%m%d_%H%M')}")

    # 2. Load Models
    if accelerator.is_main_process: logger.info("Loading Models...")
    
    # A. Generator (Policy) - Chatterbox
    model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device("cpu"))
    
    # Apply LoRA to Generator (Initialize Adapters)
    apply_lora(model, config) 
    
    # LOAD SFT WEIGHTS (The "LoRA we did implement")
    # Check default SFT output path
    sft_checkpoint = "output_trainig_partial/final_model/model.safetensors"
    if os.path.exists(sft_checkpoint):
        if accelerator.is_main_process: 
            logger.info(f"🔄 Loading SFT LoRA adapters from {sft_checkpoint}")
        try:
            # Load SFT weights onto the PEFT model
            state_dict = load_file(sft_checkpoint)
            missing, unexpected = model.t3.load_state_dict(state_dict, strict=False)
            if accelerator.is_main_process:
                logger.info(f"✅ SFT Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            if accelerator.is_main_process: logger.error(f"❌ Failed to load SFT weights: {e}")
    else:
        if accelerator.is_main_process: logger.warning(f"⚠️ SFT Checkpoint not found at {sft_checkpoint}. Starting DiffRO from SCRATCH (Base Model).")

    
    # B. Judge (Reward Model) - Frozen
    reward_model = Token2TextRewardModel(
        model.t3.hp.speech_tokens_dict_size, 
        model.t3.hp.text_tokens_dict_size
    )
    # Load pre-trained judge weights
    reward_path = "token2text_reward.pt" # Default path from pretrain_reward_model.py
    if os.path.exists(reward_path):
        state_dict = torch.load(reward_path, map_location="cpu")
        # Fix for transposed weight saving in pretrain script if needed
        if "emb.weight" in state_dict and state_dict["emb.weight"].shape != reward_model.emb.weight.shape:
             w = state_dict["emb.weight"].t() # [D, V] -> [V, D]
             if w.shape[0] == reward_model.emb.weight.shape[0] - 1:
                 padding_row = torch.zeros(1, w.shape[1])
                 w = torch.cat([w, padding_row], dim=0)
             state_dict["emb.weight"] = w
             
        reward_model.load_state_dict(state_dict)
        if accelerator.is_main_process: logger.info(f"✅ Loaded Judge from {reward_path}")
    else:
        if accelerator.is_main_process: logger.warning(f"⚠️ Judge not found at {reward_path}! DiffRO will fail to converge.")
    
    reward_model.to(accelerator.device).eval()
    for p in reward_model.parameters(): p.requires_grad = False # Freeze Judge
    
    # 3. Setup Data & Optimizer
    dataset = ArabicDataset(config, model)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    # Optimize only Generator parameters (LoRA)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.t3.parameters()), 
        lr=config.learning_rate
    )
    
    model.t3, optimizer, dataloader = accelerator.prepare(model.t3, optimizer, dataloader)
    
    # Move other components manually
    model.s3gen.to(accelerator.device)
    model.ve.to(accelerator.device)
    
    # 4. Training Loop
    if accelerator.is_main_process: logger.info("🚀 Starting DiffRO Training...")
    
    global_step = 0
    
    for epoch in range(config.num_epochs):
        model.t3.train()
        for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model.t3):
                # A. Prepare Batch (Get ground truth text and conditioning)
                prepared = prepare_batch(model, batch, config)
                if prepared is None: continue
                
                t3_cond = prepared["t3_cond"]
                text_tokens = prepared["text_tokens"]
                
                # B. Forward Generator (Get Logits)
                speech_tokens = prepared["speech_tokens"]
                speech_token_lens = prepared["speech_token_lens"]
                
                out = model.t3(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=prepared["text_token_lens"],
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens,
                    training=True
                )
                
                # logits: [B, T_speech, V_speech]
                generator_logits = out.speech_logits[:, :-1, :] 
                
                # C. Differentiable Link: Gumbel Softmax
                soft_speech_tokens = gumbel_softmax_sampling(generator_logits, temperature=1.0, hard=False)
                
                # D. Forward Judge
                judge_emb_matrix = reward_model.emb.weight[:-1] # [V, D] (Exclude padding token)
                judge_input_emb = torch.matmul(soft_speech_tokens, judge_emb_matrix)
                
                # (Simple Sinusoidal PE logic replicated here for judge input)
                B, T, D = judge_input_emb.shape
                div_term = torch.exp(torch.arange(0, D, 2, device=accelerator.device) * (-torch.log(torch.tensor(10000.0)) / D))
                pe = torch.zeros(T, D, device=accelerator.device)
                position = torch.arange(0, T, dtype=torch.float, device=accelerator.device).unsqueeze(1)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                judge_input_emb = judge_input_emb + pe.unsqueeze(0)
                
                # Run Judge Transformer
                judge_out = reward_model.transformer(judge_input_emb)
                predicted_text_logits = reward_model.head(judge_out) # [B, T_speech, V_text]
                
                # E. Calculate DiffRO Loss (CTC Loss against Ground Truth Text)
                targets = text_tokens # [B, L_text]
                log_probs = torch.nn.functional.log_softmax(predicted_text_logits, dim=-1).permute(1, 0, 2) # [T, B, V]
                
                input_lengths = speech_token_lens - 1 # Adjusted for shift
                target_lengths = prepared["text_token_lens"]
                
                loss_diffro = torch.nn.functional.ctc_loss(
                    log_probs, 
                    targets, 
                    input_lengths, 
                    target_lengths, 
                    blank=0, 
                    zero_infinity=True
                )
                
                # F. Total Loss
                loss_ce_speech = F.cross_entropy(
                    generator_logits.reshape(-1, generator_logits.size(-1)),
                    speech_tokens[:, 1:].reshape(-1),
                    ignore_index=-100
                )
                
                loss = loss_ce_speech + 0.1 * loss_diffro
                
                # Backward
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
                if accelerator.is_main_process:
                    wandb.log({
                        "train/loss": loss.item(), 
                        "train/diffro_ctc": loss_diffro.item(),
                        "train/ce": loss_ce_speech.item()
                    })
                    global_step += 1
        
        # Save checkpoints
        if accelerator.is_main_process:
            save_path = os.path.join(config.output_dir, f"epoch_{epoch}")
            os.makedirs(save_path, exist_ok=True)
            # Save only trainable params (LoRA adapters)
            unwrapped_t3 = accelerator.unwrap_model(model.t3)
            trainable_names = {name for name, param in unwrapped_t3.named_parameters() if param.requires_grad}
            t3_state_dict = {k: v.cpu() for k, v in unwrapped_t3.state_dict().items() if k in trainable_names}
            save_file(t3_state_dict, os.path.join(save_path, "model.safetensors"))
            logger.info(f"Saved checkpoint to {save_path}")

    if accelerator.is_main_process:
        wandb.finish()
        logger.info("Training Complete.")

if __name__ == "__main__":
    config = ASRConfig()
    train_diffro(config)
