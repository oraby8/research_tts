import os
import copy
import torch.distributions as dist
try:
    import editdistance
except ImportError:
    pass
try:
    from funasr import AutoModel
except ImportError:
    pass

import torch
import logging
import librosa
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import save_file, load_file
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.models.s3tokenizer import S3_SR
from chatterbox.models.t3.modules.cond_enc import T3Cond

from config import ASRConfig
from dataset import ArabicDataset, collate_fn, TestArabicDataset
import wandb
# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def freeze_strategically(model, config):
    """
    Strategic freezing for Egyptian dialect learning

    FREEZE:
    - Text encoder (protect phoneme knowledge)
    - Early acoustic layers (protect low-level audio)
    - S3Gen codec (protect audio quality)
    - VoiceEncoder (protect speaker embedding)

    UNFREEZE:
    - Prosody/duration predictors (learn Egyptian rhythm!)
    - Late acoustic decoder layers (learn speaker + dialect)
    """

    # 1. ALWAYS freeze codec and voice encoder
    for param in model.s3gen.parameters():
        param.requires_grad = False
    for param in model.ve.parameters():
        param.requires_grad = False

    logger.info("✅ Frozen: S3Gen codec, VoiceEncoder")

    # 2. Freeze/unfreeze T3 layers strategically
    total_params = 0
    trainable_params = 0

    for name, param in model.t3.named_parameters():
        # FREEZE text encoder (protect phoneme knowledge)
        if config.freeze_text_encoder and ("text_enc" in name or "text_emb" in name):
            param.requires_grad = False
            total_params += param.numel()
            continue

        # UNFREEZE prosody/duration (learn Egyptian rhythm!)
        if config.unfreeze_prosody and (
            "duration" in name or "prosody" in name or "pitch" in name
        ):
            param.requires_grad = True
            total_params += param.numel()
            trainable_params += param.numel()
            continue

        # Handle decoder layers strategically
        if "decoder" in name or "dec_layer" in name:
            # Extract layer number if possible
            try:
                # Try to find layer number in name
                import re

                layer_match = re.search(r"layer[._](\d+)|\.(\d+)\.", name)
                if layer_match:
                    layer_num = int(layer_match.group(1) or layer_match.group(2))

                    # Freeze early layers (0-11), unfreeze late layers (12+)
                    if config.unfreeze_late_acoustic and layer_num >= 12:
                        param.requires_grad = True
                        trainable_params += param.numel()
                    elif config.freeze_early_acoustic:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                        trainable_params += param.numel()
                else:
                    # If can't parse layer number, use config default
                    if config.unfreeze_late_acoustic:
                        param.requires_grad = True
                        trainable_params += param.numel()
                    else:
                        param.requires_grad = False
            except:
                # Fallback: unfreeze if unfreeze_late_acoustic is True
                if config.unfreeze_late_acoustic:
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False

            total_params += param.numel()
            continue

        # Default: keep trainable
        param.requires_grad = True
        total_params += param.numel()
        trainable_params += param.numel()

    logger.info(f"📊 Partial Fine-Tune Strategy:")
    logger.info(f"  Total T3 params: {total_params:,}")
    logger.info(
        f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
    )
    logger.info(
        f"  Text encoder: {'FROZEN ❄️' if config.freeze_text_encoder else 'Trainable'}"
    )
    logger.info(
        f"  Early acoustic: {'FROZEN ❄️' if config.freeze_early_acoustic else 'Trainable'}"
    )
    logger.info(f"  Prosody: {'Trainable 🔥' if config.unfreeze_prosody else 'FROZEN'}")
    logger.info(
        f"  Late acoustic: {'Trainable 🔥' if config.unfreeze_late_acoustic else 'FROZEN'}"
    )



class Token2TextRewardModel(torch.nn.Module):
    """Dummy Token2Text Model for True DiffRO (Requires Pre-training!)"""
    def __init__(self, speech_vocab_size, text_vocab_size, hidden_dim=256):
        super().__init__()
        self.emb = torch.nn.Linear(speech_vocab_size, hidden_dim, bias=False) # Maps one-hot to dense
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)
        self.proj = torch.nn.Linear(hidden_dim, text_vocab_size)
    def forward(self, speech_one_hot):
        x = self.emb(speech_one_hot)
        return self.proj(self.transformer(x))

def compute_loss(model, prepared, config, ref_model=None, extra_model=None):
    """
    GLM-TTS GRPO (Group Relative Policy Optimization) Loss
    Features:
    - Group-based Advantage
    - Dynamic Sampling (resample if homogeneous rewards)
    - Asymmetric PPO Clipping (epsilon_low=0.2, epsilon_high=0.3)
    """
    import torch.distributions as dist
    
    IGNORE_ID = -100
    t3_cond, text_tokens, text_token_lens = prepared["t3_cond"], prepared["text_tokens"], prepared["text_token_lens"]
    speech_tokens, speech_token_lens = prepared["speech_tokens"], prepared["speech_token_lens"]
    
    out = model.t3(t3_cond=t3_cond, text_tokens=text_tokens, text_token_lens=text_token_lens, speech_tokens=speech_tokens, speech_token_lens=speech_token_lens, training=True)
    logits_speech = out.speech_logits[:, :-1, :] # [B, T, V]
    
    with torch.no_grad():
        ref_out = ref_model(t3_cond=t3_cond, text_tokens=text_tokens, text_token_lens=text_token_lens, speech_tokens=speech_tokens, speech_token_lens=speech_token_lens, training=True)
        ref_logits = ref_out.speech_logits[:, :-1, :] # [B, T, V]
        
    B, T, V = logits_speech.shape
    G = getattr(config, 'grpo_group_size', 4)  # Use 4 by default for VRAM limits
    
    # 1. Dynamic Sampling Strategy (Up to 3 resamples to prevent zero-advantage gradient vanishing)
    max_resamples = 3
    valid_sample = False
    
    # Pre-compute distribution of current and ref policy
    dist_theta = dist.Categorical(logits=logits_speech)
    dist_ref = dist.Categorical(logits=ref_logits)
    
    # For matching Token2TextRewardModel expectation
    t3_model = model.t3.module if hasattr(model.t3, "module") else model.t3
    max_text_vocab = t3_model.hp.text_tokens_dict_size
    
    for _ in range(max_resamples):
        # Sample G times for each prompt
        # dist_theta.sample((G,)) returns [G, B, T] -> permute to [B, G, T]
        sampled_tokens = dist_theta.sample((G,)).permute(1, 0, 2)
        sampled_tokens_flat = sampled_tokens.reshape(B * G, T)
        
        rewards = torch.zeros(B * G, device=logits_speech.device)
        
        # 2. Reward Computation
        if extra_model is not None:
            with torch.no_grad():
                # One-hot encode sampled tokens
                sampled_one_hot = torch.nn.functional.one_hot(sampled_tokens_flat, num_classes=V).float()
                
                # Get predicted text logits [B*G, T, TextVocab] -> mean over time [B*G, TextVocab]
                predicted_text_logits = extra_model(sampled_one_hot).mean(dim=1)
                
                # Create target text for the repeated batch
                text_tokens_repeated = text_tokens.unsqueeze(1).expand(B, G, -1).reshape(B * G, -1)
                text_token_lens_repeated = text_token_lens.unsqueeze(1).expand(B, G).reshape(B * G)
                
                target_one_hot = torch.zeros(B * G, max_text_vocab, device=logits_speech.device)
                for i in range(B * G):
                    valid_len = text_token_lens_repeated[i]
                    valid_text_tokens = text_tokens_repeated[i, :valid_len]
                    target_one_hot[i].scatter_(0, valid_text_tokens, 1.0)
                    
                # Loss is (B*G, max_text_vocab). Use sum over vocab to get sample-level loss
                # We do this manually to get non-reduced loss
                loss_tensor = torch.nn.functional.binary_cross_entropy_with_logits(
                    predicted_text_logits, target_one_hot, reduction='none'
                )
                reward_loss = loss_tensor.mean(dim=-1) # Mean over vocabulary [B*G]
                rewards = -reward_loss # Negative loss is reward
        
        rewards = rewards.view(B, G)
        
        # 3. Advantage Calculation (Group Relative)
        mean_rewards = rewards.mean(dim=1, keepdim=True)
        std_rewards = rewards.std(dim=1, keepdim=True)
        
        # If std > 1e-4, we found a good diverse batch! Break out.
        if std_rewards.mean().item() > 1e-4:
            valid_sample = True
            break
            
    # Normalize Advantages
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-4) # [B, G]
    advantages_flat = advantages.view(B * G, 1) # [B*G, 1] for broadcasting over T
    
    # Evaluate Log Probs
    # Re-evaluating under dist_theta to keep gradients flowing
    logits_speech_repeated = logits_speech.unsqueeze(1).expand(B, G, T, V).reshape(B * G, T, V)
    dist_theta_repeated = dist.Categorical(logits=logits_speech_repeated)
    log_probs = dist_theta_repeated.log_prob(sampled_tokens_flat) # [B*G, T]
    
    with torch.no_grad():
        ref_logits_repeated = ref_logits.unsqueeze(1).expand(B, G, T, V).reshape(B * G, T, V)
        dist_ref_repeated = dist.Categorical(logits=ref_logits_repeated)
        ref_log_probs = dist_ref_repeated.log_prob(sampled_tokens_flat) # [B*G, T]
    
    # 4. Asymmetric GRPO / PPO Clipping Objective
    epsilon_high = 0.3
    epsilon_low = 0.2
    
    ratio = torch.exp(log_probs - ref_log_probs) # [B*G, T]
    
    # Apply padding mask [B, T] -> [B*G, T]
    mask = (torch.arange(T, device=logits_speech.device)[None, :] < (speech_token_lens - 1)[:, None])
    mask_repeated = mask.unsqueeze(1).expand(B, G, T).reshape(B * G, T).float()
    
    # Policy surrogate losses
    surr1 = ratio * advantages_flat
    surr2 = torch.clamp(ratio, 1.0 - epsilon_low, 1.0 + epsilon_high) * advantages_flat
    
    # Actor loss is negative min of surrogates
    actor_loss_per_token = -torch.min(surr1, surr2)
    actor_loss = (actor_loss_per_token * mask_repeated).sum() / mask_repeated.sum()
    
    # 5. KL Divergence Penalty (Optional, to bound the divergence further)
    # Using approx KL (log_probs - ref_log_probs) -> often preferred in PPO
    # but let's use exact categorical KL since we have the full logits
    prob_theta = torch.nn.functional.softmax(logits_speech, dim=-1)
    kl_div = torch.sum(prob_theta * (torch.nn.functional.log_softmax(logits_speech, dim=-1) - torch.nn.functional.log_softmax(ref_logits, dim=-1)), dim=-1)
    kl_loss = (kl_div * mask).sum() / mask.sum()
    
    loss_speech = actor_loss + 0.1 * kl_loss
    
    # Text AR loss (kept to maintain text generation stability)
    logits_text = out.text_logits[:, :-1, :]
    target_text = text_tokens[:, 1:].masked_fill(~(torch.arange(text_tokens.size(1)-1, device=logits_text.device)[None, :] < (text_token_lens - 1)[:, None]), IGNORE_ID)
    loss_text = torch.nn.functional.cross_entropy(logits_text.reshape(-1, logits_text.size(-1)), target_text.reshape(-1), ignore_index=IGNORE_ID)
    
    return loss_text, loss_speech


def prepare_batch(model, batch, config):
    """Prepare batch - same as before"""
    device = config.device
    audios = batch["audio"]
    texts = batch["text"]

    batch_size = len(texts)

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
        speech_tokens, speech_token_lens = s3_tokenizer.forward(
            audio_16k_list, max_len=1000
        )
        speech_tokens = speech_tokens.to(device)
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        return None

    t3_model = model.t3.module if hasattr(model.t3, "module") else model.t3
    max_speech_vocab = t3_model.hp.speech_tokens_dict_size
    max_text_pos_emb = t3_model.hp.max_text_tokens + 2  # Position embedding limit
    max_speech_pos_emb = t3_model.hp.max_speech_tokens + 4  # Position embedding limit

    if speech_tokens.max() >= max_speech_vocab:
        logger.warning(
            f"🧹 CLAMPING INVALID SPEECH TOKENS: {speech_tokens.max()} >= {max_speech_vocab}"
        )
        speech_tokens = torch.clamp(speech_tokens, max=max_speech_vocab - 1)

    # Check speech token length doesn't exceed position embedding limit
    if speech_tokens.shape[1] + 2 > max_speech_pos_emb:
        logger.warning(
            f"✂️ TRUNCATING SPEECH: {speech_tokens.shape[1] + 2} > {max_speech_pos_emb}"
        )
        speech_tokens = speech_tokens[:, :max_speech_pos_emb - 2]
        if isinstance(speech_token_lens, list):
            speech_token_lens = [min(l, max_speech_pos_emb - 2) for l in speech_token_lens]
        else:
            speech_token_lens = torch.clamp(speech_token_lens, max=max_speech_pos_emb - 2)

    if isinstance(speech_token_lens, list):
        speech_token_lens = torch.tensor(speech_token_lens, device=device)
    else:
        speech_token_lens = speech_token_lens.to(device)

    sot_speech = t3_model.hp.start_speech_token
    eot_speech = t3_model.hp.stop_speech_token

    if sot_speech >= max_speech_vocab or eot_speech >= max_speech_vocab:
        logger.error(f"🚨 Invalid special tokens")
        return None

    speech_tokens = F.pad(speech_tokens, (1, 0), value=sot_speech)
    speech_tokens = F.pad(speech_tokens, (0, 1), value=eot_speech)
    speech_token_lens = speech_token_lens + 2

    # Final validation after padding
    if speech_tokens.max() >= max_speech_vocab:
        logger.error(
            f"🚨 SPEECH TOKEN OUT OF BOUNDS AFTER PADDING: {speech_tokens.max()} >= {max_speech_vocab}"
        )
        return None

    text_tokens_list = []
    for idx, text in enumerate(texts):
        try:
            tokens = model.tokenizer.text_to_tokens(
                text, language_id=config.language_id
            )
            tokens = tokens.squeeze(0)
            # Validate tokens are within vocabulary
            max_text_vocab = t3_model.hp.text_tokens_dict_size
            if tokens.max() >= max_text_vocab:
                logger.warning(
                    f"🧹 CLAMPING TEXT TOKENS: text='{text[:50]}...', max_token={tokens.max()}"
                )
                tokens = torch.clamp(tokens, max=max_text_vocab - 1)
            # Check length doesn't exceed position embedding limit (account for SOT/EOT)
            if len(tokens) + 2 > max_text_pos_emb:
                logger.warning(
                    f"✂️ TRUNCATING TEXT: {len(tokens) + 2} > {max_text_pos_emb}, text='{text[:50]}...'"
                )
                tokens = tokens[:max_text_pos_emb - 2]
            text_tokens_list.append(tokens)
        except Exception as e:
            logger.error(f"Text tokenization failed: {e}")
            return None

    max_text_len = max(t.shape[0] for t in text_tokens_list)
    text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long, device=device)
    text_token_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i, t in enumerate(text_tokens_list):
        text_tokens[i, : t.shape[0]] = t.to(device)
        text_token_lens[i] = t.shape[0]

    max_text_vocab = t3_model.hp.text_tokens_dict_size
    sot_text = t3_model.hp.start_text_token
    eot_text = t3_model.hp.stop_text_token

    # Validate text tokens are within vocabulary
    if text_tokens.max() >= max_text_vocab:
        logger.error(f"🚨 INVALID TEXT TOKEN: {text_tokens.max()} >= {max_text_vocab}")
        return None

    if sot_text >= max_text_vocab or eot_text >= max_text_vocab:
        logger.error(
            f"🚨 Invalid text special tokens: sot={sot_text}, eot={eot_text}, vocab={max_text_vocab}"
        )
        return None

    text_tokens = F.pad(text_tokens, (1, 0), value=sot_text)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot_text)
    text_token_lens = text_token_lens + 2

    # Final validation after padding
    if text_tokens.max() >= max_text_vocab:
        logger.error(
            f"🚨 TEXT TOKEN OUT OF BOUNDS AFTER PADDING: {text_tokens.max()} >= {max_text_vocab}"
        )
        return None

    try:
        ref_audio = audio_16k_list[0]
        ref_audio = ref_audio[: model.ENC_COND_LEN]

        cond_speech_tokens = None
        if plen := t3_model.hp.speech_cond_prompt_len:
            cond_speech_tokens, _ = s3_tokenizer.forward([ref_audio], max_len=plen)
            cond_speech_tokens = cond_speech_tokens.to(device)

            if cond_speech_tokens.max() >= max_speech_vocab:
                logger.error(f"🚨 Invalid conditioning token")
                return None

            cond_speech_tokens = cond_speech_tokens.expand(batch_size, -1)

        ve_embed = torch.from_numpy(
            model.ve.embeds_from_wavs([ref_audio], sample_rate=S3_SR)
        )
        ve_embed = ve_embed.mean(dim=0, keepdim=True).to(device)
        ve_embed = ve_embed.expand(batch_size, -1)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=cond_speech_tokens,
            emotion_adv=0.5 * torch.ones(batch_size, 1, 1, device=device),
        )
    except Exception as e:
        logger.error(f"Conditioning preparation failed: {e}")
        return None

    return {
        "t3_cond": t3_cond,
        "text_tokens": text_tokens,
        "text_token_lens": text_token_lens,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_token_lens,
    }


@torch.no_grad()
def evaluate(model, eval_dataloader, accelerator, config, ref_model=None, extra_model=None):
    model.t3.eval()

    total_loss = 0
    total_batches = 0

    for batch in eval_dataloader:
        prepared = prepare_batch(model, batch, config)

        try:
            if prepared is None:
                loss_text = loss_speech = 0.0 * sum(p.sum() for p in model.t3.parameters() if p.requires_grad)
                loss = loss_speech + 0.1 * loss_text
            else:
                loss_text, loss_speech = compute_loss(model, prepared, config, ref_model, extra_model)
                if loss_speech.item() == 0 and loss_text.item() == 0:
                    loss = 0.0 * sum(p.sum() for p in model.t3.parameters() if p.requires_grad)
                else:
                    loss = loss_speech + 0.1 * loss_text
        except Exception as e:
            if accelerator.is_main_process:
                logger.error(f"❌ Error during evaluate forward: {e}")
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            loss = 0.0 * sum(p.sum() for p in model.t3.parameters() if p.requires_grad)

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)

    model.t3.train()
    return avg_loss
    

def train(config: ASRConfig):
    """Main training loop - PARTIAL FINE-TUNE with Multi-GPU support via Accelerate"""

    # Initialize Accelerate with DDP kwargs for unused parameters and mixed precision
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision="fp16",  # Enable mixed precision to save memory
    )
    config.device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)

    if accelerator.is_main_process:
        wandb.init(
            project=config.report_project_name,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(config),
        )

    if accelerator.is_main_process:
        logger.info("Loading ChatterboxMultilingualTTS...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device("cpu"))

    if accelerator.is_main_process: logger.info("Init True DiffRO Models...")
    ref_model = copy.deepcopy(model.t3)
    for p in ref_model.parameters(): p.requires_grad = False
    ref_model.eval().to(accelerator.device)
    extra_model = Token2TextRewardModel(model.t3.hp.speech_tokens_dict_size, model.t3.hp.text_tokens_dict_size).to(accelerator.device)
    reward_model_path = getattr(config, "reward_model_path", "token2text_reward.pt")
    if os.path.exists(reward_model_path):
        if accelerator.is_main_process: 
            logger.info(f"Loading pre-trained Token2Text weights from {reward_model_path}")
        state_dict = torch.load(reward_model_path, map_location="cpu")
        if "emb.weight" in state_dict and state_dict["emb.weight"].shape != extra_model.emb.weight.shape:
            if state_dict["emb.weight"].shape == (extra_model.emb.weight.shape[1], extra_model.emb.weight.shape[0]):
                state_dict["emb.weight"] = state_dict["emb.weight"].t()
        extra_model.load_state_dict(state_dict)
    else:
        if accelerator.is_main_process: 
            logger.warning(f"⚠️ No Token2Text weights found at {reward_model_path}. Starting with randomly initialized reward model!")


    if accelerator.is_main_process:
        logger.info(f"📊 Model Configuration:")
        logger.info(f"  Speech vocab: {model.t3.hp.speech_tokens_dict_size} tokens")
        logger.info(f"  Text vocab: {model.t3.hp.text_tokens_dict_size} tokens")

    freeze_strategically(model, config)

    if accelerator.is_main_process:
        total_trainable = sum(p.numel() for p in model.t3.parameters() if p.requires_grad)
        wandb.log({"trainable_parameters": total_trainable})

    # ✅ NEW: Resume support for partial fine-tune
    resume_step = 0
    resume_epoch = 0

    if hasattr(config, "resume_from_checkpoint") and config.resume_from_checkpoint:
        checkpoint_path = os.path.join(
            config.resume_from_checkpoint, "model.safetensors"
        )

        if os.path.exists(checkpoint_path):
            if accelerator.is_main_process:
                logger.info(f"🔄 Resuming from: {config.resume_from_checkpoint}")

            try:
                # Load checkpoint weights
                checkpoint_weights = load_file(checkpoint_path, device="cpu")
                missing, unexpected = model.t3.load_state_dict(
                    checkpoint_weights, strict=False
                )

                if accelerator.is_main_process:
                    logger.info(f"✅ Loaded {len(checkpoint_weights)} parameters")
                    logger.info(
                        f"   Missing: {len(missing)}, Unexpected: {len(unexpected)}"
                    )

                # Try to extract step/epoch from checkpoint name
                checkpoint_name = os.path.basename(config.resume_from_checkpoint)

                if "checkpoint-" in checkpoint_name:
                    resume_step = int(checkpoint_name.split("-")[1])
                    if accelerator.is_main_process:
                        logger.info(f"   Resuming from step: {resume_step}")

                elif "epoch_" in checkpoint_name:
                    resume_epoch = int(checkpoint_name.split("_")[1]) + 1
                    if accelerator.is_main_process:
                        logger.info(f"   Resuming from epoch: {resume_epoch}")

            except Exception as e:
                if accelerator.is_main_process:
                    logger.error(f"❌ Failed to load checkpoint: {e}")
                    logger.warning(f"Starting fresh training instead")
        else:
            if accelerator.is_main_process:
                logger.warning(f"⚠️ Checkpoint not found: {checkpoint_path}")
                logger.warning(f"Starting fresh training instead")

    # Move modules to device manually
    model.t3.to(accelerator.device)
    model.s3gen.to(accelerator.device)
    model.ve.to(accelerator.device)
    
    # Pre-warm GPU memory allocator if requested
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if accelerator.is_main_process:
        logger.info("Loading dataset...")
    dataset = ArabicDataset(config, model)
    val_dataset = TestArabicDataset(config, model)

    
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
    )


    # ✅ Optimizer for ALL trainable parameters with Differential Learning Rates
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for n, p in model.t3.named_parameters()
                    if p.requires_grad and ("text_enc" in n or "text_emb" in n)
                ],
                "lr": config.text_encoder_lr,
            },
            {
                "params": [
                    p
                    for n, p in model.t3.named_parameters()
                    if p.requires_grad and not ("text_enc" in n or "text_emb" in n)
                ],
                "lr": config.learning_rate,
            },
        ],
        weight_decay=0.05,  # Increased for stronger regularization
    )

    total_steps = (
        len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    model.t3, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(model.t3, optimizer,
                                                                                            train_dataloader,eval_dataloader, scheduler)

    # ✅ Load Optimizer & Scheduler states after accelerator.prepare()
    if hasattr(config, "resume_from_checkpoint") and config.resume_from_checkpoint:
        opt_path = os.path.join(config.resume_from_checkpoint, "optimizer_and_scheduler.pt")
        if os.path.exists(opt_path):
            try:
                states = torch.load(opt_path, map_location="cpu")
                optimizer.load_state_dict(states["optimizer"])
                scheduler.load_state_dict(states["scheduler"])
                if accelerator.is_main_process:
                    logger.info("✅ Loaded optimizer and scheduler states")
            except Exception as e:
                if accelerator.is_main_process:
                    logger.error(f"❌ Failed to load optimizer states: {e}")

    if accelerator.is_main_process:
        logger.info(f"🚀 Starting PARTIAL FINE-TUNE for {config.num_epochs} epochs...")
        logger.info(f"  Total steps: {total_steps}")
        logger.info(
            f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps * accelerator.num_processes}"
        )
        logger.info(f"  Learning rate: {config.learning_rate}")

    global_step = resume_step
    skipped_batches = 0
    best_eval_loss = float("inf")
    patience_counter = 0  # Counter for early stopping
    last_valid_prepared = None
    
    for epoch in range(resume_epoch, config.num_epochs):
        model.t3.train()
        epoch_loss = 0
        epoch_batches = 0

        # Only show progress bar on main process
        if accelerator.is_main_process:
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = train_dataloader

        for batch_idx, batch in enumerate(pbar):
            did_step = False
            with accelerator.accumulate(model.t3):
                try:
                    prepared_failed = False
                    try:
                        prepared = prepare_batch(model, batch, config)
                        if prepared is None:
                            prepared_failed = True
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"❌ Error during prep: {e}")
                        prepared_failed = True
                        
                    if not prepared_failed:
                        last_valid_prepared = prepared
                    elif last_valid_prepared is not None:
                        prepared = last_valid_prepared
                        skipped_batches += 1
                        
                    if prepared is None:
                        loss = 0.0 * sum(p.sum() for p in model.t3.parameters() if p.requires_grad)
                    else:
                        loss_text, loss_speech = compute_loss(model, prepared, config, ref_model, extra_model)

                        if loss_speech.item() == 0 and loss_text.item() == 0:
                            skipped_batches += 1
                            loss = 0.0 * (loss_speech + loss_text)
                        else:
                            loss = loss_speech + 0.1 * loss_text

                        if prepared_failed:
                            loss = 0.0 * loss

                except Exception as e:
                    if accelerator.is_main_process:
                        logger.error(f"❌ Error during forward: {e}")
                    if "out of memory" in str(e).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Fallback to zeroing gradients while satisfying DDP backward
                    skipped_batches += 1
                    try:
                        if last_valid_prepared is not None:
                            loss_t, loss_s = compute_loss(model, last_valid_prepared, config, ref_model, extra_model)
                            loss = 0.0 * (loss_t + loss_s)
                        else:
                            loss = 0.0 * sum(p.sum() for p in model.t3.parameters() if p.requires_grad)
                    except:
                        loss = 0.0 * sum(p.sum() for p in model.t3.parameters() if p.requires_grad)
                    
                    
                # Continue normally with the backward pass, so DDP hooks fire and syncs correctly!
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # ✅ Stricter gradient clipping for full fine-tune
                    accelerator.clip_grad_norm_(model.t3.parameters(), 0.5)
                    did_step = True

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if loss.item() > 0:
                    epoch_loss += loss.item()
                    epoch_batches += 1

            # ✅ Evaluate and log OUTSIDE accumulate context
            if did_step:
                global_step += 1

                # 🔥 Evaluate every X steps
                if global_step % config.eval_steps == 0 and global_step > 0:
                    accelerator.wait_for_everyone()
                
                    # Evaluate securely across all GPUs
                    eval_loss = evaluate(model, eval_dataloader, accelerator, config, ref_model, extra_model)
                    
                    # Reduce loss globally so the main process knows the real average
                    eval_loss_tensor = torch.tensor(eval_loss, device=accelerator.device)
                    eval_loss_tensor = accelerator.reduce(eval_loss_tensor, reduction="mean")
                    eval_loss = eval_loss_tensor.item()
                    
                    accelerator.wait_for_everyone()
                
                    if accelerator.is_main_process:
                        logger.info(f"📊 Eval @ step {global_step} | Loss: {eval_loss:.4f}")
                        wandb.log(
                            {
                                "eval/loss": eval_loss,
                            },
                            step=global_step,
                        )

                        # 🔥 SAVE ONLY IF BETTER
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            patience_counter = 0  # Reset patience on improvement
                
                            logger.info(f"🏆 New best model! Saving checkpoint...")
                
                            unwrapped_t3 = accelerator.unwrap_model(model.t3)
                            save_checkpoint(unwrapped_t3, optimizer, scheduler, config, global_step, best=True)
                            wandb.log({"eval/best_loss": best_eval_loss}, step=global_step)
                        else:
                            patience_counter += 1
                            logger.info(f"⚠️ Eval loss did not improve. Patience: {patience_counter}/{config.early_stopping_patience}")

                    accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    pbar.set_postfix(
                        {
                            "loss": f"{epoch_loss / max(epoch_batches, 1):.4f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                            "step": global_step,
                            "skip": skipped_batches,
                            "patience": patience_counter,
                        }
                    )
                    wandb.log(
                        {
                            "train/loss": loss.item() if 'loss' in locals() else 0,
                            "train/avg_epoch_loss": epoch_loss / max(epoch_batches, 1),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/step": global_step,
                            "train/skipped_batches": skipped_batches,
                        },
                        step=global_step,
                    )

                if global_step % config.save_steps == 0:
                    # Sync all processes before saving
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped_t3 = accelerator.unwrap_model(model.t3)
                        save_checkpoint(unwrapped_t3, optimizer, scheduler, config, global_step)
                    accelerator.wait_for_everyone()

            # Early stopping check after inner loop updates
            if patience_counter >= config.early_stopping_patience:
                if accelerator.is_main_process:
                    logger.warning(f"🛑 Early stopping triggered! Validation loss hasn't improved for {config.early_stopping_patience} evaluations.")
                break

        avg_loss = epoch_loss / max(epoch_batches, 1)
        if accelerator.is_main_process:
            logger.info(
                f"✅ Epoch {epoch} completed | Avg Loss: {avg_loss:.4f} | Skipped: {skipped_batches}"
            )

            # ✅ Save after each epoch
            unwrapped_t3 = accelerator.unwrap_model(model.t3)
            save_checkpoint(unwrapped_t3, optimizer, scheduler, config, global_step, epoch_final=epoch)
            
        if patience_counter >= config.early_stopping_patience:
            break

    if accelerator.is_main_process:
        unwrapped_t3 = accelerator.unwrap_model(model.t3)
        save_checkpoint(unwrapped_t3, optimizer, scheduler, config, global_step, final=True)
        logger.info(f"🎉 Training completed! Total skipped batches: {skipped_batches}")

    if accelerator.is_main_process:
        wandb.finish()


def save_checkpoint(t3_model, optimizer, scheduler, config, step, final=False, epoch_final=None, best=False):
    """Save model checkpoint - PARTIAL T3 WEIGHTS + OPTIMIZER"""
    if final:
        save_path = os.path.join(config.output_dir, "final_model")
    elif epoch_final is not None:
        save_path = os.path.join(config.output_dir, f"epoch_{epoch_final}")
    elif best:
        save_path = os.path.join(config.output_dir, f"best_checkpoint")
    else:
        save_path = os.path.join(config.output_dir, f"checkpoint-{step}")

    os.makedirs(save_path, exist_ok=True)

    # ✅ Save ONLY trainable T3 parameters to prevent memory bloat
    trainable_names = {name for name, param in t3_model.named_parameters() if param.requires_grad}
    t3_state_dict = {
        k: v.cpu()
        for k, v in t3_model.state_dict().items()
        if k in trainable_names
    }

    save_file(t3_state_dict, os.path.join(save_path, "model.safetensors"))
    
    # ✅ Save optimizer and scheduler states
    if optimizer is not None and scheduler is not None:
        torch.save({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, os.path.join(save_path, "optimizer_and_scheduler.pt"))

    # Save config for reference
    import json

    training_config = {
        "freeze_text_encoder": config.freeze_text_encoder,
        "freeze_early_acoustic": config.freeze_early_acoustic,
        "unfreeze_prosody": config.unfreeze_prosody,
        "unfreeze_late_acoustic": config.unfreeze_late_acoustic,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
    }

    with open(os.path.join(save_path, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    logger.info(f"💾 Saved checkpoint: {save_path} ({len(t3_state_dict)} parameters)")


if __name__ == "__main__":
    config = ASRConfig()
    train(config)
