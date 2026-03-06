import os
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

from config import FinetuneConfig
from dataset import ArabicDataset, collate_fn, TestArabicDataset
import wandb
# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


from peft import LoraConfig, get_peft_model, TaskType

def apply_lora(model, config):
    """
    Apply LoRA to the T3 model.
    This effectively freezes the base model and only trains the adapters.
    """
    
    # 1. ALWAYS freeze codec and voice encoder (kept from original logic)
    for param in model.s3gen.parameters():
        param.requires_grad = False
    for param in model.ve.parameters():
        param.requires_grad = False
        
    logger.info("✅ Frozen: S3Gen codec, VoiceEncoder")

    # 2. Configure LoRA
    # Retrieve config or default
    r = getattr(config, "lora_rank", 16)
    alpha = getattr(config, "lora_alpha", 32)
    target_modules_str = getattr(config, "lora_target_modules", "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj")
    
    # Parse target modules string to list
    if isinstance(target_modules_str, str):
        target_modules = [m.strip() for m in target_modules_str.split(",")]
    else:
        target_modules = target_modules_str
        
    logger.info(f"LoRA Config: Rank={r}, Alpha={alpha}, Targets={target_modules}")
    
    lora_config = LoraConfig(
        r=r, # Rank
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type=None, 
        modules_to_save=["text_head", "speech_head"] 
    )
    
    # Apply LoRA to the internal transformer backbone
    model.t3.tfmr = get_peft_model(model.t3.tfmr, lora_config)
    
    # Ensure heads are trainable if not included in LoRA
    for param in model.t3.text_head.parameters():
        param.requires_grad = True
    for param in model.t3.speech_head.parameters():
        param.requires_grad = True
        
    model.t3.tfmr.print_trainable_parameters()


def compute_loss(model, prepared, config):
    """Compute loss with improved DDP handling and accuracy metrics"""
    IGNORE_ID = -100

    t3_cond = prepared["t3_cond"]
    text_tokens = prepared["text_tokens"]
    text_token_lens = prepared["text_token_lens"]
    speech_tokens = prepared["speech_tokens"]
    speech_token_lens = prepared["speech_token_lens"]

    t3_model = model.t3.module if hasattr(model.t3, "module") else model.t3
    max_speech_vocab = t3_model.hp.speech_tokens_dict_size

    def get_dummy_return(device):
        dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
        # Connect to computation graph to avoid DDP hangs
        for p in t3_model.parameters():
            if p.requires_grad:
                dummy_loss = dummy_loss + 0.0 * p.sum()
        dummy_acc = torch.tensor(0.0, device=device)
        return dummy_loss, dummy_loss, dummy_acc, dummy_acc

    try:
        out = model.t3(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )
    except Exception as e:
        logger.error(f"Forward pass failed: {e}")
        device = speech_tokens.device
        return get_dummy_return(device)

    device = out.text_logits.device

    # Text loss
    logits_text = out.text_logits[:, :-1, :]
    target_text = text_tokens[:, 1:]

    mask_text = (
        torch.arange(target_text.size(1), device=device)[None, :]
        < (text_token_lens - 1)[:, None]
    )
    target_text = target_text.masked_fill(~mask_text, IGNORE_ID)

    loss_text = F.cross_entropy(
        logits_text.reshape(-1, logits_text.size(-1)),
        target_text.reshape(-1),
        ignore_index=IGNORE_ID,
        label_smoothing=0.05,
    )

    with torch.no_grad():
        valid_text_mask = target_text != IGNORE_ID
        if valid_text_mask.any():
            acc_text = (logits_text.argmax(dim=-1)[valid_text_mask] == target_text[valid_text_mask]).float().mean()
        else:
            acc_text = torch.tensor(0.0, device=device)

    # Speech loss
    logits_speech = out.speech_logits[:, :-1, :]
    target_speech = speech_tokens[:, 1:]

    mask_speech = (
        torch.arange(target_speech.size(1), device=device)[None, :]
        < (speech_token_lens - 1)[:, None]
    )
    target_speech = target_speech.masked_fill(~mask_speech, IGNORE_ID)

    valid_mask = (target_speech >= 0) & (target_speech < max_speech_vocab)
    target_speech = target_speech.masked_fill(
        ~valid_mask & (target_speech != IGNORE_ID), IGNORE_ID
    )

    loss_speech = F.cross_entropy(
        logits_speech.reshape(-1, logits_speech.size(-1)),
        target_speech.reshape(-1),
        ignore_index=IGNORE_ID,
        label_smoothing=0.05,
    )

    with torch.no_grad():
        valid_speech_mask = target_speech != IGNORE_ID
        if valid_speech_mask.any():
            acc_speech = (logits_speech.argmax(dim=-1)[valid_speech_mask] == target_speech[valid_speech_mask]).float().mean()
        else:
            acc_speech = torch.tensor(0.0, device=device)

    if torch.isnan(loss_text) or torch.isnan(loss_speech):
        logger.warning("⚠️ NaN loss detected!")
        return get_dummy_return(device)

    return loss_text, loss_speech, acc_text, acc_speech


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
        logger.error(
            f"🚨 INVALID TOKEN IN BATCH: {speech_tokens.max()} >= {max_speech_vocab}"
        )
        return None

    # Check speech token length doesn't exceed position embedding limit
    if speech_tokens.shape[1] + 2 > max_speech_pos_emb:
        logger.error(
            f"🚨 SPEECH TOO LONG: {speech_tokens.shape[1] + 2} > {max_speech_pos_emb} (position embedding limit)"
        )
        return None

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
                logger.error(
                    f"🚨 TEXT TOKEN OUT OF VOCAB: text='{text[:50]}...', max_token={tokens.max()}, vocab_size={max_text_vocab}"
                )
                return None
            # Check length doesn't exceed position embedding limit (account for SOT/EOT)
            if len(tokens) + 2 > max_text_pos_emb:
                logger.error(
                    f"🚨 TEXT TOO LONG: {len(tokens) + 2} > {max_text_pos_emb} (position embedding limit), text='{text[:50]}...'"
                )
                return None
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
def evaluate(model, eval_dataloader, accelerator, config):
    model.t3.eval()

    total_loss = 0
    total_batches = 0

    for batch in eval_dataloader:
        prepared = prepare_batch(model, batch, config)

        if prepared is None:
            continue

        loss_text, loss_speech, _, _ = compute_loss(model, prepared, config)
        loss = loss_speech + 0.1 * loss_text

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)

    model.t3.train()
    return avg_loss
    

def train(config: FinetuneConfig):
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

    if accelerator.is_main_process:
        logger.info(f"📊 Model Configuration:")
        logger.info(f"  Speech vocab: {model.t3.hp.speech_tokens_dict_size} tokens")
        logger.info(f"  Text vocab: {model.t3.hp.text_tokens_dict_size} tokens")

    apply_lora(model, config)

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


    # ✅ Optimizer for ALL trainable parameters (LoRA + Heads)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.t3.parameters()),
        lr=config.learning_rate,
        weight_decay=0.05,
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
                    prepared = prepare_batch(model, batch, config)

                    # Function to create graph-connected dummy loss
                    def get_graph_dummy_loss():
                        dummy = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                        t3_model_ref = model.t3.module if hasattr(model.t3, "module") else model.t3
                        for p in t3_model_ref.parameters():
                            if p.requires_grad:
                                dummy = dummy + 0.0 * p.sum()
                        return dummy

                    acc_t, acc_s = 0.0, 0.0
                    
                    if prepared is None:
                        skipped_batches += 1
                        loss = get_graph_dummy_loss()
                    else:
                        loss_text, loss_speech, acc_text, acc_speech = compute_loss(model, prepared, config)
                        acc_t = acc_text.item()
                        acc_s = acc_speech.item()

                        if loss_speech.item() == 0 and loss_text.item() == 0:
                            skipped_batches += 1
                            loss = get_graph_dummy_loss()
                        else:
                            loss = loss_speech + 0.1 * loss_text

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

                except Exception as e:
                    if accelerator.is_main_process:
                        logger.error(f"❌ Error in batch {batch_idx}: {e}")

                    # Ensure we still step the optimizer to keep DDP in sync even on error
                    loss = torch.tensor(
                        0.0, device=accelerator.device, requires_grad=True
                    )
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    skipped_batches += 1
                    continue

            # ✅ Evaluate and log OUTSIDE accumulate context
            if did_step:
                global_step += 1

                # 🔥 Evaluate every 100 steps
                if global_step % config.eval_steps == 0 and global_step > 0:
                    accelerator.wait_for_everyone()
                
                    eval_loss = evaluate(model, eval_dataloader, accelerator, config)
                
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
    config = FinetuneConfig()
    train(config)
