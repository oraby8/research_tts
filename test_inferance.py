import os
import io
import torch
import soundfile as sf
import traceback
from safetensors.torch import load_file
from datasets import load_dataset, Audio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from accelerate import Accelerator

# =========================================================================
# CONFIGURATION
# =========================================================================
CHECKPOINT_NAME = "best_checkpoint"
OUTPUT_DIR = "./output_trainig_partial"
TEST_OUTPUTS = "./test_swivid_outputs"
NUM_SAMPLES = 1500
NUM_WORKERS = 16 # Reduced per GPU to avoid overhead

def test_on_swivid():
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Starting parallel inference on {accelerator.num_processes} GPUs...")
        os.makedirs(TEST_OUTPUTS, exist_ok=True)

    # 1. Load Model
    if accelerator.is_main_process:
        print("Loading base model...")
    
    # Load model on each GPU
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    # Load checkpoint weights
    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME, "model.safetensors")
    if os.path.exists(checkpoint_path):
        if accelerator.is_main_process:
            print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load to CPU first then move to device to save VRAM spiked during load if needed
        weights = load_file(checkpoint_path, device="cpu")
        model.t3.load_state_dict(weights, strict=False)
        model.t3.to(device)
        if accelerator.is_main_process:
            print(f"  Loaded {len(weights)} parameters\n")
    else:
        if accelerator.is_main_process:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    model.t3.eval()
    model.s3gen.eval()
    model.ve.eval()

    # Optional: compile for speed
    # Note: torch.compile can sometimes take a long time or have issues with DDP, 
    # but for pure inference it's usually fine.
    if accelerator.is_main_process:
        print("Compiling model with torch.compile...")
    model.t3 = torch.compile(model.t3, mode="reduce-overhead")

    # 2. Load and Shard Dataset
    if accelerator.is_main_process:
        print("Loading SWivid/Habibi (UAE)...")
    
    # We only load on main process and then shard, or load on all and shard to save memory.
    # Since streaming=True, loading on all is fast.
    ds = load_dataset("SWivid/Habibi", "UAE", split="test", streaming=True).cast_column(
        "audio", Audio(decode=False)
    )

    if accelerator.is_main_process:
        print("Pre-loading samples...")
    
    # Pre-load only up to NUM_SAMPLES
    raw_samples = []
    for idx, item in enumerate(ds, start=1):
        if idx > NUM_SAMPLES:
            break
        text = item.get("text", "")
        audio_info = item.get("audio")
        if isinstance(audio_info, dict) and "bytes" in audio_info:
            raw_samples.append((idx, text, audio_info["bytes"]))

    # Shard samples across GPUs
    # Each GPU gets a slice: [start:end:step]
    my_samples = raw_samples[accelerator.process_index::accelerator.num_processes]
    
    if accelerator.is_main_process:
        print(f"Total samples: {len(raw_samples)}")
        print(f"Samples per GPU: ~{len(my_samples)}")

    save_executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    save_futures = []

    # 3. Generation Loop
    progress_bar = tqdm(
        my_samples, 
        desc=f"GPU {accelerator.process_index} Generating", 
        disable=not accelerator.is_local_main_process
    )

    for idx, text, audio_bytes in progress_bar:
        data, sr = sf.read(io.BytesIO(audio_bytes))

        # Each sample has a unique ID, so no name collision across processes
        orig_path = os.path.join(TEST_OUTPUTS, f"sample_{idx:04d}_original.wav")
        # Write original prompt
        sf.write(orig_path, data, sr)

        try:
            with torch.no_grad():
                # Use model.generate (which uses model.t3 internally)
                wav = model.generate(
                    text,
                    language_id="ar",
                    audio_prompt_path=orig_path,
                    exaggeration=0.5,
                    cfg_weight=0.5,
                    temperature=0.8,
                )

            gen_path = os.path.join(TEST_OUTPUTS, f"sample_{idx:04d}_generated.wav")
            wav_np = wav.squeeze().cpu().numpy()
            
            # Save asynchronously
            future = save_executor.submit(sf.write, gen_path, wav_np, model.sr)
            save_futures.append(future)

        except Exception as e:
            print(f"  ❌ GPU {accelerator.process_index} Error sample {idx}: {e}")
            # traceback.print_exc()

    # 4. Wait for all saves to finish
    if accelerator.is_main_process:
        print("\nWaiting for I/O to complete...")
    for f in save_futures:
        f.result()

    save_executor.shutdown(wait=True)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Done! All samples generated and saved.")

if __name__ == "__main__":
    test_on_swivid()
