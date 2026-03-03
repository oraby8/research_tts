import os
import torch
import pandas as pd
import soundfile as sf
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import sys

# Ensure chatterbox can be imported correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from safetensors.torch import load_file
except ImportError as e:
    print(f"WARNING: Chatterbox codebase not found in PYTHONPATH: {e}")
    ChatterboxMultilingualTTS = None
    load_file = None


def generate_dpo_losers(
    sft_csv, input_audio_dir, output_audio_dir, output_csv, checkpoint_path=None
):
    if ChatterboxMultilingualTTS is None:
        raise ImportError("Chatterbox TTS not available - check PYTHONPATH")

    print("🤖 Loading Chatterbox TTS Model to generate 'Loser' Audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device=torch.device(device))

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"🔄 Loading SFT Checkpoint: {checkpoint_path}")
        try:
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path, device=device)
            model.t3.load_state_dict(state_dict, strict=False)
            print("✅ Loaded partial fine-tuned weights successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint. Proceeding with base model: {e}")

    model.t3.eval()

    # Load dataset
    df = pd.read_csv(sft_csv)

    # Convert relative paths to absolute
    if not os.path.isabs(output_audio_dir):
        output_audio_dir = os.path.abspath(output_audio_dir)
    if not os.path.isabs(input_audio_dir):
        input_audio_dir = os.path.abspath(input_audio_dir)

    os.makedirs(output_audio_dir, exist_ok=True)

    valid_dpo_pairs = []

    print(f"🔄 Generating synthetic audio for {len(df)} samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_name = str(row["file_name"])
        text = str(row["normalized_transcription"])

        # Original Audio is the Winner
        human_audio_path = os.path.join(input_audio_dir, file_name)

        if not os.path.exists(human_audio_path):
            continue

        try:
            # Synthesize Loser Audio
            loser_filename = file_name.replace(".wav", "_synth_loser.wav")
            loser_path = os.path.join(output_audio_dir, loser_filename)

            # We use the human audio as the reference speaker / prompt conditioning
            wav, sr = sf.read(human_audio_path, dtype="float32")

            # Use Chatterbox standard inference function
            # Note: Tweak temperature higher (e.g., 0.8) to encourage slight flaws for DPO!
            generated_wav = model.generate(
                text,
                language_id="ar",
                audio_prompt_path=human_audio_path,
                temperature=0.8,
                cfg_weight=0.5,
            )

            # Debug information for tensor format
            print(
                f"📊 Generated tensor info: shape={generated_wav.shape}, dtype={generated_wav.dtype}, device={generated_wav.device}"
            )

            # Safety check for empty tensor
            if generated_wav.numel() == 0:
                raise ValueError("Generated audio tensor is empty")

            # Save the synthesized loser audio
            audio_data = generated_wav.cpu().numpy()

            # Safety check for empty array
            if audio_data.size == 0:
                raise ValueError("Generated audio array is empty")

            # Debug information for numpy array
            print(
                f"📊 Numpy array info: shape={audio_data.shape}, dtype={audio_data.dtype}"
            )

            # Ensure correct format for soundfile.write()
            # Handle both 1D and 2D arrays properly
            if audio_data.ndim == 2:
                if audio_data.shape[0] == 1:
                    audio_data = audio_data.squeeze(0)  # Remove batch dimension
                    print(f"📊 Removed batch dimension, new shape: {audio_data.shape}")
                elif audio_data.shape[1] == 1:
                    # Handle case where channels are last
                    audio_data = audio_data.squeeze(1)
                    print(
                        f"📊 Removed channel dimension, new shape: {audio_data.shape}"
                    )
                else:
                    # If shape is [channels, samples], transpose to [samples, channels]
                    audio_data = audio_data.T
                    print(f"📊 Transposed audio array, new shape: {audio_data.shape}")
            elif audio_data.ndim == 1:
                print(f"📊 Already 1D array, shape: {audio_data.shape}")
            else:
                raise ValueError(
                    f"Unexpected audio tensor dimensions: {audio_data.ndim}D"
                )

            # Ensure float32 format (soundfile requirement)
            if audio_data.dtype != np.float32:
                print(f"📊 Converting dtype from {audio_data.dtype} to float32")
                audio_data = audio_data.astype("float32")

            # Check for NaN or infinite values
            if not np.isfinite(audio_data).all():
                print(f"⚠️ Warning: Audio contains NaN/inf values, replacing with zeros")
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize if values are outside [-1, 1] range
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                print(
                    f"⚠️ Warning: Audio clipping detected (max={max_val:.3f}), normalizing"
                )
                audio_data = audio_data / max_val
            elif max_val == 0:
                print(f"⚠️ Warning: Audio is silent (max amplitude = 0)")
                raise ValueError("Generated audio is completely silent")
            else:
                print(
                    f"✅ Audio amplitude range: [{audio_data.min():.3f}, {audio_data.max():.3f}]"
                )

            # Ensure sample rate is valid
            if not isinstance(model.sr, int) or model.sr <= 0:
                raise ValueError(f"Invalid sample rate: {model.sr}")

            # Final validation
            if len(audio_data) < 100:  # Less than ~4ms at 24kHz
                print(f"⚠️ Warning: Audio is very short ({len(audio_data)} samples)")

            print(f"📝 Saving audio to: {loser_path} with sample rate: {model.sr}")
            sf.write(loser_path, audio_data, model.sr)
            print(f"✅ Successfully saved: {loser_path}")

            # Record the triplet
            valid_dpo_pairs.append(
                {
                    "transcription": text,
                    "file_name_win": os.path.abspath(human_audio_path),
                    "file_name_lose": os.path.abspath(loser_path),
                }
            )

        except Exception as e:
            print(f"❌ Failed to synthesize audio for {file_name}: {e}")
            import traceback

            traceback.print_exc()

    # Save final DPO Metadata
    dpo_df = pd.DataFrame(valid_dpo_pairs)
    dpo_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(
        f"🎉 Successfully generated {len(dpo_df)} DPO triplets. Saved to {output_csv}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate Synthetic Loser Audio for DPO"
    )
    parser.add_argument(
        "--sft_csv",
        type=str,
        required=True,
        help="Input CSV from Stage 2 (containing file_name, normalized_transcription)",
    )
    parser.add_argument(
        "--human_audio_dir",
        type=str,
        required=True,
        help="Directory of the real human VAD chunks",
    )
    parser.add_argument(
        "--synth_audio_dir",
        type=str,
        default="./dpo_synth_losers",
        help="Output directory for generated loser chunks",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="processed_dpo_metadata.csv",
        help="Final DPO metadata CSV",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="(Optional) Path to your SFT model.safetensors",
    )

    args = parser.parse_args()

    generate_dpo_losers(
        args.sft_csv,
        args.human_audio_dir,
        args.synth_audio_dir,
        args.output_csv,
        args.model_checkpoint,
    )
