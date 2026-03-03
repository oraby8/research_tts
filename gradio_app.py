import os
import torch
import traceback
import gradio as gr
from safetensors.torch import load_file
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import numpy as np

# =========================================================================
# CONFIGURATION
# =========================================================================
CHECKPOINT_NAME = "best_checkpoint"
OUTPUT_DIR = "./output_trainig_partial"

# Determine device automatically
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Global model variable
model = None

def load_model():
    global model
    if model is not None:
        return model

    print(f"Loading base model to {DEVICE}...")
    model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)

    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME, "model.safetensors")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        weights = load_file(checkpoint_path, device=DEVICE)
        model.t3.load_state_dict(weights, strict=False)
        print(f"✅ Loaded {len(weights)} parameters\n")
    else:
        print(f"⚠️ Checkpoint not found at {checkpoint_path}. Using base model weights.")

    model.t3.eval()
    model.s3gen.eval()
    model.ve.eval()

    # Torch compile often fails on MPS/CPU, so we only use it if on CUDA
    if DEVICE == "cuda":
        print("Compiling model with torch.compile...")
        try:
            model.t3 = torch.compile(model.t3, mode="reduce-overhead")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")

    return model

def synthesize_audio(text, ref_audio, lang_id, exaggeration, cfg_weight, temperature):
    """
    Inference function called by Gradio.
    """
    global model
    if model is None:
        load_model()

    if not text.strip():
        return None, "❌ Error: Please enter some text."

    if not ref_audio:
        return None, "❌ Error: Please provide a reference audio."

    try:
        # Generate the audio
        with torch.no_grad():
            wav = model.generate(
                text,
                language_id=lang_id,
                audio_prompt_path=ref_audio, # Gradio provides the temp file path directly
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

        # Squeeze to 1D array for Gradio audio output
        wav_np = wav.squeeze().cpu().numpy()

        return (model.sr, wav_np), "✅ Success!"

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Generation Error: {str(e)}"

# =========================================================================
# GRADIO INTERFACE
# =========================================================================
with gr.Blocks(title="Chatterbox TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗣️ Chatterbox Multilingual TTS Inference")
    gr.Markdown("Generate speech using your fine-tuned Chatterbox model. Provide a reference audio clip to clone the voice and style.")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Text to Synthesize",
                lines=4,
                placeholder="Enter text here..."
            )

            ref_audio_input = gr.Audio(
                label="Reference Audio Prompt",
                type="filepath",
            )

            with gr.Accordion("Advanced Settings", open=False):
                lang_input = gr.Dropdown(
                    choices=["ar", "en", "fr", "es", "de"], # Add more if your model supports them
                    value="ar",
                    label="Language ID"
                )
                exaggeration_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.5, step=0.1,
                    label="Exaggeration (Prosody Variance)"
                )
                cfg_weight_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.5, step=0.1,
                    label="CFG Weight (Adherence to prompt)"
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.8, step=0.1,
                    label="Temperature (Creativity/Randomness)"
                )

            generate_btn = gr.Button("🎤 Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio", type="numpy")
            status_output = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=synthesize_audio,
        inputs=[
            text_input,
            ref_audio_input,
            lang_input,
            exaggeration_slider,
            cfg_weight_slider,
            temperature_slider
        ],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    # Preload model on startup (optional, but makes first click faster)
    print("Initializing Gradio app...")
    try:
        load_model()
    except Exception as e:
        print(f"⚠️ Failed to preload model. It will try again on first request. Error: {e}")

    # Launch the web server
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
