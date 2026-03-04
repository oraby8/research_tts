# TTS Models Overview: Architecture, Method, Data & Training Pipelines

This report provides an overview of the architecture, method, data pipeline, and training pipeline for three state-of-the-art TTS papers: **CosyVoice 3**, **GLM-TTS**, and **MiniMax-Speech**.

---

## 1. CosyVoice 3 (arXiv:2505.17589)
**Title:** CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training

### Architecture
*   **Core:** A two-stage system consisting of a **1.5B parameter Autoregressive Transformer LM** for text-to-token generation and a **300M Conditional Flow Matching (CFM)** model (using a Diffusion Transformer/DiT backbone) for token-to-waveform generation.
*   **Tokenizer:** Uses a novel speech tokenizer derived from **MinMo** (a large audio understanding model). It inserts a Finite Scalar Quantization (FSQ) module into MinMo's encoder.
*   **Vocoder:** Implicit in the CFM stage.

### Method
*   **Supervised Multi-task Tokenizer:** The tokenizer is trained via supervised multi-task learning (ASR, SER, LID, etc.) to capture paralinguistic info better than standard acoustic tokens.
*   **Differentiable Reward Optimization (DiffRO):** A post-training technique where an ASR-like "Token2Text" model serves as a reward function. It uses Gumbel-Softmax to directly optimize speech tokens via back-propagation (unlike standard RL).
*   **Pronunciation Inpainting:** Supports mixed phoneme/text input to fix specific pronunciation errors (e.g., polyphones).

### Data Pipeline
1.  **Detection:** Speaker diarization and Voice Activity Detection (VAD) to segment speech.
2.  **Denoising:** Uses **MossFormer2** to reduce noise.
3.  **Transcription:** Consensus-based ASR using multiple models (Faster-Whisper, Canary, SeamlessM4T); keeps transcripts only if they have <15% pair-wise WER.
4.  **Processing:** Punctuation adjustment via forced alignment (MFA), volume standardization, and filtering outliers based on audio-text length ratios.

### Training Pipeline
*   **Pre-training:** Trained on **1 million hours** of multilingual data (scaled up from 10k hours in v2).
*   **Post-training:** Refined using DiffRO to align tokens with text and instruction-following goals.
*   **Fine-tuning:** Supports Supervised Fine-Tuning (SFT) for specific speakers and "Polyglot" training to enable monolingual speakers to speak other languages.

---

## 2. GLM-TTS (arXiv:2512.14291)
**Title:** GLM-TTS Technical Report

### Architecture
*   **Core:** Two-stage architecture: **Text-to-Token Autoregressive** model followed by a **Token-to-Waveform Diffusion** model.
*   **Tokenizer:** An optimized **Whisper-VQ** tokenizer with a doubled token rate (25Hz) and expanded vocabulary (32k) to reduce pronunciation glitches.
*   **Vocoder:** Introduces **Vocos2D**, a GAN-based vocoder using 2D convolutions for better frequency subband modeling.

### Method
*   **RL Alignment (GRPO):** Uses Group Relative Policy Optimization (GRPO) with multiple rewards:
    *   **CER:** Pronunciation accuracy.
    *   **SIM:** Speaker similarity.
    *   **Emotion:** Expressiveness.
    *   **Laughter:** Paralinguistic realism.
*   **Phoneme-in:** A hybrid input mechanism allowing precise control by mixing text and phonemes for polyphones/rare words.
*   **LoRA Customization:** Efficient "Premium Voice Customization" by fine-tuning only ~15% of parameters.

### Data Pipeline
1.  **Standardization:** Audio format unification and VAD segmentation.
2.  **Cleaning:** Source separation (Mel-Band Roformer) and denoising.
3.  **Filtering:** WER-based filtering (keeps data with <5% WER).
4.  **Punctuation:** Optimized based on character pronunciation duration from forced alignment.

### Training Pipeline
*   **Pre-training:** Trained on **100k hours** of high-quality data (1/10th of CosyVoice 3, emphasizing data efficiency).
*   **Alignment:** RL stage using the GRPO multi-reward framework to improve human-likeness and stability.
*   **Adaptation:** Parameter-efficient fine-tuning (LoRA) for voice cloning.

---

## 3. MiniMax-Speech (arXiv:2505.07916)
**Title:** MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder

### Architecture
*   **Core:** **Autoregressive Transformer** coupled with a **Latent Flow Matching** module.
*   **Speaker Encoder:** A **learnable speaker encoder** trained jointly with the AR model (end-to-end), rather than using a fixed pre-trained verification model.
*   **Decoder:** Uses **Flow-VAE**, a hybrid of VAE and Flow matching. The flow model predicts continuous latent features extracted by the VAE encoder, which are then decoded to waveform.

### Method
*   **Intrinsic Zero-Shot:** The speaker encoder extracts timbre from reference audio *without* requiring its text transcription, enabling robust cross-lingual cloning and reducing mismatch errors.
*   **Flow-VAE:** Improves upon standard Mel-spectrogram approaches by modeling VAE latents, constrained by KL divergence to follow a normal distribution, enhancing reconstruction quality.

### Data Pipeline
1.  **Verification:** Dual ASR verification to ensure high transcription accuracy.
2.  **Punctuation:** Refined using both VAD and ASR timestamps.
3.  **Noise Handling:** Preserves original steady-state noise for naturalness (unlike aggressive denoising in others).
4.  **Consistency:** Multi-speaker verification model ensures timbre consistency within audio files.

### Training Pipeline
*   **Joint Training:** The AR model and Speaker Encoder are optimized together.
*   **Latent Modeling:** The Flow Matching component is trained to generate the continuous latent representations of the Flow-VAE.
*   **Extensions:** Supports LoRA for emotion control and fine-tuning for "Professional Voice Cloning" (PVC).

---

## Summary Comparison

| Feature | CosyVoice 3 | GLM-TTS | MiniMax-Speech |
| :--- | :--- | :--- | :--- |
| **Core Arch** | AR Transformer + CFM (DiT) | AR Transformer + Diffusion | AR Transformer + Flow-VAE |
| **Tokenizer** | **MinMo** (Supervised Multi-task) | Optimized **Whisper-VQ** (25Hz) | **Encoder-VQ-Decoder** (CTC) |
| **Key Method** | **Scaling** (1M hrs) + **DiffRO** (Post-training) | **GRPO-RL** (Multi-reward alignment) | **Learnable Speaker Encoder** (Joint training) |
| **Voice Cloning**| Zero-shot / Polyglot SFT | Zero-shot / LoRA Customization | **Intrinsic Zero-shot** (No text needed) |
| **Data Scale** | ~1,000,000 hours | ~100,000 hours | Not specified (supports 32 langs) |
| **Vocoder** | Flow Matching (CFM) | **Vocos2D** | **Flow-VAE** Decoder |
