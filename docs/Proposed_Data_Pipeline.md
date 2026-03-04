# Analysis of TTS Data Pipelines & Proposed Architecture

## 1. Similarities in Recent SOTA Approaches

Analyzing **CosyVoice 3**, **GLM-TTS**, and **MiniMax-Speech**, several convergent trends appear in their data processing pipelines. They all move beyond simple "download and train" strategies into rigorous, multi-stage filtering and refinement workflows.

### Key Shared Patterns:

1.  **Dual/Multi-ASR Verification (The "Consensus" Strategy):**
    *   **Observation:** None of the papers trust a single ASR model.
    *   **CosyVoice 3:** Uses *three* models (Whisper, Canary, SeamlessM4T) and requires consensus (<15% pairwise WER).
    *   **GLM-TTS:** Uses "Double-check" with Paraformer/SenseVoice (CN) and Whisper/Reverb (EN).
    *   **MiniMax:** Explicitly mentions "Dual ASR verification."
    *   **Takeaway:** High-quality transcriptions are non-negotiable. If ASR models disagree, the data is likely noisy or unintelligible and should be discarded.

2.  **Acoustic-Driven Punctuation Restoration:**
    *   **Observation:** Text punctuation often doesn't match speech prosody (pauses).
    *   **CosyVoice 3:** Uses **MFA (Montreal Forced Aligner)** to detect silence duration; adds commas for pauses >300ms, removes them for <50ms.
    *   **GLM-TTS:** Uses forced alignment to check character duration and adjust punctuation based on thresholds.
    *   **MiniMax:** Refines punctuation using VAD and ASR timestamps.
    *   **Takeaway:** Punctuation must be *re-generated* based on the actual audio, not just the raw text, to teach the model proper prosody.

3.  **Strict Segmentation & Filtering:**
    *   **VAD & Diarization:** All use VAD (Voice Activity Detection) and Speaker Diarization (often `pyannote.audio`) to ensure clean, single-speaker segments.
    *   **Outlier Removal:** They filter based on **Audio-Text Length Ratios** (CosyVoice 3) or **Timbre Consistency** (MiniMax) to remove data that is technically valid but statistically abnormal (e.g., speaking too fast/slow or wrong speaker).

4.  **Audio Cleaning (Divergence point):**
    *   **GLM-TTS** & **CosyVoice** lean towards aggressive cleaning (Source Separation/Denoising).
    *   **MiniMax** prefers preserving *steady-state noise* for naturalness.

---

## 2. Proposed "Best-of-Breed" Data Pipeline

Based on these architectures, here is a design for a robust, production-grade data pipeline. This pipeline prioritizes **quality over quantity** and utilizes open-source tools to replicate the methods described in the papers.

### **Phase 1: Ingestion & Segmentation**
**Goal:** Turn raw audio (podcasts, videos, audiobooks) into clean, single-speaker chunks.

1.  **Format Standardization:**
    *   Convert all inputs to `WAV`, 24kHz (or 44.1kHz), Mono.
2.  **Source Separation (Optional but Recommended):**
    *   *Tool:* `audio-separator` (using `MDX-Net` or `Mel-Band Roformer`).
    *   *Action:* Separate Vocals from Instrumental/Noise. Discard the instrumental track.
3.  **Diarization & VAD:**
    *   *Tool:* `pyannote.audio` (pipeline 3.1).
    *   *Action:* Identify speaker turns. Segment audio by speaker.
    *   *Filter:* Discard segments < 1 second or > 30 seconds.

### **Phase 2: The "Consensus" Transcription**
**Goal:** Ensure text is 99% accurate without human review.

1.  **Primary ASR:**
    *   *Tool:* `Faster-Whisper` (Large-v3).
    *   *Action:* Transcribe the segment.
2.  **Secondary ASR:**
    *   *Tool:* `FunASR` (Paraformer) for Chinese, or `NVIDIA Canary` / `SeamlessM4T` for Multilingual.
    *   *Action:* Transcribe the *same* segment.
3.  **Consensus Check:**
    *   *Action:* Calculate **Character Error Rate (CER)** or **Word Error Rate (WER)** between Primary and Secondary transcripts.
    *   *Threshold:* If WER > 10-15%, **DISCARD** the segment. (If models disagree, the audio is ambiguous).

### **Phase 3: Alignment & Refinement**
**Goal:** Align text perfectly with audio and fix prosody.

1.  **Text Normalization:**
    *   *Tool:* `WeTextProcessing` or simple regex.
    *   *Action:* Convert numbers to words ("1990" -> "nineteen ninety").
2.  **Forced Alignment:**
    *   *Tool:* `Montreal Forced Aligner (MFA)`.
    *   *Action:* Generate a `.TextGrid` aligning every phoneme/word to timestamps.
3.  **Punctuation Adjustment (The "CosyVoice" Method):**
    *   *Logic:*
        *   If silence between words > 300ms → Insert `,` (comma).
        *   If silence between words > 800ms → Insert `.` (period).
        *   If existing comma has silence < 50ms → Remove it.

### **Phase 4: Final Quality Filtering**
**Goal:** Remove bad audio that survived previous steps.

1.  **Timbre Consistency (The "MiniMax" Method):**
    *   *Tool:* `WavLM` or `ERes2Net`.
    *   *Action:* Extract speaker embedding for every clip of Speaker X. Calculate the centroid (average) embedding.
    *   *Filter:* Discard any clip with Cosine Similarity < 0.75 to the centroid (removes wrong speakers/bad quality segments).
2.  **Length Ratio Filter:**
    *   *Action:* Calculate `Character Count / Audio Duration`.
    *   *Filter:* Remove the top/bottom 5% (outliers that are speaking impossibly fast or slow).

### Summary Workflow Diagram

```mermaid
graph TD
    A[Raw Audio] --> B[Source Separation (Roformer)]
    B --> C[Diarization & VAD (Pyannote)]
    C --> D[Segmented Audio]
    
    D --> E[ASR 1: Whisper Large v3]
    D --> F[ASR 2: Paraformer/Canary]
    
    E & F --> G{WER Check < 15%}
    G -- Fail --> H[Discard]
    G -- Pass --> I[Forced Alignment (MFA)]
    
    I --> J[Punctuation Adjustment]
    J --> K[Speaker Embedding Check (WavLM)]
    
    K -- Low Sim --> L[Discard]
    K -- High Sim --> M[Final Dataset]
```
