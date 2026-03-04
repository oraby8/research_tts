import os
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment

def load_audio(file_path, target_sr=24000):
    """
    Load audio file and resample to target sample rate.
    Returns:
        waveform (torch.Tensor): Audio tensor [channels, time]
        sr (int): Sample rate
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    waveform, sr = torchaudio.load(file_path)
    
    # Mix to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        
    return waveform, target_sr

def save_audio(waveform, file_path, sr=24000):
    """
    Save audio tensor to file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
        
    torchaudio.save(file_path, waveform, sr)

def normalize_loudness(file_path, target_db=-23.0):
    """
    Normalize audio loudness using PyDub (simple implementation).
    For production, consider pyloudnorm.
    """
    audio = AudioSegment.from_file(file_path)
    change_in_dBFS = target_db - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    return normalized_audio

def get_audio_duration(file_path):
    info = sf.info(file_path)
    return info.duration

def convert_to_wav(input_path, output_path, sr=24000):
    """
    Convert any audio format to WAV with specific SR using ffmpeg via pydub or torchaudio.
    Using torchaudio load/save is cleaner for python.
    """
    waveform, _ = load_audio(input_path, target_sr=sr)
    save_audio(waveform, output_path, sr=sr)
