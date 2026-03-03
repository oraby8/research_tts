
import os
import torch
import pandas as pd
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset

class ArabicDataset(Dataset):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.sr = model.sr
        
        # Load metadata
        metadata_path = os.path.join(config.data_dir, "processed_metadata.csv")
        self.df = pd.read_csv(metadata_path)
        
        # Filter long files (> 30s)
        if "duration_seconds" in self.df.columns:
            max_dur = 30.0
            initial_len = len(self.df)
            self.df = self.df[self.df["duration_seconds"] <= max_dur]
            if initial_len - len(self.df) > 0:
                print(f"🧹 Filtered {initial_len - len(self.df)} samples > {max_dur}s from training set")

        # NEW (handles 2 columns):
        self.df["normalized_text"] = self.df["transcription"]
        
        # Wav directory
        self.wav_dir = config.data_dir
        
        # Filter valid files
        valid_rows = []
        for idx, row in self.df.iterrows():
            wav_path = os.path.join(self.wav_dir, row["file_name"])
            
            if os.path.exists(wav_path):
                valid_rows.append(row)
        
        self.df = pd.DataFrame(valid_rows)
        print(f"Dataset loaded: {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        wav_path = os.path.join(self.wav_dir, row["file_name"])
        wav, sr = sf.read(wav_path, dtype='float32')
        
        # Convert to tensor
        wav = torch.from_numpy(wav)
        
        # Handle stereo -> mono
        if wav.dim() > 1:
            wav = wav.mean(dim=-1)
        
        # Resample if needed (use torchaudio for quality)
        if sr != self.sr:
            wav = wav.unsqueeze(0)  # Add channel dim
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav = resampler(wav).squeeze(0)
        
        # NO truncation needed - chunks are already 2-4s!
        
        # Get text
        text = row["normalized_text"]        
        return {
            "audio": wav,
            "text": text,
            "filename": row["file_name"]
        }

class TestArabicDataset(Dataset):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.sr = model.sr
        
        # Load metadata
        metadata_path = os.path.join(self.config.test_data_dir, "processed_metadata.csv")
        self.df = pd.read_csv(metadata_path)

        # Filter long files (> 30s)
        if "duration_seconds" in self.df.columns:
            max_dur = 30.0
            initial_len = len(self.df)
            self.df = self.df[self.df["duration_seconds"] <= max_dur]
            if initial_len - len(self.df) > 0:
                print(f"🧹 Filtered {initial_len - len(self.df)} samples > {max_dur}s from test set")

        # NEW (handles 2 columns):
        self.df["normalized_text"] = self.df["transcription"]
        
        # Wav directory
        self.wav_dir = config.test_data_dir
        
        # Filter valid files
        valid_rows = []
        for idx, row in self.df.iterrows():
            wav_path = os.path.join(self.wav_dir, row["file_name"])
            
            if os.path.exists(wav_path):
                valid_rows.append(row)
        
        self.df = pd.DataFrame(valid_rows)
        print(f"Dataset loaded: {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio
        wav_path = os.path.join(self.wav_dir, row["file_name"])
        wav, sr = sf.read(wav_path, dtype='float32')
        
        # Convert to tensor
        wav = torch.from_numpy(wav)
        
        # Handle stereo -> mono
        if wav.dim() > 1:
            wav = wav.mean(dim=-1)
        
        # Resample if needed (use torchaudio for quality)
        if sr != self.sr:
            wav = wav.unsqueeze(0)  # Add channel dim
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav = resampler(wav).squeeze(0)
        
        # NO truncation needed - chunks are already 2-4s!
        
        # Get text
        text = row["normalized_text"]        
        return {
            "audio": wav,
            "text": text,
            "filename": row["file_name"]
        }
        
def collate_fn(batch):
    """Collate batch with padding"""
    # Find max audio length
    max_len = max(item["audio"].shape[0] for item in batch)
    
    # Pad audio
    audios = []
    for item in batch:
        audio = item["audio"]
        if audio.shape[0] < max_len:
            padding = torch.zeros(max_len - audio.shape[0])
            audio = torch.cat([audio, padding])
        audios.append(audio)
    
    return {
        "audio": torch.stack(audios),
        "text": [item["text"] for item in batch],
        "filename": [item["filename"] for item in batch]
    }