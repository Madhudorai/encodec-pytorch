import os
import random
import glob
from pathlib import Path

import soundfile as sf
import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)

from utils import convert_audio


class MultiChannelAudioDataset(torch.utils.data.Dataset):
    """Custom dataset for 32-channel audio files with mono segment extraction."""
    
    def __init__(self, config, transform=None, mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        
        self.data_root = config.datasets.data_root
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels
        self.tensor_cut = config.datasets.tensor_cut
        self.fixed_length = config.datasets.fixed_length
        self.transform = transform
        
        # Define folder splits
        if mode == 'train':
            self.folders = ['Beach', 'Busy Street', 'Park', 'Pedestrian Zone', 'Quiet Street', 'Shopping Centre']
        else:  # test/validation
            self.folders = ['Woodland', 'Train Station']
        
        # Collect all audio files from specified folders
        self.audio_files = []
        for folder in self.folders:
            folder_path = Path(self.data_root) / folder
            if folder_path.exists():
                # Find all audio files in the folder (assuming common audio extensions)
                audio_extensions = ['*.wav', '*.flac', '*.mp3', '*.m4a']
                for ext in audio_extensions:
                    self.audio_files.extend(glob.glob(str(folder_path / ext)))
            else:
                logger.warning(f"Folder {folder_path} does not exist")
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in folders: {self.folders}")
        
        logger.info(f"Found {len(self.audio_files)} audio files for {mode} mode")
        logger.info(f"Folders: {self.folders}")

    def __len__(self):
        return self.fixed_length if self.fixed_length and len(self.audio_files) > self.fixed_length else len(self.audio_files)

    def get(self, idx=None):
        """Get uncropped, untransformed audio with random channel selection."""
        if idx is not None and idx >= len(self.audio_files):
            raise StopIteration
        if idx is None:
            idx = random.randrange(len(self.audio_files))
        
        try:
            audio_path = self.audio_files[idx]
            logger.debug(f'Loading {audio_path}')
            
            # Load audio file with soundfile (supports multi-channel)
            audio_data, sample_rate = sf.read(audio_path)
            
            # Convert to torch tensor
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data).float()
            
            # Handle different audio shapes
            if len(audio_data.shape) == 1:
                # Mono audio - expand to 32 channels (duplicate)
                audio_data = audio_data.unsqueeze(0).expand(32, -1)
            elif len(audio_data.shape) == 2:
                # Multi-channel audio - transpose to [channels, samples]
                if audio_data.shape[0] > audio_data.shape[1]:
                    audio_data = audio_data.T
                # Ensure we have at least 32 channels (pad if necessary)
                if audio_data.shape[0] < 32:
                    padding = torch.zeros(32 - audio_data.shape[0], audio_data.shape[1])
                    audio_data = torch.cat([audio_data, padding], dim=0)
                elif audio_data.shape[0] > 32:
                    # If more than 32 channels, take first 32
                    audio_data = audio_data[:32]
            else:
                raise ValueError(f"Unexpected audio shape: {audio_data.shape}")
            
            # Randomly select one channel (1-32, 0-indexed as 0-31)
            channel_idx = random.randint(0, 31)
            mono_audio = audio_data[channel_idx]  # Shape: [samples]
            
            # Convert to mono tensor with channel dimension
            mono_audio = mono_audio.unsqueeze(0)  # Shape: [1, samples]
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                mono_audio = convert_audio(mono_audio, sample_rate, self.sample_rate, self.channels)
            
            return mono_audio, self.sample_rate
            
        except Exception as e:
            logger.warning(f"Error loading {self.audio_files[idx]}: {e}")
            # Return a random sample instead
            return self[random.randint(0, len(self.audio_files) - 1)]

    def __getitem__(self, idx):
        waveform, sample_rate = self.get(idx)

        if self.transform:
            waveform = self.transform(waveform)

        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1] - self.tensor_cut - 1)
                waveform = waveform[:, start:start + self.tensor_cut]
                return waveform, sample_rate
            else:
                return waveform, sample_rate
        
        return waveform, sample_rate


def pad_sequence(batch):
    """Make all tensors in a batch the same length by padding with zeros."""
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    """Collate function for the dataloader."""
    tensors = []
    for waveform, _ in batch:
        tensors += [waveform]
    
    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    return tensors
