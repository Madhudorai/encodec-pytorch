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
            
            # First, get file info to determine number of channels
            info = sf.info(audio_path)
            num_channels = info.channels
            file_duration = info.frames / info.samplerate
            
            # Randomly select one channel (1-32, 0-indexed as 0-31)
            # If file has fewer channels, cycle through them
            channel_idx = random.randint(0, min(31, num_channels - 1))
            
            # Randomly select start time for 1-second segment
            max_start_time = max(0, file_duration - 1.0)  # Leave 1 second at end
            start_time = random.uniform(0, max_start_time)
            
            # Load only the specific channel and segment
            # soundfile.read can load specific channels and time ranges
            audio_data, sample_rate = sf.read(
                audio_path, 
                start=int(start_time * info.samplerate),
                frames=int(1.0 * info.samplerate),  # 1 second
                always_2d=False  # Get 1D array for single channel
            )
            
            # If we got 2D array, take the specific channel
            if len(audio_data.shape) == 2:
                audio_data = audio_data[:, channel_idx]
            
            # Convert to torch tensor
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data).float()
            
            # Ensure we have exactly 1 second of audio
            target_length = int(1.0 * self.sample_rate)
            if len(audio_data) < target_length:
                # Pad with zeros if too short
                padding = torch.zeros(target_length - len(audio_data))
                audio_data = torch.cat([audio_data, padding])
            elif len(audio_data) > target_length:
                # Truncate if too long
                audio_data = audio_data[:target_length]
            
            # Convert to mono tensor with channel dimension
            mono_audio = audio_data.unsqueeze(0)  # Shape: [1, samples]
            
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
