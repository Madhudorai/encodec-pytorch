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


class PairedChannelDataset(torch.utils.data.Dataset):
    """Custom dataset for paired channel audio files with same time segment from two different channels."""
    
    def __init__(self, config, transform=None, mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        
        self.data_root = config.datasets.data_root
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels
        self.tensor_cut = config.datasets.tensor_cut
        self.fixed_length = config.datasets.fixed_length
        self.transform = transform
        self.mode = mode
        
        # Define folder splits (same as original)
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
        
        # For validation, create fixed pairs
        if mode == 'test':
            self.fixed_pairs = self._create_fixed_validation_pairs()
        
        logger.info(f"Found {len(self.audio_files)} audio files for {mode} mode")
        logger.info(f"Folders: {self.folders}")

    def __len__(self):
        return self.fixed_length if self.fixed_length > 0 else len(self.audio_files)

    def _create_fixed_validation_pairs(self):
        """Create fixed validation pairs for consistent evaluation."""
        fixed_pairs = []
        random.seed(42)  # Fixed seed for reproducible validation
        
        for i in range(min(100, len(self.audio_files))):  # Use up to 100 validation pairs
            audio_path = self.audio_files[i % len(self.audio_files)]
            
            # Get file info
            info = sf.info(audio_path)
            num_channels = info.channels
            file_duration = info.frames / info.samplerate
            
            # Fixed channel and time selection
            # Ensure we have at least 2 channels for pairing
            if num_channels < 2:
                continue
                
            channel1_idx = i % min(16, num_channels)  # First channel: 0-15
            channel2_idx = (i + 1) % min(16, num_channels)  # Second channel: 1-16, wrapping around
            
            # Ensure different channels
            if channel1_idx == channel2_idx:
                channel2_idx = (channel2_idx + 1) % min(16, num_channels)
            
            max_start_time = max(0, file_duration - 1.0)
            if max_start_time > 0:
                start_time = (i * 0.1) % max_start_time  # Fixed pattern
            else:
                start_time = 0.0
            
            fixed_pairs.append({
                'audio_path': audio_path,
                'channel1_idx': channel1_idx,
                'channel2_idx': channel2_idx,
                'start_time': start_time,
                'sample_rate': info.samplerate
            })
        
        random.seed()  # Reset to random seed
        return fixed_pairs

    def get(self, idx=None):
        """Get uncropped, untransformed audio pair with random channel selection."""
        if idx is not None and idx >= len(self):
            raise StopIteration
        if idx is None:
            idx = random.randrange(len(self))
        
        try:
            # For validation, use fixed pairs
            if self.mode == 'test' and hasattr(self, 'fixed_pairs'):
                pair = self.fixed_pairs[idx % len(self.fixed_pairs)]
                audio_path = pair['audio_path']
                channel1_idx = pair['channel1_idx']
                channel2_idx = pair['channel2_idx']
                start_time = pair['start_time']
                sample_rate = pair['sample_rate']
            else:
                # For training, use random selection
                audio_path = self.audio_files[idx % len(self.audio_files)]
                logger.debug(f'Loading pair from {audio_path}')
                
                # First, get file info to determine number of channels
                info = sf.info(audio_path)
                num_channels = info.channels
                file_duration = info.frames / info.samplerate
                
                # Ensure we have at least 2 channels for pairing
                if num_channels < 2:
                    # If single channel, duplicate it (fallback)
                    channel1_idx = 0
                    channel2_idx = 0
                else:
                    # Randomly select two different channels
                    channel1_idx = random.randint(0, min(15, num_channels - 1))
                    channel2_idx = random.randint(0, min(15, num_channels - 1))
                    
                    # Ensure different channels
                    while channel1_idx == channel2_idx and num_channels > 1:
                        channel2_idx = random.randint(0, min(15, num_channels - 1))
                
                # Randomly select start time for 1-second segment
                max_start_time = max(0, file_duration - 1.0)  # Leave 1 second at end
                start_time = random.uniform(0, max_start_time)
                sample_rate = info.samplerate
            
            # Load both channels
            audio_data, sample_rate = sf.read(
                audio_path, 
                start=int(start_time * sample_rate),
                frames=int(1.0 * sample_rate),  # 1 second
                always_2d=False  # Get 1D array for single channel
            )
            
            # If we got 2D array, take the specific channels
            if len(audio_data.shape) == 2:
                audio1 = audio_data[:, channel1_idx]
                audio2 = audio_data[:, channel2_idx]
            else:
                # Single channel case - duplicate
                audio1 = audio_data
                audio2 = audio_data
            
            # Convert to torch tensors
            if isinstance(audio1, np.ndarray):
                audio1 = torch.from_numpy(audio1).float()
            if isinstance(audio2, np.ndarray):
                audio2 = torch.from_numpy(audio2).float()
            
            # Ensure we have exactly 1 second of audio
            target_length = int(1.0 * self.sample_rate)
            if len(audio1) < target_length:
                # Pad with zeros if too short
                padding1 = torch.zeros(target_length - len(audio1))
                audio1 = torch.cat([audio1, padding1])
            elif len(audio1) > target_length:
                # Truncate if too long
                audio1 = audio1[:target_length]
                
            if len(audio2) < target_length:
                # Pad with zeros if too short
                padding2 = torch.zeros(target_length - len(audio2))
                audio2 = torch.cat([audio2, padding2])
            elif len(audio2) > target_length:
                # Truncate if too long
                audio2 = audio2[:target_length]
            
            # Convert to mono tensors with channel dimension
            mono_audio1 = audio1.unsqueeze(0)  # Shape: [1, samples]
            mono_audio2 = audio2.unsqueeze(0)  # Shape: [1, samples]
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                mono_audio1 = convert_audio(mono_audio1, sample_rate, self.sample_rate, self.channels)
                mono_audio2 = convert_audio(mono_audio2, sample_rate, self.sample_rate, self.channels)
            
            return mono_audio1, mono_audio2, self.sample_rate
            
        except Exception as e:
            logger.warning(f"Error loading audio pair: {e}")
            # Return a random sample instead
            return self[random.randint(0, len(self) - 1)]

    def __getitem__(self, idx):
        waveform1, waveform2, sample_rate = self.get(idx)

        if self.transform:
            waveform1 = self.transform(waveform1)
            waveform2 = self.transform(waveform2)

        if self.tensor_cut > 0:
            if waveform1.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform1.size()[1] - self.tensor_cut - 1)
                waveform1 = waveform1[:, start:start + self.tensor_cut]
                waveform2 = waveform2[:, start:start + self.tensor_cut]
                return waveform1, waveform2, sample_rate
        
        return waveform1, waveform2, sample_rate


def pad_sequence_pairs(batch):
    """Make all tensor pairs in a batch the same length by padding with zeros."""
    waveforms1 = []
    waveforms2 = []
    
    for waveform1, waveform2, _ in batch:
        waveforms1.append(waveform1.permute(1, 0))
        waveforms2.append(waveform2.permute(1, 0))
    
    # Pad both sequences
    waveforms1 = torch.nn.utils.rnn.pad_sequence(waveforms1, batch_first=True, padding_value=0.)
    waveforms2 = torch.nn.utils.rnn.pad_sequence(waveforms2, batch_first=True, padding_value=0.)
    
    # Permute back to [B, C, T]
    waveforms1 = waveforms1.permute(0, 2, 1)
    waveforms2 = waveforms2.permute(0, 2, 1)
    
    return waveforms1, waveforms2


def collate_fn_pairs(batch):
    """Collate function for the paired dataloader."""
    waveforms1, waveforms2 = pad_sequence_pairs(batch)
    return waveforms1, waveforms2
