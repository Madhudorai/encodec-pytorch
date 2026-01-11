import os
import random
import torch
import librosa
import audioread
from pathlib import Path
import logging
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

from utils import convert_audio


class EigenscapeDataset(torch.utils.data.Dataset):
    """Dataset class for Eigenscape multi-channel audio files.
    
    Each WAV file has 32 channels. This dataset randomly selects N channels
    from each file and returns them as separate samples for training.
    """
    
    def __init__(self, config, transform=None, mode='train'):
        assert mode in ['train', 'val', 'test'], 'dataset mode must be train, val, or test'
        
        self.config = config
        self.transform = transform
        self.mode = mode
        self.fixed_length = config.datasets.fixed_length
        self.tensor_cut = config.datasets.tensor_cut
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels  # Should be 1 for mono output
        self.num_channels_to_load = config.datasets.num_channels_to_load
        
        # Get base path and folders
        self.base_path = Path(config.datasets.eigenscape_base_path)
        if mode == 'train':
            self.folders = config.datasets.eigenscape_train_folders
        else:  # val or test
            self.folders = config.datasets.eigenscape_valid_folders
        
        # Load all audio files from specified folders
        self.audio_files = self._load_audio_files()
        
        logger.info(f"Loaded {len(self.audio_files)} audio files from {len(self.folders)} folders for {mode} mode")
        logger.info(f"Will randomly select {self.num_channels_to_load} channels from each 32-channel file")
        
        # For validation and test, create fixed segments for consistent evaluation
        if mode in ['val', 'test']:
            self.fixed_segments = self._create_fixed_segments()
    
    def _load_audio_files(self):
        """Load all WAV files from the specified folders."""
        audio_files = []
        
        for folder_name in self.folders:
            folder_path = self.base_path / folder_name
            if not folder_path.exists():
                logger.warning(f"Folder not found: {folder_path}")
                continue
            
            # Find all WAV files in the folder
            wav_files = list(folder_path.glob("*.wav"))
            audio_files.extend(wav_files)
            logger.info(f"Found {len(wav_files)} WAV files in {folder_name}")
        
        if len(audio_files) == 0:
            raise ValueError(f"No audio files found in folders: {self.folders}")
        
        return audio_files
    
    def _create_fixed_segments(self):
        """Create fixed segments for validation/testing."""
        segments = []
        
        # For validation, create 2k fixed segments (same every epoch)
        if self.mode == 'val':
            num_segments = 2000
            random.seed(42)  # Fixed seed for reproducible validation segments
        else:  # test mode
            num_segments = 1000
            random.seed(42)  # Fixed seed for reproducible test evaluation
        
        # Create segments from all available audio files
        # Cycle through files multiple times to reach num_segments (similar to training)
        for i in range(num_segments):
            audio_path = self.audio_files[i % len(self.audio_files)]
            
            try:
                # Get file info
                info = sf.info(audio_path)
                file_duration = info.frames / info.samplerate
                
                # Random start time
                max_start_time = max(0, file_duration - 1.0)  # Leave 1 second at end
                start_time = random.uniform(0, max_start_time)
                
                # Randomly select channels
                selected_channels = random.sample(range(32), self.num_channels_to_load)
                
                segments.append({
                    'audio_path': audio_path,
                    'start_time': start_time,
                    'sample_rate': info.samplerate,
                    'selected_channels': selected_channels
                })
            except Exception as e:
                logger.warning(f"Failed to get info for {audio_path}: {e}")
                continue
        
        if self.mode == 'test':
            random.seed()  # Reset to random seed only for test mode
        # For validation, keep fixed seed (don't reset)
        
        logger.info(f"Created {len(segments)} segments for {self.mode}")
        return segments
    
    def _load_multi_channel_audio(self, audio_path, start_time=None, duration=None, selected_channels=None):
        """Load multi-channel audio and select specific channels.
        
        Args:
            audio_path: Path to the WAV file
            start_time: Optional start time in seconds
            duration: Optional duration in seconds
            selected_channels: List of channel indices to select (0-31)
            
        Returns:
            waveform: Audio tensor [num_selected_channels, T] or [C, T] where C=1 for mono
        """
        try:
            # Load the full multi-channel audio
            if start_time is not None and duration is not None:
                # Load specific segment
                waveform, sample_rate = librosa.load(
                    audio_path,
                    sr=self.sample_rate,
                    mono=False,  # Keep multi-channel
                    offset=start_time,
                    duration=duration
                )
            else:
                # Load full file
                waveform, sample_rate = librosa.load(
                    audio_path,
                    sr=self.sample_rate,
                    mono=False  # Keep multi-channel
                )
            
            # Convert to tensor
            waveform = torch.as_tensor(waveform)  # [C, T] where C=32
            
            # Select random channels if not specified
            if selected_channels is None:
                selected_channels = random.sample(range(waveform.shape[0]), self.num_channels_to_load)
            
            # Select the specified channels
            selected_waveform = waveform[selected_channels, :]  # [num_channels_to_load, T]
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                # Convert to [T, C] for resampling, then back
                selected_waveform = selected_waveform.transpose(0, 1)  # [T, C]
                selected_waveform = convert_audio(
                    selected_waveform,
                    sample_rate,
                    self.sample_rate,
                    self.num_channels_to_load
                )
                selected_waveform = selected_waveform.transpose(0, 1)  # [C, T]
            
            return selected_waveform, selected_channels
            
        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}")
            # Return silence if loading fails
            return torch.zeros(self.num_channels_to_load, int(self.sample_rate * (duration or 1.0))), selected_channels or list(range(self.num_channels_to_load))
    
    def __len__(self):
        if self.mode in ['val', 'test'] and hasattr(self, 'fixed_segments'):
            return len(self.fixed_segments)
        # For training, always use fixed_length if set (allows cycling through files)
        # This ensures consistent epoch size regardless of number of audio files
        if self.mode == 'train' and self.fixed_length:
            return self.fixed_length
        # Fallback: return number of audio files if fixed_length not set
        return len(self.audio_files)
    
    def get(self, idx=None):
        """Get uncropped, untransformed audio with random channel selection."""
        if idx is not None and idx >= len(self):
            raise StopIteration
        if idx is None:
            idx = random.randrange(len(self))
        
        try:
            if self.mode in ['val', 'test'] and hasattr(self, 'fixed_segments'):
                # Use fixed segments for validation/test
                segment = self.fixed_segments[idx % len(self.fixed_segments)]
                audio_path = segment['audio_path']
                start_time = segment['start_time']
                selected_channels = segment['selected_channels']
                
                waveform, _ = self._load_multi_channel_audio(
                    audio_path,
                    start_time=start_time,
                    duration=1.0,
                    selected_channels=selected_channels
                )
                sample_rate = self.sample_rate
            else:
                # For training, load random segment
                audio_path = self.audio_files[idx % len(self.audio_files)]
                
                # Get file info to determine duration
                info = sf.info(audio_path)
                file_duration = info.frames / info.samplerate
                
                # Random start time
                max_start_time = max(0, file_duration - 1.0)
                start_time = random.uniform(0, max_start_time)
                
                # Randomly select channels
                waveform, selected_channels = self._load_multi_channel_audio(
                    audio_path,
                    start_time=start_time,
                    duration=1.0,
                    selected_channels=None  # Random selection
                )
                sample_rate = self.sample_rate
                
        except (audioread.exceptions.NoBackendError, ZeroDivisionError, FileNotFoundError) as e:
            logger.warning(f"Not able to load audio: {e}")
            # Return a random sample instead
            return self[random.randint(0, len(self) - 1)]
        
        # Return waveform [num_channels_to_load, T] and metadata
        return waveform, sample_rate, selected_channels
    
    def __getitem__(self, idx):
        """Get item with transformation and cropping."""
        waveform, sample_rate, selected_channels = self.get(idx)
        
        if self.transform:
            waveform = self.transform(waveform)
        
        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1] - self.tensor_cut - 1)
                waveform = waveform[:, start:start + self.tensor_cut]
            else:
                # If audio is shorter than tensor_cut, pad with zeros
                if waveform.size()[1] < self.tensor_cut:
                    padding_size = self.tensor_cut - waveform.size()[1]
                    padding = torch.zeros(waveform.size()[0], padding_size)
                    waveform = torch.cat([waveform, padding], dim=1)
        
        # Return waveform [num_channels_to_load, T] and selected_channels list
        return waveform, sample_rate, selected_channels


def pad_sequence_multi_channel(batch):
    """Make all tensors in a batch the same length by padding with zeros.
    
    Handles multi-channel audio where each sample has shape [num_channels_to_load, T].
    """
    # Extract waveforms and channel info
    waveforms = [item[0] for item in batch]  # List of [C, T] tensors
    sample_rates = [item[1] for item in batch]
    selected_channels_list = [item[2] for item in batch]
    
    # Pad sequences: convert to [T, C] format for padding
    batch_tc = [item.permute(1, 0) for item in waveforms]  # [T, C]
    batch_padded = torch.nn.utils.rnn.pad_sequence(batch_tc, batch_first=True, padding_value=0.)
    batch_padded = batch_padded.permute(0, 2, 1)  # [B, C, T]
    
    return batch_padded, sample_rates, selected_channels_list


def collate_fn(batch):
    """Collate function for the dataloader.
    
    Returns:
        waveforms: Batched tensor [B, num_channels_to_load, T]
        sample_rates: List of sample rates
        selected_channels_list: List of lists, each containing selected channel indices
    """
    waveforms, sample_rates, selected_channels_list = pad_sequence_multi_channel(batch)
    return waveforms, sample_rates, selected_channels_list

