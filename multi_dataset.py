import os
import random
import pandas as pd
import torch
import librosa
import audioread
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

from utils import convert_audio


class MultiDataset(torch.utils.data.Dataset):
    """Multi-dataset class that can handle multiple CSV files for training, validation, and testing."""
    
    def __init__(self, config, transform=None, mode='train'):
        assert mode in ['train', 'val', 'test'], 'dataset mode must be train, val, or test'
        
        self.config = config
        self.transform = transform
        self.mode = mode
        self.fixed_length = config.datasets.fixed_length
        self.tensor_cut = config.datasets.tensor_cut
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels
        
        # Load datasets based on mode
        self.audio_files = self._load_datasets()
        
        # Separate files by dataset for mixing strategies
        self.dataset_files = self._organize_files_by_dataset()
        
        logger.info(f"Loaded {len(self.audio_files)} audio files for {mode} mode")
        
        # Log dataset composition
        for dataset_name, files in self.dataset_files.items():
            logger.info(f"  {dataset_name}: {len(files)} files")
        
        # For validation and test, create fixed segments for consistent evaluation
        if mode in ['val', 'test']:
            self.fixed_segments = self._create_fixed_segments()

    def _load_datasets(self):
        """Load audio files from multiple datasets based on mode."""
        audio_files = []
        
        # Define dataset configurations
        datasets_config = {
            'jamendo': {
                'train': self.config.datasets.jamendo_train_csv,
                'val': self.config.datasets.jamendo_valid_csv,
                'test': self.config.datasets.jamendo_test_csv
            },
            'common_voice': {
                'train': self.config.datasets.common_voice_train_csv,
                'val': self.config.datasets.common_voice_valid_csv,
                'test': self.config.datasets.common_voice_test_csv
            },
            'fsd50k': {
                'train': self.config.datasets.fsd50k_train_csv,
                'val': self.config.datasets.fsd50k_valid_csv,
                'test': self.config.datasets.fsd50k_test_csv
            },
            'dns_challenge4': {
                'train': self.config.datasets.dns_challenge4_train_csv,
                'val': self.config.datasets.dns_challenge4_valid_csv,
                'test': self.config.datasets.dns_challenge4_test_csv
            }
        }
        
        # Add more datasets as needed
        # datasets_config['librispeech'] = {
        #     'train': self.config.datasets.librispeech_train_csv,
        #     'val': self.config.datasets.librispeech_valid_csv,
        #     'test': self.config.datasets.librispeech_test_csv
        # }
        
        # Load files from each dataset
        for dataset_name, csv_paths in datasets_config.items():
            csv_path = csv_paths[self.mode]
            
            if csv_path and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, on_bad_lines='skip')
                    if len(df) > 0:
                        # Assume the first column contains the audio file paths
                        dataset_files = df.iloc[:, 0].tolist()
                        audio_files.extend(dataset_files)
                        logger.info(f"Loaded {len(dataset_files)} files from {dataset_name} {self.mode}")
                    else:
                        logger.warning(f"Empty CSV file: {csv_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {dataset_name} {self.mode}: {e}")
            else:
                logger.warning(f"CSV file not found: {csv_path}")
        
        if len(audio_files) == 0:
            raise ValueError(f"No audio files found for {self.mode} mode")
        
        return audio_files

    def _organize_files_by_dataset(self):
        """Organize audio files by dataset for mixing strategies."""
        dataset_files = {
            'jamendo': [],
            'common_voice': [],
            'fsd50k': [],
            'dns_challenge4': []
        }
        
        for file_path in self.audio_files:
            dataset_name = self._get_dataset_name(file_path)
            if dataset_name in dataset_files:
                dataset_files[dataset_name].append(file_path)
        
        return dataset_files

    def _create_fixed_segments(self):
        """Create segments for validation/testing."""
        import random
        import soundfile as sf
        
        segments = []
        
        # For validation, create 10k random segments every epoch
        if self.mode == 'val':
            num_segments = 10000
        else:  # test mode
            num_segments = 1000
            random.seed(42)  # Fixed seed for reproducible test evaluation
        
        # Create segments from all available audio files
        for i in range(min(num_segments, len(self.audio_files) * 20)):  # Allow more segments per file
            audio_path = self.audio_files[i % len(self.audio_files)]
            
            try:
                # Get file info
                info = sf.info(audio_path)
                file_duration = info.frames / info.samplerate
                
                # Random start time
                max_start_time = max(0, file_duration - 1.0)  # Leave 1 second at end
                start_time = random.uniform(0, max_start_time)
                
                segments.append({
                    'audio_path': audio_path,
                    'start_time': start_time,
                    'sample_rate': info.samplerate
                })
            except Exception as e:
                logger.warning(f"Failed to get info for {audio_path}: {e}")
                continue
        
        if self.mode == 'test':
            random.seed()  # Reset to random seed only for test mode
        
        logger.info(f"Created {len(segments)} segments for {self.mode}")
        return segments

    def _get_dataset_name(self, file_path):
        """Extract dataset name from file path."""
        if 'jamendo' in file_path.lower():
            return 'jamendo'
        elif 'common_voice' in file_path.lower() or 'commonvoice' in file_path.lower():
            return 'common_voice'
        elif 'fsd50k' in file_path.lower():
            return 'fsd50k'
        elif 'dns_challenge4' in file_path.lower() or 'dns' in file_path.lower():
            return 'dns_challenge4'
        elif 'librispeech' in file_path.lower():
            return 'librispeech'
        else:
            return 'unknown'

    def _load_audio_segment(self, audio_path, duration=1.0):
        """Load a random 1-second segment from an audio file."""
        try:
            # Get file info to determine duration
            import soundfile as sf
            info = sf.info(audio_path)
            file_duration = info.frames / info.samplerate
            
            # Random start time
            max_start_time = max(0, file_duration - duration)
            start_time = random.uniform(0, max_start_time)
            
            # Load the segment
            waveform, sample_rate = librosa.load(
                audio_path,
                sr=self.sample_rate,
                mono=self.channels == 1,
                offset=start_time,
                duration=duration
            )
            
            # Convert to tensor and add channel dimension if needed
            waveform = torch.as_tensor(waveform)
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
                waveform = waveform.expand(self.channels, -1)
            
            # Ensure exact length to avoid tensor size mismatches
            target_length = int(self.sample_rate * duration)
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            elif waveform.shape[1] < target_length:
                padding = torch.zeros(self.channels, target_length - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)
            
            return waveform
            
        except Exception as e:
            logger.warning(f"Failed to load segment from {audio_path}: {e}")
            # Return silence if loading fails
            return torch.zeros(self.channels, int(self.sample_rate * duration))

    def _normalize_audio(self, waveform):
        """Normalize audio by file (per-file normalization)."""
        # Calculate RMS and normalize
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            waveform = waveform / rms
        return waveform

    def _apply_random_gain(self, waveform, min_gain_db=-10, max_gain_db=6):
        """Apply random gain between min_gain_db and max_gain_db."""
        gain_db = random.uniform(min_gain_db, max_gain_db)
        gain_linear = 10 ** (gain_db / 20)
        return waveform * gain_linear

    def _sample_mixing_strategy(self):
        """Sample one of the four mixing strategies based on probabilities."""
        rand = random.random()
        
        if rand < 0.32:
            return 's1'  # Single source from Jamendo (0.32)
        elif rand < 0.64:  # 0.32 + 0.32
            return 's2'  # Single source from other datasets (0.32)
        elif rand < 0.88:  # 0.64 + 0.24
            return 's3'  # Mix two sources (0.24)
        else:  # 0.88 to 1.0
            return 's4'  # Mix three sources (0.12)

    def _get_mixed_audio(self):
        """Get audio sample based on mixing strategy."""
        strategy = self._sample_mixing_strategy()
        
        if strategy == 's1':
            # Single source from Jamendo (probability 0.32)
            if len(self.dataset_files['jamendo']) > 0:
                audio_path = random.choice(self.dataset_files['jamendo'])
                waveform = self._load_audio_segment(audio_path)
            else:
                # Fallback to any available dataset
                all_files = [f for files in self.dataset_files.values() for f in files]
                audio_path = random.choice(all_files)
                waveform = self._load_audio_segment(audio_path)
        
        elif strategy == 's2':
            # Single source from other datasets (probability 0.10 each)
            other_datasets = ['common_voice', 'fsd50k', 'dns_challenge4']
            available_datasets = [d for d in other_datasets if len(self.dataset_files[d]) > 0]
            
            if available_datasets:
                dataset = random.choice(available_datasets)
                audio_path = random.choice(self.dataset_files[dataset])
                waveform = self._load_audio_segment(audio_path)
            else:
                # Fallback to any available dataset
                all_files = [f for files in self.dataset_files.values() for f in files]
                audio_path = random.choice(all_files)
                waveform = self._load_audio_segment(audio_path)
        
        elif strategy == 's3':
            # Mix two sources from all datasets (probability 0.24)
            all_files = [f for files in self.dataset_files.values() for f in files]
            if len(all_files) >= 2:
                # Sample two different files
                audio_paths = random.sample(all_files, 2)
                waveform1 = self._load_audio_segment(audio_paths[0])
                waveform2 = self._load_audio_segment(audio_paths[1])
                waveform = waveform1 + waveform2
            else:
                # Fallback to single source
                audio_path = random.choice(all_files)
                waveform = self._load_audio_segment(audio_path)
        
        elif strategy == 's4':
            # Mix three sources from all datasets except music (probability 0.12)
            non_music_files = []
            for dataset in ['common_voice', 'fsd50k', 'dns_challenge4']:
                non_music_files.extend(self.dataset_files[dataset])
            
            if len(non_music_files) >= 3:
                # Sample three different files
                audio_paths = random.sample(non_music_files, 3)
                waveform1 = self._load_audio_segment(audio_paths[0])
                waveform2 = self._load_audio_segment(audio_paths[1])
                waveform3 = self._load_audio_segment(audio_paths[2])
                waveform = waveform1 + waveform2 + waveform3
            elif len(non_music_files) >= 2:
                # Fallback to two sources
                audio_paths = random.sample(non_music_files, 2)
                waveform1 = self._load_audio_segment(audio_paths[0])
                waveform2 = self._load_audio_segment(audio_paths[1])
                waveform = waveform1 + waveform2
            else:
                # Fallback to single source
                all_files = [f for files in self.dataset_files.values() for f in files]
                audio_path = random.choice(all_files)
                waveform = self._load_audio_segment(audio_path)
        
        # Apply normalization and random gain
        waveform = self._normalize_audio(waveform)
        waveform = self._apply_random_gain(waveform)
        
        return waveform

    def __len__(self):
        if self.mode in ['val', 'test'] and hasattr(self, 'fixed_segments'):
            return len(self.fixed_segments)
        return self.fixed_length if self.fixed_length and len(self.audio_files) > self.fixed_length else len(self.audio_files)

    def get(self, idx=None):
        """Get uncropped, untransformed audio with mixing strategy for training and validation."""
        if idx is not None and idx >= len(self):
            raise StopIteration
        if idx is None:
            idx = random.randrange(len(self))
        
        try:
            # For validation, use mixing strategy
            if self.mode == 'val':
                # Use mixing strategy for validation
                waveform = self._get_mixed_audio()
                sample_rate = self.sample_rate
            # For test, use fixed segments (no mixing)
            elif self.mode == 'test' and hasattr(self, 'fixed_segments'):
                segment = self.fixed_segments[idx % len(self.fixed_segments)]
                audio_path = segment['audio_path']
                start_time = segment['start_time']
                sample_rate = segment['sample_rate']
                
                # Load specific segment
                waveform, sample_rate = librosa.load(
                    audio_path,
                    sr=self.sample_rate,
                    mono=self.channels == 1,
                    offset=start_time,
                    duration=1.0  # 1 second
                )
                
                # Add channel dimension if loaded audio was mono
                waveform = torch.as_tensor(waveform)
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
                    waveform = waveform.expand(self.channels, -1)
            else:
                # For training, use mixing strategy
                waveform = self._get_mixed_audio()
                sample_rate = self.sample_rate
                
        except (audioread.exceptions.NoBackendError, ZeroDivisionError, FileNotFoundError) as e:
            logger.warning(f"Not able to load audio: {e}")
            # Return a random sample instead
            return self[random.randint(0, len(self) - 1)]

        return waveform, sample_rate

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
                # If audio is shorter than tensor_cut, pad with zeros
                if waveform.size()[1] < self.tensor_cut:
                    padding_size = self.tensor_cut - waveform.size()[1]
                    padding = torch.zeros(waveform.size()[0], padding_size)
                    waveform = torch.cat([waveform, padding], dim=1)
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
