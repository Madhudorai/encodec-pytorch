"""
Combined dataset that merges EigenscapeDataset and MultiDataset (spatial audio datasets).

This allows training with both Eigenscape and spatial audio datasets (STARSS, DCASE, LOCATA, MARCO).
"""

import torch
import logging
from torch.utils.data import ConcatDataset

logger = logging.getLogger(__name__)

try:
    from . import eigenscape_dataset
    # multi_dataset is in root directory - should be importable when running from project root
    import multi_dataset
except ImportError as e:
    logger.warning(f"Failed to import dataset modules: {e}")


class CombinedSpatialDataset(torch.utils.data.Dataset):
    """Combined dataset that merges Eigenscape and spatial audio datasets.
    
    This dataset combines:
    - EigenscapeDataset: Multi-channel audio from Eigenscape
    - MultiDataset: Spatial audio datasets from CSV files (STARSS, DCASE, LOCATA, MARCO)
    
    Both datasets should return the same format: (waveform, sample_rate) or (waveform, sample_rate, channels)
    """
    
    def __init__(self, config, transform=None, mode='train', use_eigenscape=True, use_spatial=True):
        """
        Args:
            config: Configuration object
            transform: Optional transform
            mode: 'train', 'val', or 'test'
            use_eigenscape: Whether to include Eigenscape dataset
            use_spatial: Whether to include spatial audio datasets (from CSV)
        """
        self.config = config
        self.transform = transform
        self.mode = mode
        
        datasets = []
        
        # Add Eigenscape dataset if enabled
        if use_eigenscape:
            try:
                eigenscape_ds = eigenscape_dataset.EigenscapeDataset(config=config, transform=transform, mode=mode)
                datasets.append(eigenscape_ds)
                logger.info(f"✓ Added EigenscapeDataset: {len(eigenscape_ds)} samples")
            except Exception as e:
                logger.warning(f"Failed to create EigenscapeDataset: {e}")
        
        # Add spatial audio datasets if enabled
        if use_spatial:
            try:
                spatial_ds = multi_dataset.MultiDataset(config=config, transform=transform, mode=mode)
                datasets.append(spatial_ds)
                logger.info(f"✓ Added MultiDataset (spatial): {len(spatial_ds)} samples")
            except Exception as e:
                logger.warning(f"Failed to create MultiDataset: {e}")
        
        if len(datasets) == 0:
            raise ValueError("No datasets available! Enable at least one of use_eigenscape or use_spatial.")
        
        # Combine datasets using ConcatDataset
        if len(datasets) == 1:
            self.dataset = datasets[0]
        else:
            self.dataset = ConcatDataset(datasets)
        
        logger.info(f"Combined dataset total size: {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

