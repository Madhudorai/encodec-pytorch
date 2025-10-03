#!/usr/bin/env python3
"""
Download checkpoint script for EnCodec training.
Downloads checkpoints from Weights & Biases for resuming training.
Default: Downloads 74th epoch from madhudorai24/mono-encodec-nq2
"""

import os
import argparse
import wandb
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_checkpoint(entity, project, run_id, epoch, save_dir="./checkpoints_mono_nq2/", training_mode="mono"):
    """
    Download checkpoints for a specific epoch from a W&B run.
    
    Args:
        entity (str): W&B entity name
        project (str): W&B project name
        run_id (str): W&B run ID
        epoch (int): Epoch number to download
        save_dir (str): Directory to save the downloaded checkpoints
        training_mode (str): "mono" or "paired" - determines checkpoint file naming
    """
    try:
        # Initialize wandb API
        api = wandb.Api()
        
        # Get the run by ID
        run_path = f"{entity}/{project}/{run_id}"
        logger.info(f"Connecting to run: {run_path}")
        run = api.run(run_path)
        logger.info(f"✓ Found run: {run.name} (ID: {run.id})")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Target checkpoint files based on training mode
        if training_mode == "paired":
            checkpoint_files = [
                f"bs8_cut24000_length13800_epoch{epoch}_disc_lr0.0003.pt",
                f"bs8_cut24000_length13800_epoch{epoch}_lr0.0003.pt"
            ]
        else:  # mono
            checkpoint_files = [
                f"bs16_cut24000_length27600_epoch{epoch}_disc_lr0.0003.pt",
                f"bs16_cut24000_length27600_epoch{epoch}_lr0.0003.pt"
            ]
        
        downloaded_files = []
        
        # Look for the specific epoch artifact
        for artifact in run.logged_artifacts():
            artifact_name = artifact.name.lower()
            if f'epoch_{epoch}' in artifact_name or f'epoch{epoch}' in artifact_name:
                logger.info(f"✓ Found epoch {epoch} artifact: {artifact.name}")
                artifact_dir = artifact.download(root=save_dir)
                artifact_path = Path(artifact_dir)
                
                # Look for the specific checkpoint files
                for checkpoint_file in checkpoint_files:
                    matching_files = list(artifact_path.rglob(checkpoint_file))
                    if matching_files:
                        downloaded_files.extend(matching_files)
                        logger.info(f"✓ Found {checkpoint_file}")
                    else:
                        # Also look for files with similar patterns (in case naming is different)
                        pattern_files = list(artifact_path.rglob(f"*epoch{epoch}*.pt"))
                        if pattern_files:
                            downloaded_files.extend(pattern_files)
                            logger.info(f"✓ Found similar file: {pattern_files[0].name}")
                
                break  # Found our target epoch, no need to check other artifacts
            else:
                logger.debug(f"Skipping artifact {artifact.name} (not epoch {epoch})")
        
        if not downloaded_files:
            logger.warning(f"Could not find epoch {epoch} checkpoint files: {checkpoint_files}")
            logger.info("Available artifacts:")
            for artifact in run.logged_artifacts():
                logger.info(f"  - {artifact.name}")
        
        return downloaded_files
        
    except Exception as e:
        logger.error(f"Error downloading checkpoint: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download checkpoints from Weights & Biases")
    parser.add_argument("--entity", default="madhudorai24", help="W&B entity name")
    parser.add_argument("--project", default="mono-encodec-nq2", help="W&B project name")
    parser.add_argument("--run_id", default="s2pnxx7w", help="W&B run ID")
    parser.add_argument("--epoch", type=int, default=74, help="Epoch number to download")
    parser.add_argument("--save_dir", default="/user/i/iran/encodec-pytorch/checkpoints_mono_nq2/", help="Directory to save checkpoints")
    parser.add_argument("--mode", choices=["mono", "paired"], default="mono", help="Training mode: mono or paired")
    
    args = parser.parse_args()
    
    logger.info(f"Downloading epoch {args.epoch} checkpoints from {args.entity}/{args.project} ({args.mode} mode)")
    downloaded_files = download_checkpoint(args.entity, args.project, args.run_id, args.epoch, args.save_dir, args.mode)
    
    if downloaded_files:
        logger.info(f"✅ Successfully downloaded {len(downloaded_files)} checkpoint files:")
        for file in downloaded_files:
            logger.info(f"  - {file}")
    else:
        logger.error("❌ Download failed - no checkpoint files found")

if __name__ == "__main__":
    main()