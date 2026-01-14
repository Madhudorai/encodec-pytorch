#!/usr/bin/env python3
"""
Checkpoint management script for EnCodec training.
- Downloads checkpoints from Weights & Biases for resuming training
- Uploads local checkpoints to Weights & Biases for sharing/backup
Default: Downloads 74th epoch from madhudorai24/mono-encodec-nq2
"""

import os
import argparse
import wandb
import logging
from pathlib import Path
from typing import List, Optional

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


def upload_checkpoint_to_wandb(checkpoint_path: str, disc_checkpoint_path: Optional[str],
                               entity: str, project: str, run_id: str, epoch: int,
                               artifact_name: Optional[str] = None):
    """
    Upload local checkpoint files to W&B as an artifact.
    
    Args:
        checkpoint_path: Path to model checkpoint file
        disc_checkpoint_path: Optional path to discriminator checkpoint file
        entity: W&B entity name
        project: W&B project name
        run_id: W&B run ID (or run name)
        epoch: Epoch number
        artifact_name: Optional custom artifact name (default: auto-generated)
    """
    try:
        logger.info(f"Uploading checkpoint (epoch {epoch}) to W&B...")
        
        # Initialize wandb API
        api = wandb.Api()
        
        # Get the run by ID or name
        run_path = f"{entity}/{project}/{run_id}"
        logger.info(f"Connecting to run: {run_path}")
        try:
            run = api.run(run_path)
        except Exception as e:
            logger.error(f"Failed to find run {run_path}: {e}")
            logger.info("Creating new run for upload...")
            # If run doesn't exist, we'll create a new one
            wandb.init(entity=entity, project=project, name=run_id, job_type="checkpoint_upload")
            run = wandb.run
        
        logger.info(f"✓ Found run: {run.name} (ID: {run.id})")
        
        # Verify checkpoint files exist
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        disc_file = None
        if disc_checkpoint_path:
            disc_file = Path(disc_checkpoint_path)
            if not disc_file.exists():
                logger.warning(f"Discriminator checkpoint file not found: {disc_checkpoint_path}")
                disc_file = None
        
        # Create artifact name
        if artifact_name is None:
            artifact_name = f"model_epoch_{epoch}"
        
        # Create artifact
        artifact = wandb.Artifact(artifact_name, type="model")
        
        # Add checkpoint files
        artifact.add_file(str(checkpoint_file), name=checkpoint_file.name)
        logger.info(f"✓ Added model checkpoint: {checkpoint_file.name}")
        
        if disc_file:
            artifact.add_file(str(disc_file), name=disc_file.name)
            logger.info(f"✓ Added discriminator checkpoint: {disc_file.name}")
        
        # Upload artifact
        logger.info(f"Uploading artifact '{artifact_name}' to W&B...")
        run.log_artifact(artifact)
        
        # Wait for artifact to be processed
        artifact.wait()
        
        logger.info(f"✅ Successfully uploaded checkpoint (epoch {epoch}) to W&B")
        logger.info(f"   Artifact: {artifact_name}")
        logger.info(f"   Run: {run_path}")
        
        # Finish wandb if we created a new run
        if hasattr(wandb, 'run') and wandb.run is not None:
            wandb.finish()
        
        return artifact_name
        
    except Exception as e:
        logger.error(f"Error uploading checkpoint to W&B: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download or upload checkpoints to/from Weights & Biases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download checkpoint
  python download_checkpoint.py --download --run_id s2pnxx7w --epoch 74
  
  # Upload checkpoint
  python download_checkpoint.py --upload --checkpoint path/to/checkpoint.pt --disc_checkpoint path/to/disc.pt --run_id my_run --epoch 150
        """
    )
    
    # Mode selection
    parser.add_argument("--download", action="store_true", help="Download checkpoint from W&B")
    parser.add_argument("--upload", action="store_true", help="Upload checkpoint to W&B")
    
    # Common arguments
    parser.add_argument("--entity", default="madhudorai24", help="W&B entity name")
    parser.add_argument("--project", default="mono-encodec-nq2", help="W&B project name")
    parser.add_argument("--run_id", default="s2pnxx7w", help="W&B run ID or run name")
    parser.add_argument("--epoch", type=int, default=74, help="Epoch number")
    
    # Download arguments
    parser.add_argument("--save_dir", default="/user/i/iran/encodec-pytorch/checkpoints_mono_nq2/", 
                       help="Directory to save downloaded checkpoints")
    parser.add_argument("--mode", choices=["mono", "paired"], default="mono", 
                       help="Training mode: mono or paired (for download)")
    
    # Upload arguments
    parser.add_argument("--checkpoint", type=str, 
                       help="Path to model checkpoint file (required for upload)")
    parser.add_argument("--disc_checkpoint", type=str, 
                       help="Path to discriminator checkpoint file (optional for upload)")
    parser.add_argument("--artifact_name", type=str, 
                       help="Custom artifact name (default: model_epoch_{epoch})")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.download and not args.upload:
        parser.error("Must specify either --download or --upload")
    
    if args.download and args.upload:
        parser.error("Cannot specify both --download and --upload")
    
    # Download mode
    if args.download:
        logger.info(f"Downloading epoch {args.epoch} checkpoints from {args.entity}/{args.project} ({args.mode} mode)")
        downloaded_files = download_checkpoint(args.entity, args.project, args.run_id, args.epoch, args.save_dir, args.mode)
        
        if downloaded_files:
            logger.info(f"✅ Successfully downloaded {len(downloaded_files)} checkpoint files:")
            for file in downloaded_files:
                logger.info(f"  - {file}")
        else:
            logger.error("❌ Download failed - no checkpoint files found")
    
    # Upload mode
    elif args.upload:
        if not args.checkpoint:
            parser.error("--checkpoint is required for upload")
        
        logger.info(f"Uploading epoch {args.epoch} checkpoints to {args.entity}/{args.project}")
        artifact_name = upload_checkpoint_to_wandb(
            args.checkpoint,
            args.disc_checkpoint,
            args.entity,
            args.project,
            args.run_id,
            args.epoch,
            args.artifact_name
        )
        
        if artifact_name:
            logger.info(f"✅ Successfully uploaded checkpoint as artifact: {artifact_name}")

if __name__ == "__main__":
    main()