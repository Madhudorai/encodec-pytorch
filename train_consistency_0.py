import logging
import os
import warnings
from collections import defaultdict
import random
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import hydra
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchaudio
import wandb

import eigenscape_dataset as data
from eigenscape_dataset import collate_fn
from losses import disc_loss, total_loss
from model_consistency_0 import EncodecModelWithSliceConsistency
from msstftd import MultiScaleSTFTDiscriminator
from scheduler import WarmupCosineLrScheduler
from utils import (count_parameters, save_master_checkpoint, set_seed)
from balancer import Balancer
from cal_metrics import calculate_si_snr

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def calculate_confidence_interval(values, confidence=0.95):
    """Calculate mean and 95% confidence interval for a list of values."""
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    n = len(values)
    
    # Calculate 95% confidence interval using t-distribution
    from scipy import stats
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_val * (std / np.sqrt(n))
    
    return mean, mean - margin_error, mean + margin_error


def compute_cosine_weight_factor(epoch, start_epoch, max_epoch, start_weight, end_weight):
    """Compute cosine annealing/ramp-up weight factor.
    
    Args:
        epoch: Current epoch number
        start_epoch: Starting epoch (usually 1)
        max_epoch: Maximum epoch number
        start_weight: Starting weight value
        end_weight: Ending weight value
    
    Returns:
        Weight factor interpolated between start_weight and end_weight using cosine schedule
    """
    if max_epoch <= start_epoch:
        return end_weight
    
    # Compute progress from 0 to 1
    progress = (epoch - start_epoch) / (max_epoch - start_epoch)
    progress = max(0.0, min(1.0, progress))  # Clamp to [0, 1]
    
    # Cosine interpolation: weight = start + (end - start) * (1 - cos(π * progress)) / 2
    import math
    cosine_factor = (1 - math.cos(math.pi * progress)) / 2
    weight = start_weight + (end_weight - start_weight) * cosine_factor
    
    return weight


def train_one_step(epoch, optimizer, optimizer_disc, model, disc_model, trainloader, config, scheduler, disc_scheduler, scaler=None, scaler_disc=None, balancer=None, wandb_logger=None, model_weight_factor=1.0, consistency_weight_factor=1.0):
    """Train one step function with slice consistency loss and cosine weight scheduling.
    
    Args:
        model_weight_factor: Scaling factor for model losses (loss_g, loss_w). Cosine anneals from 1.0 to 0.5.
        consistency_weight_factor: Scaling factor for consistency losses. Cosine ramps from 0.1 to 1.0.
    """
    """Train one step function with slice consistency loss."""
    model.train()
    disc_model.train()
    data_length = len(trainloader)
    
    # Initialize variables to accumulate losses  
    accumulated_loss_g = 0.0
    accumulated_losses_g = defaultdict(float)
    accumulated_loss_w = 0.0
    accumulated_loss_disc = 0.0
    accumulated_loss_slice = 0.0
    accumulated_losses_slice = defaultdict(float)

    for idx, batch_data in enumerate(trainloader):
        # Handle new dataset format: (waveforms, sample_rates, selected_channels_list)
        if isinstance(batch_data, tuple) and len(batch_data) == 3:
            input_wav, sample_rates, selected_channels_list = batch_data
        else:
            # Fallback for old format
            input_wav = batch_data
        
        input_wav = input_wav.contiguous()
        if torch.cuda.is_available():
            input_wav = input_wav.cuda()
        optimizer.zero_grad()
        
        with autocast(enabled=config.common.amp):
            output, loss_w, _, slice_consistency_output = model(input_wav, return_slice_consistency=True)
            
            # Reshape discriminator input to [B*C, 1, T] for consistency with encoder/decoder
            B, C, T = input_wav.shape
            input_wav_reshaped = input_wav.reshape(B * C, 1, T) if C > 1 else input_wav
            output_reshaped = output.reshape(B * C, 1, T) if C > 1 else output
            
            logits_real, fmap_real = disc_model(input_wav_reshaped)
            logits_fake, fmap_fake = disc_model(output_reshaped)
            losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                input_wav, 
                output, 
                sample_rate=config.model.sample_rate,
            )
        
        # Compute total generator loss
        if config.common.amp: 
            loss_g = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f']
            
            # Apply model weight scaling (cosine annealing: 1.0 -> 0.5)
            loss_g = loss_g * model_weight_factor
            loss_w_scaled = loss_w * model_weight_factor
            
            # Add slice consistency loss if available
            if slice_consistency_output is not None:
                loss_slice = slice_consistency_output['loss']
                # Apply consistency weight scaling (cosine ramp-up: 0.1 -> 1.0)
                loss_slice = loss_slice * consistency_weight_factor
                loss_g = loss_g + loss_slice
                accumulated_loss_slice += loss_slice.item()
                for k, v in slice_consistency_output['loss_dict'].items():
                    # Scale individual loss components for logging
                    scaled_v = v.item() * consistency_weight_factor if isinstance(v, torch.Tensor) else v * consistency_weight_factor
                    accumulated_losses_slice[k] += scaled_v
            
            loss_g = loss_g + loss_w_scaled
            
            scaler.scale(loss_g).backward()  
            scaler.step(optimizer)  
            scaler.update()  
        else:
            if balancer is not None:
                # Balancer handles backward on losses_g through output with retain_graph=True
                # so we can backward additional losses (loss_w, loss_slice) afterward
                balancer.backward(losses_g, output, retain_graph=True)
                loss_g = sum([l * balancer.weights[k] for k, l in losses_g.items()])
            else:
                loss_g = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f']
            
            # Apply model weight scaling (cosine annealing: 1.0 -> 0.5)
            loss_g = loss_g * model_weight_factor
            loss_w_scaled = loss_w * model_weight_factor
            
            # Add slice consistency loss if available
            loss_slice = None
            if slice_consistency_output is not None:
                loss_slice = slice_consistency_output['loss']
                # Apply consistency weight scaling (cosine ramp-up: 0.1 -> 1.0)
                loss_slice = loss_slice * consistency_weight_factor
                accumulated_loss_slice += loss_slice.item()
                for k, v in slice_consistency_output['loss_dict'].items():
                    # Scale individual loss components for logging
                    scaled_v = v.item() * consistency_weight_factor if isinstance(v, torch.Tensor) else v * consistency_weight_factor
                    accumulated_losses_slice[k] += scaled_v
                # Add to loss_g for accumulation (convert to tensor if needed)
                if balancer is not None:
                    # loss_g is scalar, convert loss_slice to scalar for accumulation
                    loss_g = loss_g + loss_slice.item()
                else:
                    # loss_g is tensor, add loss_slice tensor
                    loss_g = loss_g + loss_slice
            
            # Backward: combine all losses before backward to avoid graph violation
            if balancer is None:
                # Combine all losses (loss_g, loss_w_scaled, loss_slice) and backward once
                combined_loss = loss_g + loss_w_scaled
                combined_loss.backward()
            else:
                # Balancer already backwarded on output, so we need to backward
                # loss_w_scaled and loss_slice separately with retain_graph=True
                # (they share the same computational graph as output)
                loss_w_scaled.backward(retain_graph=True)
                if loss_slice is not None:
                    loss_slice.backward(retain_graph=True)
            
            optimizer.step()

        # Accumulate losses  
        if isinstance(loss_g, torch.Tensor):
            accumulated_loss_g += loss_g.item()
        else:
            accumulated_loss_g += loss_g
        for k, l in losses_g.items():
            accumulated_losses_g[k] += l.item()
        # loss_w_scaled is defined in both AMP and non-AMP paths
        accumulated_loss_w += loss_w_scaled.item()

        # Update discriminator with probability
        optimizer_disc.zero_grad()
        train_discriminator = torch.BoolTensor([config.model.train_discriminator 
                               and epoch >= config.lr_scheduler.warmup_epoch 
                               and random.random() < 0.5])
        if torch.cuda.is_available():
            train_discriminator = train_discriminator.cuda()

        if train_discriminator:
            with autocast(enabled=config.common.amp):
                # Reshape discriminator input to [B*C, 1, T] for consistency with encoder/decoder
                B, C, T = input_wav.shape
                input_wav_reshaped = input_wav.reshape(B * C, 1, T) if C > 1 else input_wav
                output_reshaped = output.reshape(B * C, 1, T) if C > 1 else output
                
                logits_real, _ = disc_model(input_wav_reshaped)
                logits_fake, _ = disc_model(output_reshaped.detach())
                loss_disc = disc_loss(logits_real, logits_fake)
            if config.common.amp: 
                scaler_disc.scale(loss_disc).backward()
                scaler_disc.step(optimizer_disc)  
                scaler_disc.update()  
            else:
                loss_disc.backward() 
                optimizer_disc.step()

            accumulated_loss_disc += loss_disc.item()
        
        scheduler.step()
        disc_scheduler.step()

    # Print epoch summary
    avg_loss_g = accumulated_loss_g / data_length
    avg_loss_w = accumulated_loss_w / data_length
    avg_loss_disc = accumulated_loss_disc / data_length if accumulated_loss_disc > 0 else 0.0
    avg_loss_slice = accumulated_loss_slice / data_length if accumulated_loss_slice > 0 else 0.0
    
    log_msg = f"| TRAIN | epoch: {epoch} | loss_g: {avg_loss_g:.4f} | loss_w: {avg_loss_w:.4f}"
    
    if avg_loss_slice > 0:
        log_msg += f" | loss_slice: {avg_loss_slice:.4f}"
    
    log_msg += f" | lr_G: {optimizer.param_groups[0]['lr']:.6e} | lr_D: {optimizer_disc.param_groups[0]['lr']:.6e}"
    log_msg += f" | w_model: {model_weight_factor:.3f} | w_consistency: {consistency_weight_factor:.3f}"
    
    if config.model.train_discriminator and epoch >= config.lr_scheduler.warmup_epoch:
        log_msg += f" | loss_disc: {avg_loss_disc:.4f}"
    
    logger.info(log_msg)
    
    # Weights & Biases logging
    if wandb_logger:
        log_dict = {
            'epoch': epoch,
            'train/loss_g': avg_loss_g,
            'train/loss_w': avg_loss_w,
            'train/lr_g': optimizer.param_groups[0]['lr'],
            'train/lr_d': optimizer_disc.param_groups[0]['lr'],
            'train/model_weight_factor': model_weight_factor,
            'train/consistency_weight_factor': consistency_weight_factor,
        }
        for k, l in accumulated_losses_g.items():
            log_dict[f'train/{k}'] = l / data_length
        
        if avg_loss_slice > 0:
            log_dict['train/loss_slice'] = avg_loss_slice
            for k, v in accumulated_losses_slice.items():
                log_dict[f'train/slice_{k}'] = v / data_length
        
        if config.model.train_discriminator and epoch >= config.lr_scheduler.warmup_epoch:
            log_dict['train/loss_disc'] = avg_loss_disc
        wandb_logger.log(log_dict)


@torch.no_grad()
def validate(epoch, model, disc_model, valloader, config, wandb_logger=None):
    """Validation function with bandwidth-specific metrics."""
    model.eval()
    disc_model.eval()
    
    total_loss_g = 0.0
    total_loss_disc = 0.0
    total_si_snr = 0.0
    num_samples = 0
    
    # Track metrics per bandwidth - store individual values for confidence intervals
    bandwidth_metrics = defaultdict(lambda: {
        'si_snr': [], 'count': 0
    })
    
    # Track consistency accuracy metrics - CODEBOOK 0 ONLY
    slice_consistency_codebook0 = defaultdict(list)  # Original full vs original slice (codebook 0 only)
    
    augmentation_consistency_codebook0 = defaultdict(list)  # Original full vs perturbed full (codebook 0 only)
    
    inter_channel_consistency_codebook0 = defaultdict(list)  # Between random channels (codebook 0 only)
    
    for idx, batch_data in enumerate(valloader):
        # Handle new dataset format: (waveforms, sample_rates, selected_channels_list)
        if isinstance(batch_data, tuple) and len(batch_data) == 3:
            input_wav, sample_rates, selected_channels_list = batch_data
        else:
            # Fallback for old format
            input_wav = batch_data
        
        if torch.cuda.is_available():
            input_wav = input_wav.cuda()
        
        # Test all bandwidths
        for bandwidth in config.model.target_bandwidths:
            model.bandwidth = bandwidth
            
            # Enable slice consistency during validation to compute accuracy metrics
            # Pass batch_idx for deterministic slice positions and perturbations
            if model.use_slice_consistency:
                output, loss_w, frames, slice_consistency_output = model(input_wav, return_slice_consistency=True, batch_idx=idx)
            else:
                output = model(input_wav, batch_idx=idx)
                slice_consistency_output = None
            
            # Reshape discriminator input to [B*C, 1, T] for consistency with encoder/decoder
            B, C, T = input_wav.shape
            input_wav_reshaped = input_wav.reshape(B * C, 1, T) if C > 1 else input_wav
            output_reshaped = output.reshape(B * C, 1, T) if C > 1 else output
            
            logits_real, fmap_real = disc_model(input_wav_reshaped)
            logits_fake, fmap_fake = disc_model(output_reshaped)
            loss_disc = disc_loss(logits_real, logits_fake)
            losses_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output)
            
            total_loss_g += sum([l.item() for l in losses_g.values()])
            total_loss_disc += loss_disc.item()
            
            # Extract consistency accuracy metrics if available - CODEBOOK 0 ONLY
            if slice_consistency_output is not None:
                # Slice consistency metrics (original full vs original slice) - codebook 0 only
                if 'slice_consistency_codebook0_accuracy' in slice_consistency_output:
                    slice_consistency_codebook0[bandwidth].append(slice_consistency_output['slice_consistency_codebook0_accuracy'])
                
                # Augmentation consistency metrics (original full vs perturbed full) - codebook 0 only
                if 'augmentation_consistency_codebook0_accuracy' in slice_consistency_output:
                    augmentation_consistency_codebook0[bandwidth].append(slice_consistency_output['augmentation_consistency_codebook0_accuracy'])
                
                # Inter-channel consistency metrics (between random channels) - codebook 0 only
                if 'inter_channel_consistency_codebook0_accuracy' in slice_consistency_output:
                    inter_channel_consistency_codebook0[bandwidth].append(slice_consistency_output['inter_channel_consistency_codebook0_accuracy'])
            
            # Calculate comprehensive metrics for each sample in batch
            batch_size = input_wav.shape[0]
            for i in range(batch_size):
                ref_sample = input_wav[i].squeeze().cpu()
                rec_sample = output[i].squeeze().cpu()
                
                # Calculate only SI-SNR metric
                try:
                    si_snr = calculate_si_snr(ref_sample.numpy(), rec_sample.numpy())
                except Exception as e:
                    si_snr = None
                
                # Update totals and collect individual values
                if si_snr is not None:
                    total_si_snr += si_snr
                    bandwidth_metrics[bandwidth]['si_snr'].append(si_snr)
                
                bandwidth_metrics[bandwidth]['count'] += 1
                num_samples += 1
    
    avg_loss_g = total_loss_g / (len(valloader) * len(config.model.target_bandwidths))
    avg_loss_disc = total_loss_disc / (len(valloader) * len(config.model.target_bandwidths))
    avg_si_snr = total_si_snr / num_samples if num_samples > 0 else 0.0
    
    log_msg = f"| VAL  | epoch: {epoch} | loss_g: {avg_loss_g:.4f} | loss_disc: {avg_loss_disc:.4f} | SI-SNR: {avg_si_snr:.2f} dB"
    logger.info(log_msg)
    
    # Log bandwidth-specific metrics with confidence intervals
    for bandwidth, metrics in bandwidth_metrics.items():
        count = metrics['count']
        if count > 0:
            # Calculate confidence intervals
            si_snr_mean, si_snr_ci_low, si_snr_ci_high = calculate_confidence_interval(metrics['si_snr'])
            
            logger.info(f"  Bandwidth {bandwidth} kbps (n={count}):")
            si_snr_margin = (si_snr_ci_high - si_snr_ci_low) / 2
            logger.info(f"    SI-SNR: {si_snr_mean:.2f}±{si_snr_margin:.2f} dB")
            
            # Log slice consistency accuracy metrics (original full vs original slice) - CODEBOOK 0 ONLY
            if bandwidth in slice_consistency_codebook0 and len(slice_consistency_codebook0[bandwidth]) > 0:
                slice_cb0 = slice_consistency_codebook0[bandwidth]
                avg_slice_cb0 = sum(slice_cb0) / len(slice_cb0)
                logger.info(f"    Slice Consistency Codebook 0: {avg_slice_cb0:.4f}")
            
            # Log augmentation consistency accuracy metrics (original full vs perturbed full) - CODEBOOK 0 ONLY
            if bandwidth in augmentation_consistency_codebook0 and len(augmentation_consistency_codebook0[bandwidth]) > 0:
                aug_cb0 = augmentation_consistency_codebook0[bandwidth]
                avg_aug_cb0 = sum(aug_cb0) / len(aug_cb0)
                logger.info(f"    Augmentation Consistency Codebook 0: {avg_aug_cb0:.4f}")
            
            # Log inter-channel consistency accuracy metrics (between random channels) - CODEBOOK 0 ONLY
            if bandwidth in inter_channel_consistency_codebook0 and len(inter_channel_consistency_codebook0[bandwidth]) > 0:
                inter_cb0 = inter_channel_consistency_codebook0[bandwidth]
                avg_inter_cb0 = sum(inter_cb0) / len(inter_cb0)
                logger.info(f"    Inter-Channel Consistency Codebook 0: {avg_inter_cb0:.4f}")
    
    # Weights & Biases logging
    if wandb_logger:
        val_log_dict = {
            'epoch': epoch,
            'val/loss_g': avg_loss_g,
            'val/loss_disc': avg_loss_disc,
            'val/si_snr': avg_si_snr,
        }
        
        # Log bandwidth-specific metrics with confidence intervals
        for bandwidth, metrics in bandwidth_metrics.items():
            count = metrics['count']
            if count > 0:
                si_snr_mean, si_snr_ci_low, si_snr_ci_high = calculate_confidence_interval(metrics['si_snr'])
                
                val_log_dict[f'val/si_snr_bw_{bandwidth}'] = si_snr_mean
                val_log_dict[f'val/si_snr_bw_{bandwidth}_ci_low'] = si_snr_ci_low
                val_log_dict[f'val/si_snr_bw_{bandwidth}_ci_high'] = si_snr_ci_high
                
                # Log slice consistency accuracy metrics (original full vs original slice) - CODEBOOK 0 ONLY
                if bandwidth in slice_consistency_codebook0 and len(slice_consistency_codebook0[bandwidth]) > 0:
                    slice_cb0 = slice_consistency_codebook0[bandwidth]
                    avg_slice_cb0 = sum(slice_cb0) / len(slice_cb0)
                    val_log_dict[f'val/slice_consistency_codebook0_bw_{bandwidth}'] = avg_slice_cb0
                
                # Log augmentation consistency accuracy metrics (original full vs perturbed full) - CODEBOOK 0 ONLY
                if bandwidth in augmentation_consistency_codebook0 and len(augmentation_consistency_codebook0[bandwidth]) > 0:
                    aug_cb0 = augmentation_consistency_codebook0[bandwidth]
                    avg_aug_cb0 = sum(aug_cb0) / len(aug_cb0)
                    val_log_dict[f'val/augmentation_consistency_codebook0_bw_{bandwidth}'] = avg_aug_cb0
                
                # Log inter-channel consistency accuracy metrics (between random channels) - CODEBOOK 0 ONLY
                if bandwidth in inter_channel_consistency_codebook0 and len(inter_channel_consistency_codebook0[bandwidth]) > 0:
                    inter_cb0 = inter_channel_consistency_codebook0[bandwidth]
                    avg_inter_cb0 = sum(inter_cb0) / len(inter_cb0)
                    val_log_dict[f'val/inter_channel_consistency_codebook0_bw_{bandwidth}'] = avg_inter_cb0
        
        wandb_logger.log(val_log_dict)


def train(config):
    """Main training function."""
    # Remove existing logging handlers
    logger.handlers.clear()

    # Set up logging
    file_handler = logging.FileHandler(f"{config.checkpoint.save_folder}/train_consistency_0_bs{config.datasets.batch_size}_lr{config.optimization.lr}.log")
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
    file_handler.setFormatter(formatter)

    # Print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Initialize Weights & Biases
    wandb_logger = None
    if config.get('wandb', {}).get('enabled', True):
        try:
            wandb.init(
                project=config.get('wandb', {}).get('project', 'eigenscape-encodec-consistency-0'),
                name=config.get('wandb', {}).get('name', f'consistency_0_bs{config.datasets.batch_size}_lr{config.optimization.lr}'),
                config=dict(config),
                dir=config.checkpoint.save_folder,
            )
            wandb_logger = wandb
            logger.info("✓ Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")
            wandb_logger = None

    # Set seed
    if config.common.seed is not None:
        set_seed(config.common.seed)

    # Set up datasets
    trainset = data.EigenscapeDataset(config=config, mode='train')
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=config.datasets.num_workers,
        pin_memory=config.datasets.pin_memory
    )
    
    
    # Set up models
    # Get slice consistency config
    slice_consistency_config = config.model.get('slice_consistency', None)
    # Get perturb encoder config
    perturb_encoder_config = config.model.get('perturb_encoder', None)
    # Get inter-channel consistency config
    inter_channel_consistency_config = config.model.get('inter_channel_consistency', None)
    
    # Model processes each channel separately as mono (channels=1)
    # Reshapes [B, C, T] -> [B*C, 1, T] to process each channel separately
    model = EncodecModelWithSliceConsistency._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        channels=1,
        causal=config.model.causal, 
        model_norm=config.model.norm, 
        audio_normalize=config.model.audio_normalize,
        segment=eval(str(config.model.segment)) if config.model.segment is not None else None, 
        name=config.model.name,
        ratios=config.model.ratios,
        slice_consistency=slice_consistency_config,
        perturb_encoder=perturb_encoder_config,
        inter_channel_consistency=inter_channel_consistency_config,
    )
    
    disc_model = MultiScaleSTFTDiscriminator(
        in_channels=1,
        out_channels=1,
        filters=config.model.filters,
        hop_lengths=config.model.disc_hop_lengths,
        win_lengths=config.model.disc_win_lengths,
        n_ffts=config.model.disc_n_ffts,
    )

    # Log model information
    logger.info(model)
    logger.info(disc_model)
    logger.info(config)
    logger.info(f"EnCodec Model Parameters: {count_parameters(model)} | Disc Model Parameters: {count_parameters(disc_model)}")
    logger.info(f"Model train mode: {model.training} | Quantizer train mode: {model.quantizer.training}")
    logger.info(f"Target bandwidths: {config.model.target_bandwidths}")
    logger.info(f"Slice consistency enabled: {model.use_slice_consistency}")
    logger.info(f"Perturb encoder enabled: {model.perturb_encoder is not None}")
    if model.perturb_encoder is not None:
        logger.info(f"  Perturb methods: {model.perturb_encoder.perturb_methods}")
        logger.info(f"  Perturb all audio: {model.perturb_encoder.perturb_all_audio}")
        logger.info(f"  Perturb slice audio: {model.perturb_encoder.perturb_slice_audio}")

    # Resume training if specified
    loaded_epoch = 0
    model_checkpoint = None
    disc_model_checkpoint = None
    if config.checkpoint.resume:
        assert config.checkpoint.checkpoint_path != '', "resume path is empty"
        assert config.checkpoint.disc_checkpoint_path != '', "disc resume path is empty"

        if not os.path.exists(config.checkpoint.checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {config.checkpoint.checkpoint_path}")
        if not os.path.exists(config.checkpoint.disc_checkpoint_path):
            raise FileNotFoundError(f"Discriminator checkpoint not found: {config.checkpoint.disc_checkpoint_path}")

        logger.info(f"Loading model checkpoint from: {config.checkpoint.checkpoint_path}")
        logger.info(f"Loading discriminator checkpoint from: {config.checkpoint.disc_checkpoint_path}")

        model_checkpoint = torch.load(config.checkpoint.checkpoint_path, map_location='cpu')
        disc_model_checkpoint = torch.load(config.checkpoint.disc_checkpoint_path, map_location='cpu')
        
        # Load model state dict (should be compatible since base model is the same)
        model_state_dict = model_checkpoint['model_state_dict']
        # Filter out slice_consistency module if it exists in checkpoint but not in model
        if hasattr(model, 'slice_consistency') and model.slice_consistency is None:
            # Remove slice_consistency keys if they exist
            model_state_dict = {k: v for k, v in model_state_dict.items() 
                              if not k.startswith('slice_consistency')}
        
        model.load_state_dict(model_state_dict, strict=False)
        disc_model.load_state_dict(disc_model_checkpoint['model_state_dict'])
        loaded_epoch = model_checkpoint['epoch']
        
        logger.info(f"✓ Successfully loaded model weights from epoch {loaded_epoch}")
        logger.info(f"  Starting training from epoch 1 (fresh training with loaded weights)")

    if torch.cuda.is_available():
        model.cuda()
        disc_model.cuda()

    logger.info(f"Training: {len(trainloader)} batches (batch_size={config.datasets.batch_size})")
    logger.info(f"Validation: 2k fixed segments per epoch (batch_size={config.datasets.batch_size})")

    # Set up optimizers and schedulers
    params = [p for p in model.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': config.optimization.lr}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params': disc_params, 'lr': config.optimization.disc_lr}], betas=(0.5, 0.9))
    
    scheduler = WarmupCosineLrScheduler(
        optimizer, 
        max_iter=config.common.max_epoch * len(trainloader), 
        eta_ratio=0.1, 
        warmup_iter=config.lr_scheduler.warmup_epoch * len(trainloader), 
        warmup_ratio=1e-4
    )
    disc_scheduler = WarmupCosineLrScheduler(
        optimizer_disc, 
        max_iter=config.common.max_epoch * len(trainloader), 
        eta_ratio=0.1, 
        warmup_iter=config.lr_scheduler.warmup_epoch * len(trainloader), 
        warmup_ratio=1e-4
    )

    scaler = GradScaler() if config.common.amp else None
    scaler_disc = GradScaler() if config.common.amp else None  

    # Load optimizer and scheduler states if resuming
    # (Even though we start from epoch 1, we load optimizer/scheduler states for continuity)
    if config.checkpoint.resume:
        if 'optimizer_state_dict' in model_checkpoint.keys():
            optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
            logger.info(f"✓ Loaded generator optimizer state from epoch {loaded_epoch}")
        
        if 'optimizer_state_dict' in disc_model_checkpoint.keys():
            optimizer_disc.load_state_dict(disc_model_checkpoint['optimizer_state_dict'])
            logger.info(f"✓ Loaded discriminator optimizer state from epoch {loaded_epoch}")
        
        if 'scheduler_state_dict' in model_checkpoint.keys():
            scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
            logger.info(f"✓ Loaded generator scheduler state from epoch {loaded_epoch}")
            
        if 'scheduler_state_dict' in disc_model_checkpoint.keys():
            disc_scheduler.load_state_dict(disc_model_checkpoint['scheduler_state_dict'])
            logger.info(f"✓ Loaded discriminator scheduler state from epoch {loaded_epoch}")

    # Start training from epoch 1 (fresh training with loaded weights)
    # This allows fresh logging and training continuation
    start_epoch = 1
    
    # Instantiate loss balancer
    balancer = Balancer(dict(config.balancer.weights)) if hasattr(config, 'balancer') else None
    if balancer:
        logger.info(f'Loss balancer with weights {balancer.weights} instantiated')
    
    # Training loop
    for epoch in range(start_epoch, config.common.max_epoch + 1):
        # Compute cosine weight factors for this epoch
        # Model weights: start at 1.0, gradually decrease to 0.5
        model_weight_factor = compute_cosine_weight_factor(
            epoch, start_epoch, config.common.max_epoch, 
            start_weight=1.0, end_weight=0.5
        )
        # Consistency weights: start at 0.1, gradually increase to 1.0
        consistency_weight_factor = compute_cosine_weight_factor(
            epoch, start_epoch, config.common.max_epoch,
            start_weight=0.1, end_weight=1.0
        )
        
        train_one_step(
            epoch, optimizer, optimizer_disc, 
            model, disc_model, trainloader, config,
            scheduler, disc_scheduler, scaler, scaler_disc, balancer, wandb_logger,
            model_weight_factor=model_weight_factor,
            consistency_weight_factor=consistency_weight_factor
        )
        
        # Validation
        if epoch % config.common.val_interval == 0 and epoch > 0:
            # Create validation dataset with fixed segments (same every epoch)
            # Note: dataset is recreated but uses fixed seed, so segments are identical
            valset = data.EigenscapeDataset(config=config, mode='val')
            valloader = torch.utils.data.DataLoader(
                valset,
                batch_size=config.datasets.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=config.datasets.num_workers,
                pin_memory=config.datasets.pin_memory
            )
            validate(epoch, model, disc_model, valloader, config, wandb_logger)
        
        
        # Save checkpoint
        if epoch % config.common.save_interval == 0:
            if not os.path.exists(config.checkpoint.save_folder):
                os.makedirs(config.checkpoint.save_folder)
            model_path = f'{config.checkpoint.save_location}epoch{epoch}_lr{config.optimization.lr}.pt'
            disc_path = f'{config.checkpoint.save_location}epoch{epoch}_disc_lr{config.optimization.lr}.pt'
            
            try:
                save_master_checkpoint(epoch, model, optimizer, scheduler, model_path)  
                save_master_checkpoint(epoch, disc_model, optimizer_disc, disc_scheduler, disc_path)
                logger.info(f"✓ Saved checkpoints for epoch {epoch}")
            except Exception as e:
                logger.error(f"❌ Failed to save checkpoints for epoch {epoch}: {e}")
                raise  # Re-raise to stop training if checkpoint saving fails
            
            # Log model artifacts to wandb (with timeout to prevent hanging)
            # Skip artifact upload if disabled in config or if upload fails/times out
            upload_artifacts = config.get('wandb', {}).get('upload_artifacts', True)
            if wandb_logger and upload_artifacts:
                try:
                    artifact = wandb.Artifact(f'consistency_0_model_epoch_{epoch}', type='model')
                    
                    # CRITICAL: Wrap add_file in timeout - file I/O can hang on network filesystems
                    # artifact.add_file() reads files from disk and can block indefinitely
                    def add_files_to_artifact():
                        artifact.add_file(model_path)
                        artifact.add_file(disc_path)
                    
                    # Use thread pool with timeout for file I/O (reading checkpoints)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future_add = executor.submit(add_files_to_artifact)
                        try:
                            future_add.result(timeout=60)  # 1 minute timeout for file I/O
                            logger.info(f"✓ Added checkpoint files to artifact for epoch {epoch}")
                        except FutureTimeoutError:
                            logger.warning(f"⚠ Adding files to artifact timed out for epoch {epoch} (file I/O hang). Skipping upload...")
                            future_add.cancel()
                            raise  # Re-raise to skip upload
                        except Exception as e:
                            logger.warning(f"⚠ Failed to add files to artifact for epoch {epoch}: {e}. Skipping upload...")
                            raise  # Re-raise to skip upload
                    
                    # Use thread pool with timeout to prevent blocking indefinitely on network upload
                    # Timeout: 300 seconds (5 minutes) - adjust if checkpoints are very large
                    def upload_artifact():
                        wandb_logger.log_artifact(artifact)
                    
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(upload_artifact)
                        try:
                            future.result(timeout=300)  # 5 minute timeout
                            logger.info(f"✓ Successfully uploaded wandb artifact for epoch {epoch}")
                        except FutureTimeoutError:
                            logger.warning(f"⚠ Wandb artifact upload timed out for epoch {epoch} (exceeded 5 minutes). Continuing training...")
                            # Cancel the future (though it may continue in background)
                            future.cancel()
                        except Exception as e:
                            logger.warning(f"⚠ Wandb artifact upload failed for epoch {epoch}: {e}. Continuing training...")
                except Exception as e:
                    logger.warning(f"⚠ Failed to create/upload wandb artifact for epoch {epoch}: {e}. Continuing training...")
            elif wandb_logger and not upload_artifacts:
                logger.info(f"Skipping wandb artifact upload for epoch {epoch} (disabled in config - checkpoints saved locally only)")
            
            # Keep all checkpoints locally - no deletion
            # All checkpoints are saved in: {config.checkpoint.save_folder}/
            logger.info(f"Checkpoints saved locally for epoch {epoch} (keeping all checkpoints, no deletion)")
    
    # Finish wandb run
    if wandb_logger:
        wandb.finish()


@hydra.main(config_path='config', config_name='config_consistency_0')
def main(config):
    # Disable cudnn
    torch.backends.cudnn.enabled = False
    
    # Memory optimization
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if not os.path.exists(config.checkpoint.save_folder):
        os.makedirs(config.checkpoint.save_folder)
    
    train(config)


if __name__ == '__main__':
    main()
