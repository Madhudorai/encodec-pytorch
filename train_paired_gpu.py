import logging
import os
import warnings
from collections import defaultdict
import random
from pathlib import Path

import hydra
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torchaudio
import wandb

import paired_channel_dataset as data
from paired_channel_dataset import collate_fn_pairs
from losses import disc_loss, total_loss
from pairwise_losses import pairwise_losses
from model import EncodecModel
from msstftd import MultiScaleSTFTDiscriminator
from scheduler import WarmupCosineLrScheduler
from utils import (count_parameters, save_master_checkpoint, set_seed)
from balancer import Balancer

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define train one step function for paired training
def train_one_step_paired(epoch, optimizer, optimizer_disc, model, disc_model, trainloader, config, scheduler, disc_scheduler, scaler=None, scaler_disc=None, balancer=None, wandb_logger=None):
    """train one step function for paired channel training

    Args:
        epoch (int): current epoch
        optimizer (_type_) : generator optimizer
        optimizer_disc (_type_): discriminator optimizer
        model (_type_): generator model
        disc_model (_type_): discriminator model
        trainloader (_type_): train dataloader
        config (_type_): hydra config file
        scheduler (_type_): adjust generate model learning rate
        disc_scheduler (_type_): adjust discriminator model learning rate
        scaler (_type_): gradient scaler for mixed precision
        scaler_disc (_type_): gradient scaler for discriminator
        balancer (_type_): loss balancer
    """
    model.train()
    disc_model.train()
    data_length = len(trainloader)
    
    # Initialize variables to accumulate losses  
    accumulated_loss_g = 0.0
    accumulated_losses_g = defaultdict(float)
    accumulated_loss_w = 0.0
    accumulated_loss_disc = 0.0
    accumulated_loss_sim = 0.0
    accumulated_loss_div = 0.0

    for idx, (input_wav1, input_wav2) in enumerate(trainloader):
        # Process both channels
        input_wav1 = input_wav1.contiguous().cuda()  # [B, 1, T]
        input_wav2 = input_wav2.contiguous().cuda()  # [B, 1, T]
        
        optimizer.zero_grad()
        
        with autocast(enabled=config.common.amp):
            # Process first channel
            output1, loss_w1, _, embeddings1 = model(input_wav1, return_embeddings=True)
            logits_real1, fmap_real1 = disc_model(input_wav1)
            logits_fake1, fmap_fake1 = disc_model(output1)
            losses_g1 = total_loss(
                fmap_real1, 
                logits_fake1, 
                fmap_fake1, 
                input_wav1, 
                output1, 
                sample_rate=config.model.sample_rate,
            )
            
            # Process second channel
            output2, loss_w2, _, embeddings2 = model(input_wav2, return_embeddings=True)
            logits_real2, fmap_real2 = disc_model(input_wav2)
            logits_fake2, fmap_fake2 = disc_model(output2)
            losses_g2 = total_loss(
                fmap_real2, 
                logits_fake2, 
                fmap_fake2, 
                input_wav2, 
                output2, 
                sample_rate=config.model.sample_rate,
            )
            
            # Compute pairwise losses
            pairwise_losses_dict = pairwise_losses(
                embeddings1, 
                embeddings2, 
                sim_weight=config.pairwise.sim_weight,
                div_weight=config.pairwise.div_weight
            )
        
        # Combine individual losses
        if config.common.amp: 
            loss_g1 = 3*losses_g1['l_g'] + 3*losses_g1['l_feat'] + losses_g1['l_t']/10 + losses_g1['l_f'] + loss_w1
            loss_g2 = 3*losses_g2['l_g'] + 3*losses_g2['l_feat'] + losses_g2['l_t']/10 + losses_g2['l_f'] + loss_w2
            loss_g = loss_g1 + loss_g2 + pairwise_losses_dict['l_sim'] + pairwise_losses_dict['l_div']
            
            scaler.scale(loss_g).backward()  
            scaler.step(optimizer)  
            scaler.update()   
            scheduler.step()  
        else:
            if balancer is not None:
                # Apply balancer to individual losses
                balancer.backward(losses_g1, output1, retain_graph=True)
                balancer.backward(losses_g2, output2, retain_graph=True)
                loss_g1 = sum([l * balancer.weights[k] for k, l in losses_g1.items()])
                loss_g2 = sum([l * balancer.weights[k] for k, l in losses_g2.items()])
            else:
                loss_g1 = 3*losses_g1['l_g'] + 3*losses_g1['l_feat'] + losses_g1['l_t']/10 + losses_g1['l_f'] 
                loss_g2 = 3*losses_g2['l_g'] + 3*losses_g2['l_feat'] + losses_g2['l_t']/10 + losses_g2['l_f']
            
            # Combine all losses into a single backward pass
            combined_loss = loss_g1 + loss_g2 + loss_w1 + loss_w2 + pairwise_losses_dict['l_sim'] + pairwise_losses_dict['l_div']
            combined_loss.backward()
            optimizer.step()

        # Accumulate losses  
        if config.common.amp:
            accumulated_loss_g += loss_g.item()
        else:
            accumulated_loss_g += combined_loss.item()
        
        for k, l in losses_g1.items():
            accumulated_losses_g[k] += l.item()
        for k, l in losses_g2.items():
            accumulated_losses_g[k] += l.item()
        accumulated_loss_w += (loss_w1.item() + loss_w2.item())
        accumulated_loss_sim += pairwise_losses_dict['l_sim'].item()
        accumulated_loss_div += pairwise_losses_dict['l_div'].item()

        # only update discriminator with probability from paper (configure)
        optimizer_disc.zero_grad()
        train_discriminator = torch.BoolTensor([config.model.train_discriminator 
                               and epoch >= config.lr_scheduler.warmup_epoch 
                               and random.random() < 0.5]).cuda()

        if train_discriminator:
            with autocast(enabled=config.common.amp):
                # Discriminator loss for both channels
                logits_real1, _ = disc_model(input_wav1)
                logits_fake1, _ = disc_model(output1.detach())
                logits_real2, _ = disc_model(input_wav2)
                logits_fake2, _ = disc_model(output2.detach())
                
                loss_disc1 = disc_loss(logits_real1, logits_fake1)
                loss_disc2 = disc_loss(logits_real2, logits_fake2)
                loss_disc = loss_disc1 + loss_disc2
                
            if config.common.amp: 
                scaler_disc.scale(loss_disc).backward()
                scaler_disc.step(optimizer_disc)  
                scaler_disc.update()  
            else:
                loss_disc.backward() 
                optimizer_disc.step()

            # Accumulate discriminator loss  
            accumulated_loss_disc += loss_disc.item()
        
        scheduler.step()
        disc_scheduler.step()

        if idx % config.common.log_interval == 0 or idx == data_length - 1: 
            log_msg = (  
                f"Epoch {epoch} {idx+1}/{data_length}\t"
                f"Avg loss_G: {accumulated_loss_g / (idx + 1):.4f}\t"
                f"Avg loss_W: {accumulated_loss_w / (idx + 1):.4f}\t"
                f"Avg loss_sim: {accumulated_loss_sim / (idx + 1):.6e}\t"
                f"Avg loss_div: {accumulated_loss_div / (idx + 1):.6e}\t"
                f"lr_G: {optimizer.param_groups[0]['lr']:.6e}\t"
                f"lr_D: {optimizer_disc.param_groups[0]['lr']:.6e}\t"
            ) 
            
            # Weights & Biases logging
            if wandb_logger:
                log_dict = {
                    'epoch': epoch,
                    'step': (epoch-1) * len(trainloader) + idx,
                    'train/loss_g': accumulated_loss_g / (idx + 1),
                    'train/loss_w': accumulated_loss_w / (idx + 1),
                    'train/loss_sim': accumulated_loss_sim / (idx + 1),
                    'train/loss_div': accumulated_loss_div / (idx + 1),
                    'train/lr_g': optimizer.param_groups[0]['lr'],
                    'train/lr_d': optimizer_disc.param_groups[0]['lr'],
                }
                for k, l in accumulated_losses_g.items():
                    log_dict[f'train/{k}'] = l / (idx + 1)
                if config.model.train_discriminator and epoch >= config.lr_scheduler.warmup_epoch:
                    log_dict['train/loss_disc'] = accumulated_loss_disc / (idx + 1)
                    log_msg += f"loss_disc: {accumulated_loss_disc / (idx + 1) :.4f}"  
                wandb_logger.log(log_dict)
            
            logger.info(log_msg) 

@torch.no_grad()
def test_paired(epoch, model, disc_model, testloader, config, wandb_logger=None):
    model.eval()
    for idx, (input_wav1, input_wav2) in enumerate(testloader):
        input_wav1 = input_wav1.cuda()
        input_wav2 = input_wav2.cuda()

        # Process both channels
        output1, embeddings1 = model(input_wav1, return_embeddings=True)
        output2, embeddings2 = model(input_wav2, return_embeddings=True)
        
        logits_real1, fmap_real1 = disc_model(input_wav1)
        logits_fake1, fmap_fake1 = disc_model(output1)
        logits_real2, fmap_real2 = disc_model(input_wav2)
        logits_fake2, fmap_fake2 = disc_model(output2)
        
        loss_disc1 = disc_loss(logits_real1, logits_fake1)
        loss_disc2 = disc_loss(logits_real2, logits_fake2)
        loss_disc = loss_disc1 + loss_disc2
        
        losses_g1 = total_loss(fmap_real1, logits_fake1, fmap_fake1, input_wav1, output1) 
        losses_g2 = total_loss(fmap_real2, logits_fake2, fmap_fake2, input_wav2, output2)
        
        # Compute pairwise losses
        pairwise_losses_dict = pairwise_losses(
            embeddings1, 
            embeddings2, 
            sim_weight=config.pairwise.sim_weight,
            div_weight=config.pairwise.div_weight
        )

    log_msg = (f'| TEST | epoch: {epoch} | '
               f'loss_g: {sum([l.item() for l in losses_g1.values()]) + sum([l.item() for l in losses_g2.values()])} | '
               f'loss_disc: {loss_disc.item():.4f} | '
               f'loss_sim: {pairwise_losses_dict["l_sim"].item():.4f} | '
               f'loss_div: {pairwise_losses_dict["l_div"].item():.4f}') 
    
    # Weights & Biases logging
    if wandb_logger:
        test_log_dict = {
            'epoch': epoch,
            'test/loss_g': sum([l.item() for l in losses_g1.values()]) + sum([l.item() for l in losses_g2.values()]),
            'test/loss_disc': loss_disc.item(),
            'test/loss_sim': pairwise_losses_dict['l_sim'].item(),
            'test/loss_div': pairwise_losses_dict['l_div'].item(),
        }
        for k, l in losses_g1.items():
            test_log_dict[f'test/{k}_1'] = l.item()
        for k, l in losses_g2.items():
            test_log_dict[f'test/{k}_2'] = l.item()
        wandb_logger.log(test_log_dict)
    
    logger.info(log_msg)

    # save a sample reconstruction (not cropped) - use first channel
    input_wav1, input_wav2, _ = testloader.dataset.get()
    input_wav1 = input_wav1.cuda()
    input_wav2 = input_wav2.cuda()
    output1 = model(input_wav1.unsqueeze(0), return_embeddings=True)[0].squeeze(0)
    output2 = model(input_wav2.unsqueeze(0), return_embeddings=True)[0].squeeze(0)
    
    # Audio samples are logged to WandB - no need to save disk files
    
    # Log audio samples to wandb
    if wandb_logger:
        try:
            # Ensure audio tensors are properly formatted for wandb
            input_audio1 = input_wav1.cpu().squeeze()
            output_audio1 = output1.cpu().squeeze()
            input_audio2 = input_wav2.cpu().squeeze()
            output_audio2 = output2.cpu().squeeze()
            
            # Ensure audio is 1D for wandb
            if input_audio1.dim() > 1:
                input_audio1 = input_audio1.squeeze()
            if output_audio1.dim() > 1:
                output_audio1 = output_audio1.squeeze()
            if input_audio2.dim() > 1:
                input_audio2 = input_audio2.squeeze()
            if output_audio2.dim() > 1:
                output_audio2 = output_audio2.squeeze()
            
            wandb_logger.log({
                'audio/ground_truth_ch1': wandb.Audio(input_audio1.numpy(), sample_rate=config.model.sample_rate),
                'audio/reconstruction_ch1': wandb.Audio(output_audio1.numpy(), sample_rate=config.model.sample_rate),
                'audio/ground_truth_ch2': wandb.Audio(input_audio2.numpy(), sample_rate=config.model.sample_rate),
                'audio/reconstruction_ch2': wandb.Audio(output_audio2.numpy(), sample_rate=config.model.sample_rate),
            })
        except Exception as e:
            logger.warning(f"Failed to log audio to wandb: {e}")

def train_paired(config):
    """train main function for paired channel training."""
    # remove the logging handler "somebody" added
    logger.handlers.clear()

    # set logger
    file_handler = logging.FileHandler(f"{config.checkpoint.save_folder}/train_paired_encodec_bs{config.datasets.batch_size}_lr{config.optimization.lr}.log")
    formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
    file_handler.setFormatter(formatter)

    # print to screen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Initialize Weights & Biases
    wandb_logger = None
    if config.get('wandb', {}).get('enabled', True):  # Enable by default
        try:
            wandb.init(
                project=config.get('wandb', {}).get('project', 'paired-encodec-nq2'),
                name=config.get('wandb', {}).get('name', f'paired_encodec_bs{config.datasets.batch_size}_lr{config.optimization.lr}'),
                config=dict(config),
                dir=config.checkpoint.save_folder,
            )
            wandb_logger = wandb
            logger.info("✓ Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")
            wandb_logger = None

    # set seed
    if config.common.seed is not None:
        set_seed(config.common.seed)

    # set train dataset
    trainset = data.PairedChannelDataset(config=config, mode='train')
    testset = data.PairedChannelDataset(config=config, mode='test')
    
    # set encodec model and discriminator model
    model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, 
        model_norm=config.model.norm, 
        audio_normalize=config.model.audio_normalize,
        segment=eval(config.model.segment), 
        name=config.model.name,
        ratios=config.model.ratios,
        n_q=config.model.n_q,  # Use n_q from config
    )
    
    disc_model = MultiScaleSTFTDiscriminator(
        in_channels=config.model.channels,
        out_channels=config.model.channels,
        filters=config.model.filters,
        hop_lengths=config.model.disc_hop_lengths,
        win_lengths=config.model.disc_win_lengths,
        n_ffts=config.model.disc_n_ffts,
    )

    # log model, disc model parameters and train mode
    logger.info(model)
    logger.info(disc_model)
    logger.info(config)
    logger.info(f"Encodec Model Parameters: {count_parameters(model)} | Disc Model Parameters: {count_parameters(disc_model)}")
    logger.info(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")

    # resume training
    resume_epoch = 0
    if config.checkpoint.resume:
        # check the checkpoint_path
        assert config.checkpoint.checkpoint_path != '', "resume path is empty"
        assert config.checkpoint.disc_checkpoint_path != '', "disc resume path is empty"

        # Check if checkpoint files exist
        if not os.path.exists(config.checkpoint.checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {config.checkpoint.checkpoint_path}")
        if not os.path.exists(config.checkpoint.disc_checkpoint_path):
            raise FileNotFoundError(f"Discriminator checkpoint not found: {config.checkpoint.disc_checkpoint_path}")

        logger.info(f"Loading model checkpoint from: {config.checkpoint.checkpoint_path}")
        logger.info(f"Loading discriminator checkpoint from: {config.checkpoint.disc_checkpoint_path}")

        model_checkpoint = torch.load(config.checkpoint.checkpoint_path, map_location='cpu')
        disc_model_checkpoint = torch.load(config.checkpoint.disc_checkpoint_path, map_location='cpu')
        
        # Validate checkpoint structure
        if 'model_state_dict' not in model_checkpoint:
            raise KeyError("Model checkpoint missing 'model_state_dict' key")
        if 'model_state_dict' not in disc_model_checkpoint:
            raise KeyError("Discriminator checkpoint missing 'model_state_dict' key")
        if 'epoch' not in model_checkpoint:
            raise KeyError("Model checkpoint missing 'epoch' key")
        
        model.load_state_dict(model_checkpoint['model_state_dict'])
        disc_model.load_state_dict(disc_model_checkpoint['model_state_dict'])
        resume_epoch = model_checkpoint['epoch']
        
        if resume_epoch >= config.common.max_epoch:
            raise ValueError(f"resume epoch {resume_epoch} is larger than total epochs {config.common.max_epoch}")
        
        logger.info(f"✓ Successfully loaded checkpoints and resuming from epoch {resume_epoch}")
        logger.info(f"  - Model checkpoint epoch: {resume_epoch}")
        logger.info(f"  - Total epochs to train: {config.common.max_epoch}")
        logger.info(f"  - Epochs remaining: {config.common.max_epoch - resume_epoch}")

    model.cuda()
    disc_model.cuda()

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.datasets.batch_size,
        shuffle=True, 
        collate_fn=collate_fn_pairs,
        num_workers=config.datasets.num_workers,
        pin_memory=config.datasets.pin_memory)
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.datasets.batch_size,
        shuffle=False, 
        collate_fn=collate_fn_pairs,
        num_workers=config.datasets.num_workers,
        pin_memory=config.datasets.pin_memory)
    
    logger.info(f"There are {len(trainloader)} data pairs to train the Paired EnCodec")
    logger.info(f"There are {len(testloader)} data pairs to test the Paired EnCodec")

    # set optimizer and scheduler, warmup scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    disc_params = [p for p in disc_model.parameters() if p.requires_grad]
    optimizer = optim.Adam([{'params': params, 'lr': config.optimization.lr}], betas=(0.5, 0.9))
    optimizer_disc = optim.Adam([{'params':disc_params, 'lr': config.optimization.disc_lr}], betas=(0.5, 0.9))
    scheduler = WarmupCosineLrScheduler(optimizer, max_iter=config.common.max_epoch*len(trainloader), eta_ratio=0.1, warmup_iter=config.lr_scheduler.warmup_epoch*len(trainloader), warmup_ratio=1e-4)
    disc_scheduler = WarmupCosineLrScheduler(optimizer_disc, max_iter=config.common.max_epoch*len(trainloader), eta_ratio=0.1, warmup_iter=config.lr_scheduler.warmup_epoch*len(trainloader), warmup_ratio=1e-4)

    scaler = GradScaler() if config.common.amp else None
    scaler_disc = GradScaler() if config.common.amp else None  

    if config.checkpoint.resume and 'scheduler_state_dict' in model_checkpoint.keys() and 'scheduler_state_dict' in disc_model_checkpoint.keys(): 
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(model_checkpoint['scheduler_state_dict'])
        optimizer_disc.load_state_dict(disc_model_checkpoint['optimizer_state_dict'])
        disc_scheduler.load_state_dict(disc_model_checkpoint['scheduler_state_dict'])
        logger.info(f"load optimizer and disc_optimizer state_dict from {resume_epoch}")

    
    start_epoch = max(1, resume_epoch+1) # start epoch is 1 if not resume
    # instantiate loss balancer
    balancer = Balancer(dict(config.balancer.weights)) if hasattr(config, 'balancer') else None
    if balancer:
        logger.info(f'Loss balancer with weights {balancer.weights} instantiated')
    
    test_paired(0, model, disc_model, testloader, config, wandb_logger)
    
    for epoch in range(start_epoch, config.common.max_epoch+1):
        train_one_step_paired(
            epoch, optimizer, optimizer_disc, 
            model, disc_model, trainloader, config,
            scheduler, disc_scheduler, scaler, scaler_disc, balancer, wandb_logger)
        
        if epoch % config.common.test_interval == 0:
            test_paired(epoch, model, disc_model, testloader, config, wandb_logger)
        
        # save checkpoint and epoch
        if epoch % config.common.save_interval == 0:
            if not os.path.exists(config.checkpoint.save_folder):
                os.makedirs(config.checkpoint.save_folder)
            model_path = f'{config.checkpoint.save_location}epoch{epoch}_lr{config.optimization.lr}.pt'
            disc_path = f'{config.checkpoint.save_location}epoch{epoch}_disc_lr{config.optimization.lr}.pt'
            
            save_master_checkpoint(epoch, model, optimizer, scheduler, model_path)  
            save_master_checkpoint(epoch, disc_model, optimizer_disc, disc_scheduler, disc_path)
            
            # Log model artifacts to wandb
            if wandb_logger:
                artifact = wandb.Artifact(f'paired_model_epoch_{epoch}', type='model')
                artifact.add_file(model_path)
                artifact.add_file(disc_path)
                wandb_logger.log_artifact(artifact)
    
    # Finish wandb run
    if wandb_logger:
        wandb.finish() 

@hydra.main(config_path='config', config_name='config_paired_nq2')
def main(config):
    # disable cudnn
    torch.backends.cudnn.enabled = False
    
    # Memory optimization
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if not os.path.exists(config.checkpoint.save_folder):
        os.makedirs(config.checkpoint.save_folder)
    
    train_paired(config)

if __name__ == '__main__':
    main()
