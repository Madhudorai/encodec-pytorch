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

import multi_channel_dataset as data
from multi_channel_dataset import collate_fn
from losses import disc_loss, total_loss
from model import EncodecModel
from msstftd import MultiScaleSTFTDiscriminator
from scheduler import WarmupCosineLrScheduler
from utils import (count_parameters, save_master_checkpoint, set_seed)
from balancer import Balancer

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define train one step function
def train_one_step(epoch, optimizer, optimizer_disc, model, disc_model, trainloader, config, scheduler, disc_scheduler, scaler=None, scaler_disc=None, balancer=None, wandb_logger=None):
    """train one step function

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

    for idx, input_wav in enumerate(trainloader):
        # warmup learning rate, warmup_epoch is defined in config file,default is 5
        input_wav = input_wav.contiguous().cuda() #[B, 1, T]: eg. [2, 1, 203760]
        optimizer.zero_grad()
        
        with autocast(enabled=config.common.amp):
            output, loss_w, _ = model(input_wav) #output: [B, 1, T]: eg. [2, 1, 203760] | loss_w: [1] 
            logits_real, fmap_real = disc_model(input_wav)
            logits_fake, fmap_fake = disc_model(output)
            losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                input_wav, 
                output, 
                sample_rate=config.model.sample_rate,
            ) 
        
        if config.common.amp: 
            loss_g = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f']  + loss_w
            scaler.scale(loss_g).backward()  
            scaler.step(optimizer)  
            scaler.update()   
            scheduler.step()  
        else:
            if balancer is not None:
                balancer.backward(losses_g, output, retain_graph=True)
                loss_g = sum([l * balancer.weights[k] for k, l in losses_g.items()])
            else:
                loss_g = 3*losses_g['l_g'] + 3*losses_g['l_feat'] + losses_g['l_t']/10 + losses_g['l_f'] 
                loss_g.backward()
            loss_w.backward()
            optimizer.step()

        # Accumulate losses  
        accumulated_loss_g += loss_g.item()
        for k, l in losses_g.items():
            accumulated_losses_g[k] += l.item()
        accumulated_loss_w += loss_w.item()

        # only update discriminator with probability from paper (configure)
        optimizer_disc.zero_grad()
        train_discriminator = torch.BoolTensor([config.model.train_discriminator 
                               and epoch >= config.lr_scheduler.warmup_epoch 
                               and random.random() < 0.5]).cuda()

        if train_discriminator:
            with autocast(enabled=config.common.amp):
                logits_real, _ = disc_model(input_wav)
                logits_fake, _ = disc_model(output.detach()) # detach to avoid backpropagation to model
                loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
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
                f"Epoch {epoch} {idx+1}/{data_length}\tAvg loss_G: {accumulated_loss_g / (idx + 1):.4f}\tAvg loss_W: {accumulated_loss_w / (idx + 1):.4f}\tlr_G: {optimizer.param_groups[0]['lr']:.6e}\tlr_D: {optimizer_disc.param_groups[0]['lr']:.6e}\t"  
            ) 
            
            
            # Weights & Biases logging
            if wandb_logger:
                log_dict = {
                    'epoch': epoch,
                    'step': (epoch-1) * len(trainloader) + idx,
                    'train/loss_g': accumulated_loss_g / (idx + 1),
                    'train/loss_w': accumulated_loss_w / (idx + 1),
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
def test(epoch, model, disc_model, testloader, config, wandb_logger=None):
    model.eval()
    for idx, input_wav in enumerate(testloader):
        input_wav = input_wav.cuda()

        output = model(input_wav)
        logits_real, fmap_real = disc_model(input_wav)
        logits_fake, fmap_fake = disc_model(output)
        loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss
        losses_g = total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output) 

    log_msg = (f'| TEST | epoch: {epoch} | loss_g: {sum([l.item() for l in losses_g.values()])} | loss_disc: {loss_disc.item():.4f}') 
    
    
    # Weights & Biases logging
    if wandb_logger:
        test_log_dict = {
            'epoch': epoch,
            'test/loss_g': sum([l.item() for l in losses_g.values()]),
            'test/loss_disc': loss_disc.item(),
        }
        for k, l in losses_g.items():
            test_log_dict[f'test/{k}'] = l.item()
        wandb_logger.log(test_log_dict)
    
    logger.info(log_msg)

    # save a sample reconstruction (not cropped)
    input_wav, _ = testloader.dataset.get()
    input_wav = input_wav.cuda()
    output = model(input_wav.unsqueeze(0)).squeeze(0)
    
    # Audio samples are logged to WandB - no need to save disk files
    
    # Log audio samples to wandb
    if wandb_logger:
        try:
            # Ensure audio tensors are properly formatted for wandb
            input_audio = input_wav.cpu().squeeze()  # Remove batch and channel dimensions if present
            output_audio = output.cpu().squeeze()    # Remove batch and channel dimensions if present
            
            # Ensure audio is 1D for wandb
            if input_audio.dim() > 1:
                input_audio = input_audio.squeeze()
            if output_audio.dim() > 1:
                output_audio = output_audio.squeeze()
            
            wandb_logger.log({
                'audio/ground_truth': wandb.Audio(input_audio.numpy(), sample_rate=config.model.sample_rate),
                'audio/reconstruction': wandb.Audio(output_audio.numpy(), sample_rate=config.model.sample_rate),
            })
        except Exception as e:
            logger.warning(f"Failed to log audio to wandb: {e}")

def train(config):
    """train main function."""
    # remove the logging handler "somebody" added
    logger.handlers.clear()

    # set logger
    file_handler = logging.FileHandler(f"{config.checkpoint.save_folder}/train_encodec_bs{config.datasets.batch_size}_lr{config.optimization.lr}.log")
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
                project=config.get('wandb', {}).get('project', 'mono-encodec-nq2'),
                name=config.get('wandb', {}).get('name', f'mono_encodec_bs{config.datasets.batch_size}_lr{config.optimization.lr}'),
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
    trainset = data.MultiChannelAudioDataset(config=config, mode='train')
    testset = data.MultiChannelAudioDataset(config=config, mode='test')
    
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
        collate_fn=collate_fn,
        num_workers=config.datasets.num_workers,
        pin_memory=config.datasets.pin_memory)
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.datasets.batch_size,
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=config.datasets.num_workers,
        pin_memory=config.datasets.pin_memory)
    
    logger.info(f"There are {len(trainloader)} data to train the EnCodec")
    logger.info(f"There are {len(testloader)} data to test the EnCodec")

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
    
    test(0, model, disc_model, testloader, config, wandb_logger)
    
    for epoch in range(start_epoch, config.common.max_epoch+1):
        train_one_step(
            epoch, optimizer, optimizer_disc, 
            model, disc_model, trainloader, config,
            scheduler, disc_scheduler, scaler, scaler_disc, balancer, wandb_logger)
        
        if epoch % config.common.test_interval == 0:
            test(epoch, model, disc_model, testloader, config, wandb_logger)
        
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
                artifact = wandb.Artifact(f'model_epoch_{epoch}', type='model')
                artifact.add_file(model_path)
                artifact.add_file(disc_path)
                wandb_logger.log_artifact(artifact)
    
    # Finish wandb run
    if wandb_logger:
        wandb.finish() 

@hydra.main(config_path='config', config_name='config_mono_nq2')
def main(config):
    # disable cudnn
    torch.backends.cudnn.enabled = False
    
    # Memory optimization
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if not os.path.exists(config.checkpoint.save_folder):
        os.makedirs(config.checkpoint.save_folder)
    
    train(config)

if __name__ == '__main__':
    main()
