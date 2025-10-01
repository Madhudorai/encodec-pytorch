# Training Monitoring & Logging Setup

## 📊 **Current Logging Setup**

### **1. Console Output**
- **Real-time progress**: Epoch, step, losses, learning rates
- **Format**: `Epoch 1 100/5000	Avg loss_G: 0.1234	Avg loss_W: 0.0567	lr_G: 3.000000e-04	lr_D: 3.000000e-04`
- **Frequency**: Every `log_interval` steps (default: 100)

### **2. File Logging**
- **Location**: `./checkpoints_mono_nq2/train_encodec_bs8_lr3e-4.log`
- **Content**: Complete training log with timestamps
- **Format**: `2024-01-15 10:30:45: INFO: [train_single_gpu.py: 155]: Epoch 1 100/5000...`

### **3. TensorBoard Logging**
- **Location**: `./checkpoints_mono_nq2/runs/`
- **Metrics**: All losses, learning rates, audio samples
- **View**: `tensorboard --logdir=./checkpoints_mono_nq2/runs/`

### **4. Weights & Biases (wandb) - NEW! 🎉**
- **Project**: `mono-encodec-nq2`
- **Run Name**: `mono_encodec_bs8_lr3e-4`
- **Features**:
  - Real-time loss plots
  - Audio sample playback
  - Model artifact tracking
  - Hyperparameter tracking
  - System metrics (GPU usage, memory)

## 🎯 **What Gets Logged**

### **Training Metrics (Every 100 steps)**
```
train/loss_g          # Generator loss
train/loss_w          # Quantizer loss  
train/l_t            # Time domain loss
train/l_f            # Frequency domain loss
train/l_g            # Generator adversarial loss
train/l_feat         # Feature matching loss
train/loss_disc      # Discriminator loss (when enabled)
train/lr_g           # Generator learning rate
train/lr_d           # Discriminator learning rate
```

### **Validation Metrics (Every 5 epochs)**
```
test/loss_g          # Test generator loss
test/loss_disc       # Test discriminator loss
test/l_t, test/l_f, test/l_g, test/l_feat  # Individual test losses
```

### **Audio Samples (Every 5 epochs)**
- **Ground Truth**: Original 1-second audio sample
- **Reconstruction**: Model output audio sample
- **Location**: `./checkpoints_mono_nq2/GT_epoch{X}.wav` and `Reconstruction_epoch{X}.wav`
- **wandb**: Audio samples logged for instant playback in browser

## 💾 **Model Checkpoints**

### **Save Locations**
- **Main Model**: `./checkpoints_mono_nq2/bs8_cut24000_length0_epoch{X}_lr3e-4.pt`
- **Discriminator**: `./checkpoints_mono_nq2/bs8_cut24000_length0_epoch{X}_disc_lr3e-4.pt`
- **Frequency**: Every 2 epochs (configurable)

### **Checkpoint Contents**
```python
{
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': current_loss,
}
```

### **wandb Artifacts**
- Model checkpoints automatically uploaded to wandb
- Version controlled and easily downloadable
- Accessible from wandb dashboard

## 🚀 **How to Monitor Training**

### **1. Console Monitoring**
```bash
# Just run training and watch console
python train_single_gpu.py
```

### **2. TensorBoard**
```bash
# In another terminal
tensorboard --logdir=./checkpoints_mono_nq2/runs/
# Open http://localhost:6006
```

### **3. Weights & Biases**
```bash
# First time: login to wandb
wandb login

# Run training (wandb auto-initializes)
python train_single_gpu.py

# View in browser at https://wandb.ai
```

### **4. File Monitoring**
```bash
# Watch log file in real-time
tail -f ./checkpoints_mono_nq2/train_encodec_bs8_lr3e-4.log

# Check latest checkpoints
ls -la ./checkpoints_mono_nq2/*.pt
```

## ⚙️ **Configuration Options**

### **wandb Settings** (in `config/config_mono_nq2.yaml`)
```yaml
wandb:
  enabled: true                    # Enable/disable wandb
  project: "mono-encodec-nq2"      # Project name
  name: "mono_encodec_bs8_lr3e-4"  # Run name
```

### **Logging Frequency** (in `config/config_mono_nq2.yaml`)
```yaml
common:
  log_interval: 100    # Console/file logging every N steps
  test_interval: 5     # Validation every N epochs  
  save_interval: 2     # Checkpoint every N epochs
```

## 📈 **Expected Training Output**

### **Console Example**
```
2024-01-15 10:30:45: INFO: [train_single_gpu.py: 155]: Epoch 1 100/5000	Avg loss_G: 0.1234	Avg loss_W: 0.0567	lr_G: 3.000000e-04	lr_D: 3.000000e-04	loss_disc: 0.0890
2024-01-15 10:31:12: INFO: [train_single_gpu.py: 155]: Epoch 1 200/5000	Avg loss_G: 0.1156	Avg loss_W: 0.0523	lr_G: 3.000000e-04	lr_D: 3.000000e-04	loss_disc: 0.0823
...
2024-01-15 10:45:30: INFO: [train_single_gpu.py: 188]: | TEST | epoch: 5 | loss_g: 0.0987 | loss_disc: 0.0756
```

### **wandb Dashboard**
- **Charts**: Loss curves, learning rate schedules, audio waveforms
- **Audio**: Play ground truth vs reconstruction samples
- **System**: GPU utilization, memory usage
- **Artifacts**: Download model checkpoints

## 🔧 **Troubleshooting**

### **wandb Issues**
```bash
# If wandb fails to initialize
pip install --upgrade wandb
wandb login

# Disable wandb temporarily
# Edit config/config_mono_nq2.yaml: wandb.enabled: false
```

### **TensorBoard Issues**
```bash
# If TensorBoard doesn't start
pip install --upgrade tensorboard

# Clear old logs
rm -rf ./checkpoints_mono_nq2/runs/*
```

### **Disk Space**
```bash
# Check checkpoint sizes
du -sh ./checkpoints_mono_nq2/

# Clean old checkpoints (keep last 5)
ls -t ./checkpoints_mono_nq2/*.pt | tail -n +11 | xargs rm
```

## 🎉 **Benefits of This Setup**

✅ **Real-time monitoring** - See progress as it happens  
✅ **Multiple interfaces** - Console, TensorBoard, wandb  
✅ **Audio quality tracking** - Listen to reconstructions  
✅ **Model versioning** - Automatic checkpoint management  
✅ **Experiment tracking** - Compare different runs  
✅ **Remote monitoring** - Access wandb from anywhere  
✅ **Reproducibility** - All configs and code tracked  

The training will now provide comprehensive monitoring with both local (console, files, TensorBoard) and cloud (wandb) logging options!
