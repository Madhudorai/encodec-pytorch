# Training Monitoring & Logging Setup

## 📊 **Current Logging Setup**

### **1. Console Output**
- **Real-time progress**: Epoch, losses, learning rates
- **Format**: `| TRAIN | epoch: 1 | loss_g: 0.1234 | loss_w: 0.0567 | lr_G: 3.000000e-04 | lr_D: 3.000000e-04`
- **Frequency**: Every epoch

### **2. File Logging**
- **Location**: `./checkpoints_multi_dataset/train_multi_dataset_bs16_lr3e-4.log`
- **Content**: Complete training log with timestamps
- **Format**: `2024-01-15 10:30:45: INFO: [train_multi_dataset.py: 139]: | TRAIN | epoch: 1...`

### **3. Weights & Biases (wandb) - Primary Monitoring Tool 🎉**
- **Project**: `multi-dataset-encodec`
- **Run Name**: `multi_dataset_bs64_epochs300_24khz_mono`
- **Features**:
  - Real-time loss plots
  - Model artifact tracking
  - Hyperparameter tracking
  - Bandwidth-specific metrics with confidence intervals
  - SI-SNR metrics per bandwidth

## 🎯 **What Gets Logged**

### **Training Metrics (Every epoch)**
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

### **Validation Metrics (Every validation epoch)**
```
val/loss_g           # Validation generator loss
val/loss_disc        # Validation discriminator loss
val/si_snr           # Overall SI-SNR metric
val/si_snr_bw_1.5    # SI-SNR at 1.5 kbps bandwidth
val/si_snr_bw_3.0    # SI-SNR at 3.0 kbps bandwidth
val/si_snr_bw_6.0    # SI-SNR at 6.0 kbps bandwidth
val/si_snr_bw_12.0   # SI-SNR at 12.0 kbps bandwidth
val/si_snr_bw_24.0   # SI-SNR at 24.0 kbps bandwidth
```

### **Bandwidth-Specific Metrics**
- **SI-SNR with Confidence Intervals**: Each bandwidth gets individual SI-SNR metrics with 95% confidence intervals
- **Sample Counts**: Number of samples evaluated per bandwidth
- **Console Output**: Detailed bandwidth-specific logging with confidence intervals

## 💾 **Model Checkpoints**

### **Save Locations**
- **Main Model**: `./checkpoints_multi_dataset/bs16_cut24000_length32000_epoch{X}_lr3e-4.pt`
- **Discriminator**: `./checkpoints_multi_dataset/bs16_cut24000_length32000_epoch{X}_disc_lr3e-4.pt`
- **Frequency**: Every epoch (configurable via `save_interval`)

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
# Run training and watch console output
./run_multi_dataset_training.sh
```

### **2. Weights & Biases (Primary Method)**
```bash
# First time: login to wandb
wandb login

# Run training (wandb auto-initializes)
./run_multi_dataset_training.sh

# View in browser at https://wandb.ai
```

### **3. File Monitoring**
```bash
# Watch log file in real-time
tail -f ./checkpoints_multi_dataset/train_multi_dataset_bs16_lr3e-4.log

# Check latest checkpoints
ls -la ./checkpoints_multi_dataset/*.pt
```

## ⚙️ **Configuration Options**

### **wandb Settings** (in `config/config_multi_dataset.yaml`)
```yaml
wandb:
  enabled: true                    # Enable/disable wandb
  project: "multi-dataset-encodec" # Project name
  name: "multi_dataset_bs64_epochs300_24khz_mono"  # Run name
```

### **Logging Frequency** (in `config/config_multi_dataset.yaml`)
```yaml
common:
  val_interval: 1     # Validation every N epochs  
  save_interval: 1    # Checkpoint every N epochs
```

## 📈 **Expected Training Output**

### **Console Example**
```
2024-01-15 10:30:45: INFO: [train_multi_dataset.py: 139]: | TRAIN | epoch: 1 | loss_g: 0.1234 | loss_w: 0.0567 | lr_G: 3.000000e-04 | lr_D: 3.000000e-04 | loss_disc: 0.0890
2024-01-15 10:31:12: INFO: [train_multi_dataset.py: 214]: | VAL  | epoch: 1 | loss_g: 0.1156 | loss_disc: 0.0823 | SI-SNR: 12.34 dB
2024-01-15 10:31:12: INFO: [train_multi_dataset.py: 224]:   Bandwidth 1.5 kbps (n=100):
2024-01-15 10:31:12: INFO: [train_multi_dataset.py: 226]:     SI-SNR: 8.45±0.23 dB
2024-01-15 10:31:12: INFO: [train_multi_dataset.py: 224]:   Bandwidth 3.0 kbps (n=100):
2024-01-15 10:31:12: INFO: [train_multi_dataset.py: 226]:     SI-SNR: 12.34±0.18 dB
```

### **wandb Dashboard**
- **Charts**: Loss curves, learning rate schedules, SI-SNR per bandwidth
- **System**: GPU utilization, memory usage
- **Artifacts**: Download model checkpoints
- **Bandwidth Analysis**: Individual SI-SNR metrics with confidence intervals

## 🔧 **Troubleshooting**

### **wandb Issues**
```bash
# If wandb fails to initialize
pip install --upgrade wandb
wandb login

# Disable wandb temporarily
# Edit config/config_multi_dataset.yaml: wandb.enabled: false
```

### **Disk Space**
```bash
# Check checkpoint sizes
du -sh ./checkpoints_multi_dataset/

# Clean old checkpoints (keep last 5)
ls -t ./checkpoints_multi_dataset/*.pt | tail -n +11 | xargs rm
```

## 🎉 **Benefits of This Setup**

✅ **Real-time monitoring** - See progress as it happens  
✅ **Multiple interfaces** - Console, file logging, wandb  
✅ **Bandwidth-specific metrics** - Track performance per compression rate  
✅ **Confidence intervals** - Statistical rigor in validation metrics  
✅ **Model versioning** - Automatic checkpoint management  
✅ **Experiment tracking** - Compare different runs  
✅ **Remote monitoring** - Access wandb from anywhere  
✅ **Reproducibility** - All configs and code tracked  

The training provides comprehensive monitoring with both local (console, files) and cloud (wandb) logging options, with special focus on multi-bandwidth performance analysis!
