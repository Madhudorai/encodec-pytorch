# encodec-pytorch
>[!IMPORTANT]
>This is an unofficial implementation of the paper [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) in PyTorch.
>
>The LibriTTS960h 24khz encodec checkpoint and disc checkpoint is release in https://huggingface.co/zkniu/encodec-pytorch/tree/main
>
>I hope we can get together to do something meaningful and rebuild encodec in this repo.

## Introduction
This repository is based on [encodec](https://github.com/facebookresearch/encodec) and [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer).

Based on the [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer), new changes:
- support Weights & Biases (wandb) for monitoring the training process.
- support multi-dataset training with automatic dataset mixing.
- support bandwidth-specific metrics with confidence intervals.
- support loss balancer, thanks [@leoauri](https://github.com/leoauri). in https://github.com/ZhikangNiu/encodec-pytorch/pull/22.
- You can find all the training scripts in scripts folder

## Enviroments
The code is tested on the following environment:
- Python 3.9
- PyTorch 2.0.0 / PyTorch 1.13
- GeForce RTX 3090 x 4 / V100-16G x 8 / A40 x 3 / A100 x 1

In order to you can run the code, you can install the environment by the help of requirements.txt.

## Usage
### Training
#### 1. Prepare dataset
This repository supports multi-dataset training with automatic dataset mixing. You can use multiple datasets like:
- **Jamendo**: Music dataset for diverse audio content
- **Common Voice**: Speech dataset for voice content  
- **FSD50K**: Environmental sounds dataset
- **DNS Challenge 4**: Clean speech dataset

Use the `datasets/generate_dataset_csvs.py` script to generate train/validation/test CSV files for your datasets:
```bash
# Generate CSV files with train/val/test split
python datasets/generate_dataset_csvs.py -i /path/to/your/dataset --three_way_split --train_ratio 0.995 --val_ratio 0.0025 --test_ratio 0.0025
```

You can check the `datasets/generate_dataset_csvs.py` and `multi_dataset.py` to understand how to prepare your own dataset.
Also you can use `ln -s` to link the dataset to the `datasets` folder.
#### [Optional] Docker image
I provide a dockerfile to build a docker image with all the necessary dependencies.
1. Building the image
```shell
docker build -t encodec:v1 .
```
2. Using the image
```shell
# CPU running
docker run encodec:v1 <command> # you can add some parameters, such as -tid
# GPU running
docker run --gpus=all encodec:v1 <command>
```
#### 2. Train
You can use the following command to train the model with multi-dataset support:
```bash
# Multi-dataset training (recommended)
./run_multi_dataset_training.sh
```

Or run directly with Python:
```bash
python train_multi_dataset.py \
    --config-name=config_multi_dataset \
    common.max_epoch=400 \
    datasets.batch_size=16 \
    datasets.fixed_length=32000 \
    model.sample_rate=24000 \
    model.channels=1 \
    model.target_bandwidths=[1.5,3.0,6.0,12.0,24.0] \
    wandb.enabled=true \
    wandb.project=multi-dataset-encodec \
    wandb.name=multi_dataset_bs16_epochs400_24khz_mono
```
#### 3. Test
```bash
python test.py
```
Runs checkpoint model on demo ground truth audio files. logs spectrograms, reconstructed audio for different bandwidths to wandb 

**Key Features:**
- **Multi-dataset training**: Automatically mixes multiple datasets (Jamendo, Common Voice, FSD50K, DNS Challenge 4)
- **Bandwidth-specific metrics**: Tracks SI-SNR performance for each compression rate (1.5, 3.0, 6.0, 12.0, 24.0 kbps)
- **Weights & Biases integration**: Real-time monitoring with confidence intervals
- **Automatic dataset balancing**: Intelligent mixing of different audio types

**Configuration:**
- Edit `config/config_multi_dataset.yaml` to customize dataset paths and training parameters
- The model supports multiple bandwidths simultaneously during training
- Validation includes comprehensive metrics per bandwidth with statistical confidence intervals

**Notes:**
1. The multi-dataset approach provides better generalization across different audio types
2. Training includes automatic dataset mixing for robust performance
3. Monitor training progress via Weights & Biases dashboard
4. Checkpoints are saved with bandwidth-specific naming conventions
5. **The code is actively maintained for multi-dataset training scenarios**

#### Legacy Training (Single Dataset)
For single-dataset training, you can still use the legacy approach:
```bash
python train_single_gpu.py \
    --config-name=config \
    datasets.train_csv_path=YOUR_TRAIN_DATA.csv \
    common.max_epoch=100 \
    datasets.batch_size=8 \
    optimization.lr=5e-5
```

#### Slurm
Usage will depend on your cluster setup, but see `scripts/train.sbatch` for an example. This uses a container with the dependencies installed. Run `sbatch scripts/train.sbatch` from the repository root to use.

### Test
I have add a shell script to compress and decompress the audio by different bandwidth, you can use the `compression.sh` to test your model. 

The script can be used as follows:
```shell
sh compression.sh INPUT_WAV_FILE [MODEL_NAME] [CHECKPOINT]
```
- INPUT_WAV_FILE is the wav file you want to test
- MODEL_NAME is the model name, default is `encodec_24khz`,support `encodec_48khz`, `my_encodec`,`encodec_bw`
- CHECKPOINT is the checkpoint path, when your MODEL_NAME is `my_encodec`,you can point out the checkpoint

if you want to test the model at a specific bandwidth, you can use the following command:
```shell
python main.py -r -b [bandwidth] -f [INPUT_FILE] [OUTPUT_WAV_FILE] -m [MODEL_NAME] -c [CHECKPOINT]
```
main.py from the [encodec](https://github.com/facebookresearch/encodec) , you can use the `-h` to check the help information.

## Acknowledgement
Thanks to the following repositories:
- [encodec](https://github.com/facebookresearch/encodec)
- [EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer)
- [melgan-neurips](https://github.com/descriptinc/melgan-neurips): audio_to_mel.py

## LICENSE
The code is same as [encodec](https://github.com/facebookresearch/encodec) LICENSE.

