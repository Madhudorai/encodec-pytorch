#!/usr/bin/env python3
"""
Prepare spatial audio datasets using data_utils processing.

This script downloads, processes, and prepares spatial audio datasets:
- STARSS: Masks overlapping events, proper train/test splits
- DCASE: Simple move to organized directories
- LOCATA: Converts to DCASE format, extracts Eigenmike, augments
- MARCO: Extracts Eigenmike files, resamples to 24kHz, keeps 32 channels, augments

After processing, generates CSV files for training.

Usage:
    python datasets/prepare_all_data.py [--base_path /scratch/spatial_audio_data]
"""

import os
import sys
from tqdm import tqdm 
from glob import glob
import scipy.io.wavfile as wav
from scipy import signal
import numpy as np
import shutil
import csv
import pickle
import pandas as pd
from pathlib import Path

# Add data_utils to path
data_utils_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_utils')
sys.path.insert(0, data_utils_path)
from starss_utils import *
from locata_utils import Locata2DecaseFormat, mic_util
from marco_utils import file_dist_dict, marco_aug_data, marco_test_files, marco_val_files

import warnings
warnings.filterwarnings("ignore")

# Base paths - can be overridden via environment variable or modified here
BASE_DATA_PATH = os.environ.get('SPATIAL_DATA_BASE_PATH', '/scratch/spatial_audio_data')
DOWNLOAD_PATH = os.path.join(BASE_DATA_PATH, 'download')
INPUT_BASE_PATH = os.path.join(BASE_DATA_PATH, 'input')
METADATA_PATH = os.path.join(INPUT_BASE_PATH, 'metadata_dev')
MIC_PATH = os.path.join(INPUT_BASE_PATH, 'mic_dev')
CSV_PATH = os.path.join(BASE_DATA_PATH, 'csvs')

def get_links(dataset):
    """Get download links for each dataset."""
    links = {}
    if dataset == 'starss':
        links = {
            'metadata': 'https://zenodo.org/records/7880637/files/metadata_dev.zip',
            'micdata': 'https://zenodo.org/records/7880637/files/mic_dev.zip'
        }
    elif dataset == 'dcase':
        links = {
            'all_data': 'https://huggingface.co/datasets/sakshamsingh1/sound_distance/resolve/main/dcase.zip'
        }
    elif dataset == 'locata':
        links = {
            'dev': 'https://zenodo.org/records/3630471/files/dev.zip',
            'eval': 'https://zenodo.org/records/3630471/files/eval.zip'
        }
    elif dataset == 'marco':
        links = {
            'organ': 'https://zenodo.org/records/3477602/files/06%203D-MARCo%20Samples_Organ.zip',
            'paino_1': 'https://zenodo.org/records/3477602/files/07%203D-MARCo%20Samples_Piano%20solo%201.zip',
            'paino_2': 'https://zenodo.org/records/3477602/files/08%203D-MARCo%20Samples_Piano%20solo%202.zip',
            'acapella': 'https://zenodo.org/records/3477602/files/09%203D-MARCo%20Samples_Acappella.zip',
            'quartet': 'https://zenodo.org/records/3477602/files/04%203D-MARCo%20Samples_Quartet.zip',
        }
    return links

def get_cmd(link, save_dir):
    """Get wget command for downloading."""
    cmd = f'wget {link} -P {save_dir}'
    return cmd

def aug_data(meta_dir, mic_dir, aug_folds):
    """Channel permutation augmentation (from data_utils)."""
    aug_list = {}
    aug_list['aug1'] = [2, 4, 1, 3]
    aug_list['aug2'] = [4, 2, 3, 1]
    aug_list['aug3'] = [1, 2, 3, 4]
    aug_list['aug4'] = [2, 1, 4, 3]
    aug_list['aug5'] = [3, 1, 4, 2]
    aug_list['aug6'] = [1, 3, 2, 4]
    aug_list['aug7'] = [4, 3, 2, 1]
    aug_list['aug8'] = [3, 4, 1, 2]

    for file_path in tqdm(glob(os.path.join(mic_dir, '*.wav'))):
        mic_file = os.path.basename(file_path)
        meta_file = mic_file.replace('.wav', '.csv')

        if int(mic_file.split('_')[0][4:]) in aug_folds:
            fs, audio = wav.read(file_path)
            for aug in aug_list:
                chan_seq = aug_list[aug]
                chan_seq = [i - 1 for i in chan_seq]
                audio_out = audio[:, chan_seq]
                new_file = os.path.basename(file_path).replace('.wav', '_' + aug + '.wav')
                new_file_path = os.path.join(mic_dir, new_file)
                wav.write(new_file_path, fs, audio_out)

                # copy the metadata file
                meta_file = os.path.basename(file_path).replace('.wav', '.csv')
                meta_file_path = os.path.join(meta_dir, meta_file)
                new_meta_file_path = os.path.join(meta_dir, new_file.replace('.wav', '.csv'))
                os.system('cp ' + meta_file_path + ' ' + new_meta_file_path)
            os.remove(os.path.join(meta_dir, meta_file))
            os.remove(os.path.join(mic_dir, mic_file))

############################# STARSS #############################
def download_starss(save_path):
    print('Downloading STARSS dataset')
    links = get_links('starss')
    for link_type, link in links.items():
        print(f'Downloading {link_type}..........')
        wget_command = get_cmd(link, save_path)
        os.system(wget_command)
    
def unzip_starss(save_path):
    print('Unzipping STARSS dataset')
    os.system(f'unzip -q {save_path}/metadata_dev.zip -d {save_path}')
    os.system(f'unzip -q {save_path}/mic_dev.zip -d {save_path}')
    os.system(f'rm {save_path}/metadata_dev.zip')
    os.system(f'rm {save_path}/mic_dev.zip')

def mask_single_event():
    mic_dir = os.path.join(DOWNLOAD_PATH,'mic_dev')
    meta_dir = os.path.join(DOWNLOAD_PATH,'metadata_dev')

    save_meta_dir = os.path.join(METADATA_PATH, 'starss')
    save_mic_dir = os.path.join(MIC_PATH, 'starss')
    
    os.makedirs(save_meta_dir, exist_ok=True)
    os.makedirs(save_mic_dir, exist_ok=True)

    for fold in ['train','test']:
        print(f'Processing {fold} ....')

        mic_files = glob(f'{mic_dir}/*{fold}*/*.wav')
        mic_files = sorted(mic_files)
        meta_files = glob(f'{meta_dir}/*{fold}*/*.csv')
        meta_files = sorted(meta_files)

        for i, meta_file in tqdm(enumerate(meta_files), desc=f'Processing {fold} files'):
            meta_data = load_file(meta_file)
            mic_file = mic_files[i]

            #sanity check
            assert os.path.basename(meta_file).split('.')[0] == os.path.basename(mic_file).split('.')[0]

            ov_list = get_cons_ov_list(meta_data)
            zero_list = get_cons_zero_list(meta_data)

            fs, mic_data = wav.read(mic_file)
            mic_data = mic_data.T

            con_audio = None
            for interval in zero_list:
                this_audio = get_audio_chunk(interval, mic_data)
                smooth_audio = smooth_audio_edge(this_audio)
                noise_audio = add_normal_noise(smooth_audio)
                if con_audio is None:
                    con_audio = noise_audio
                else:
                    con_audio = np.concatenate((con_audio, noise_audio),axis=1)

            meta_save_path = os.path.join(save_meta_dir,os.path.basename(meta_file))
            is_non_empty = remove_ov_and_save_meta(meta_file, meta_save_path)

            if is_non_empty:
                mask_mic = mask_ov(mic_data, ov_list, con_audio)        
                mic_save_path = os.path.join(save_mic_dir,os.path.basename(mic_file))
                wav.write(mic_save_path, fs, mask_mic.T.astype(np.int16))

def rename_starss():
    print('Renaming STARSS dataset')
    save_meta_dir = os.path.join(METADATA_PATH, 'starss')
    save_mic_dir = os.path.join(MIC_PATH, 'starss')
    
    for file in starss_rename_list:
        file = file.split('.')[0]

        #metadata
        file_old = file + '.csv'
        file_new = 'fold15' + file[5:] + '.csv'
        if os.path.exists(os.path.join(save_meta_dir, file_old)):
            os.system(f'mv {save_meta_dir}/{file_old} {save_meta_dir}/{file_new}')

        #micdata
        file_old = file + '.wav'
        file_new = 'fold15' + file[5:] + '.wav'
        if os.path.exists(os.path.join(save_mic_dir, file_old)):
            os.system(f'mv {save_mic_dir}/{file_old} {save_mic_dir}/{file_new}')

def data_prep_starss():
    download_starss(DOWNLOAD_PATH)
    unzip_starss(DOWNLOAD_PATH)
    mask_single_event()
    rename_starss()

############################# DCASE #############################
def download_dcase(save_path):
    print('Downloading DCASE dataset')
    links = get_links('dcase')
    for link_type, link in links.items():
        print(f'Downloading {link_type}..........')
        wget_command = get_cmd(link, save_path)
        os.system(wget_command)

def unzip_dcase(save_path):
    print('Unzipping DCASE dataset')
    os.system(f'unzip -q {save_path}/dcase.zip -d {save_path}')
    os.system(f'rm {save_path}/dcase.zip')

def move_dcase():
    curr_path = os.path.join(DOWNLOAD_PATH, "zenodo_upload")
    meta_curr_path = os.path.join(curr_path,'meta_data')
    mic_curr_path = os.path.join(curr_path,'mic_data')

    meta_save_path = os.path.join(METADATA_PATH, 'dcase')
    mic_save_path = os.path.join(MIC_PATH, 'dcase')
    
    os.makedirs(meta_save_path, exist_ok=True)
    os.makedirs(mic_save_path, exist_ok=True)

    if os.path.exists(meta_curr_path):
        os.system(f'mv {meta_curr_path}/*.csv {meta_save_path}/ 2>/dev/null || true')
    if os.path.exists(mic_curr_path):
        os.system(f'mv {mic_curr_path}/*.wav {mic_save_path}/ 2>/dev/null || true')
    if os.path.exists(curr_path):
        shutil.rmtree(curr_path)

def data_prep_dcase():
    download_dcase(DOWNLOAD_PATH)
    unzip_dcase(DOWNLOAD_PATH)
    move_dcase()

############################# LOCATA #############################
def download_locata(save_path):
    print('Downloading LOCATA dataset')
    links = get_links('locata')
    for link_type, link in links.items():
        print(f'Downloading {link_type}..........')
        wget_command = get_cmd(link, save_path)
        os.system(wget_command)

def unzip_locata(save_path):
    print('Unzipping LOCATA dataset')
    os.system(f'unzip -q {save_path}/dev.zip -d {save_path}')
    os.system(f'unzip -q {save_path}/eval.zip -d {save_path}')
    os.system(f'rm {save_path}/dev.zip')
    os.system(f'rm {save_path}/eval.zip')

def get_loc_metadata():
    out_dir = os.path.join(METADATA_PATH, 'aug_locata')
    base_input_dir = DOWNLOAD_PATH
    
    os.makedirs(out_dir, exist_ok=True)
    task_list = ["1","3","5"]
    splits = ['eval','dev']
    for split in splits:
        input_path = os.path.join(base_input_dir, split)
        if os.path.exists(input_path):
            Locata2DecaseFormat(task_list, input_path, out_dir, arrays=["eigenmike"], is_dev=True, coord_system="polar")

def get_loc_mic_data():
    read_dir = DOWNLOAD_PATH
    save_dir = os.path.join(MIC_PATH, 'aug_locata')
    
    os.makedirs(save_dir, exist_ok=True)
    mic_util(read_dir, save_dir)

def augment_locata():
    print('Augmenting LOCATA dataset')
    mic_dir = os.path.join(MIC_PATH, 'aug_locata')
    meta_dir = os.path.join(METADATA_PATH, 'aug_locata')
    aug_folds = [11,13]
    aug_data(meta_dir, mic_dir, aug_folds)

def delete_locata():
    print('Cleaning up LOCATA download files')
    for dir_name in ['dev', 'eval']:
        dir_path = os.path.join(DOWNLOAD_PATH, dir_name)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

def data_prep_locata():
    download_locata(DOWNLOAD_PATH)
    unzip_locata(DOWNLOAD_PATH)
    get_loc_metadata()
    get_loc_mic_data()
    augment_locata()
    delete_locata()

##################################### MARCO #####################################
def download_marco(save_path):
    print('Downloading MARCO dataset')
    links = get_links('marco')
    for link_type, link in links.items():
        print(f'Downloading {link_type}..........')
        wget_command = get_cmd(link, save_path)
        os.system(wget_command)

def unzip_marco(save_path):
    print('Unzipping MARCO dataset')
    os.system(f'unzip -q "{save_path}/06 3D-MARCo Samples_Organ.zip" -d {save_path}')
    os.system(f'unzip -q "{save_path}/07 3D-MARCo Samples_Piano solo 1.zip" -d {save_path}')
    os.system(f'unzip -q "{save_path}/08 3D-MARCo Samples_Piano solo 2.zip" -d {save_path}')
    os.system(f'unzip -q "{save_path}/09 3D-MARCo Samples_Acappella.zip" -d {save_path}')
    os.system(f'unzip -q "{save_path}/04 3D-MARCo Samples_Quartet.zip" -d {save_path}')
    os.system(f'rm "{save_path}/06 3D-MARCo Samples_Organ.zip"')
    os.system(f'rm "{save_path}/07 3D-MARCo Samples_Piano solo 1.zip"')
    os.system(f'rm "{save_path}/08 3D-MARCo Samples_Piano solo 2.zip"')
    os.system(f'rm "{save_path}/09 3D-MARCo Samples_Acappella.zip"')
    os.system(f'rm "{save_path}/04 3D-MARCo Samples_Quartet.zip"')

def get_mic_marco(read_dir):
    """Extract Eigenmike WAV files from MARCO dataset."""
    save_dir = os.path.join(read_dir,'temp_mic_data') 
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    ins_dirs = ['Acappella','Organ','Piano solo 1','Piano solo 2','Quartet']
    ins_dir_2 = ['Single sources at different positions']
    eigen_files = []

    for ins_dir in ins_dirs:
        files = glob(f'{read_dir}/{ins_dir}/*wav')
        for file in files:
            if 'Eigenmike' in file:
                eigen_files.append(file)

    files = glob(f'{read_dir}/{ins_dir_2[0]}/*/*wav')
    for file in files:
        if 'Eigenmike' in file:
            eigen_files.append(file)

    for src in eigen_files:
        file = os.path.basename(src)
        dst = os.path.join(save_dir,file)
        shutil.copyfile(src, dst)       

    for ins_dir in ins_dirs:
        ins_path = os.path.join(read_dir,ins_dir)
        if os.path.exists(ins_path):
            shutil.rmtree(ins_path)
    
    if os.path.exists(os.path.join(read_dir,ins_dir_2[0])):
        shutil.rmtree(os.path.join(read_dir,ins_dir_2[0]))

def resample_marco_32ch(read_dir):
    """Resample MARCO files to 24kHz, keeping all 32 channels (not just tetrahedral)."""
    mic_dir = os.path.join(read_dir,'temp_mic_data') 
    NEW_SR = 24000

    def resample_and_save(file):
        rec_path = os.path.join(mic_dir,file)
        save_path = os.path.join(mic_dir,file)

        fs, audio = wav.read(rec_path)
        # Resample to 24kHz, keep all 32 channels
        audio_new = signal.resample(audio, int(len(audio) * float(NEW_SR) / fs))
        # Normalize and convert to int16
        audio_new = audio_new.astype('float32')/np.iinfo(np.int32).max
        audio_new = (audio_new * 32767).astype('int16')
        wav.write(save_path, NEW_SR, audio_new)

    recs = os.listdir(mic_dir)
    for rec in tqdm(recs, desc='Resampling MARCO to 24kHz'):
        _ = resample_and_save(rec)

def get_meta_data_marco(read_dir):
    """Generate metadata CSV files for MARCO."""
    save_dir = os.path.join(read_dir,'temp_meta_data') 
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    mic_dir = os.path.join(read_dir,'temp_mic_data') 
    def make_csv(file):
        save_file = file.split('.')[0]+'.csv'
        save_path = os.path.join(save_dir,save_file)

        rec_file = os.path.join(mic_dir,file)
        fs, audio = wav.read(rec_file)
        
        fs *= 0.1
        num_pts = int(audio.shape[0]//fs)

        class_no = -1
        source_no = -1
        azimuth = -1
        elevation = -1

        r = file_dist_dict.get(file, 3.0)  # Default distance if not found

        with open(save_path, 'w') as f:
            writer = csv.writer(f)
            for i in range(num_pts):
                writer.writerow([i, class_no, source_no, azimuth, elevation, r])

    recs = os.listdir(mic_dir)
    for rec in tqdm(recs, desc='Generating MARCO metadata'):
        make_csv(rec)
    
def aug_data_marco(read_dir):
    """Apply channel permutation augmentation to MARCO."""
    print('Augmenting MARCO dataset')
    meta_dir = os.path.join(read_dir,'temp_meta_data') 
    mic_dir = os.path.join(read_dir,'temp_mic_data')
    marco_aug_data(meta_dir, mic_dir)

def rename_marco(read_dir):
    """Rename MARCO files with proper fold prefixes."""
    meta_dir = os.path.join(read_dir,'temp_meta_data') 
    mic_dir = os.path.join(read_dir,'temp_mic_data')

    save_meta_dir = os.path.join(METADATA_PATH, 'aug_marco')
    save_mic_dir = os.path.join(MIC_PATH, 'aug_marco')
    
    os.makedirs(save_meta_dir, exist_ok=True)
    os.makedirs(save_mic_dir, exist_ok=True)

    for file in os.listdir(meta_dir):
        file_pre = file.split('.')[0]
        if file_pre in marco_test_files:
            fold_pre = 'fold10_'
        elif file_pre in marco_val_files:
            fold_pre = 'fold17_'
        else:
            fold_pre = 'fold9_'

        #metadata
        file_old = file_pre + '.csv'
        file_new = fold_pre + file_pre + '.csv'
        if os.path.exists(os.path.join(meta_dir, file_old)):
            os.system(f'mv {meta_dir}/{file_old} {save_meta_dir}/{file_new}')

        #micdata
        file_old = file_pre + '.wav'
        file_new = fold_pre + file_pre + '.wav'
        if os.path.exists(os.path.join(mic_dir, file_old)):
            os.system(f'mv {mic_dir}/{file_old} {save_mic_dir}/{file_new}')

def del_temp_marco(read_dir):
    """Clean up temporary MARCO directories."""
    for temp_dir in ['temp_meta_data', 'temp_mic_data']:
        temp_path = os.path.join(read_dir, temp_dir)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

def data_prep_marco():    
    download_marco(DOWNLOAD_PATH)
    unzip_marco(DOWNLOAD_PATH)
    get_mic_marco(DOWNLOAD_PATH)
    resample_marco_32ch(DOWNLOAD_PATH)  # Keep 32 channels, resample to 24kHz
    get_meta_data_marco(DOWNLOAD_PATH)
    aug_data_marco(DOWNLOAD_PATH)
    rename_marco(DOWNLOAD_PATH)
    del_temp_marco(DOWNLOAD_PATH)

############################# Generate CSV Files #############################
def get_fold_from_filename(filename):
    """Extract fold prefix from filename (e.g., 'fold7_file.wav' -> 'fold7')."""
    import re
    match = re.match(r'^(fold\d+)_', filename)
    if match:
        return match.group(1)
    return None

def generate_csv_for_dataset(dataset_name, mic_dir, output_dir):
    """Generate train/val/test CSV files based on fold naming."""
    if not os.path.exists(mic_dir):
        print(f"Warning: Directory {mic_dir} does not exist. Skipping {dataset_name}.")
        return
    
    wav_files = glob(os.path.join(mic_dir, '*.wav'))
    if len(wav_files) == 0:
        print(f"Warning: No WAV files found in {mic_dir}. Skipping {dataset_name}.")
        return
    
    print(f"\nProcessing {dataset_name}:")
    print(f"  Found {len(wav_files)} WAV files in {mic_dir}")
    
    # Fold mapping for splits
    fold_mapping = {
        'starss': {
            'train': [],
            'test': ['fold15'],
            'val': []
        },
        'dcase': {
            'train': [],
            'test': [],
            'val': []
        },
        'aug_locata': {
            'train': [],
            'test': ['fold11', 'fold12', 'fold13'],
            'val': []
        },
        'aug_marco': {
            'train': ['fold9'],
            'test': ['fold10'],
            'val': ['fold17']
        }
    }
    
    mapping = fold_mapping.get(dataset_name, {'train': [], 'test': [], 'val': []})
    
    # Organize files by split
    splits = {'train': [], 'val': [], 'test': []}
    
    for wav_file in wav_files:
        filename = os.path.basename(wav_file)
        fold = get_fold_from_filename(filename)
        
        assigned = False
        for split_name, fold_prefixes in mapping.items():
            if fold and fold in fold_prefixes:
                splits[split_name].append(wav_file)
                assigned = True
                break
        
        if not assigned:
            splits['train'].append(wav_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV files for each split
    for split_name, files in splits.items():
        if len(files) > 0:
            csv_path = os.path.join(output_dir, f'{dataset_name}_{split_name}.csv')
            df = pd.DataFrame(files)
            df.to_csv(csv_path, index=False, header=False)
            print(f"  {split_name}: {len(files)} files -> {csv_path}")
        else:
            print(f"  {split_name}: 0 files (skipped)")
    
    # Also create a combined CSV
    all_files = [f for files in splits.values() for f in files]
    if len(all_files) > 0:
        csv_path = os.path.join(output_dir, f'{dataset_name}_all.csv')
        df = pd.DataFrame(all_files)
        df.to_csv(csv_path, index=False, header=False)
        print(f"  all: {len(all_files)} files -> {csv_path}")

if __name__ == '__main__':
    # Create base directories
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    os.makedirs(METADATA_PATH, exist_ok=True)
    os.makedirs(MIC_PATH, exist_ok=True)
    os.makedirs(CSV_PATH, exist_ok=True)
    
    print(f"Using base data path: {BASE_DATA_PATH}")
    print(f"Download path: {DOWNLOAD_PATH}")
    print(f"Metadata path: {METADATA_PATH}")
    print(f"Mic data path: {MIC_PATH}")
    print(f"CSV path: {CSV_PATH}")
    print()
    
    data_prep_starss()
    data_prep_dcase()
    data_prep_locata()
    data_prep_marco()
    
    # Generate CSV files
    print(f"\n{'='*60}")
    print("Generating CSV files")
    print(f"{'='*60}")
    
    datasets_to_process = ['starss', 'dcase', 'aug_locata', 'aug_marco']
    for dataset_name in datasets_to_process:
        mic_dir = os.path.join(MIC_PATH, dataset_name)
        generate_csv_for_dataset(dataset_name, mic_dir, CSV_PATH)
    
    print(f"\n{'='*60}")
    print("All datasets processed and CSV files generated!")
    print(f"{'='*60}")
    print(f"\nCSV files saved to: {CSV_PATH}")
    print(f"WAV files saved to: {MIC_PATH}")
    print("\nNext step: Run training with config_spatial_consistency.yaml")
