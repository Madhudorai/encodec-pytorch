import os
import argparse
from pathlib import Path
import pandas as pd  

def generate_csv(file_dir, csv_path):
    file_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith('.flac') or file.endswith('.wav') or file.endswith('.mp3'):
                file_list.append(os.path.join(root, file))
    print(f"file length:{len(file_list)}")
    csv_path = Path(csv_path)
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True)
        
    data = pd.DataFrame(file_list)
    data.to_csv(csv_path, index=False, header=False)

def split_train_test_csv(csv_path, threshold=0.8):
    try:
        from sklearn.model_selection import train_test_split  
    except ImportError as E:
        print("please pip install pandas slearn")
        
    data = pd.read_csv(csv_path)  
    train_data, test_data = train_test_split(data, train_size=threshold, random_state=42)  
   
    # Save files in the same directory as the output path
    csv_path_obj = Path(csv_path)
    base_name = csv_path_obj.stem
    output_dir = csv_path_obj.parent
    
    train_data.to_csv(output_dir / f'{base_name}_train.csv', index=False, header=False)  
    test_data.to_csv(output_dir / f'{base_name}_test.csv', index=False, header=False)

def split_train_val_test_csv(csv_path, input_dir, train_ratio=0.995, val_ratio=0.0025, test_ratio=0.0025):
    """Split CSV into train/val/test with custom ratios"""
    try:
        from sklearn.model_selection import train_test_split  
    except ImportError as E:
        print("please pip install pandas sklearn")
        return
        
    data = pd.read_csv(csv_path)  
    
    # Handle case where train_ratio is 0.0
    if train_ratio == 0.0:
        # All data goes to val+test split
        train_data = pd.DataFrame()  # Empty dataframe
        temp_data = data
    else:
        # First split: train vs (val+test)
        train_data, temp_data = train_test_split(data, train_size=train_ratio, random_state=42)
    
    # Second split: val vs test from the remaining data
    val_size = val_ratio / (val_ratio + test_ratio)  # Proportion of val in the remaining data
    val_data, test_data = train_test_split(temp_data, train_size=val_size, random_state=42)
    
    base_name = Path(input_dir).name
    output_dir = Path(input_dir)
    
    # Save files (only save train file if it has data)
    if len(train_data) > 0:
        train_data.to_csv(output_dir / f'{base_name}_train.csv', index=False, header=False)  
    val_data.to_csv(output_dir / f'{base_name}_val.csv', index=False, header=False)
    test_data.to_csv(output_dir / f'{base_name}_test.csv', index=False, header=False)
    
    print(f"Split complete:")
    print(f"  Train: {len(train_data)} files ({len(train_data)/len(data)*100:.3f}%)")
    print(f"  Val: {len(val_data)} files ({len(val_data)/len(data)*100:.3f}%)")
    print(f"  Test: {len(test_data)} files ({len(test_data)/len(data)*100:.3f}%)")
    
    # Clean up intermediate CSV file
    csv_path_obj = Path(csv_path)
    if csv_path_obj.exists():
        csv_path_obj.unlink()
        print(f"✓ Cleaned up intermediate file: {csv_path}") 

if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('-i','--input_file_dir', type=str, default='./LibriSpeech/train-clean-100')
    arg.add_argument('-s','--split', action='store_true', default=False,help='split dataset')
    arg.add_argument('-t','--threshold',type=float,default=0.8)
    arg.add_argument('--three_way_split', action='store_true', default=False,help='split into train/val/test with custom ratios')
    arg.add_argument('--train_ratio',type=float,default=0.995,help='train ratio for three-way split')
    arg.add_argument('--val_ratio',type=float,default=0.0025,help='validation ratio for three-way split')
    arg.add_argument('--test_ratio',type=float,default=0.0025,help='test ratio for three-way split')
    args = arg.parse_args()
    
    # Generate temporary CSV file name based on input directory
    input_dir_name = Path(args.input_file_dir).name
    temp_csv_path = f'/tmp/{input_dir_name}_temp.csv'
    
    generate_csv(args.input_file_dir, temp_csv_path)
    if args.three_way_split:
        split_train_val_test_csv(temp_csv_path, args.input_file_dir, args.train_ratio, args.val_ratio, args.test_ratio)
    elif args.split:
        split_train_test_csv(temp_csv_path,threshold=args.threshold)