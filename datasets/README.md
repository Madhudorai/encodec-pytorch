# datasets
You need to use the `generate_train_file.py` script to generate the datasets train csv. 
The script will generate the csv files for the train datasets. 

For Common Voice, we randomly sample 99.5% of the dataset for train, 0.25% for valid and 0.25% for test splits.

DNS Challenge 4 Clean speech, we sample 98% for train, 1% for valid and 1% for test.

For FSD50K using the dev set for training and splitting the eval set between validation and test

For Jamendo dataset,96% for train, 2% for valid and 2% for test

# Usage

python datasets/generate_dataset_csvs.py -i /path/to/your/dataset/directory

2 way split
python datasets/generate_dataset_csvs.py -i /path/to/your/dataset/directory --split -t 0.8

3 way split
python datasets/generate_dataset_csvs.py -i /path/to/your/dataset/directory --three_way_split --train_ratio 0.995 --val_ratio 0.0025 --test_ratio 0.0025


