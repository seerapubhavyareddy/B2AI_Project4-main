import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def check_labels(data):
    labels = data['label'].unique()
    return set(labels) == {0, 1}

def stratified_kfold_split(data, n_splits=5, random_state=42):
    patients = data['patient'].unique()
    np.random.seed(random_state)
    np.random.shuffle(patients)

    patient_labels = data.groupby('patient')['label'].agg(lambda x: x.value_counts().idxmax())
    patient_labels = patient_labels.loc[patients]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for train_idx, test_idx in skf.split(patients, patient_labels):
        train_patients, test_patients = patients[train_idx], patients[test_idx]

        train_data = data[data['patient'].isin(train_patients)]
        test_data = data[data['patient'].isin(test_patients)]

        folds.append((train_data, test_data, train_patients, test_patients))

    return folds

def process_data_type(main_path, data_type, n_splits=5):
    print(f"Processing data type: {data_type}")
    main_file_path = os.path.join(main_path, f'{data_type}_combined_files.txt')
    output_dir = os.path.join(main_path, data_type)

    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(main_file_path, sep=' ', header=None, names=['path', 'label'])
    data['patient'] = data['path'].apply(lambda x: x.split('/')[-2])

    folds = stratified_kfold_split(data, n_splits=n_splits)

    for i, (train_data, test_data, train_patients, test_patients) in enumerate(folds):
        fold_dir = os.path.join(output_dir, f'fold{i+1}')
        os.makedirs(fold_dir, exist_ok=True)

        if check_labels(train_data) and check_labels(test_data):
            print(f"Each dataset in fold {i+1} of {data_type} contains both labels.")
        else:
            print(f"One or more datasets in fold {i+1} of {data_type} are missing a label.")

        train_data[['path', 'label']].to_csv(os.path.join(fold_dir, 'train.txt'), sep=' ', header=False, index=False)
        test_data[['path', 'label']].to_csv(os.path.join(fold_dir, 'test.txt'), sep=' ', header=False, index=False)

        print(f"Data split for fold {i+1} of {data_type} completed and saved to txt files.")

        print(f"Label distribution for training set in fold {i+1} of {data_type}:")
        print(train_data['label'].value_counts().sort_index())
        print(f"Label distribution for testing set in fold {i+1} of {data_type}:")
        print(test_data['label'].value_counts().sort_index())

        print(f"\nNumber of patients in training set for fold {i+1} of {data_type}:", len(train_patients))
        print(f"Patients in training set for fold {i+1} of {data_type}:", train_patients)
        print(f"\nNumber of patients in testing set for fold {i+1} of {data_type}:", len(test_patients))
        print(f"Patients in testing set for fold {i+1} of {data_type}:", test_patients)

def main():
    main_path = '/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/MachineLearning/Data_PreProcessing'
    data_types = ['fimo', 'deep', 'rp', 'reg']

    for data_type in data_types:
        process_data_type(main_path, data_type)

if __name__ == "__main__":
    main()
