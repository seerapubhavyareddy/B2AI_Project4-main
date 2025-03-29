import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import config as config
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer

def load_selected_features(csv_file_path, num_features):
    df = pd.read_csv(csv_file_path)
    selected_features = df['Selected Features'].values[:num_features]
    return [str(int(x)) for x in selected_features]

def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    paths = df.iloc[:, 0].values
    X = df.iloc[:, 1:-1].values
    return paths, X, df.columns[1:-1]

def load_annotations(annotation_file):
    df = pd.read_csv(annotation_file, sep=' ', header=None, names=['path', 'label'])
    return df['path'].values, df['label'].values.astype(int)

def standardize_features(X, scaler):
    return scaler.transform(X)

def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()

def write_test_results(y_test, y_pred, paths, save_path):
    df = pd.DataFrame({
        'recording_path': paths,
        'Prediction': y_pred,
        'Actual': y_test
    })
    results_path = os.path.join(save_path, f'test_results.txt')
    df.to_csv(results_path, index=False, sep='\t')

def process_fold(data_type, fold, threshold):
    random_state = 42
    
    # Load annotations
    annotations_dir = os.path.join(config.MAIN_DATA_DIR, data_type, f'fold{fold}')
    train_annotations_file = os.path.join(annotations_dir, 'train.txt')
    test_annotations_file = os.path.join(annotations_dir, 'test.txt')
    train_paths, y_train = load_annotations(train_annotations_file)
    test_paths, y_test = load_annotations(test_annotations_file)
    
    # Load feature data
    combine_csv_file_path = os.path.join(config.MAIN_SAVE_DIR, data_type, 
                                        f'{data_type}_combined_features{config.WAYS}.csv')
    combined_paths, X_combined, feature_names = load_data(combine_csv_file_path)

    # Match features with labels
    train_indices = np.where(np.isin(combined_paths, train_paths))[0]
    test_indices = np.where(np.isin(combined_paths, test_paths))[0]
    
    X_train_full = X_combined[train_indices]
    train_path_full = combined_paths[train_indices]
    X_test_full = X_combined[test_indices]
    test_path_full = combined_paths[test_indices]
    y_test_full = y_test

    save_base_path = os.path.join(config.MAIN_SAVE_DIR, data_type, 'CFS', 
                                 f'fold{fold}', f'threshold_{threshold}')
    os.makedirs(save_base_path, exist_ok=True)
    
    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_full = imputer.fit_transform(X_train_full)
    X_test_full = imputer.transform(X_test_full)
    
    if config.Standard:
        scaler = StandardScaler().fit(X_train_full)
        X_train_full = standardize_features(X_train_full, scaler)
        joblib.dump(scaler, os.path.join(save_base_path, 'scaler.pkl'))

    # Load selected features
    selected_features_path = os.path.join(config.MAIN_SAVE_DIR, data_type, 'CFS', 
                                        f'fold{fold}', f'threshold_{threshold}', 
                                        f'{data_type}_fold{fold}_selected_features_{config.FEATURE}_{config.MAX_FEATURES}_threshold{threshold}_std{config.Standard}.csv')
    
    if not os.path.exists(selected_features_path):
        raise FileNotFoundError(f"Selected features file not found at {selected_features_path}")
        
    selected_features_names = load_selected_features(selected_features_path, config.MAX_FEATURES)
    selected_features_indices = [list(feature_names).index(feature) for feature in selected_features_names]

    if not selected_features_indices:
        raise ValueError("No valid features were found in the selection process")

    X_train_selected = X_train_full[:, selected_features_indices]
    X_test_selected = X_test_full[:, selected_features_indices]

    if config.OVERSAMPLING == 'SMOTE':
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_selected, y_train

    # Train model
    if config.GridSearch:
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=random_state),
            param_grid=config.PARAM_GRID,
            cv=3,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train_resampled, y_train_resampled)
        rf_classifier = grid_search.best_estimator_
    else:
        rf_classifier = RandomForestClassifier(
            random_state=random_state,
            n_estimators=200,
            max_features='sqrt',
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            bootstrap=True
        )
        rf_classifier.fit(X_train_resampled, y_train_resampled)

    # Get chunk-level predictions and probabilities
    y_pred_chunks = rf_classifier.predict(X_test_selected)
    y_prob_chunks = rf_classifier.predict_proba(X_test_selected)[:, 1] if len(np.unique(y_train_resampled)) == 2 else None
    
    # Extract subject IDs from paths
    # Assuming the path format includes subject ID, e.g., "path/to/subject_123/chunk_456.wav"
    # Modify this pattern extraction according to your actual file naming convention
    def extract_subject_id(path):
        # Extract subject ID from path - adjust this according to your naming convention
        # This is a placeholder implementation
        parts = os.path.basename(path).split('_')
        if len(parts) >= 2:
            return parts[0]  # Assumes first part is subject ID
        return os.path.basename(path)  # Fallback

    # Group test data by subject
    subject_ids = [extract_subject_id(path) for path in test_path_full]
    unique_subjects = np.unique(subject_ids)
    
    # Create dictionaries to store subject-level data
    subject_true = {}
    subject_pred = {}
    subject_prob = {}
    
    # Aggregate predictions by subject
    for i, subj_id in enumerate(subject_ids):
        if subj_id not in subject_true:
            # Initialize for first occurrence of this subject
            subject_true[subj_id] = y_test_full[i]
            subject_pred[subj_id] = []
            if y_prob_chunks is not None:
                subject_prob[subj_id] = []
        
        # Add predictions for this subject
        subject_pred[subj_id].append(y_pred_chunks[i])
        if y_prob_chunks is not None:
            subject_prob[subj_id].append(y_prob_chunks[i])
    
    # Convert to subject-level predictions (majority vote)
    y_true_subject = np.array([subject_true[subj] for subj in unique_subjects])
    y_pred_subject = np.array([np.round(np.mean(subject_pred[subj])) for subj in unique_subjects])
    
    # For AUC, use average probabilities if available
    if y_prob_chunks is not None:
        y_prob_subject = np.array([np.mean(subject_prob[subj]) for subj in unique_subjects])
    else:
        y_prob_subject = None
    
    # Calculate subject-level metrics
    accuracy = accuracy_score(y_true_subject, y_pred_subject)
    f1 = f1_score(y_true_subject, y_pred_subject, average='weighted')
    precision = precision_score(y_true_subject, y_pred_subject, average='weighted')
    recall = recall_score(y_true_subject, y_pred_subject, average='weighted')
    
    if y_prob_subject is not None and len(np.unique(y_true_subject)) == 2:
        auc = roc_auc_score(y_true_subject, y_prob_subject)
    else:
        auc = 'N/A'

    # Save results
    model_filename = f'rf_model_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.pkl'
    joblib.dump(rf_classifier, os.path.join(save_base_path, model_filename))

    report_filename = f'classification_report_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.txt'
    
    # Save both chunk-level and subject-level results
    with open(os.path.join(save_base_path, report_filename), 'w') as f:
        f.write("SUBJECT-LEVEL METRICS\n")
        f.write("=====================\n")
        f.write(f'Test Accuracy: {accuracy}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'AUC: {auc}\n\n')
        
        f.write(f'Number of test subjects: {len(unique_subjects)}\n')
        f.write(f'Number of positive subjects: {np.sum(y_true_subject)}\n')
        f.write(f'Number of negative subjects: {len(y_true_subject) - np.sum(y_true_subject)}\n\n')
        
        f.write("CHUNK-LEVEL METRICS (for reference)\n")
        f.write("==================================\n")
        chunk_accuracy = accuracy_score(y_test_full, y_pred_chunks)
        chunk_f1 = f1_score(y_test_full, y_pred_chunks, average='weighted')
        chunk_precision = precision_score(y_test_full, y_pred_chunks, average='weighted')
        chunk_recall = recall_score(y_test_full, y_pred_chunks, average='weighted')
        chunk_auc = roc_auc_score(y_test_full, y_prob_chunks) if y_prob_chunks is not None and len(np.unique(y_test_full)) == 2 else 'N/A'
        
        f.write(f'Chunk Accuracy: {chunk_accuracy}\n')
        f.write(f'Chunk F1 Score: {chunk_f1}\n')
        f.write(f'Chunk Precision: {chunk_precision}\n')
        f.write(f'Chunk Recall: {chunk_recall}\n')
        f.write(f'Chunk AUC: {chunk_auc}\n\n')
        
        f.write(classification_report(y_test_full, y_pred_chunks))
    
    # Write detailed subject-level results
    subject_results_df = pd.DataFrame({
        'subject_id': unique_subjects,
        'true_label': y_true_subject,
        'predicted_label': y_pred_subject,
        'prediction_probability': y_prob_subject if y_prob_subject is not None else np.nan,
        'num_chunks': [len(subject_pred[subj]) for subj in unique_subjects]
    })
    
    subject_results_df.to_csv(os.path.join(save_base_path, 'subject_level_results.csv'), index=False)
    
    # Write chunk-level results for reference
    write_test_results(y_test_full, y_pred_chunks, test_path_full, save_base_path)
    
    # Create confusion matrices for both chunk and subject level
    # Chunk-level confusion matrix
    cm_chunk = confusion_matrix(y_test_full, y_pred_chunks)
    plot_confusion_matrix(
        cm_chunk, 
        classes=[0, 1], 
        save_path=os.path.join(save_base_path, 
                              f'chunk_confusion_matrix_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'),
        title='Chunk-Level Confusion Matrix'
    )
    
    # Subject-level confusion matrix
    cm_subject = confusion_matrix(y_true_subject, y_pred_subject)
    plot_confusion_matrix(
        cm_subject, 
        classes=[0, 1], 
        save_path=os.path.join(save_base_path, 
                              f'subject_confusion_matrix_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'),
        title='Subject-Level Confusion Matrix'
    )

    return {
        'data_type': data_type,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'num_subjects': len(unique_subjects)
    }

def main():
    summary_results = []
    for data_type in config.DATA_TYPES:
        print(f"\nProcessing data type: {data_type}")
        for fold in config.Fold:
            for threshold in config.thredhold:
                print(f"Processing fold {fold} with threshold {threshold}")
                try:
                    results = process_fold(data_type, fold, threshold)
                    results['fold'] = fold
                    results['threshold'] = threshold
                    summary_results.append(results)
                except Exception as e:
                    print(f"Error processing {data_type}, fold {fold}, threshold {threshold}: {str(e)}")
                    continue
    
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        
        # Create summary tables for each data type
        for data_type in config.DATA_TYPES:
            data_type_results = summary_df[summary_df['data_type'] == data_type]
            if not data_type_results.empty:
                summary_table = data_type_results.pivot(index='fold', columns='threshold', values='accuracy')
                summary_table = summary_table.round(3)
                summary_save_path = os.path.join(config.MAIN_SAVE_DIR, data_type, 'summary_results.csv')
                summary_table.to_csv(summary_save_path)
                print(f'Summary results for {data_type} saved to {summary_save_path}')

if __name__ == "__main__":
    main()