import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
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

# Add the extract_subject_id function that's missing in the XGBoost code
def extract_subject_id(path):
    # Extract subject ID from path - adjust this according to your naming convention
    # This is the same implementation as in the RF/CFS code
    parts = os.path.basename(path).split('_')
    if len(parts) >= 2:
        return parts[0]  # Assumes first part is subject ID
    return os.path.basename(path)  # Fallback

def process_fold(data_type, fold, threshold, feature_selection):
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
    y_train_full = y_train
    y_test_full = y_test

    # Create save path based on feature selection method
    save_base_path = os.path.join(config.MAIN_SAVE_DIR, data_type, 'xgboost', feature_selection,
                                 f'fold{fold}', f'threshold_{threshold}')
    os.makedirs(save_base_path, exist_ok=True)
    
    # Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_full = imputer.fit_transform(X_train_full)
    X_test_full = imputer.transform(X_test_full)
    
    if config.Standard:
        scaler = StandardScaler().fit(X_train_full)
        X_train_full = standardize_features(X_train_full, scaler)
        X_test_full = standardize_features(X_test_full, scaler)
        joblib.dump(scaler, os.path.join(save_base_path, 'scaler.pkl'))

    # Load selected features based on feature selection method
    selected_features_path = os.path.join(config.MAIN_SAVE_DIR, data_type, feature_selection, 
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
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train_full)
    else:
        X_train_resampled, y_train_resampled = X_train_selected, y_train_full

    # XGBoost specific parameters
    if config.GridSearch:
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        grid_search = GridSearchCV(
            estimator=XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train_resampled, y_train_resampled)
        xgb_classifier = grid_search.best_estimator_
    else:
        xgb_classifier = XGBClassifier(
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8
        )
        xgb_classifier.fit(X_train_resampled, y_train_resampled)

    # Get chunk-level predictions
    y_pred = xgb_classifier.predict(X_test_selected)
    y_pred_proba = xgb_classifier.predict_proba(X_test_selected)[:, 1] if len(np.unique(y_test_full)) == 2 else None

    # Calculate chunk-level metrics
    chunk_accuracy = accuracy_score(y_test_full, y_pred)
    chunk_f1 = f1_score(y_test_full, y_pred, average='weighted')
    chunk_precision = precision_score(y_test_full, y_pred, average='weighted')
    chunk_recall = recall_score(y_test_full, y_pred, average='weighted')
    chunk_auc = roc_auc_score(y_test_full, y_pred_proba) if y_pred_proba is not None else 'N/A'
    chunk_report = classification_report(y_test_full, y_pred)
    
    # Extract subject IDs using the same method as in RF/CFS code
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
            if y_pred_proba is not None:
                subject_prob[subj_id] = []
        
        # Add predictions for this subject
        subject_pred[subj_id].append(y_pred[i])
        if y_pred_proba is not None:
            subject_prob[subj_id].append(y_pred_proba[i])
    
    # Convert to subject-level predictions (majority vote)
    y_true_subject = np.array([subject_true[subj] for subj in unique_subjects])
    y_pred_subject = np.array([np.round(np.mean(subject_pred[subj])) for subj in unique_subjects])
    
    # For AUC, use average probabilities if available
    if y_pred_proba is not None:
        y_prob_subject = np.array([np.mean(subject_prob[subj]) for subj in unique_subjects])
    else:
        y_prob_subject = None
    
    # Calculate subject-level metrics
    subject_accuracy = accuracy_score(y_true_subject, y_pred_subject)
    subject_f1 = f1_score(y_true_subject, y_pred_subject, average='weighted')
    subject_precision = precision_score(y_true_subject, y_pred_subject, average='weighted')
    subject_recall = recall_score(y_true_subject, y_pred_subject, average='weighted')
    
    if y_prob_subject is not None and len(np.unique(y_true_subject)) == 2:
        subject_auc = roc_auc_score(y_true_subject, y_prob_subject)
    else:
        subject_auc = 'N/A'
    
    subject_report = classification_report(y_true_subject, y_pred_subject)
    subject_cm = confusion_matrix(y_true_subject, y_pred_subject)

    # Save results with feature selection method in filename
    model_filename = f'xgb_model_{feature_selection}_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.pkl'
    joblib.dump(xgb_classifier, os.path.join(save_base_path, model_filename))

    # Save the classification report with both chunk and subject metrics
    report_filename = f'classification_report_{feature_selection}_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.txt'
    with open(os.path.join(save_base_path, report_filename), 'w') as f:
        f.write(f'Feature Selection Method: {feature_selection}\n')
        f.write(f'Threshold: {threshold}\n\n')
        
        f.write("SUBJECT-LEVEL METRICS\n")
        f.write("=====================\n")
        f.write(f'Test Accuracy: {subject_accuracy}\n')
        f.write(f'F1 Score: {subject_f1}\n')
        f.write(f'Precision: {subject_precision}\n')
        f.write(f'Recall: {subject_recall}\n')
        f.write(f'AUC: {subject_auc}\n\n')
        
        f.write(f'Number of test subjects: {len(unique_subjects)}\n')
        f.write(f'Number of positive subjects: {np.sum(y_true_subject)}\n')
        f.write(f'Number of negative subjects: {len(y_true_subject) - np.sum(y_true_subject)}\n\n')
        f.write(subject_report + '\n\n')
        
        f.write("CHUNK-LEVEL METRICS (for reference)\n")
        f.write("==================================\n")
        f.write(f'Test Accuracy: {chunk_accuracy}\n')
        f.write(f'F1 Score: {chunk_f1}\n')
        f.write(f'Precision: {chunk_precision}\n')
        f.write(f'Recall: {chunk_recall}\n')
        f.write(f'AUC: {chunk_auc}\n\n')
        f.write(chunk_report)

    # Write test results
    write_test_results(y_test_full, y_pred, test_path_full, save_base_path)
    
    # Save subject-level results
    subject_results_df = pd.DataFrame({
        'subject_id': unique_subjects,
        'true_label': y_true_subject,
        'predicted_label': y_pred_subject,
        'prediction_probability': y_prob_subject if y_prob_subject is not None else np.nan,
        'num_chunks': [len(subject_pred[subj]) for subj in unique_subjects]
    })
    
    subject_results_df.to_csv(os.path.join(save_base_path, 'subject_level_results.csv'), index=False)
    
    # Create confusion matrices
    # Chunk-level confusion matrix
    chunk_cm = confusion_matrix(y_test_full, y_pred)
    plot_confusion_matrix(
        chunk_cm, 
        classes=[0, 1], 
        save_path=os.path.join(save_base_path, 
                              f'chunk_confusion_matrix_{feature_selection}_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'),
        title='Chunk-Level Confusion Matrix'
    )
    
    # Subject-level confusion matrix
    plot_confusion_matrix(
        subject_cm, 
        classes=[0, 1], 
        save_path=os.path.join(save_base_path, 
                              f'subject_confusion_matrix_{feature_selection}_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'),
        title='Subject-Level Confusion Matrix'
    )

    # Feature importance plot for XGBoost
    plt.figure(figsize=(10, 6))
    feature_importance = xgb_classifier.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx])
    plt.yticks(pos, np.array(selected_features_names)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title(f'XGBoost Feature Importance ({feature_selection}, threshold={threshold})')
    plt.savefig(os.path.join(save_base_path, f'feature_importance_{feature_selection}_{config.MAX_FEATURES}_features.png'))
    plt.close()

    # Return subject-level metrics instead of chunk-level metrics
    return {
        'data_type': data_type,
        'feature_selection': feature_selection,
        'threshold': threshold,
        'accuracy': subject_accuracy,
        'f1': subject_f1,
        'precision': subject_precision,
        'recall': subject_recall,
        'auc': subject_auc,
        'chunk_accuracy': chunk_accuracy,
        'chunk_f1': chunk_f1,
        'chunk_precision': chunk_precision,
        'chunk_recall': chunk_recall,
        'chunk_auc': chunk_auc,
        'num_subjects': len(unique_subjects)
    }

def main():
    summary_results = []
    feature_selection_methods = ['CFS']  # Add more methods if needed
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    for data_type in config.DATA_TYPES:
        print(f"\nProcessing data type: {data_type}")
        for feature_selection in feature_selection_methods:
            for fold in config.Fold:
                for threshold in thresholds:
                    print(f"Processing {feature_selection}, fold {fold}, threshold {threshold}")
                    try:
                        results = process_fold(data_type, fold, threshold, feature_selection)
                        results['fold'] = fold
                        summary_results.append(results)
                    except Exception as e:
                        print(f"Error processing {data_type}, {feature_selection}, fold {fold}, threshold {threshold}: {str(e)}")
                        continue
    
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        
        # Create summary tables for each data type and feature selection method
        for data_type in config.DATA_TYPES:
            for feature_selection in feature_selection_methods:
                data_type_results = summary_df[
                    (summary_df['data_type'] == data_type) & 
                    (summary_df['feature_selection'] == feature_selection)
                ]
                if not data_type_results.empty:
                    summary_table = data_type_results.pivot(
                        index='fold',
                        columns='threshold',
                        values=['accuracy', 'f1', 'precision', 'recall', 'auc']
                    )
                    summary_table = summary_table.round(3)
                    summary_save_path = os.path.join(
                        config.MAIN_SAVE_DIR, 
                        data_type, 
                        'xgboost',
                        feature_selection,
                        f'summary_results_{feature_selection}.csv'
                    )
                    summary_table.to_csv(summary_save_path)
                    print(f'Summary results for {data_type} ({feature_selection}) saved to {summary_save_path}')

if __name__ == "__main__":
    main()