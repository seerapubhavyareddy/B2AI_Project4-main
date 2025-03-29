import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import config as config
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

def extract_subject_id(path):
    """
    Extract subject ID from the file path.
    Handles formats like:
    - Patient-XX
    - sub-XXXXX-XXXXX-XXXXX-XXXXX-XXXXX
    """
    # Extract subject ID from path components
    filename = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))
    
    # First check the directory name for subject ID
    if dirname.startswith("sub-"):
        # Extract only the subject ID part (removing any session info)
        return dirname.split('_')[0]
    
    if dirname.startswith("Patient-"):
        return dirname
    
    # Then check the filename
    parts = filename.split('_')
    for part in parts:
        if part.startswith("sub-"):
            return part
        if part.startswith("Patient-"):
            return part
    
    # If we can't find a specific pattern, return the directory name
    return dirname

def load_selected_features(csv_file_path, num_features):
    df = pd.read_csv(csv_file_path)
    selected_features = df['Selected Features'].values[:num_features]
    return selected_features

def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    paths = df.iloc[:, 0].values
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    print("Columns ", y)
    return paths, X, y, df.columns[1:-1]

def load_annotations(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    paths = []
    labels = []
    for line in lines:
        path, label = line.strip().split()
        paths.append(path)
        labels.append(int(label))
    return np.array(paths), np.array(labels)

def clean_data(X):
    # Initialize imputer for handling NaN values
    imputer = SimpleImputer(strategy='mean')
    
    # Fit and transform the data
    X_cleaned = imputer.fit_transform(X)
    
    # Handle any infinite values by replacing them with large finite values
    X_cleaned = np.nan_to_num(X_cleaned, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    
    return X_cleaned

def standardize_features(features, scaler):
    return scaler.transform(features)

def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    random_state = 42
    # config.FEATURE = 'PCA'
    config.NUMBER_FEATURES = [3,10,20,30,40,50,60,70,80,90,100,110,120,130]
    
    # Initialize results dictionary
    all_results = {}
    
    # Main path now includes 'xgboost'
    main_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, 'xgboost', f'PCA_{config.PCA_component}')
    
    for fold in config.Fold:
        all_results[fold] = {}
        
        for num_features in config.NUMBER_FEATURES:
            print(f'************************ Fold {fold} - Features {num_features} *****************************************')
            
            # Modified directory structure
            save_path = os.path.join(main_path, f'fold{fold}', f'features_{num_features}')
            os.makedirs(save_path, exist_ok=True)

            # Load selected features
            selected_features_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE,
                f'{config.DATA_TYPE}_selected_features_{config.FEATURE}_{config.MAX_FEATURES}_std{config.Standard}.csv')
            selected_features_names = load_selected_features(selected_features_path, num_features)

            # Load and process data
            combine_csv_file_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, 
                f'{config.DATA_TYPE}_combined_features{config.WAYS}.csv')
            combined_paths, X_combined, y_combined, feature_names = load_data(combine_csv_file_path)

            # Clean the combined data first
            X_combined = clean_data(X_combined)

            # Load annotations
            annotations_dir = os.path.join(config.MAIN_DATA_DIR, config.DATA_TYPE, f'fold{fold}')
            train_paths, y_train = load_annotations(os.path.join(annotations_dir, 'train.txt'))
            test_paths, y_test = load_annotations(os.path.join(annotations_dir, 'test.txt'))

            # Process data and train model
            train_indices = np.where(np.isin(combined_paths, train_paths))[0]
            test_indices = np.where(np.isin(combined_paths, test_paths))[0]

            X_train_full = X_combined[train_indices]
            y_train = y_combined[train_indices]
            X_test_full = X_combined[test_indices]
            y_test = y_combined[test_indices]
            # Store test paths for later use in subject ID extraction
            test_path_full = combined_paths[test_indices]

            selected_features_indices = [list(feature_names).index(str(feature)) for feature in selected_features_names]
            X_train_selected = X_train_full[:, selected_features_indices]
            X_test_selected = X_test_full[:, selected_features_indices]

            # Verify data is clean before SMOTE
            print("Checking for NaN values before SMOTE:")
            print("Training data NaNs:", np.isnan(X_train_selected).sum())
            print("Training labels NaNs:", np.isnan(y_train).sum())

            # Apply SMOTE if configured
            if config.OVERSAMPLING == 'SMOTE':
                smote = SMOTE(random_state=random_state)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train_selected, y_train

            # Standardization
            if config.Standard:
                scaler_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'scaler_fold{fold}.pkl')
                scaler = joblib.load(scaler_path)
                X_train_standardized = standardize_features(X_train_resampled, scaler)
                X_test_standardized = standardize_features(X_test_selected, scaler)
            else:
                X_train_standardized = X_train_resampled
                X_test_standardized = X_test_selected

            # Train model
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
                grid_search.fit(X_train_standardized, y_train_resampled)
                print(f"Best parameters: {grid_search.best_params_}")
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
                xgb_classifier.fit(X_train_standardized, y_train_resampled)

            # Predictions and metrics at the chunk level
            y_pred_chunks = xgb_classifier.predict(X_test_standardized)
            y_pred_proba = xgb_classifier.predict_proba(X_test_standardized)[:, 1] if len(np.unique(y_test)) == 2 else None
            
            # Calculate chunk-level metrics
            chunk_accuracy = accuracy_score(y_test, y_pred_chunks)
            chunk_f1 = f1_score(y_test, y_pred_chunks, average='weighted')
            chunk_precision = precision_score(y_test, y_pred_chunks, average='weighted')
            chunk_recall = recall_score(y_test, y_pred_chunks, average='weighted')
            chunk_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'
            chunk_report = classification_report(y_test, y_pred_chunks)
            chunk_cm = confusion_matrix(y_test, y_pred_chunks)
            
            # Now calculate subject-level metrics
            # Extract subject IDs
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
                    subject_true[subj_id] = y_test[i]
                    subject_pred[subj_id] = []
                    if y_pred_proba is not None:
                        subject_prob[subj_id] = []
                
                # Add predictions for this subject
                subject_pred[subj_id].append(y_pred_chunks[i])
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
            
            print(f'Chunk-level Accuracy: {chunk_accuracy}')
            print(f'Subject-level Accuracy: {subject_accuracy}')

            # Save results with PCA prefix
            model_filename = f'PCA_xgb_model_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.pkl'
            joblib.dump(xgb_classifier, os.path.join(save_path, model_filename))

            # Save classification report with both chunk and subject metrics
            report_filename = f'PCA_classification_report_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.txt'
            with open(os.path.join(save_path, report_filename), 'w') as f:
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
            
            # Save subject-level results
            subject_results_df = pd.DataFrame({
                'subject_id': unique_subjects,
                'true_label': y_true_subject,
                'predicted_label': y_pred_subject,
                'prediction_probability': y_prob_subject if y_prob_subject is not None else np.nan,
                'num_chunks': [len(subject_pred[subj]) for subj in unique_subjects]
            })
            
            subject_results_df.to_csv(os.path.join(save_path, f'subject_level_results_{num_features}.csv'), index=False)

            # Save confusion matrices for both levels
            # Chunk-level confusion matrix
            plot_confusion_matrix(
                chunk_cm, 
                classes=[0, 1], 
                save_path=os.path.join(save_path, f'PCA_chunk_confusion_matrix_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'),
                title='Chunk-Level Confusion Matrix'
            )
            
            # Subject-level confusion matrix
            plot_confusion_matrix(
                subject_cm, 
                classes=[0, 1], 
                save_path=os.path.join(save_path, f'PCA_subject_confusion_matrix_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'),
                title='Subject-Level Confusion Matrix'
            )

            # Feature importance plot
            feature_importances = xgb_classifier.feature_importances_
            sorted_idx = np.argsort(feature_importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(feature_importances)), feature_importances[sorted_idx], align='center')
            plt.xticks(range(len(feature_importances)), np.array(selected_features_names)[sorted_idx], rotation=90)
            plt.xlabel('Features')
            plt.ylabel('Feature Importance Score')
            plt.title(f'XGBoost Feature Importance Analysis - {num_features} Features')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'PCA_feature_importance_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'))
            plt.close()

            # Store both chunk and subject level results
            all_results[fold][num_features] = {
                'chunk': {
                    'accuracy': chunk_accuracy,
                    'f1': chunk_f1,
                    'precision': chunk_precision,
                    'recall': chunk_recall,
                    'auc': chunk_auc
                },
                'subject': {
                    'accuracy': subject_accuracy,
                    'f1': subject_f1,
                    'precision': subject_precision,
                    'recall': subject_recall,
                    'auc': subject_auc,
                    'num_subjects': len(unique_subjects)
                }
            }

    # Generate separate summary tables for chunk and subject levels
    # Chunk-level summary
    chunk_summary_rows = []
    for fold in all_results:
        for num_features in all_results[fold]:
            metrics = all_results[fold][num_features]['chunk']
            row = {
                'Fold': fold,
                'Num_Features': num_features,
                'Accuracy': metrics['accuracy'],
                'F1': metrics['f1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'AUC': metrics['auc']
            }
            chunk_summary_rows.append(row)
    
    chunk_summary_table = pd.DataFrame(chunk_summary_rows)
    chunk_summary_save_path = os.path.join(main_path, 'chunk_summary_results_xgboost_pca.csv')
    chunk_summary_table.to_csv(chunk_summary_save_path, index=False)
    print(f'Chunk-level summary results saved to {chunk_summary_save_path}')
    
    # Subject-level summary
    subject_summary_rows = []
    for fold in all_results:
        for num_features in all_results[fold]:
            metrics = all_results[fold][num_features]['subject']
            row = {
                'Fold': fold,
                'Num_Features': num_features,
                'Accuracy': metrics['accuracy'],
                'F1': metrics['f1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'AUC': metrics['auc'],
                'Num_Subjects': metrics['num_subjects']
            }
            subject_summary_rows.append(row)
    
    subject_summary_table = pd.DataFrame(subject_summary_rows)
    subject_summary_save_path = os.path.join(main_path, 'subject_summary_results_xgboost_pca.csv')
    subject_summary_table.to_csv(subject_summary_save_path, index=False)
    print(f'Subject-level summary results saved to {subject_summary_save_path}')

    # Create performance plots for both levels
    # Chunk-level plot
    plt.figure(figsize=(12, 6))
    chunk_mean_accuracy = chunk_summary_table.groupby('Num_Features')['Accuracy'].mean()
    plt.plot(chunk_mean_accuracy.index, chunk_mean_accuracy.values, marker='o', label='Chunk-level')
    
    # Subject-level plot on the same figure
    subject_mean_accuracy = subject_summary_table.groupby('Num_Features')['Accuracy'].mean()
    plt.plot(subject_mean_accuracy.index, subject_mean_accuracy.values, marker='s', label='Subject-level')
    
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Accuracy')
    plt.title('PCA Feature Selection Performance')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(main_path, 'pca_performance_comparison.png'))
    plt.close()
    
    # Also save the original performance plot for backward compatibility
    plt.figure(figsize=(12, 6))
    plt.plot(chunk_mean_accuracy.index, chunk_mean_accuracy.values, marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Mean Accuracy')
    plt.title('PCA Feature Selection Performance (Chunk-level)')
    plt.grid(True)
    plt.savefig(os.path.join(main_path, 'pca_performance.png'))
    plt.close()

if __name__ == "__main__":
    main()