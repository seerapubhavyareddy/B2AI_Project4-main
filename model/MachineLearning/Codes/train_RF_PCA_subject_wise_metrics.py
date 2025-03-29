import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import config as config
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

def load_selected_features(csv_file_path, num_features):
    df = pd.read_csv(csv_file_path)
    selected_features = df['Selected Features'].values[:num_features]  # Select the top N features
    return selected_features

def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    paths = df.iloc[:, 0].values  # First column as paths
    X = df.iloc[:, 1:-1].values  # All columns except the first (path) and the last one (label)
    y = df.iloc[:, -1].values   # The last column
    print("Columns ", y)
    return paths, X, y, df.columns[1:-1]

def load_annotations(annotation_file):
    df = pd.read_csv(annotation_file, sep=' ', header=None, names=['path', 'label'])
    return df['path'].values, df['label'].values

def clean_data(X):
    # Initialize imputer for handling NaN values
    imputer = SimpleImputer(strategy='mean')
    
    # Fit and transform the data
    X_cleaned = imputer.fit_transform(X)
    
    # Handle any infinite values by replacing them with large finite values
    X_cleaned = np.nan_to_num(X_cleaned, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    
    return X_cleaned

def standardize_features(X, scaler):
    X_standardized = scaler.transform(X)
    return X_standardized

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
    print(f'Test results written to {results_path}')

def extract_subject_id(path):
    """
    Extract subject ID from the file path.
    Handles formats like:
    - Patient-XX
    - sub-XXXXX
    And complex paths with UUID-style identifiers
    """
    # Extract subject ID from path components
    filename = os.path.basename(path)
    dirname = os.path.basename(os.path.dirname(path))
    
    # First check the directory name for subject ID
    if dirname.startswith("sub-"):
        # Extract only the subject ID part (removing any session info)
        return dirname.split('_')[0]
    
    # Then check the filename
    parts = filename.split('_')
    for part in parts:
        if part.startswith("sub-"):
            return part
    
    # Look in parent directory as fallback
    if dirname.startswith("Patient-"):
        return dirname
    
    # If we can't find a specific pattern, return the directory name
    return dirname


def calculate_subject_metrics(y_test, y_pred, y_pred_proba, test_paths):
    """
    Calculate metrics at the subject level.
    
    Args:
        y_test: True labels for test chunks
        y_pred: Predicted labels for test chunks
        y_pred_proba: Predicted probabilities for test chunks (for positive class)
        test_paths: Paths to test files for extracting subject IDs
        
    Returns:
        Dictionary with subject-level metrics and arrays of subject-level data
    """
    # Extract subject IDs
    subject_ids = [extract_subject_id(path) for path in test_paths]
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
    accuracy = accuracy_score(y_true_subject, y_pred_subject)
    f1 = f1_score(y_true_subject, y_pred_subject, average='weighted')
    precision = precision_score(y_true_subject, y_pred_subject, average='weighted')
    recall = recall_score(y_true_subject, y_pred_subject, average='weighted')
    
    if y_prob_subject is not None and len(np.unique(y_true_subject)) == 2:
        auc = roc_auc_score(y_true_subject, y_prob_subject)
    else:
        auc = 'N/A'
    
    # Create subject summary dataframe
    subject_summary = pd.DataFrame({
        'subject_id': unique_subjects,
        'true_label': y_true_subject,
        'predicted_label': y_pred_subject,
        'prediction_probability': y_prob_subject if y_prob_subject is not None else np.nan,
        'num_chunks': [len(subject_pred[subj]) for subj in unique_subjects]
    })
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_subject, y_pred_subject)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'y_true_subject': y_true_subject,
        'y_pred_subject': y_pred_subject,
        'y_prob_subject': y_prob_subject,
        'subject_summary': subject_summary,
        'confusion_matrix': cm,
        'num_subjects': len(unique_subjects),
        'num_positive': np.sum(y_true_subject),
        'num_negative': len(y_true_subject) - np.sum(y_true_subject)
    }

def main():
    random_state = 42  # Set a fixed random state for reproducibility

    main_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'PCA_{config.PCA_component}')

    summary_results = {}

    for fold in config.Fold:
        fold_results = {}
        for num_features in config.NUMBER_FEATURES:
            print(f'************************ Fold {fold} - Features {num_features} *****************************************')
            save_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'PCA_{config.PCA_component}', f'fold{fold}', f'features_{num_features}')
            os.makedirs(save_path, exist_ok=True)

            # Load selected features
            selected_features_path = os.path.join(main_path, f'{config.DATA_TYPE}_selected_features_{config.FEATURE}_{config.MAX_FEATURES}_std{config.Standard}.csv')
            selected_features_names = load_selected_features(selected_features_path, num_features)

            # Load the combined data
            combine_csv_file_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, f'{config.DATA_TYPE}_combined_features{config.WAYS}.csv')
            combined_paths, X_combined, y_combined, feature_names = load_data(combine_csv_file_path)

            # Clean the combined data first
            X_combined = clean_data(X_combined)

            annotations_dir = os.path.join(config.MAIN_DATA_DIR, config.DATA_TYPE, f'fold{fold}')
            train_annotations_file = os.path.join(annotations_dir, 'train.txt')
            test_annotations_file = os.path.join(annotations_dir, 'test.txt')

            train_paths, y_train = load_annotations(train_annotations_file)
            test_paths, y_test = load_annotations(test_annotations_file)

            # Find the indices of the training and test paths in the combined data
            train_indices = np.where(np.isin(combined_paths, train_paths))[0]
            test_indices = np.where(np.isin(combined_paths, test_paths))[0]

            # Extract the corresponding features for training and testing
            X_train_full = X_combined[train_indices]
            train_path_full = combined_paths[train_indices]
            y_train = y_combined[train_indices]  # Ensure y_train corresponds to train_indices
            X_test_full = X_combined[test_indices]
            test_path_full = combined_paths[test_indices]
            y_test = y_combined[test_indices]  # Ensure y_test corresponds to test_indices

            # Convert selected feature names to indices
            selected_features_indices = [list(feature_names).index(str(feature)) for feature in selected_features_names]

            # Select the top features from the datasets
            X_train_selected = X_train_full[:, selected_features_indices]
            X_test_selected = X_test_full[:, selected_features_indices]

            if config.OVERSAMPLING == 'SMOTE':
                smote = SMOTE(random_state=random_state)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
            else:
                X_train_resampled, y_train_resampled = X_train_selected, y_train

            if config.Standard:
                # Standardize the datasets using the previously saved scaler
                scaler_path = os.path.join(main_path, f'scaler_fold{fold}.pkl')
                scaler = joblib.load(scaler_path)
                X_train_standardized = standardize_features(X_train_resampled, scaler)
                X_test_standardized = standardize_features(X_test_selected, scaler)
            else:
                X_train_standardized = X_train_resampled
                X_test_standardized = X_test_selected

            if config.GridSearch:
                grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=random_state), param_grid=config.PARAM_GRID, cv=3, n_jobs=-1, verbose=2)
                # Fit the grid search to the resampled data
                grid_search.fit(X_train_standardized, y_train_resampled)

                # Print the best parameters
                print(f"Best parameters: {grid_search.best_params_}")

                # Use the best model for predictions
                best_rf = grid_search.best_estimator_
                y_pred = best_rf.predict(X_test_standardized)
                y_pred_proba = best_rf.predict_proba(X_test_standardized)[:, 1] if len(np.unique(y_test)) == 2 else None
                rf_classifier = best_rf
            else:
                # Train RandomForest model using the selected features
                rf_classifier = RandomForestClassifier(random_state=random_state, n_estimators=200, max_features='sqrt', max_depth=20, min_samples_split=5, min_samples_leaf=2, bootstrap=True)
                rf_classifier.fit(X_train_standardized, y_train_resampled)

                # Use the trained model for predictions
                y_pred = rf_classifier.predict(X_test_standardized)
                y_pred_proba = rf_classifier.predict_proba(X_test_standardized)[:, 1] if len(np.unique(y_test)) == 2 else None

            # Evaluate the model at chunk level
            chunk_accuracy = accuracy_score(y_test, y_pred)
            chunk_f1 = f1_score(y_test, y_pred, average='weighted')
            chunk_precision = precision_score(y_test, y_pred, average='weighted')
            chunk_recall = recall_score(y_test, y_pred, average='weighted')
            chunk_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 'N/A'
            chunk_report = classification_report(y_test, y_pred)
            chunk_cm = confusion_matrix(y_test, y_pred)

            print(f'Chunk-level Test Accuracy: {chunk_accuracy}')

            # Calculate subject-level metrics
            subject_metrics = calculate_subject_metrics(y_test, y_pred, y_pred_proba, test_path_full)
            
            print(f'Subject-level Test Accuracy: {subject_metrics["accuracy"]}')

            # Save the best model
            joblib.dump(rf_classifier, os.path.join(save_path, f'rf_model_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.pkl'))

            # Save the classification report with both chunk and subject metrics
            with open(os.path.join(save_path, f'PCA_classification_report_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.txt'), 'w') as f:
                f.write("SUBJECT-LEVEL METRICS\n")
                f.write("=====================\n")
                f.write(f'Test Accuracy: {subject_metrics["accuracy"]}\n')
                f.write(f'F1 Score: {subject_metrics["f1"]}\n')
                f.write(f'Precision: {subject_metrics["precision"]}\n')
                f.write(f'Recall: {subject_metrics["recall"]}\n')
                f.write(f'AUC: {subject_metrics["auc"]}\n\n')
                
                f.write(f'Number of test subjects: {subject_metrics["num_subjects"]}\n')
                f.write(f'Number of positive subjects: {subject_metrics["num_positive"]}\n')
                f.write(f'Number of negative subjects: {subject_metrics["num_negative"]}\n\n')
                
                f.write("CHUNK-LEVEL METRICS (for reference)\n")
                f.write("==================================\n")
                f.write(f'Test Accuracy: {chunk_accuracy}\n')
                f.write(f'F1 Score: {chunk_f1}\n')
                f.write(f'Precision: {chunk_precision}\n')
                f.write(f'Recall: {chunk_recall}\n')
                f.write(f'AUC: {chunk_auc}\n\n')
                f.write(chunk_report)

            # Write the test results to a file (chunk level)
            write_test_results(y_test, y_pred, test_path_full, save_path)

            # Save the subject-level results
            subject_metrics["subject_summary"].to_csv(os.path.join(save_path, f'subject_level_results_{num_features}.csv'), index=False)

            # Save the confusion matrices
            # Chunk-level confusion matrix
            plot_confusion_matrix(chunk_cm, classes=[0, 1], 
                                save_path=os.path.join(save_path, f'chunk_confusion_matrix_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'),
                                title='Chunk-Level Confusion Matrix')
            
            # Subject-level confusion matrix
            plot_confusion_matrix(subject_metrics["confusion_matrix"], classes=[0, 1], 
                                save_path=os.path.join(save_path, f'subject_confusion_matrix_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'),
                                title='Subject-Level Confusion Matrix')

            # Feature importance analysis
            feature_importances = rf_classifier.feature_importances_
            sorted_idx = np.argsort(feature_importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(feature_importances)), feature_importances[sorted_idx], align='center')
            plt.xticks(range(len(feature_importances)), np.array(selected_features_names)[sorted_idx], rotation=90)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance Analysis - {num_features} Features')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'feature_importance_{num_features}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png'))
            plt.close()

            # Collect results for both chunk and subject level
            fold_results[num_features] = {
                'chunk': {
                    'accuracy': chunk_accuracy,
                    'f1': chunk_f1,
                    'precision': chunk_precision,
                    'recall': chunk_recall,
                    'auc': chunk_auc
                },
                'subject': {
                    'accuracy': subject_metrics["accuracy"],
                    'f1': subject_metrics["f1"],
                    'precision': subject_metrics["precision"],
                    'recall': subject_metrics["recall"],
                    'auc': subject_metrics["auc"]
                }
            }

        summary_results[fold] = fold_results

    # Generate summary tables for both chunk and subject level
    # Chunk-level summary
    chunk_summary_table = pd.DataFrame(columns=['Fold'] + config.NUMBER_FEATURES)
    for fold in config.Fold:
        fold_data = summary_results[fold]
        row = {'Fold': fold}
        for num_features in config.NUMBER_FEATURES:
            row[num_features] = fold_data[num_features]['chunk']['accuracy']
        chunk_summary_table = pd.concat([chunk_summary_table, pd.DataFrame([row])], ignore_index=True)

    # Subject-level summary
    subject_summary_table = pd.DataFrame(columns=['Fold'] + config.NUMBER_FEATURES)
    for fold in config.Fold:
        fold_data = summary_results[fold]
        row = {'Fold': fold}
        for num_features in config.NUMBER_FEATURES:
            row[num_features] = fold_data[num_features]['subject']['accuracy']
        subject_summary_table = pd.concat([subject_summary_table, pd.DataFrame([row])], ignore_index=True)

    # Save summary tables
    chunk_summary_save_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, 'chunk_summary_results.csv')
    chunk_summary_table.to_csv(chunk_summary_save_path, index=False)
    print(f'Chunk-level summary results saved to {chunk_summary_save_path}')
    
    subject_summary_save_path = os.path.join(config.MAIN_SAVE_DIR, config.DATA_TYPE, 'subject_summary_results.csv')
    subject_summary_table.to_csv(subject_summary_save_path, index=False)
    print(f'Subject-level summary results saved to {subject_summary_save_path}')

if __name__ == "__main__":
    main()