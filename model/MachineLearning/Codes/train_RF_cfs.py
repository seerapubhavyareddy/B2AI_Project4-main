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
    y_train_full = y_train
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
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train_full)
    else:
        X_train_resampled, y_train_resampled = X_train_selected, y_train_full

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

    y_pred = rf_classifier.predict(X_test_selected)

    # Calculate metrics
    accuracy = accuracy_score(y_test_full, y_pred)
    f1 = f1_score(y_test_full, y_pred, average='weighted')
    precision = precision_score(y_test_full, y_pred, average='weighted')
    recall = recall_score(y_test_full, y_pred, average='weighted')
    auc = roc_auc_score(y_test_full, rf_classifier.predict_proba(X_test_selected)[:, 1]) if len(np.unique(y_test_full)) == 2 else 'N/A'

    # Save results
    model_filename = f'rf_model_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.pkl'
    joblib.dump(rf_classifier, os.path.join(save_base_path, model_filename))

    report_filename = f'classification_report_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.txt'
    with open(os.path.join(save_base_path, report_filename), 'w') as f:
        f.write(f'Test Accuracy: {accuracy}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'AUC: {auc}\n\n')
        f.write(classification_report(y_test_full, y_pred))

    write_test_results(y_test_full, y_pred, test_path_full, save_base_path)
    
    cm = confusion_matrix(y_test_full, y_pred)
    plot_confusion_matrix(
        cm, 
        classes=[0, 1], 
        save_path=os.path.join(save_base_path, 
                              f'confusion_matrix_{config.MAX_FEATURES}_features_{config.OVERSAMPLING}_GRIDSEARCH{config.GridSearch}.png')
    )

    return {
        'data_type': data_type,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
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