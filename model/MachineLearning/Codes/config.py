# config_psd.py

# Constants
TARGET_SAMPLE_RATE = 16000  # Target sample rate for resampling
CHUNK_LENGTH_S = 4  # Chunk length in seconds
CHUNK_LENGTH = TARGET_SAMPLE_RATE * CHUNK_LENGTH_S  # Convert seconds to samples


# Data type and paths
DATA_TYPE = 'rp'  # Options: 'deep', 'reg', 'fimo'
DATA_TYPES = ['reg', 'fimo', 'rp']
MAIN_DATA_DIR = '/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/MachineLearning/Data_PreProcessing'
MAIN_SAVE_DIR = '/home/b/bhavyareddyseerapu/B2AI_Project4-main/model/features'

# Feature extraction
FEATURE = ['mfcc','wavelet','zcr','spectral_centroid','spectral_bandwidth','spectral_rolloff','spectral_flatness','spectral_contrast']  # Options: 'mfcc', 'psd', 'mel', 'wavelet'

# Model type and parameters
MODEL_TYPE = 'RandomForest'

#Feature Selection
# Feature_Selection = 'CFS'
thredhold = [0.5,0.6,0.7,0.8,0.9]
#thredhold = [0.8]
#cross validation
fold = 1
Fold = [1,2,3,4,5]

#PCA Parameters
Feature_Selection = 'PCA'
MAX_FEATURES = 200
NUMBER_FEATURES = [3,10,20,30,40,50,60,70,80,90,100,110,120,130]
# BestFeature = 130
# NUMBER_FEATURES = [130]
PCA_component = 0.95
Standard = False

#Oversampling
OVERSAMPLING = 'SMOTE'

#Finetuning
GridSearch = False 


# Output file naming
WAYS = f'_{FEATURE}_std{Standard}'
# RandomForest parameter grid for GridSearchCV
PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_features': [None, 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
