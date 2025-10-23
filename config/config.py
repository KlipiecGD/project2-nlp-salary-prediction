RANDOM_SEED = 42
FILEPATH = "data/Train_rev1.csv"
MODELS_DIR = "trained_models/"
PREPROCESSORS_DIR = "fitted_preprocessors/"
TFIDF_FEATURES_DIR = "tfidf_features/"
EMBEDDINGS_DIR = "embeddings/"
REPORTS_DIR = "reports/"
PLOTS_DIR = "reports/figures/"

TEST_SIZE = 0.2
VALID_SIZE = 0.5  # proportion of validation in test split (i.e., 0.1 of total data if test_size=0.2)

TITLE_COLUMN = "Title"
DESC_COLUMN = "FullDescription"
TARGET_COLUMN = "SalaryNormalized"
CATEGORICAL_COLUMNS = ["LocationNormalized", "ContractType", "Category"]
HIGH_CARDINALITY_COLUMNS = ["Company", "LocationNormalized", "SourceName"]

NUM_WORKERS = 0
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
MAX_SEQ_LENGTH = 320
EMBEDDING_DIM = 300
DELTA = 0.001
SCHEDULER_PATIENCE = 0
SCHEDULER_FACTOR = 0.5
EARLY_STOPPING_PATIENCE = 2
DROPOUT_RATE = 0.3
CAT_HIDDEN_SIZE = 128
REG_HIDDEN_SIZE = 256
EMB_HIDDEN_SIZE = 256
EMBEDDING_DIM = 300
W2V_EMBEDDING_DIM = 256
RECURRENT_HIDDEN_SIZE = 128
NUM_FILTERS = 64
NUM_RESIDUAL_BLOCKS = 2 # 1,2 or 3
LOSS_FUNCTION = "mse"  # Options: 'mse', 'mae'
OPTIMIZER = "adam"  # Options: 'adam', 'sgd'

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L12-v2"
PRETRAINED_MODEL = "fasttext"


