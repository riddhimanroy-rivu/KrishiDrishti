"""
KrishiDrishti Configuration Module
Defines all global paths, hyperparameters, dataset configs, and styling constants.
"""

from pathlib import Path
import os

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Cache and environment configuration for reliable execution
os.environ["MPLCONFIGDIR"] = str(BASE_DIR / ".cache" / "matplotlib")
os.environ["KERAS_HOME"] = str(BASE_DIR / ".cache" / "keras")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Core Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
PREPROCESS_CONFIG_PATH = MODELS_DIR / "preprocessing_config.json"

OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"
METRICS_DIR = OUTPUTS_DIR / "metrics"

ASSETS_DIR = BASE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"

# Create directories if they do not exist
for dir_path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    METRICS_DIR,
    ASSETS_DIR,
    IMAGES_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Image & Model Specifications
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
RANDOM_SEED = 42

# Training Hyperparameters
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
MIN_LR = 1e-7

# Train / Validation / Test Splitting Ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Supported file extensions for validation
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Standard PlantVillage Tomato classes for reference & fallback
DEFAULT_TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

# Project Branding
PROJECT_NAME = "KrishiDrishti"
TAGLINE = "Smarter Farming Through Artificial Intelligence"
ACADEMIC_SUBJECT = "Neural Network & Deep Learning (PCCCS582)"
UN_SDG_TARGET = "UN SDG 2 - Zero Hunger (Target 2.4: Sustainable Agriculture & Resilient Practices)"
