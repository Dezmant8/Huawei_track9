"""Project-wide configuration: paths, hyperparameters, and constants."""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
BENCHMARK_DIR = os.path.join(BASE_DIR, "data", "benchmark")
BENCHMARK_IMAGES_DIR = os.path.join(BENCHMARK_DIR, "images")
MODEL_DIR = os.path.join(BASE_DIR, "models", "rotation_classifier")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TRIVIA_DIR = os.path.join(BASE_DIR, "TRivia-main")

# Reproducibility
SEED = 42

# SynthTabNet dataset
PARTS = ["fintabnet", "marketing", "pubtabnet", "sparse"]
SAMPLES_PER_PART = 250
ANGLES = [0, 90, 180, 270]
NUM_ROTATION_CLASSES = len(ANGLES)

DATASET_URLS = {
    "fintabnet": "https://ds4sd-public-artifacts.s3.eu-de.cloud-object-storage.appdomain.cloud/datasets/synthtabnet_public/v2.0.0/fintabnet.zip",
    "marketing": "https://ds4sd-public-artifacts.s3.eu-de.cloud-object-storage.appdomain.cloud/datasets/synthtabnet_public/v2.0.0/marketing.zip",
    "pubtabnet": "https://ds4sd-public-artifacts.s3.eu-de.cloud-object-storage.appdomain.cloud/datasets/synthtabnet_public/v2.0.0/pubtabnet.zip",
    "sparse": "https://ds4sd-public-artifacts.s3.eu-de.cloud-object-storage.appdomain.cloud/datasets/synthtabnet_public/v2.0.0/sparse.zip",
}

# VLM inference
MIN_PIXELS = 256 * 28 * 28   # ~448x448, Qwen2.5-VL minimum
MAX_PIXELS = 512 * 28 * 28   # ~634x634, conservative for MPS memory
MAX_NEW_TOKENS = 4096

TRIVIA_SYSTEM_PROMPT = (
    "You are an AI specialized in recognizing and extracting table from images. "
    "Your mission is to analyze the table image and generate the result in OTSL format "
    "using specified tags. Output only the results without any other words and explanation."
)

QWEN_TABLE_PROMPT = (
    "Extract the table from this image and output it as an HTML table using "
    "<table>, <tr>, <td> tags with rowspan and colspan attributes where needed. "
    "Output only the HTML table, nothing else."
)

GENERATION_PARAMS = {
    "trivia": {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "do_sample": False,
        "repetition_penalty": 1.05,
    },
    "qwen": {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
        "do_sample": False,
    },
}

# Rotation classifier
CLASSIFIER_ARCH = "resnet18"
CLASSIFIER_LR = 1e-3
CLASSIFIER_EPOCHS = 10
CLASSIFIER_BATCH_SIZE = 64
CLASSIFIER_PATIENCE = 3
CLASSIFIER_RESIZE = 256
CLASSIFIER_CROP = 224
CLASSIFIER_TRAIN_PER_PART = 5000
CLASSIFIER_VAL_PER_PART = 500
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Metrics and analysis
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95
SIGNIFICANCE_ALPHA = 0.05
