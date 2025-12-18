
import os

# ================================
# API CONFIGURATION
# ================================
API_KEY = 'MmI5OWMyOWRhMmFhNGFkNTljNDhjNTEzMTIwZWRjNzM6NTlhYjI3ZDdiQzM1NEFiMkI5MjNmRTcwOUM0QUYwZEQ='
API_VERSION = '2024-10-21'
RESOURCE_ENDPOINT = 'https://unified-api.ucsf.edu/general'
JUDGE_MODEL = 'gpt-5-mini-2025-08-07'

# ================================
# ANALYSIS CONFIGURATION
# ================================
TEMPERATURE = 1.0 
RANDOM_SEED = 42
CONCURRENCY = 128
RATE_DELAY = 0.1
MAX_RETRIES = 5
SUBSAMPLE_SIZE = 5  # Set to integer for testing, None for full run

# ================================
# DATA PATHS
# ================================
# Dictionary mapping strategy names to directory paths
# Assumes these directories are relative to the notebook/script location or absolute
STRATEGIES = {
    'strategy_one': 'strategy_one_results',
    'strategy_two': 'strategy_two_results',
    'strategy_three': 'strategy_three_results',
    'strategy_four': 'strategy_four_results'
}

# ================================
# OUTPUT DIRECTORY
# ================================
OUTPUT_DIR = './reasoning_analysis_results'
# Sub-directories for organized results
LLM_JUDGE_DIR = os.path.join(OUTPUT_DIR, 'llm_judge_analysis')
TRADITIONAL_DIR = os.path.join(OUTPUT_DIR, 'traditional_analysis')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LLM_JUDGE_DIR, exist_ok=True)
os.makedirs(TRADITIONAL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ================================
# VISUALIZATION CONFIGURATION
# ================================
FIG_SIZE_SINGLE = (10, 6)
FIG_SIZE_MULTI = (16, 12)
FONT_SCALE = 1.1
DPI = 300
