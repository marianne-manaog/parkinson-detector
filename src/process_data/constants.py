"""
Constants to aid data preparation.
"""

CSV_FMT_STR = '.csv'
DATA_DIR_STR = 'data'

NEW_COL_NAMES = ['subject_id', 'jitter_percent',
                 'jitter_abs', 'rap', 'ppq', 'apq_3', 'apq_5', 'apq_11']
PROCESSED_SUFFIX = '_processed'

# Random state to get n samples from the majority class,
# where n is the number of samples in the minority class
RANDOM_STATE = 0

# Z-score threshold to identify and remove outliers
Z_SCORE_THRESH = 3
