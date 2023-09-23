"""
Constants to aid data preparation.
"""

CSV_FMT_STR = '.csv'
DATA_DIR_STR = 'data'

NEW_COL_NAMES = ['subject_id', 'jitter_percent',
                 'jitter_abs', 'rap', 'ppq', 'apq_3', 'apq_5', 'apq_11']
PROCESSED_SUFFIX = '_processed'

# Z-score threshold to identify and remove outliers
Z_SCORE_THRESH = 3
