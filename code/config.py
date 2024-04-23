""" Configuration file for the project."""
WAV_DIR_PATH = '../data/raw/waveforms/'
RAW_DIR_PATH = '../data/raw/'

PROCESSED_DIR_PATH = '../data/processed/'
TRAIN_DIR_PATH = '../data/processed/train/'
WAV_TRAIN_DIR_PATH = '../data/processed/train/waveforms/'
AUGMENTED_WAV_DIR_PATH = '../data/processed/train/augmented_waveforms'

LMS_TRAIN_DIR_PATH = '../data/processed/train/spectrograms/'
AUGMENTED_LMS_DIR_PATH = '../data/processed/train/augmented_spectrograms'

SAMPLE_RATE = 48000

VAL_WAV_DIR_PATH = '../data/processed/val/waveforms'
TEST_WAV_DIR_PATH = '../data/processed/test/waveforms/'

VAL_LMS_DIR_PATH = '../data/processed/val/spectrograms/'
TEST_LMS_DIR_PATH = '../data/processed/test/spectrograms/'

TRAIN_WAV_MFCCS = '../data/processed/train/waveforms/features/'
TRAIN_LMS_MFCCS = '../data/processed/train/spectrograms/features/'
TRAIN_AUG_WAV_MFCCS = '../data/processed/train/augmented_waveforms/features/'
TRAIN_AUG_LMS_MFCCS = '../data/processed/train/augmented_spectrograms/features/'

TEST_WAV_MFCCS = '../data/processed/test/waveforms/features/'
TEST_LMS_MFCCS = '../data/processed/test/spectrograms/features/'

VAL_WAV_MFCCS = '../data/processed/val/waveforms/features/'
VAL_LMS_MFCCS = '../data/processed/val/spectrograms/features/'
