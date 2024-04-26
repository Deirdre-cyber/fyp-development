import os
import sys
import subprocess
import shutil
import logging
import data_loading
import config

from data_loading import load_data_from_all, load_data_from_dir, create_dir, move_files
from data_splitting import split_data
from data_augmentation import augment_wav_data_pipeline,augment_spectrogram_data_pipeline
from spectrogram_conversion import convert_to_spectrogram
from feature_extraction import extract_features

logging.basicConfig(filename="example.log", level=logging.INFO)
logger = logging.getLogger(__name__)

WAV_DIR = config.WAV_DIR_PATH
RAW_DIR = config.RAW_DIR_PATH
WAV_TRAIN_DIR = config.WAV_TRAIN_DIR_PATH
LMS_TRAIN_DIR = config.LMS_TRAIN_DIR_PATH
AUGMENTED_WAV_DIR = config.AUGMENTED_WAV_DIR_PATH
AUGMENTED_LMS_DIR = config.AUGMENTED_LMS_DIR_PATH

TEST_WAV_DIR = config.TEST_WAV_DIR_PATH
VAL_WAV_DIR = config.VAL_WAV_DIR_PATH

TEST_LMS_DIR = config.TEST_LMS_DIR_PATH
VAL_LMS_DIR = config.VAL_LMS_DIR_PATH

def install_dependencies():
    """ Install dependencies from requirements.txt file """
    try:
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("Error occurred while installing dependencies.")

def main():
    """ Main function """

    # if os.path.exists(config.PROCESSED_DIR_PATH):
    #     shutil.rmtree(config.PROCESSED_DIR_PATH)
    # os.makedirs(config.PROCESSED_DIR_PATH)

   # convert_to_spectrogram(WAV_DIR)
    #print("Spectrogram conversion complete.") # for debugging
    
    # train_dir, val_dir, test_dir = create_dir()
    # print("Directories created.", train_dir, val_dir, test_dir) # for debugging
    
    # audio_file_paths, audio_file_labels = load_data_from_all(RAW_DIR)

    # split_data(audio_file_paths, audio_file_labels, train_dir, val_dir, test_dir)
    # print("Data splitting complete.") # for debugging

    augment_wav_data_pipeline(WAV_TRAIN_DIR)
    print("Wav augmentation complete.") # for debugging

    augment_spectrogram_data_pipeline(LMS_TRAIN_DIR)
    print("Spectrogram augmentation complete.") # for debugging

    for wav_directory in [WAV_TRAIN_DIR, AUGMENTED_WAV_DIR, VAL_WAV_DIR, TEST_WAV_DIR]:
        extract_features(wav_directory, "wav")
    print("Wav feature extraction complete.") # for debugging

    for lms_directory in [LMS_TRAIN_DIR, AUGMENTED_LMS_DIR, VAL_LMS_DIR, TEST_LMS_DIR]:
        extract_features(lms_directory, "lms")
    print("LMS Feature extraction complete.") # for debugging


if __name__ == "__main__":
    #install_dependencies()
    main()
