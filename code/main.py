import os
import sys
import subprocess
import shutil
import logging
import data_loading
import config

from sklearn.model_selection import train_test_split
from data_loading import load_data_from_all, load_data_from_dir, create_dir, move_files
from data_splitting import split_data
from data_augmentation import augment_wav_data_pipeline,augment_spectrogram_data_pipeline
from spectrogram_conversion import convert_to_spectrogram
from feature_extraction import extract_features_wav, extract_features_lms

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
    try:
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("Error occurred while installing dependencies.")

def main():
    """ Main function """


    #convert_to_spectrogram(WAV_DIR)
    #print("Spectrogram conversion complete.") # for debugging
    
    #train_dir, val_dir, test_dir = create_dir()
    #print("Directories created.", train_dir, val_dir, test_dir) # for debugging
    
    #audio_file_paths, audio_file_labels = load_data_from_all(RAW_DIR)

    #split_data(audio_file_paths, audio_file_labels, train_dir, val_dir, test_dir)
    #print("Data splitting complete.") # for debugging

    #augment_wav_data_pipeline(WAV_TRAIN_DIR)
    #print("Wav augmentation complete.") # for debugging

    #augment_spectrogram_data_pipeline(LMS_TRAIN_DIR)
    #print("Spectrogram augmentation complete.") # for debugging

    extract_features_wav(WAV_TRAIN_DIR)
    extract_features_wav(AUGMENTED_WAV_DIR)
    extract_features_wav(VAL_WAV_DIR)
    extract_features_wav(TEST_WAV_DIR)
    print("Wav feature extraction complete.") # for debugging

    extract_features_lms(LMS_TRAIN_DIR)
    extract_features_lms(AUGMENTED_LMS_DIR)
    extract_features_lms(VAL_LMS_DIR)
    extract_features_lms(TEST_LMS_DIR)
    print("LMS Feature extraction complete.") # for debugging

    # train model using embeddings...

if __name__ == "__main__":
    #install_dependencies()
    main()
