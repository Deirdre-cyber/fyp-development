""" Extract the features from the data. """
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import shutil
import logging
import matplotlib.pyplot as plt

import librosa
import resampy
import numpy as np

import config
from data_loading import load_data_from_dir

logging.basicConfig(filename="example.log", level=logging.INFO)
logger = logging.getLogger(__name__)

SR = config.SAMPLE_RATE

def extract_features_wav(directory):
    """ Extract the features from the data. """
    
    output_dir = os.path.join(directory, 'features')
    output_dir = os.path.normpath(output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_dir = os.path.normpath(output_dir)

    logging.info("Loading the data")

    file_paths, file_labels = load_data_from_dir(directory)
    mfccs_list = []

    for file_path,file_labels in zip(file_paths, file_labels):
        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # pad or truncate mfccs if not of length 775
        if mfccs.shape[1] < 775:
            mfccs = np.pad(mfccs, ((0, 0), (0, 775 - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > 775:
            mfccs = mfccs[:, :775]
        
        file_name = os.path.basename(file_path)
        label = file_name.split('_')[0]
        output_file_path = os.path.join(output_dir, "{}.npy".format(os.path.splitext(file_name)[0]))

        
        np.save(output_file_path, mfccs)
        mfccs_list.append((mfccs, file_labels))

    return mfccs_list

def extract_features_lms(directory):
    """ Extract the features from the data. """

    output_dir = os.path.join(directory, 'features')
    output_dir = os.path.normpath(output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_dir = os.path.normpath(output_dir)

    logging.info("Loading the data")

    file_paths, file_labels = load_data_from_dir(directory)
    mfccs_list = []

    for file_path, file_labels in zip(file_paths, file_labels):
        mel_spectrogram = plt.imread(file_path)
        spec_array = np.array(mel_spectrogram)
        spec_gray = np.mean(spec_array, axis=2)
        spec_norm = spec_gray / 255.0
        spec_db = librosa.power_to_db(spec_norm)
        mfccs = librosa.feature.mfcc(S=spec_db, n_mfcc=40)

        # pad or truncate mfccs if not of length 775
        if mfccs.shape[1] < 775:
            mfccs = np.pad(mfccs, ((0, 0), (0, 775 - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > 775:
            mfccs = mfccs[:, :775]
        
        file_name = os.path.basename(file_path)
        label = file_name.split('_')[0]
        output_file_path = os.path.join(output_dir, "{}.npy".format(os.path.splitext(file_name)[0]))
        
        np.save(output_file_path, mfccs)
        mfccs_list.append((mfccs, file_labels))

    return mfccs_list
