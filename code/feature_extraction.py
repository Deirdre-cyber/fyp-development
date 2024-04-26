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

def extract_features(directory, feature_type):
    """ Extract the features from the data. """

    output_dir = os.path.join(directory, 'features')
    output_dir = os.path.normpath(output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_dir = os.path.normpath(output_dir)

    logging.info("Loading the data")

    file_paths, file_labels = load_data_from_dir(directory)
    features_list = []

    for file_path, file_label in zip(file_paths, file_labels):
        if feature_type == 'wav':
            y, sr = librosa.load(file_path)
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        elif feature_type == 'lms':
            mel_spectrogram = plt.imread(file_path)
            spec_array = np.array(mel_spectrogram)
            spec_gray = np.mean(spec_array, axis=2)
            spec_norm = spec_gray / 255.0
            spec_db = librosa.power_to_db(spec_norm)
            features = librosa.feature.mfcc(S=spec_db, n_mfcc=40)

        # pad or truncate features if not of length 775 to mitigate error of differnet shapes
        if features.shape[1] < 775:
            features = np.pad(features, ((0, 0), (0, 775 - features.shape[1])), mode='constant')
        elif features.shape[1] > 775:
            features = features[:, :775]
        
        label = os.path.basename(file_path).split('_')[0]

        output_file_path = os.path.join(output_dir, "{}.npy".format(os.path.splitext(file_name)[0]))
        
        np.save(output_file_path, features)
        features_list.append((features, file_labels))

    return features_list
