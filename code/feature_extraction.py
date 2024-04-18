""" Extract the features from the data. """
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import shutil
import logging
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

import librosa
import resampy
import numpy as np

import config
from data_loading import load_data_from_dir


logging.basicConfig(filename="example.log", level=logging.INFO)
logger = logging.getLogger(__name__)

SR = config.SAMPLE_RATE


def extract_features_wav(directory):
    """ Extract the features from the wav files in the directory. """

    output_dir = os.path.join(config.AUGMENTED_WAV_DIR_PATH , 'embeddings')
    output_dir = os.path.normpath(output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_dir = os.path.normpath(output_dir)

    wav_paths, wav_labels = load_data_from_dir(directory)

    features = []

    for wav_path, _ in zip(wav_paths, wav_labels):

        wav_path = os.path.normpath(wav_path)
        
        logger.info("Extracting features from %s...", wav_path) # debug
        waveform, _ = librosa.load(wav_path, sr=SR)

        logger.info("Converting waveform to tensor")
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)

        logger.info("Reshaping waveform tensor to match VGGish input shape")
        waveform_tensor = tf.expand_dims(waveform_tensor, axis=0)

        logger.info("Extracting embeddings from waveform")
       # embeddings = vggish_slim.forward(waveform_tensor, SR)
        # features.append(embeddings)

        # output_file_path = os.path.join(output_dir, os.path.basename(wav_path))
        # np.save(output_file_path, embeddings)

        # logger.info("Finished extracting features from %s", wav_path)

    logger.info("Finished extracting features from all wav files")


def extract_features_lms(directory):
    """ Extract the features from the log-mel spectrograms in the directory. """

    output_dir = os.path.join(directory, 'embeddings')
    output_dir = os.path.normpath(output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_dir = os.path.normpath(output_dir)

    for filename in os.listdir(directory):
        if filename.endswith(".png"):

            lms_file = os.path.join(directory, filename)

            lms_file = os.path.normpath(lms_file)

            logger.info("Extracting features from %s...", lms_file)

            spectrogram_image = plt.imread(lms_file)

            logger.info("Extracting embeddings from spectrogram")

            embeddings = extract_features_vggish(spectrogram_image)
            
            # Store features to output directory
            output_file_path = os.path.join(output_dir, os.path.basename(lms_file))
            np.save(output_file_path, embeddings)
            
            logger.info("Finished extracting features from %s", lms_file)

def preprocess_mel_spectrogram(mel_spectrogram):
    
    resized_spectrogram = tf.image.resize(mel_spectrogram, (96, 64))
    
    normalised_spectrogram = resized_spectrogram / 255.0
    
    mel_spectrogram = tf.convert_to_tensor(normalised_spectrogram, dtype=tf.float32)

    logger.info("Preprocessed mel spectrogram size before transpose: %s", mel_spectrogram.shape)
    mel_spectrogram = tf.transpose(mel_spectrogram, perm=[1, 2])
    logger.info("Preprocessed mel spectrogram size before transpose: %s", mel_spectrogram.shape)
    
    return mel_spectrogram

def extract_features_vggish(mel_spectrogram):
    # Preprocess the mel spectrogram
    input_spec = preprocess_mel_spectrogram(mel_spectrogram)
    # Extract embeddings using VGGish model
    embeddings = model(input_spec)
    return embeddings
