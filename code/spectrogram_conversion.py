""" File to convert audio signal to spectrogram """
import os
import logging
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt

from data_loading import load_data
import config

logging.basicConfig(filename="example.log", level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_spectrogram(directory):
    """ Convert the audio files to mel spectrograms and save them to the output directory. """
    output_dir = os.path.join(directory, 'spectrograms')
    output_dir = os.path.normpath(output_dir)

    logging.info("Loading the data")

    ignored_directories = ['augmented', 'spectrograms']  # ignore for now

    training_path = []
    training_labels = []

    for root, _, filenames in os.walk(config.TRAIN_DIR):
        if not any(directory in root for directory in ignored_directories):
            training_path.extend([os.path.join(root, filename)
                                 for filename in filenames])
            training_labels.extend([os.path.basename(root)] * len(filenames))

    logging.info(
        "Computing mel spectrogram for each audio file in the training set and saving to %s", output_dir)
    for file_path, _ in zip(training_path, training_labels):
        logger.info("Processing file: %s", file_path)
        file_path = os.path.normpath(file_path)
        audio, sr = lr.load(file_path, sr=None)

        # """
        # Decide on appropriate number of mels - Task FIN-53 -Research Hyperparameter tuning (Spectrogram)

        # n_fft : length of the FFT window
        # hop_length : number of samples between successive frames
        # n_mels : number of mel bands to generate
        # fmax : maximum frequency
        # """
        mel_spectrogram = lr.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, fmax=sr / 2.0)
        mel_spectrogram = lr.power_to_db(mel_spectrogram, ref=np.max)

        file_name = os.path.basename(file_path)

        output_file_path = os.path.join(output_dir, file_name[:-4] + '.png')
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spectrogram, cmap='viridis', origin='lower')
        plt.axis("off")
        plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    logging.info("Total of %d files processed", len(training_path))

    return output_dir
