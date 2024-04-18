""" File to convert audio signal to spectrogram """
import os
import shutil
import logging
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt

from data_loading import load_data_from_all, load_data_from_dir
import config

logging.basicConfig(filename="example.log", level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_spectrogram(audio_file_directory):
    """ Convert the audio files to mel spectrograms and save them to the output directory. """  

    output_dir = os.path.join(config.RAW_DIR_PATH, 'spectrograms')
    output_dir = os.path.normpath(output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_dir = os.path.normpath(output_dir)

    logging.info("Loading the data")

    audio_files = [wav_file for wav_file in os.listdir(audio_file_directory) if wav_file.endswith('.wav')]

    logging.info(
        "Computing mel spectrogram for each audio file in the raw directory and saving to %s", output_dir)

    for wav_file in audio_files:
        # logger.info("Processing file: %s", wav_file) # debug
        file_path = os.path.join(audio_file_directory, wav_file)
        logging.info(file_path)
        
        # """
        # n_fft : length of the FFT window (256 - 1024)
        # hop_length : number of samples between successive frames (64 - 256)
        # n_mels : number of mel bands to generate (20 - 128)
        # fmin : min (20hz)
        # fmax : maximum frequency
        # """
        try:
            audio, sr = lr.load(file_path, sr=None)
            mel_spectrogram = lr.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, fmax=sr / 2.0)
            mel_spectrogram = lr.power_to_db(mel_spectrogram, ref=np.max)

            file_name = os.path.basename(file_path)

            output_file_path = os.path.join(output_dir, file_name[:-4] + '.png')
            plt.figure(figsize=(10, 6))
            plt.imshow(mel_spectrogram, cmap='plasma', origin='lower') # viridis, plasmas, inferno, magma, or cividis
            plt.axis("off")
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        except Exception as e:
            logging.error("Error processing file %s",wav_file)

    logging.info("Total of %d files processed", len(audio_files))

    return output_dir
