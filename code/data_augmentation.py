""" Augment the data and preprocess it for training """
import os
import random
import shutil
import logging

import numpy as np
import librosa as lr
import soundfile as sf
import matplotlib.pyplot as plt
import SpecAugment

from SpecAugment import spec_augment_tensorflow # issue with import

import config
from data_loading import load_data

logging.basicConfig(filename="example.log", level=logging.INFO)
logger = logging.getLogger(__name__)

def augment_audio(audio, technique):
    """ Augment the audio data using the specified technique. """
    sr = config.SAMPLE_RATE
    if technique == 'noise':
        return inject_noise(audio, 0.005)
    if technique == 'time':
        return shift_time(audio, sr, 0.5, 'right')
    if technique == 'pitch':
        return lr.effects.pitch_shift(audio, sr=sr, n_steps=2)
    if technique == 'stretch':
        return lr.effects.time_stretch(audio, rate=0.8)
    if technique == 'compress':
        return lr.effects.time_stretch(audio, rate=1.2)
    return audio

def inject_noise(audio_data, noise_factor):
    """ Inject random noise into the audio data. """
    noise = np.random.randn(len(audio_data))
    augmented_data = audio_data + noise_factor * noise
    augmented_data = augmented_data.astype(type(audio_data[0]))
    return augmented_data

def shift_time(data, sampling_rate, shift_max, shift_direction):
    """ Shift the audio data by a random amount. """
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
    augmented_data = np.roll(data, shift)
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def spec_augment_files(spectrogram):
    """ Apply SpecAugment to the audio data. """
    augmented_data = spec_augment_tensorflow.spec_augment(mel_spectrogram=spectrogram)
    return augmented_data

def select_random_files(file_paths, file_labels, percentage=0.65):
    """ Randomly select a percentage of WAV files and their corresponding labels. """
    logger.info("Selecting %s percent of the data", percentage * 100)
    num_idx_to_select = int(len(file_paths) * percentage)

    valid_indices = [i for i, path in enumerate(file_paths) if "augmented" not in path]
    selected_indices = random.sample(valid_indices, min(num_idx_to_select, len(valid_indices)))

    selected_file_paths = [os.path.relpath(file_paths[i]) for i in selected_indices]
    selected_labels = [file_labels[i] for i in selected_indices]
    logger.info("Selected %s files and labels", len(selected_file_paths))

    return selected_file_paths, selected_labels

def load_wav_files(file_paths, file_labels):
    """ Load the WAV files and their corresponding labels. """
    logger.info("Loading WAV files")
    all_wav_files = []

    try:
        for file_path in file_paths:
            #logger.info("Loading WAV file: %s", file_path)
            audio, _ = lr.load(file_path, sr=config.SAMPLE_RATE)
            all_wav_files.append(audio)
    except FileNotFoundError as e:
        logger.error("Error loading WAV file: %s. Error: %s", file_path, e)
    except IOError as e:
        logger.error("IOError when loading WAV file: %s. Error: %s", file_path, e)

    return all_wav_files, file_labels

def augment_data(wav_files, labels):
    """ Augment the data using one of 5 techniques each time. """
    augmented_audio_files = []
    technique_count = {'noise': 0, 'time': 0, 'pitch': 0, 'stretch': 0, 'compress': 0}


    for audio in wav_files:
        technique = random.choice(['noise', 'time', 'pitch', 'stretch', 'compress']) # to test 15 vs 65%
        # count the number of times each technique is used
        technique_count[technique] += 1
        #logging.info("Augmenting audio using %s technique", technique)
        augmented_audio = augment_audio(audio, technique)
        augmented_audio_files.append(augmented_audio)
    logger.info("Number of times each technique was used: %s", technique_count)
    return augmented_audio_files, labels

# update augment to augment spectrogram files also
def augment_wav_data_pipeline(directory):
    """ Augment the data and save it to a new directory. """
    output_dir = os.path.join(directory, 'augmented')
    output_dir = os.path.normpath(output_dir)
    #techniques = ['noise', 'time', 'pitch', 'stretch', 'compress']

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    output_dir = os.path.normpath(output_dir)

    training_path, training_labels = load_data(directory)

    #for technique in techniques:
    random_15_path, random_15_label = select_random_files(training_path, training_labels)
    random_wav_files, random_labels = load_wav_files(random_15_path, random_15_label)
    augmented_wav_files, _ = augment_data(random_wav_files, random_labels)
    logger.info("number of augmented files: %s", len(augmented_wav_files))

    for idx, wav in enumerate(augmented_wav_files):
        random_file_path = random_15_path[idx]
        file_name = os.path.basename(random_file_path)
        output_file_path = os.path.join(output_dir, file_name)
        sf.write(output_file_path, wav, config.SAMPLE_RATE)

    logger.info("Augmented data saved to: %s", output_dir)
    logger.info("Length of augmented data: %s", len(os.listdir(output_dir)))


def augment_spectrogram_data_pipeline(directory):
    """
    Augment spectrograms in a directory using spec_augment_tensorflow. """

    augmentation_probability = 0.65

    logger.info("Augmenting spectrograms in directory: %s", directory)
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            mel_spectrogram = plt.imread(file_path)

            if np.random.rand() < augmentation_probability:
                warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram=mel_spectrogram)

                output_file_path = os.path.join(root, filename)
                plt.imsave(output_file_path, warped_masked_spectrogram)

                logger.info("Augmented spectrogram saved: %s", output_file_path)
    logger.info("Spectrogram augmentation complete.")
