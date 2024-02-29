""" File to convert audio signal to spectrogram """
import os
import logging
import numpy as np
import librosa as lr
import soundfile as sf
import matplotlib.pyplot as plt
import config
from data_loading import load_data

logging.basicConfig(filename="example.log", level=logging.INFO)
logger = logging.getLogger(__name__)

output_dir = os.path.join(config.TRAIN_DIR, 'spectrograms')
output_dir = os.path.normpath(output_dir)

# Load the data
logging.info("Loading the data")
training_path, training_labels = load_data(config.TRAIN_DIR)

# compute mel spectrogram for all loaded files
logging.info("Computing mel spectrogram")
for file_path, label in zip(training_path, training_labels):
    file_path = os.path.normpath(file_path)
    audio, sr = lr.load(file_path, sr=None)

    # decide on appropriate number of mels
    mel_spectrogram = lr.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spectrogram = lr.power_to_db(mel_spectrogram, ref=np.max)

    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, file_name)
    #sf.write(output_file_path, mel_spectrogram, sr, format='wav')
    logging.info("Spectrogram saved to %s", output_file_path)

    # Display the mel spectrogram - debugging
    plt.figure(figsize=(10, 4))
    lr.display.specshow(mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram of the first audio file')
    plt.show()