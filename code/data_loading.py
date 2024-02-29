import os
import shutil
import logging

import config

logging.basicConfig(filename="example.log", level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = config.PROCESSED_DIR

def load_data(dir_path):
    """ Load the data from the directory into python. """
    logging.info('Loading data from %s', dir_path)
    subdirectories = [file_name.path for file_name in os.scandir(dir_path) if file_name.is_dir()]
    file_paths = []
    labels = []

    for subdir in subdirectories:
        label = os.path.basename(subdir)
        files = [os.path.join(subdir, file_name) for file_name in os.listdir(
            subdir) if file_name.endswith('.wav')]
        file_paths.extend(files)
        labels.extend([label] * len(files))
    return file_paths, labels

def create_dir():
    """ Create the directories for the processed data. """
    logging.info('Creating directories for processed data')
    train_dir = os.path.join(PROCESSED_DIR, 'train')
    val_dir = os.path.join(PROCESSED_DIR, 'val')
    test_dir = os.path.join(PROCESSED_DIR, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return train_dir, val_dir, test_dir

def move_files(file_paths, labels, directory):
    """ Move the files to the respective directories. """
    logging.info('Moving files to %s', directory)
    for file_paths, labels in zip(file_paths, labels):
        destination = os.path.join(directory, labels)
        os.makedirs(destination, exist_ok=True)
        shutil.copy(file_paths, os.path.join(destination, os.path.basename(file_paths)))