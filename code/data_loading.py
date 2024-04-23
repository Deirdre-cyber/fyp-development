import os
import shutil
import logging

import config

logging.basicConfig(filename="example.log", level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = config.PROCESSED_DIR_PATH

def load_data_from_all(dir_path):
    """ Load all data from the directory and its subdirectories into Python. """
    logging.info('Loading all data from %s', dir_path)
    file_paths = []
    labels = []

    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            try:
                file_path = os.path.join(root, file_name)
                label = os.path.basename(root)
                file_paths.append(file_path)
                labels.append(label)
            except Exception as e:
                logging.error("Error loading data from file %s: %s", file_path, e)

    logging.info('All data loaded from %s', dir_path)
    return file_paths, labels

def load_data_from_dir(dir_path):
    """ Load data only from the specified directory into Python. """
    logging.info('Loading data from %s', dir_path)
    file_paths = []
    labels = []

    try:
        files = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)]
        files = [file_path for file_path in files if os.path.isfile(file_path)]
        file_paths.extend(files)
        labels.extend([os.path.basename(dir_path)] * len(files))
    except Exception as e:
        logging.error("Error loading data from directory %s: %s", dir_path, e)

    logging.info('Data loaded from %s', dir_path)
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

    directory = os.path.normpath(directory)
    logging.info('Moving files to %s', directory)

    for file_path, label in zip(file_paths, labels):
        destination = os.path.join(directory, label)
        os.makedirs(destination, exist_ok=True)
        destination_file = os.path.join(destination, os.path.basename(file_path))
        shutil.copy(file_path, destination_file)
    logging.info('Files moved to %s', directory)
