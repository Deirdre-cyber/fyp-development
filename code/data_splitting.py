""" Split the data into train, validation and test sets. """
import config
import logging

from sklearn.model_selection import train_test_split
from data_loading import load_data
from data_loading import create_dir
from data_loading import move_files


logging.basicConfig(filename="example.log", level=logging.INFO)
logger = logging.getLogger(__name__)


RAW_DIR = config.ROOT_DIR_PATH

file_paths, labels = load_data(RAW_DIR)
train_dir, val_dir, test_dir = create_dir()

X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    file_paths, labels, test_size=0.15, stratify=labels, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.15, stratify=y_train_temp, random_state=42)

move_files(X_train, y_train, train_dir)
move_files(X_val, y_val, val_dir)
move_files(X_test, y_test, test_dir)
logging.info('Data splitting complete.')
