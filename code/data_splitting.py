""" Split the data into train, validation and test sets. """
import os
from sklearn.model_selection import train_test_split
import shutil


RAW_DIR = '../data/raw'
PROCESSED_DIR = '../data/processed'

train_dir = os.path.join(PROCESSED_DIR, 'train')
val_dir = os.path.join(PROCESSED_DIR, 'val')
test_dir = os.path.join(PROCESSED_DIR, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

subdirectories = [f.path for f in os.scandir(RAW_DIR) if f.is_dir()]

file_paths = []
labels = []

for subdir in subdirectories:
    label = os.path.basename(subdir)
    files = [os.path.join(subdir, f) for f in os.listdir(
        subdir) if f.endswith('.wav')]
    file_paths.extend(files)
    labels.extend([label] * len(files))

X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    file_paths, labels, test_size=0.15, stratify=labels, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.15, stratify=y_train_temp, random_state=42)

def move_files(files, labels, destination_dir):
    for file, label in zip(files, labels):
        destination = os.path.join(destination_dir, label)
        os.makedirs(destination, exist_ok=True)
        filename = os.path.basename(file)
        destination_file = os.path.join(destination, filename)
        shutil.move(file, destination_file)

move_files(X_train, y_train, train_dir)
move_files(X_val, y_val, val_dir)
move_files(X_test, y_test, test_dir)


