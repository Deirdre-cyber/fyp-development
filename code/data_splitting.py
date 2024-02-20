import os
import glob
from sklearn.model_selection import train_test_split


root_directory = 'DIR_ROOT_PATH'

subdirectories = glob.glob(os.path.join(root_directory, '*'))

file_paths = []
labels = []

for subdir in subdirectories:
    label = os.path.basename(subdir)  # Use directory name as label
    files = [os.path.join(subdir, f) for f in os.listdir(subdir) if f.endswith('.wav')]  # Adjust file extension if needed
    file_paths.extend(files)
    labels.extend([label] * len(files))

X_train_temp, X_test, y_train_temp, y_test = train_test_split(file_paths, labels, test_size=0.15, stratify=labels, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=0.15, stratify=y_train_temp, random_state=42)


# Store the resulting data sets in project
