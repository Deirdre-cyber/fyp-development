import os
import shutil
import unittest
from code.data_loading import load_data, create_dir, move_files



class TestDataLoading(unittest.TestCase):
    """ Test the data loading and preprocessing functions """

    def setUp(self):
        """ Create a temporary directory and populate it with test data """
        self.test_raw_dir = 'test_raw_data'
        self.test_processed_dir = 'test_processed_data'

        os.makedirs(self.test_raw_dir, exist_ok=True)
        os.makedirs(self.test_processed_dir, exist_ok=True)

        self.create_fake_data()

    def tearDown(self):
        """ Clean up the temporary directories """
        shutil.rmtree(self.test_raw_dir)
        shutil.rmtree(self.test_processed_dir)

    def create_fake_data(self):
        """ Create some fake subdirectories and .wav files for testing """
        subdirs = ['class1', 'class2']
        for subdir in subdirs:
            subdir_path = os.path.join(self.test_raw_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            for i in range(3):  # Create 3 fake .wav files in each class
                with open(os.path.join(subdir_path, f'file{i}.wav'), 'w') as f:
                    f.write('Fake audio data')

    def test_load_data(self):
        """ Test whether data loading function works as expected """
        file_paths, labels = load_data(self.test_raw_dir)

        self.assertEqual(len(file_paths), 6)
        self.assertEqual(len(labels), 6)

    def test_create_dir(self):
        """ Test whether directory creation function works as expected """
        train_dir, val_dir, test_dir = create_dir(self.test_processed_dir)

        self.assertTrue(os.path.exists(train_dir))
        self.assertTrue(os.path.exists(val_dir))
        self.assertTrue(os.path.exists(test_dir))

    def test_move_files(self):
        """ Test whether file moving function works as expected """
        file_paths = [os.path.join(self.test_raw_dir, 'class1', f'file{i}.wav') for i in range(3)]
        labels = ['class1'] * 3
        train_dir = os.path.join(self.test_processed_dir, 'train')

        move_files(file_paths, labels, train_dir)

        files_moved = os.listdir(os.path.join(train_dir, 'class1'))
        self.assertEqual(len(files_moved), 3)


if __name__ == '__main__':
    unittest.main()
