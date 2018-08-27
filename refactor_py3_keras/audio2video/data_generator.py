from bisect import bisect_right
from glob import glob
from math import ceil
from os.path import join as pjoin
from random import shuffle

import numpy as np

from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler


class BatchGenerator(Sequence):

    def __init__(self, data_directory, *,
                 batch_size, sequence_length,
                 normalize_input,
                 normalize_output,
                 shuffle=True,):
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sequence_length = sequence_length
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.dirs, self.dirs_lower_batch_indices = \
            self._lower_batch_number_per_directory()

    def __len__(self):
        return self.dirs_lower_batch_indices[-1]

    def __getitem__(self, batch_index):
        dir_index = bisect_right(self.dirs_lower_batch_indices,
                                 batch_index) - 1
        lower_batch_index = self.dirs_lower_batch_indices[dir_index]
        start_row = batch_index - lower_batch_index
        end_row = start_row + self.batch_size + self.sequence_length - 1
        X = np.load(self.dirs[dir_index][0], mmap_mode='r')[start_row: end_row]
        y = np.load(self.dirs[dir_index][1], mmap_mode='r')[start_row: end_row]
        if self.normalize_input:
            X = self.audio_scaler.transform(X)
        if self.normalize_output:
            y = self.mouth_scaler.transform(y)
        return self.create_sequences(X), self.create_sequences(y)

    def create_sequences(self, dataset):
        """Reshape dataset to inputs of sequences

        The dataset contains observations ordered in time, the dimensions are
        rows are examples and columns are features. So for example for 10
        observations with 4 features, the input dataset is a matrix of shape
        10 x 4, if we create input sequences of lenght 3, the resulting
        dataset will be of dimensions 8 x 3 x 4, That is 8 sequences of length
        3 with each point in the sequence having 4 features
        """
        return np.stack(
            [dataset[i: (i + self.sequence_length)]
             for i in range(dataset.shape[0] - (self.sequence_length - 1))],
            axis=0
        )

    def _lower_batch_number_per_directory(self):
        if self.normalize_input:
            self.audio_scaler = StandardScaler(with_mean=True, with_std=True)
        if self.normalize_output:
            self.mouth_scaler = StandardScaler(with_mean=True, with_std=True)

        directories = list(zip(
            glob(pjoin(self.data_directory, '*', '*', 'input_audio.npy')),
            glob(pjoin(self.data_directory, '*', '*', 'output_mouth.npy'))
        ))
        if self.shuffle:
            shuffle(directories)
        # collect number of batches in each directory excluding the last
        number_of_batches = []
        for audio_file, mouth_file in directories:
            audio = np.load(audio_file, mmap_mode='r')
            mouth = np.load(mouth_file, mmap_mode='r')
            number_of_batches.append(ceil(
                (audio.shape[0] - self.sequence_length + 1) / self.batch_size
            ))
            if self.normalize_input:
                self.audio_scaler.partial_fit(audio)
            if self.normalize_output:
                self.mouth_scaler.partial_fit(mouth)
        return directories, np.cumsum([0] + number_of_batches)

    def _calculate_number_of_sequences(self, npy_file):
        return np.load(npy_file,
                       mmap_mode='r').shape[0] - self.sequence_length + 1
