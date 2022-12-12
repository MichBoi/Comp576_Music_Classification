import librosa
import os
import re

import numpy as np
from numpy import floor


class GenerateSpectrogramData:
    genre_list = [
        "classical",
        "hiphop",
        "jazz",
        "metal",
        "pop",
        "reggae",
    ]

    train_X_preprocessed_data = "./gtzan/data_train_input_spetrogram.npy"
    train_Y_preprocessed_data = "./gtzan/data_train_target_spetrogram.npy"
    dev_X_preprocessed_data = "./gtzan/data_validation_input_spetrogram.npy"
    dev_Y_preprocessed_data = "./gtzan/data_validation_target_spetrogram.npy"
    test_X_preprocessed_data = "./gtzan/data_test_input_spetrogram.npy"
    test_Y_preprocessed_data = "./gtzan/data_test_target_spetrogram.npy"

    train_folder = "./gtzan/_train"
    validation_folder = "./gtzan/_validation"
    test_folder = "./gtzan/_test"
    all_files = "./gtzan"

    train_X = train_Y = None
    dev_X = dev_Y = None
    test_X = test_Y = None

    def __init__(self):
        self.train_files_list = self.path_to_audiofiles(self.train_folder)
        self.dev_files_list = self.path_to_audiofiles(self.validation_folder)
        self.test_files_list = self.path_to_audiofiles(self.test_folder)

        self.all_files_list = []
        self.all_files_list.extend(self.train_files_list)
        self.all_files_list.extend(self.dev_files_list)
        self.all_files_list.extend(self.test_files_list)

    def load_preprocess_data(self):
        # Training
        self.train_X, self.train_Y = self.extract_audio_features(self.train_files_list)
        with open(self.train_X_preprocessed_data, "wb") as f:
            np.save(f, self.train_X)
        with open(self.train_Y_preprocessed_data, "wb") as f:
            self.train_Y = self.one_hot(self.train_Y)
            np.save(f, self.train_Y)

        # Validation
        self.dev_X, self.dev_Y = self.extract_audio_features(self.dev_files_list)
        with open(self.dev_X_preprocessed_data, "wb") as f:
            np.save(f, self.dev_X)
        with open(self.dev_Y_preprocessed_data, "wb") as f:
            self.dev_Y = self.one_hot(self.dev_Y)
            np.save(f, self.dev_Y)

        # Test
        self.test_X, self.test_Y = self.extract_audio_features(self.test_files_list)
        with open(self.test_X_preprocessed_data, "wb") as f:
            np.save(f, self.test_X)
        with open(self.test_Y_preprocessed_data, "wb") as f:
            self.test_Y = self.one_hot(self.test_Y)
            np.save(f, self.test_Y)

    def load_deserialize_data(self):
        self.train_X = np.load(self.train_X_preprocessed_data)
        self.train_Y = np.load(self.train_Y_preprocessed_data)
        self.dev_X = np.load(self.dev_X_preprocessed_data)
        self.dev_Y = np.load(self.dev_Y_preprocessed_data)
        self.test_X = np.load(self.test_X_preprocessed_data)
        self.test_Y = np.load(self.test_Y_preprocessed_data)

    def extract_audio_features(self, list_of_audiofiles):
        # mel-spectrogram parameters
        SR = 12000
        N_FFT = 512
        N_MELS = 96
        HOP_LEN = 256
        DURA = 29.12  # to make it 1366 frame
        DURA_TRASH = 0

        x = []
        y = []

        for i, file in enumerate(list_of_audiofiles):
            splits = re.split("[ .]", file)
            genre = re.split("[ /]", splits[1])[3]
            y.append(genre)

            src, sr = librosa.load(file)
            n_sample = src.shape[0]
            n_sample_fit = int(DURA * SR)
            n_sample_trash = int(DURA_TRASH * SR)

            # trim tail and head
            src = src[n_sample_trash:(n_sample - n_sample_trash)]
            n_sample = n_sample - 2 * n_sample_trash

            ret = np.zeros((96, 1366), dtype=np.float32)

            if n_sample < n_sample_fit:  # if too short
                src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
                logam = librosa.amplitude_to_db
                melgram = librosa.feature.melspectrogram
                ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                                    n_fft=N_FFT, n_mels=N_MELS) ** 2,)

            elif n_sample > n_sample_fit:  # if too long
                N = int(floor(n_sample / n_sample_fit))

                src_total = src

                for i in range(0, N):
                    src = src_total[(i * n_sample_fit):(i + 1) * n_sample_fit]

                    logam = librosa.amplitude_to_db
                    melgram = librosa.feature.melspectrogram
                    retI = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                                         n_fft=N_FFT, n_mels=N_MELS) ** 2,)
                    ret = np.concatenate((ret, retI), axis=0)
            x.append(ret)
        return np.array(x), np.expand_dims(np.asarray(y), axis=1)

    def one_hot(self, Y_genre_strings):
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(self.genre_list)))
        for i, genre_string in enumerate(Y_genre_strings):
            index = self.genre_list.index(genre_string)
            y_one_hot[i, index] = 1
        return y_one_hot

    @staticmethod
    def path_to_audiofiles(dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            if file.endswith(".au"):
                directory = "%s/%s" % (dir_folder, file)
                list_of_audio.append(directory)
        return list_of_audio
