"""Given an audio (.wav) file, output a numpy array of features to disk
"""

import os
import subprocess

from pathlib import Path

import ffmpeg
import numpy as np

from scipy.io import wavfile

from .mfcc import MFCC


SAMPLE_RATE = 16000
CODEC = 'pcm_s16le'
NUMBER_CHANNELS = 1


def read_audio_features(path):
    audio = np.load(path.joinpath(path.stem + ".wav.npy"))
    return audio[:, :-1], audio[:, -1]


def get_and_save_audio_features(audio_filename, clean_up=True):
    input_path = Path(audio_filename)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"{input_path} not found. Is it an absolute path?"
        )

    # create filename for formatted version of the audio
    input_path_formatted = input_path.parents[0].joinpath(
        '-'.join([input_path.stem, 'formatted.wav'])
    )
    # create filename for normalized version of the audio
    input_path_normalized = input_path.parents[0].joinpath(
        '-'.join([input_path.stem, 'normalized.wav'])
    )
    # create filename to store audio features of the audio
    audio_features_file = input_path.parents[0].joinpath(
        '-'.join([input_path.stem, 'features.npy'])
    )

    # force audio codec pcm_s16le
    # set number of audio channels to 1
    # set audio sampling rate (in Hz)
    # save formatted audio to file and overwrite
    try:
        (ffmpeg
         .input(input_path)
         .output(str(input_path_formatted),
                 acodec=CODEC, ac=NUMBER_CHANNELS, ar=SAMPLE_RATE)
         .run(overwrite_output=True, capture_stderr=True))
    except ffmpeg.Error as e:
        raise Exception(e.stderr.decode())

    # EBU R128 loudness normalization
    # save normalized audio to file and overwrite
    try:
        subprocess.check_output(
            "ffmpeg-normalize "
            f"-f {input_path_formatted} "
            f"-o {input_path_normalized}", shell=True
        )
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode())

    # normalized audio file to numpy array
    audio_data = wavfile.read(input_path_normalized)[1]
    print(audio_data.shape, audio_data.dtype)

    # extract mfcc audio features as numpy array
    audio_features = MFCC().sig2s2mfc_energy(audio_data)
    print(audio_features.shape, audio_features.dtype)

    # save numpy array of features to npy file and overwrite
    np.save(audio_features_file, audio_features)

    if clean_up:
        os.remove(input_path_formatted)
        os.remove(input_path_normalized)
