"""Create input dataset from the audio and mouth features
"""

import bisect
import math

from pathlib import Path

import numpy as np

from tqdm import tqdm

from .audio_features import read_audio_features
from .mouth_features import read_mouth_features


FPS = 29.97
TIMEDELAY = 20
SEQUENCE_LENGTH = 100


def delay_output(input, output):
    """Shift output forwards in time

    Reshape inputs and outputs such that for input at t_TIMEDELAY the
    corresponding output is at t_0. This will allow the model to see
    into the future before emiting an output.
    """
    if len(input) - TIMEDELAY >= (SEQUENCE_LENGTH + 2) and TIMEDELAY > 0:
        return input[TIMEDELAY:], output[:-TIMEDELAY]
    else:
        return input, output


def crop_audio(audio, audiodiff, timestamps, start_frame, number_frames):
    """Select period of the audio matching the frames

    Select audio period between the start and end frame numbers of the
    matching video. This will yield a time matching audio with the
    corresponding mouth features of the video.

    Is good to be aware that this does not guarantee that there is a
    mouth feature for each audio point. This function only selects
    the audio portion between the first and last frame.
    """
    # index for before or equal timestamp corresponding to initial frame
    start_index = bisect.bisect_left(
        timestamps, (start_frame - 1) / FPS  # frame nr starts from 1
    )
    # index for after or equal timestamp corresponding to final frame
    end_index = bisect.bisect_right(
        timestamps, (start_frame + number_frames - 2) / FPS
    )

    # create a single matrix with all audio and audiodiff features
    audio_period = np.concatenate(
        (audio[start_index:end_index], audiodiff[start_index:end_index]),
        axis=1
    )
    timestamps_period = timestamps[start_index: end_index]
    return audio_period, timestamps_period


def make_mouth(audio_timestamp, mouth, n_start_frame):
    """Make mouth features for an specific timestamp

    This is basically an upsampling of the mouth features by a weighted
    average between the two closest mouth features to the audio timestamp.
    """
    # get first frame equal or a little after than first timestamp
    leftmark = math.ceil((audio_timestamp * FPS) - n_start_frame)
    # fraction of frame difference
    diff = (audio_timestamp - (n_start_frame - 1 + leftmark) / FPS) * FPS
    # weighted average between the previous and next frame
    return diff * mouth[min(len(mouth) - 1, leftmark + 1)] + \
        (1 - diff) * mouth[leftmark]


def create_datasets(data_dir):

    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise OSError(
            f"{data_path} not found or is not a directoy."
        )
    # read videos IDs
    with open(data_path.joinpath("processed_video_ids.txt")) as f:
        processed_ids = [_id.split("\t")[0].strip() for _id in f.readlines()]

    # create input and output dataset from each video
    for _id in tqdm(processed_ids):
        print(f"\nprocessing id: {_id}")
        # get directories containing the mouth features
        mouth_dirs = sorted(data_path.joinpath(_id).glob('*}}*'))
        print(len(mouth_dirs))
        # load audio features and create difference feature
        audio, timestamps = read_audio_features(data_path.joinpath(_id))
        audio_diff = audio[1:] - audio[:-1]
        print(audio.shape, audio_diff.shape, timestamps.shape)

        # align mouth features with audio features
        for _dir in mouth_dirs:
            print(_dir.stem)
            mouth, n_start_frame, n_frames = read_mouth_features(_dir)
            print(mouth.shape, n_start_frame, n_frames)
            sub_audio, sub_timestamps = crop_audio(
                audio, audio_diff, timestamps, n_start_frame, n_frames
            )
            print(sub_audio.shape, sub_timestamps.shape)

            # combine mouth features
            mouth_features = np.zeros(
                (sub_audio.shape[0], mouth.shape[1]), dtype=np.float32
            )
            for index, timestamp in enumerate(sub_timestamps):
                mouth_features[index] = \
                    make_mouth(timestamp, mouth, n_start_frame)
            print(sub_audio.shape, mouth_features.shape)

            input_audio, output_mouth = delay_output(sub_audio, mouth_features)
            print(input_audio.shape, output_mouth.shape)
            np.save(data_path.joinpath(_id, _dir, 'input_audio'),
                    input_audio)
            np.save(data_path.joinpath(_id, _dir, 'output_mouth'),
                    output_mouth)
        import os
        try:
            os.remove(data_path.joinpath(_id, 'audio_sequences.npy'))
            os.remove(data_path.joinpath(_id, 'mouth_sequences.npy'))
            os.remove(data_path.joinpath(_id, _dir, 'audio.npy'))
            os.remove(data_path.joinpath(_id, _dir, 'mouth.npy'))
            os.remove(data_path.joinpath(_id, 'audio.npy'))
            os.remove(data_path.joinpath(_id, 'mouth.npy'))
        except FileNotFoundError:
            pass
