import struct

import numpy as np

FRAME_TRANSLATOR = {
    'start': "startframe.txt",
    'n_frames': "nframe.txt"
}


def _read_mouth_features(path):
    f = open(path.joinpath("frontalfidsCoeff_unrefined.bin"), 'rb')
    t = struct.unpack('B', f.read(1))[0]
    if t != 5:
        return 0
    h = struct.unpack('i', f.read(4))[0]
    w = struct.unpack('i', f.read(4))[0]
    return np.reshape(np.array(
        struct.unpack('%df' % (h * w), f.read(4 * h * w)), float),
        (h, w)
    )


def read_frame(path):
    if path.exists:
        with open(path) as f:
            start_frame = int(f.readline())
    else:
        start_frame = 1
    return start_frame


def read_mouth_features(path):
    mouth_features = _read_mouth_features(path)
    start_frame = read_frame(path.joinpath(FRAME_TRANSLATOR['start']))
    number_of_frames = read_frame(path.joinpath(FRAME_TRANSLATOR['n_frames']))
    return mouth_features, start_frame, number_of_frames
