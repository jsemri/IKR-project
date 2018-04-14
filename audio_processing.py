from glob import glob
import math
import wave
import numpy as np

CURSOR_UP = '\033[F'
ERASE_LINE = '\033[K'

FRAME_SIZE = 256
OVER_LAP = 128
SILENCE_THRESHOLD = 20

__all__ = ["load_audio", "FRAME_SIZE", "OVER_LAP", "SILENCE_THRESHOLD"]


def load_audio(dir, audio_processor=None):
    """
    Loads all wav files from directory
    :param dir: directory containing wav files
    :param audio_processor: function to preprocess files, function takes one np.ndarray argument and returns np.ndarray
    :return: list of tuples containing two elements, first is file name and second preprocessed audio as np.ndarray
    """
    _features = list()
    _files = glob(dir + '/*.wav')
    files_count = len(_files)
    print("loading {} files...".format(files_count))
    counter = 0
    for f in _files:
        print("leaded: {}/{}".format(counter, files_count))
        preprocessed = pre_process_file(f)
        if audio_processor is not None:
            assert callable(audio_processor)
            preprocessed = audio_processor(preprocessed)
        _features.append((f, preprocessed))
        counter += 1
        print(CURSOR_UP + ERASE_LINE + CURSOR_UP)
    print(CURSOR_UP + ERASE_LINE + CURSOR_UP)
    print("Loaded {} files".format(len(_features)))
    return _features


def cal_volume(wave_data, frame_size, over_lap):
    """
    Calculate volume dynamic
    :param wave_data: np.ndarray audio data
    :param frame_size: frame size
    :param over_lap: overlap size
    :return: Array of volume dynamic
    """
    w_len = len(wave_data)
    step = frame_size - over_lap
    frame_num = int(math.ceil(w_len * 1.0 / step))
    volume = np.zeros((frame_num, 1))
    for i in range(frame_num):
        cur_frame = wave_data[np.arange(i * step, min(i * step + frame_size, w_len))]
        cur_frame = cur_frame - np.median(cur_frame)  # zero-justified
        volume[i] = np.sum(np.abs(cur_frame))
    return volume


def pre_process_file(file):
    """
    Preprocess audio file, removes first two seconds of audio all silent parts
    :param file: file url
    :return: processed audio as np.ndarray
    """
    fw = wave.open(file, 'rb')
    params = fw.getparams()
    n_channels, sample_width, frame_rate, n_frames = params[:4]
    str_data = fw.readframes(n_frames)
    fw.close()
    wave_data = np.fromstring(str_data, dtype=np.int16)
    wave_data = wave_data * 1.0 / max(abs(wave_data))  # normalization

    # calculate volume
    frame_size = FRAME_SIZE
    over_lap = OVER_LAP

    wave_data = wave_data[frame_rate * 2:]
    volume11 = cal_volume(wave_data, frame_size, over_lap)

    volume111 = list()
    for v in volume11:
        volume111.extend([v] * over_lap)
    res = list()
    for i, v1 in enumerate(volume111):
        v1 = v1[0]
        if v1 >= SILENCE_THRESHOLD:
            res.append(wave_data[i])

    return wave_data
