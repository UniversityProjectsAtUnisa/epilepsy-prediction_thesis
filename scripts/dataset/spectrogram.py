import numpy as np
from scipy import signal as s


def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = s.butter(order, normal_cutoff, btype='high', analog=False)
    y = s.lfilter(b, a, data)
    return y


def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = s.butter(order, [low, high], btype='bandstop')
    y = s.lfilter(i, u, data)
    return y


def createSpec(data):
    fs = 256
    lowcut = 117
    highcut = 123

    y = butter_bandstop_filter(data, lowcut, highcut, fs, order=6)
    lowcut = 57
    highcut = 63
    y = butter_bandstop_filter(y, lowcut, highcut, fs, order=6)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)

    Pxx = s.spectrogram(y, nfft=256, fs=256, return_onesided=True, noverlap=128)[2]
    Pxx = np.delete(Pxx, np.s_[117:123+1], axis=0)
    Pxx = np.delete(Pxx, np.s_[57:63+1], axis=0)
    Pxx = np.delete(Pxx, 0, axis=0)

    result = (10*np.log10(np.transpose(Pxx))-(10*np.log10(np.transpose(Pxx))).min())/(10*np.log10(np.transpose(Pxx))).ptp()
    return result


def to_spectrogram(data: np.ndarray):
    new_data = []
    for window in data:
        new_window = []
        for channel in window:
            channel_spectrogram = createSpec(channel)
            new_window.append(channel_spectrogram)

        new_data.append(np.array(new_window))
    return np.array(new_data)
