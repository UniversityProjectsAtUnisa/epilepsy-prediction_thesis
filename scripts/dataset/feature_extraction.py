import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, variation
from scipy.signal import welch
from scipy.integrate import simps
from pyentrp import entropy
import math

_scaler = StandardScaler()


def featureNormalization(ft):
    global _scaler
    return _scaler.fit_transform(ft)


# def removeNonNumericValues(df):
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)
#     df.dropna(inplace=True)


def averageChannels(arr):
    arr = arr.mean(axis=0)
    return arr


def featureExtractionAverage(arr, sample_rate):
    # print('Feature Extraction')
    ft = []
    time_feats = computeTimeDomainFeatures(arr)
    ft.extend(time_feats)

    # Frequency Domain Features
    freq_feats = psd(arr, sample_rate, arr.shape[0])
    ft.extend(freq_feats)
    return np.array(ft)


def psd(x, fs, win):
    bands = [0.5, 4, 8, 12, 30, 100]
    freqs, psd = welch(x, fs, nperseg=win)
    avg_power = []
    while len(bands) > 1:
        idx = np.logical_and(freqs >= bands[0], freqs <= bands[1])
        power_simps = simps(psd[idx], dx=bands[1]-bands[0])
        avg_power.append(power_simps)
        bands = np.copy(bands[1:])
    return avg_power


def computeTimeDomainFeatures(x):
    mean = np.mean(x)
    var = np.var(x)
    sk = skew(x)
    kurt = kurtosis(x)
    std = np.std(x)
    median = np.median(x)
    zcr = ((x[:-1] * x[1:]) < 0).sum() / len(x)
    if x.mean() != 0:
        cv = variation(x)
    else:
        cv = math.nan
    if x.size > 0:
        rms = np.sqrt(x.dot(x)/x.size)
    else:
        rms = math.nan
    p2p = x.max() - x.min()
    sampEn = entropy.sample_entropy(x, 1)[0]
    return mean, var, sk, kurt, std, median, zcr, cv, rms, p2p, sampEn


def featureExtractionFull(arr, sample_rate):
    # print('Feature Extraction')
    ft = []
    for s in arr:
        # Time Domain Features
        time_feats = computeTimeDomainFeatures(s)
        ft.extend(time_feats)

        # Frequency Domain Features
        freq_feats = psd(s, sample_rate, s.shape[0])
        ft.extend(freq_feats)

    return np.array(ft)


def featureExtraction(arr, sample_rate, pca_tolerance=None, exp="FULL"):
    fts = []
    chunk_size = arr.shape[1]//2
    for i in range(2):
        a = arr[i*chunk_size:(i+1)*chunk_size]
        if exp.upper() == 'FULL':
            ft = featureExtractionFull(arr, sample_rate)
        else:
            # elif exp.upper() == 'AVERAGE':
            ft = featureExtractionAverage(averageChannels(arr), sample_rate)
        fts.append(ft)
    removeNonNumericValues(ft)
    # ft = featureNormalization(ft)
    # print('Normalized features')
    # removeNonNumericValues(ft)
    # size = ft.shape
    # print('Reducing features dimension')
    # ft = dimentionalityReduction(ft, pca_tolerance)
    # removeNonNumericValues(ft)
    return np.array(fts)
