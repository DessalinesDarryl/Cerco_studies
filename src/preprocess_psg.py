import numpy as np
from scipy.signal import butter, filtfilt
import mne


def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Create a Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int
        Order of the filter.

    Returns
    -------
    b, a : ndarray
        Filter coefficients for digital IIR filter.
    """
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-5)
    high = min(highcut / nyq, 0.9999)

    if not (0 < low < high < 1):
        raise ValueError(f"Invalid cutoff frequencies: low={lowcut}, high={highcut}, fs={fs}")

    return butter(order, [low, high], btype='band')



def apply_filter(data, lowcut, highcut, fs):
    """
    Apply zero-phase bandpass filtering to a 1D signal.

    Parameters
    ----------
    data : ndarray
        Input signal.
    lowcut : float
        Lower cutoff frequency.
    highcut : float
        Upper cutoff frequency.
    fs : float
        Sampling frequency.

    Returns
    -------
    filtered : ndarray
        Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)


def soft_normalize(x):
    """
    Apply quantile-based soft normalization to a signal.

    Parameters
    ----------
    x : ndarray
        Input signal.

    Returns
    -------
    x_norm : ndarray
        Normalized signal in range approximately [-1, 1].
    """
    q05 = np.quantile(x, 0.05)
    q95 = np.quantile(x, 0.95)
    return 2 * (x - q05) / (q95 - q05) - 1


def preprocess_psg(raw, target_fs=200):
    """
    Preprocess a raw PSG recording into 30-second normalized epochs.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw PSG recording loaded using MNE.
    target_fs : int
        Target resampling frequency in Hz (default: 200 Hz).

    Returns
    -------
    epochs : ndarray
        Preprocessed data of shape (N, C, 1, T), where N is the number
        of 30-second epochs, C is the number of channels (5), and
        T = target_fs * 30.
    """
    channels = ['EEG C4', 'EEG O2', 'EOGG', 'EOGD', 'Menton']
    raw.pick_channels(channels)
    raw.resample(target_fs)

    fs = raw.info['sfreq']
    print(f"[INFO] sfreq identified = {fs}")

    data = raw.get_data()

    filtered = np.zeros_like(data)
    bands = [(0.3, 35), (0.3, 35), (0.1, 10), (0.1, 10), (10, 100)]
    for i, (low, high) in enumerate(bands):
        filtered[i, :] = apply_filter(data[i], low, high, fs)
        filtered[i, :] = soft_normalize(filtered[i, :])

    T = int(30 * target_fs)
    N = filtered.shape[1] // T
    epochs = np.zeros((N, 5, 1, T))

    for i in range(N):
        epochs[i] = filtered[:, i*T:(i+1)*T].reshape(5, 1, T)

    return epochs
