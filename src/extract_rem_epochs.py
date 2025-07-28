def extract_rem_epochs(y_pred, epoch_length=30, fs=200):
    """
    Extract REM segments from predicted labels.

    Parameters
    ----------
    y_pred : ndarray
        Predicted sleep stage labels for each 30s epoch.
    epoch_length : int
        Duration of each epoch in seconds (default: 30).
    fs : int
        Sampling frequency in Hz (default: 200).

    Returns
    -------
    rem_segments : list of tuple
        List of (start_sample, end_sample) for each REM segment.
    """
    rem_segments = []
    for i, stage in enumerate(y_pred):
        if stage == 4:
            start = i * epoch_length * fs
            end = (i + 1) * epoch_length * fs
            rem_segments.append((start, end))
    return rem_segments
