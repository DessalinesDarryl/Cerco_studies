def segment_rem_in_windows(raw, rem_segments, window_sec=4, step_sec=2):
    """
    Découpe les segments REM fournis en fenêtres glissantes, avec debug.

    Args:
        raw (mne.io.Raw): objet Raw EEG complet
        rem_segments (list): liste de tuples (start_sec, end_sec) des périodes REM
        window_sec (float): durée d'une fenêtre (en secondes)
        step_sec (float): pas de décalage entre fenêtres (en secondes)

    Returns:
        list of mne.io.Raw: fenêtres extraites du signal
    """
    windows = []
    print(f"[INFO] Fenêtrage en fenêtres de {window_sec}s avec pas de {step_sec}s.")

    for seg_idx, (start, end) in enumerate(rem_segments):
        if end - start < window_sec:
            print(f"[WARNING] Segment REM #{seg_idx} trop court ({end - start:.2f}s), ignoré.")
            continue

        print(f"[INFO] Segment REM #{seg_idx}: de {start:.2f}s à {end:.2f}s")
        t = start
        while t + window_sec <= end:
            print(f"  ↳ Fenêtre : tmin={t:.2f}s, tmax={t + window_sec:.2f}s")
            try:
                epoch = raw.copy().crop(tmin=t, tmax=t + window_sec).load_data()
                windows.append(epoch)
            except Exception as e:
                print(f"[ERROR] Impossible de couper la fenêtre [{t:.2f}s - {t + window_sec:.2f}s] : {e}")
            t += step_sec

    print(f"[INFO] Total de {len(windows)} fenêtres extraites.")
    return windows
