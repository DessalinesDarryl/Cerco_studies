from mne import Annotations

def safe_add_annotations(raw, new_ann):
    """Compat MNE : ajoute new_ann sans écraser l’existant, même si raw.add_annotations n’existe pas."""
    cur = getattr(raw, "annotations", None)
    if cur is None or len(cur) == 0:
        raw.set_annotations(new_ann)
    else:
        # Concaténation d'Annotations : new total = ancien + nouveau
        raw.set_annotations(cur + new_ann)


def annotate_microstates(raw, windows, labels, window_sec, mode="add", prefix="microstate_"):
    """
    Ajoute des annotations 'phasic' / 'tonic' sans perdre les annotations existantes.
    - mode="add" : ajoute simplement
    - mode="replace_microstates" : remplace seulement les anciennes annotations de micro-états
    - mode="replace_all" : remplace tout (comportement initial, à éviter)
    """
    onsets = [float(win.first_time) for win in windows]
    durations = [float(window_sec)] * len(windows)
    descriptions = [f"{prefix}{lab}" for lab in labels]

    # Garder la même origine temporelle que le Raw courant s'il en a une
    orig_time = raw.annotations.orig_time if (raw.annotations is not None and len(raw.annotations) > 0) else None
    new_ann = Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=orig_time)

    if mode == "add":
        safe_add_annotations(raw, new_ann)  # n'écrase pas l'existant
    elif mode == "replace_microstates":
        # on enlève d'abord d'éventuelles annotations de micro-états déjà présentes
        if raw.annotations is not None and len(raw.annotations) > 0:
            keep_idx = [i for i, d in enumerate(raw.annotations.description) if not str(d).startswith(prefix)]
            kept = raw.annotations[keep_idx] if keep_idx else Annotations([], [], [], orig_time=orig_time)
            raw.set_annotations(kept + new_ann)
        else:
            raw.set_annotations(new_ann)
    else:  # "replace_all"
        raw.set_annotations(new_ann)

    print(f"[INFO] {len(new_ann)} annotations micro-états ajoutées (total = {len(raw.annotations)})")
