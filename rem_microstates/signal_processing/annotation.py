from typing import Sequence, Tuple
from mne import Annotations

def safe_add_annotations(raw, new_ann):
    """Ajoute new_ann sans écraser l’existant (compat MNE)."""
    cur = getattr(raw, "annotations", None)
    if cur is None or len(cur) == 0:
        raw.set_annotations(new_ann)
    else:
        raw.set_annotations(cur + new_ann)


def annotate_microstates(
    raw,
    windows_or_times: Sequence,  # liste de win (avec .first_time) OU de tuples (tmin,tmax)
    labels: Sequence[str],
    window_sec: float,
    mode: str = "add",
    prefix: str = "microstate_",
):
    """
    Ajoute des annotations 'phasic'/'tonic' sans perdre l’existant.
    - windows_or_times: [win,...] avec .first_time  OU  [(tmin,tmax), ...]
    - mode: "add" | "replace_microstates" | "replace_all"
    """
    onsets, durations = [], []

    # Cas tuples (tmin,tmax)
    if len(windows_or_times) > 0 and isinstance(windows_or_times[0], (tuple, list)) \
       and len(windows_or_times[0]) == 2:
        for (tmin, tmax) in windows_or_times:
            onsets.append(float(tmin))
            durations.append(float(tmax) - float(tmin))
    else:
        # Cas objets fenêtre possédant .first_time
        onsets = [float(win.first_time) for win in windows_or_times]
        durations = [float(window_sec)] * len(windows_or_times)

    descriptions = [f"{prefix}{str(lab).strip().lower()}" for lab in labels]

    orig_time = raw.annotations.orig_time if (raw.annotations is not None and len(raw.annotations) > 0) else None
    new_ann = Annotations(onset=onsets, duration=durations, description=descriptions, orig_time=orig_time)

    if mode == "add":
        safe_add_annotations(raw, new_ann)
    elif mode == "replace_microstates":
        if raw.annotations is not None and len(raw.annotations) > 0:
            keep_idx = [i for i, d in enumerate(raw.annotations.description) if not str(d).startswith(prefix)]
            kept = raw.annotations[keep_idx] if keep_idx else Annotations([], [], [], orig_time=orig_time)
            raw.set_annotations(kept + new_ann)
        else:
            raw.set_annotations(new_ann)
    else:  # "replace_all"
        raw.set_annotations(new_ann)

    print(f"[INFO] {len(new_ann)} annotations micro-états ajoutées (total = {len(raw.annotations)})")