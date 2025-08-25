from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

try:
    import mne
except Exception:
    mne = None


@dataclass(slots=True)
class WindowProxy:
    """Proxy ultra-léger d’une fenêtre: ne garde que (raw, tmin, tmax)."""
    raw: "mne.io.BaseRaw"
    tmin: float
    tmax: float

    @property
    def first_time(self) -> float:
        return self.tmin

    def get_data(self, picks=None):
        """Tranche à la volée sans matérialiser ailleurs."""
        return self.raw.get_data(picks=picks, tmin=self.tmin, tmax=self.tmax)


def segment_rem_in_windows(
    raw,
    rem_segments: Iterable[Tuple[float, float]],
    window_sec: float = 4.0,
    step_sec: float = 4.0,
    verbose: bool = False,
) -> Iterator[WindowProxy]:
    """
    Générateur de fenêtres REM: yield WindowProxy(raw, tmin, tmax).
    Zéro liste, zéro copie complète en mémoire.
    """
    if verbose:
        print(f"[INFO] Fenêtrage en fenêtres de {window_sec}s avec pas de {step_sec}s.")
    for seg_idx, (start, end) in enumerate(rem_segments):
        start = float(start); end = float(end)
        if end - start < window_sec:
            if verbose:
                print(f"[WARNING] Segment REM #{seg_idx} trop court ({end - start:.2f}s), ignoré.")
            continue
        if verbose:
            print(f"[INFO] Segment REM #{seg_idx}: de {start:.2f}s à {end:.2f}s")
        t = start
        while t + window_sec <= end:
            if verbose:
                print(f"  >>> Fenêtre : tmin={t:.2f}s, tmax={t + window_sec:.2f}s")
            yield WindowProxy(raw=raw, tmin=t, tmax=t + window_sec)
            t += step_sec
