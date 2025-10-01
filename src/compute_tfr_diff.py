import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import re

BASE = Path("..//documents/EEG/preprocessed/bipolaire/3_results_analysis/gp2/_by_patient_category")
EXCLUDE_CATS = {"NA"}          # catégories à ignorer
EXCLUDE_TOKEN = "-A1"          # exclure les canaux contenant ceci
OUT_ROOT = "_between_groups"   # racine de sortie entre groupes
DPI = 1200

# ====== Utils ======
def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "", s)

def robust_limits(A, p_lo=5, p_hi=95):
    A = np.asarray(A)
    A = A[np.isfinite(A)]
    if A.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(A, [p_lo, p_hi])
    if lo == hi:
        eps = 1e-6
        return lo - eps, hi + eps
    return float(lo), float(hi)

def extent_from_axes(time_v, freq_v, Z):
    return [time_v[0], time_v[len(Z[0]) - 1], freq_v[0], freq_v[len(Z) - 1]]

def channel_from_name(p: Path) -> str:
    m = re.match(r"group_(.+)_power\.npy$", p.name)
    return m.group(1) if m else p.stem

def load_tfr(path: Path) -> np.ndarray:
    return np.load(path)

def align_min(A: np.ndarray, B: np.ndarray,
              timeA: np.ndarray, freqA: np.ndarray,
              timeB: np.ndarray, freqB: np.ndarray):
    """Rogne A/B et axes aux tailles minimales communes."""
    f_len = min(A.shape[0], B.shape[0], len(freqA), len(freqB))
    t_len = min(A.shape[1], B.shape[1], len(timeA), len(timeB))
    return (A[:f_len, :t_len],
            B[:f_len, :t_len],
            np.asarray(timeA[:t_len]),
            np.asarray(freqA[:f_len]))

def plot_triptych(T1, T2, diff, time_, freq_,
                  title_left, title_mid, title_right, suptitle, out_path,
                  diff_label="Delta Power (dB)", cmap_diff="RdBu_r",
                  vmin_diff=None, vmax_diff=None, ticks_diff=None):
    """Affiche T1, T2 et diff ; si vmin_diff/vmax_diff sont fournis, on force l'échelle du 3e panneau."""
    # Échelles couleurs T1/T2 (robustes auto)
    vmin1, vmax1 = robust_limits(T1)
    vmin2, vmax2 = robust_limits(T2)

    # Échelle diff
    if vmin_diff is None or vmax_diff is None:
        lo_d, hi_d = robust_limits(diff)
        vmax_d = max(abs(lo_d), abs(hi_d)) or 1e-6
        vmin_d, vmax_d = -vmax_d, vmax_d
    else:
        vmin_d, vmax_d = float(vmin_diff), float(vmax_diff)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    im1 = axes[0].imshow(T1, aspect='auto', origin='lower',
                         extent=extent_from_axes(time_, freq_, T1),
                         cmap='jet', vmin=vmin1, vmax=vmax1)
    axes[0].set_title(title_left)
    axes[0].set_xlabel("Temps (s)")
    axes[0].set_ylabel("Fréquence (Hz)")
    cbar1 = plt.colorbar(im1, ax=axes[0]); cbar1.set_label("Power (dB)")

    im2 = axes[1].imshow(T2, aspect='auto', origin='lower',
                         extent=extent_from_axes(time_, freq_, T2),
                         cmap='jet', vmin=vmin2, vmax=vmax2)
    axes[1].set_title(title_mid)
    axes[1].set_xlabel("Temps (s)")
    cbar2 = plt.colorbar(im2, ax=axes[1]); cbar2.set_label("Power (dB)")

    im3 = axes[2].imshow(diff, aspect='auto', origin='lower',
                         extent=extent_from_axes(time_, freq_, diff),
                         cmap=cmap_diff, vmin=vmin_d, vmax=vmax_d)
    axes[2].set_title(title_right)
    axes[2].set_xlabel("Temps (s)")
    if ticks_diff is not None:
        cbar3 = plt.colorbar(im3, ax=axes[2], ticks=ticks_diff)
    else:
        cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label(diff_label)

    fig.suptitle(suptitle, y=1.02, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# ====== Indexation par groupe/stage/canal =======
cats = [d for d in BASE.iterdir()
        if d.is_dir() and d.name not in EXCLUDE_CATS and not d.name.startswith("_")]
if not cats:
    print("[INFO] Aucune catégorie trouvée (hors exclusions).")

# cartographie: {cat: {stage: {"time": array, "freq": array, "files": {channel: Path}}}}
catalog = {}
for cat_dir in sorted(cats, key=lambda p: p.name):
    cat = cat_dir.name
    for stage_dir in sorted([d for d in cat_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
        stage = stage_dir.name
        time_path = stage_dir / "times.npy"
        freq_path = stage_dir / "freqs.npy"
        if not (time_path.exists() and freq_path.exists()):
            print(f"[WARN] {cat}/{stage}: times.npy ou freqs.npy absent -> skip.")
            continue
        time = np.load(time_path); freq = np.load(freq_path)
        files = {}
        for p in sorted(stage_dir.glob("group_*_power.npy")):
            if EXCLUDE_TOKEN in p.name:
                continue
            ch = channel_from_name(p)
            files[ch] = p
        if not files:
            continue
        catalog.setdefault(cat, {})[stage] = {"time": time, "freq": freq, "files": files}

# ====== Comparaisons entre groupes (même canal) =======
cat_names = sorted(catalog.keys())
for catA, catB in itertools.combinations(cat_names, 2):
    stages_common = sorted(set(catalog[catA].keys()) & set(catalog[catB].keys()))
    if not stages_common:
        print(f"[INFO] Aucune stage commun entre {catA} et {catB}.")
        continue

    for stage in stages_common:
        infoA = catalog[catA][stage]
        infoB = catalog[catB][stage]
        timeA, freqA, filesA = infoA["time"], infoA["freq"], infoA["files"]
        timeB, freqB, filesB = infoB["time"], infoB["freq"], infoB["files"]

        # canaux communs
        common_channels = sorted(set(filesA.keys()) & set(filesB.keys()))
        if not common_channels:
            print(f"[INFO] {catA} vs {catB} / {stage}: aucun canal commun.")
            continue

        for ch in common_channels:
            try:
                p1, p2 = filesA[ch], filesB[ch]
                T1 = load_tfr(p1)
                T2 = load_tfr(p2)

                # alignement (rognage au min)
                T1_, T2_, time_, freq_ = align_min(T1, T2, timeA, freqA, timeB, freqB)

                # -- Différence absolue (dB)
                diff_abs = T1_ - T2_
                # -- Différence relative (%) par rapport à B
                diff_rel = 100.0 * (T1_ - T2_) / np.maximum(np.abs(T2_), 1e-12)

                # dossiers de sortie
                out_dir = BASE / OUT_ROOT / stage / sanitize(ch)
                # titres
                title_left  = f"{catA} - {ch}"
                title_mid   = f"{catB} - {ch}"
                suptitle    = f"Comparaison TFR - {stage}"

                # ABS (auto-échelle symétrique robuste)
                title_right_abs = f"Diff (abs) - {catA} - {catB}"
                out_abs = out_dir / f"compare_{sanitize(catA)}_vs_{sanitize(catB)}_abs.png"
                plot_triptych(T1_, T2_, diff_abs, time_, freq_,
                              title_left, title_mid, title_right_abs, suptitle,
                              out_abs, diff_label="Delta Power (dB)", cmap_diff="RdBu_r")

                # REL (échelle FIXE ±100 %)
                title_right_rel = f"Diff (rel %) - {catA} - {catB}"
                out_rel = out_dir / f"compare_{sanitize(catA)}_vs_{sanitize(catB)}_rel.png"
                plot_triptych(T1_, T2_, diff_rel, time_, freq_,
                              title_left, title_mid, title_right_rel, suptitle,
                              out_rel, diff_label="Delta Power (%)", cmap_diff="RdBu_r",
                              vmin_diff=-100, vmax_diff=100, ticks_diff=[-100, -50, 0, 50, 100])

                print(f"[OK] {stage}/{ch}: {catA} vs {catB} -> abs & rel (±100 %)")

            except Exception as e:
                print(f"[ERR] {stage}/{ch}: {catA} vs {catB} -> {e}")
