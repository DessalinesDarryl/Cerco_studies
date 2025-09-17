#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convertit des .edf en .fif avec montage bipolaire configurable.

- Entrée : répertoire racine contenant des sous-dossiers patients (ex. AE129/AE129_raw.edf)
- Sortie : out_root/{patient}/{patient}_bipolar.fif
- Parallélisation avec multiprocessing
- Filtrage optionnel (passe-bande + notch)
- Filtrage des patients via un fichier texte (un ID par ligne)

Usage:
  python apply_montage_bip.py \
      --in_root /chemin/EEG/raw \
      --out_root /chemin/EEG/preprocessed/bipolaire/0_raw_bip \
      --montage gp1 \
      --patients_list data/patients_gp1.txt \
      --filter --l_freq 0.3 --h_freq 100 --notch 50 \
      --workers 4 [--overwrite]
"""

from __future__ import annotations
import os
import argparse
import multiprocessing as mp
import tempfile
from pathlib import Path
import mne

from utils.montages import get_montage

# =============== Exceptions =================

class MissingChannelsError(Exception):
    def __init__(self, base: str, missing: list[str]):
        super().__init__(f"[{base}] canaux manquants: {missing}")
        self.base = base
        self.missing = missing


# =============== Utilitaires =================

def read_patients_list(path: Path | None) -> set[str] | None:
    """Lit un fichier texte avec un ID patient par ligne, ignore lignes vides/commentaires."""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Fichier liste patients introuvable: {path}")
    keep: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # accepte éventuelle entête "Patients :" comme dans ton exemple
            if s.lower().startswith("patients"):
                continue
            keep.add(s)
    return keep or None


def discover_patients(edf_root: Path) -> list[tuple[str, Path]]:
    """
    Explore edf_root et retourne [(patient_id, chemin_edf)].
    Prend le plus gros *_raw.edf/EDF si plusieurs.
    """
    mapping = {}
    for child in sorted(edf_root.iterdir()):
        if child.is_dir():
            edf_candidates = list(child.glob("*_raw.edf")) + list(child.glob("*_raw.EDF"))
            if edf_candidates:
                mapping[child.name] = max(edf_candidates, key=lambda p: p.stat().st_size)
    return sorted(mapping.items())


def _safe_bipolar(inst: mne.io.BaseRaw, anode: str, cathode: str, new_name: str, base: str="") -> bool:
    if anode in inst.ch_names and cathode in inst.ch_names:
        try:
            mne.set_bipolar_reference(inst, anode=anode, cathode=cathode,
                                      ch_name=new_name, drop_refs=False, copy=False, verbose="ERROR")
            return True
        except Exception as e:
            print(f"[{base}] Bipolaire {new_name} échec: {e}")
    else:
        missing = [x for x in (anode, cathode) if x not in inst.ch_names]
        print(f"[{base}] Bipolaire {new_name} ignoré (manque {missing})")
    return False


def apply_bipolar_montage(inst: mne.io.BaseRaw, montage_name: str, base: str="") -> list[str]:
    """
    Applique le montage déclaré dans montages.py et ajuste les types de canaux.
    Retourne la liste finale des canaux conservés.
    """
    montage = get_montage(montage_name)
    # Création des canaux bipolaires
    for a, c, n in montage.pairs:
        _safe_bipolar(inst, a, c, n, base)

    # Détermine les types des nouveaux canaux par convention de nommage
    type_map = {}
    for ch in inst.ch_names:
        if ch.endswith("-A1") or ch.startswith("EOG"):  # e.g., "EOGD-A1"
            type_map[ch] = "eog"
        elif "-" in ch:
            # Suppose que les noms "X-Y" (non EOG) sont EEG bipolaires
            type_map[ch] = "eeg"

    # Ajoute les types pour les “raw” conservés tels quels
    for raw_name in montage.keep_raw:
        if raw_name in inst.ch_names:
            if raw_name.startswith("EMG") or raw_name in ("Menton", "JAMBG", "JAMBD"):
                type_map[raw_name] = "emg"
            elif raw_name == "ECG":
                type_map[raw_name] = "ecg"
            elif raw_name == "RONF":
                type_map[raw_name] = "misc"

    if type_map:
        inst.set_channel_types(type_map)

    # Liste désirée: tous les nouveaux canaux bipolaires + les bruts à garder
    desired = [n for _, _, n in montage.pairs] + montage.keep_raw
    present = [ch for ch in desired if ch in inst.ch_names]
    if not present:
        missing = [ch for ch in desired if ch not in inst.ch_names]
        raise MissingChannelsError(base, missing)

    inst.pick(present)
    print(f"[{base}] Montage '{montage_name}' OK. Canaux conservés: {inst.ch_names}")
    return inst.ch_names


# =============== Worker =================

SKIPPED = None  # liste partagée entre workers
OUT_ROOT = None
OVERWRITE = False
MONTAGE_NAME = "default"
DO_FILTER = False
L_FREQ = 0.3
H_FREQ = 100.0
NOTCH_FREQS = None  # e.g. [50] ou [50,100]

def process_one(item):
    base, edf_path = item
    out_dir = OUT_ROOT / base
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}_bipolar.fif"

    if out_path.exists() and not OVERWRITE:
        print(f"[{base}] Déjà traité -> skip (utiliser --overwrite pour refaire)")
        return

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

        # correction des noms de canaux : suppression du préfixe "EEG " 
        new_names = {ch: ch.replace("EEG ", "") for ch in raw.ch_names if ch.startswith("EEG ")}
        if new_names:
            raw.rename_channels(new_names)
            print(f"[{base}] Canaux renommés (suppression 'EEG '): {new_names}")

        # montage bipolaire 
        apply_bipolar_montage(raw, MONTAGE_NAME, base)

        # filtrage  
        if DO_FILTER:
            picks = mne.pick_types(raw.info, eeg=True, eog=True, emg=True, ecg=True, misc=False)
            if len(picks) > 0:
                raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, fir_design="firwin", picks=picks, verbose="ERROR")
                if NOTCH_FREQS:
                    raw.notch_filter(freqs=NOTCH_FREQS, picks=picks, verbose="ERROR")
                print(f"[{base}] Filtrage appliqué: band [{L_FREQ}-{H_FREQ}] Hz, notch={NOTCH_FREQS or 'None'}")
            else:
                print(f"[{base}] Avertissement: aucun pick pour filtrage, filtrage sauté.")

        raw.save(out_path, overwrite=True)
        print(f"[{base}] Sauvé -> {out_path}")

    except MissingChannelsError as e:
        print(f"[{base}] Sauté : canaux manquants {e.missing}")
        SKIPPED.append((e.base, e.missing))
    except Exception as e:
        print(f"[{base}] Erreur inattendue: {e}")
        SKIPPED.append((base, [str(e)]))


def _init_worker(gl_out_root: str, gl_overwrite: bool, gl_montage: str,
                 gl_do_filter: bool, gl_l_freq: float, gl_h_freq: float, gl_notch_freqs: list[int] | None,
                 skipped):
    global OUT_ROOT, OVERWRITE, MONTAGE_NAME, DO_FILTER, L_FREQ, H_FREQ, NOTCH_FREQS, SKIPPED
    OUT_ROOT = Path(gl_out_root)
    OVERWRITE = gl_overwrite
    MONTAGE_NAME = gl_montage
    DO_FILTER = gl_do_filter
    L_FREQ = gl_l_freq
    H_FREQ = gl_h_freq
    NOTCH_FREQS = gl_notch_freqs
    SKIPPED = skipped
    # Evite les collisions Matplotlib en multi-process (certaines installs)
    try:
        mpl_cache = os.path.join(tempfile.gettempdir(), f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass


# =============== Main =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", type=str, required=False, help="Dossier contenant les dossiers patients avec .edf")
    parser.add_argument("--out_root", type=str, required=False, help="Dossier de sortie pour les .fif")
    parser.add_argument("--workers", type=int, default=8, help="Nb de workers (0=CPU-1)")
    parser.add_argument("--overwrite", action="store_true", help="Écraser les fichiers déjà existants")
    parser.add_argument("--montage", type=str, default="default", help="Nom du montage déclaré dans montages.py (ex: default, gp1)")
    parser.add_argument("--patients_list", type=str, default=None, help="Chemin d'une liste de patients (un ID par ligne)")
    # Filtrage 
    parser.add_argument("--filter", action="store_true", help="Activer le filtrage (passe-bande + notch)")
    parser.add_argument("--l_freq", type=float, default=0.3, help="Borne basse du passe-bande (Hz)")
    parser.add_argument("--h_freq", type=float, default=100.0, help="Borne haute du passe-bande (Hz)")
    parser.add_argument("--notch", type=str, default="50", help='Fréquences notch, (à laisser vide si aucun notch)')
    args = parser.parse_args()

    edf_root = Path("/home/darryld/documents/EEG/raw") if args.in_root is None else Path(args.in_root)
    out_root = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/0_raw_bip_gp1") if args.out_root is None else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Liste patients à garder
    keep = read_patients_list(Path(args.patients_list)) if args.patients_list else None

    # Découverte des patients
    all_patients = discover_patients(edf_root)
    if not all_patients:
        raise SystemExit("Aucun fichier *_raw.edf ou *_raw.EDF trouvé")

    if keep:
        patients = [(pid, p) for pid, p in all_patients if pid in keep]
        missing = sorted(list(keep - set(pid for pid, _ in all_patients)))
        if missing:
            print(f"[INFO] Patients listés mais non trouvés sous {edf_root}: {missing}")
    else:
        patients = all_patients

    if not patients:
        raise SystemExit("Aucun patient à traiter après filtrage par la liste.")

    cpu = os.cpu_count() or 1
    n_workers = (cpu - 1 if args.workers == 0 else args.workers)
    n_workers = min(max(1, n_workers), len(patients))

    # Parse notch list
    notch_freqs = None
    if args.filter and args.notch:
        s = args.notch.strip()
        if s:
            notch_freqs = [int(x) for x in s.split(",") if x.strip().isdigit()]
            if not notch_freqs:
                notch_freqs = None

    print(f"Patients détectés: {[b for b,_ in patients]}")
    print(f"Lancement en parallèle avec {n_workers} workers (overwrite={args.overwrite})")
    print(f"Montage: {args.montage}")
    if args.filter:
        print(f"Filtrage activé: band [{args.l_freq}-{args.h_freq}] Hz, notch={notch_freqs or 'None'}")

    manager = mp.Manager()
    skipped = manager.list()

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        maxtasksperchild=1,
        initializer=_init_worker,
        initargs=(str(out_root), args.overwrite, args.montage,
                  bool(args.filter), float(args.l_freq), float(args.h_freq), notch_freqs,
                  skipped),
    ) as pool:
        for _ in pool.imap_unordered(process_one, patients, chunksize=1):
            pass

    # Récapitulatif
    if skipped:
        print("\n=== RÉCAP PATIENTS NON TRAITÉS ===")
        for base, missing in skipped:
            print(f"- {base}: manquant {missing}")
        recap_file = out_root / "patients_skipped.txt"
        with open(recap_file, "w", encoding="utf-8") as f:
            for base, missing in skipped:
                f.write(f"{base}: manquant {missing}\n")
        print(f"\nRécap sauvegardé dans {recap_file}")
    else:
        print("\nTous les patients ont été traités avec succès !")


if __name__ == "__main__":
    main()
