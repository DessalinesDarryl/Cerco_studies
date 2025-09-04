#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bipolar_from_edf_batch.py — Convertit des enregistrements .edf en .fif avec montage bipolaire.

- Entrée : répertoire racine contenant des sous-dossiers patients (ex. AE129/AE129_raw.edf)
- Sortie : out_root/{patient}/{patient}_bipolar.fif
- Parallélisation avec multiprocessing.Pool
- Option --overwrite pour forcer la régénération

Usage:
    python bipolar_from_edf_batch.py --in_root /chemin/EEG/raw --out_root /chemin/EEG/preprocessed/bipolaire --workers 4 [--overwrite]
"""

from __future__ import annotations
import os
import multiprocessing as mp
import tempfile
from pathlib import Path
import argparse
import mne

# ---------------- Montage ----------------

EEG_BIP  = ["Fp2-C4", "C4-O2", "T4-O2", "Cz-Pz", "Fp1-C3", "C3-O1", "Fp1-T3", "T3-O1"]
EOG_BIP  = ["EOGD-A1", "EOGG-A1"]
KEEP_RAW = ["Menton", "JAMBG", "JAMBD", "RONF", "EMG1", "EMG2", "ECG"]

class MissingChannelsError(Exception):
    def __init__(self, base: str, missing: list[str]):
        super().__init__(f"[{base}] canaux manquants: {missing}")
        self.base = base
        self.missing = missing

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

def apply_bipolar_montage(inst: mne.io.BaseRaw, base: str="") -> list[str]:
    pairs = [
        ("EOGD", "A1",  "EOGD-A1"),
        ("EOGG", "A1",  "EOGG-A1"),
        ("Fp2",  "C4",  "Fp2-C4"),
        ("C4",   "O2",  "C4-O2"),
        ("T4",   "O2",  "T4-O2"),
        ("Cz",   "Pz",  "Cz-Pz"),
        ("Fp1",  "C3",  "Fp1-C3"),
        ("C3",   "O1",  "C3-O1"),
        ("Fp1",  "T3",  "Fp1-T3"),
        ("T3",   "O1",  "T3-O1"),
    ]
    for a, c, n in pairs:
        _safe_bipolar(inst, a, c, n, base)

    type_map = {}
    for ch in EEG_BIP:
        if ch in inst.ch_names: type_map[ch] = "eeg"
    for ch in EOG_BIP:
        if ch in inst.ch_names: type_map[ch] = "eog"
    for ch in ("Menton","EMG1","EMG2","JAMBG","JAMBD"):
        if ch in inst.ch_names: type_map[ch] = "emg"
    if "ECG" in inst.ch_names: type_map["ECG"] = "ecg"
    if "RONF" in inst.ch_names: type_map["RONF"] = "misc"
    if type_map:
        inst.set_channel_types(type_map)

    desired = EOG_BIP + EEG_BIP + KEEP_RAW
    present = [ch for ch in desired if ch in inst.ch_names]
    if not present:
        missing = [ch for ch in desired if ch not in inst.ch_names]
        raise MissingChannelsError(base, missing)

    inst.pick(present)
    print(f"[{base}] Montage bipolaire OK. Canaux conservés: {inst.ch_names}")
    return inst.ch_names

# ---------------- Découverte fichiers EDF ----------------

def discover_patients(edf_root: Path) -> list[tuple[str, Path]]:
    mapping = {}
    for child in sorted(edf_root.iterdir()):
        if child.is_dir():
            edf_candidates = [p for p in child.glob("*_raw.edf")] + [p for p in child.glob("*_raw.EDF")]
            if edf_candidates:
                mapping[child.name] = max(edf_candidates, key=lambda p: p.stat().st_size)
    return sorted(mapping.items())

# ---------------- Worker ----------------

SKIPPED = None  # liste partagée entre workers

def process_one(item, out_root: Path, overwrite: bool):
    base, edf_path = item
    out_dir = out_root / base
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base}_bipolar.fif"

    if out_path.exists() and not overwrite:
        print(f"[{base}] Déjà traité -> skip (utiliser --overwrite pour refaire)")
        return

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        # --- correction des noms de canaux : suppression du préfixe "EEG " ---
        new_names = {ch: ch.replace("EEG ", "") for ch in raw.ch_names if ch.startswith("EEG ")}
        if new_names:
            raw.rename_channels(new_names)
            print(f"[{base}] Canaux renommés (suppression du préfixe 'EEG '): {new_names}")

        apply_bipolar_montage(raw, base)
        raw.save(out_path, overwrite=True)
        print(f"[{base}] Sauvé -> {out_path}")
    except MissingChannelsError as e:
        print(f"[{base}] Sauté : canaux manquants {e.missing}")
        SKIPPED.append((e.base, e.missing))
    except Exception as e:
        print(f"[{base}] Erreur inattendue: {e}")
        SKIPPED.append((base, [str(e)]))

def _init_worker(gl_out_root: str, gl_overwrite: bool, skipped):
    global OUT_ROOT, OVERWRITE, SKIPPED
    OUT_ROOT = Path(gl_out_root)
    OVERWRITE = gl_overwrite
    SKIPPED = skipped
    try:
        mpl_cache = os.path.join(tempfile.gettempdir(), f"mplcache_{os.getpid()}")
        os.environ["MPLCONFIGDIR"] = mpl_cache
        os.makedirs(mpl_cache, exist_ok=True)
    except Exception:
        pass

def _worker_wrapper(item):
    return process_one(item, OUT_ROOT, OVERWRITE)

# ---------------- Main ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_root", type=str, required=False, help="Dossier contenant les dossiers patients avec .edf")
    parser.add_argument("--out_root", type=str, required=False, help="Dossier de sortie pour les .fif")
    parser.add_argument("--workers", type=int, default=8, help="Nb de workers (0=CPU-1)")
    parser.add_argument("--overwrite", action="store_true", help="Écraser les fichiers déjà existants")
    args = parser.parse_args()

    edf_root = Path("/home/darryld/documents/EEG/raw") if args.in_root == None else Path(args.in_root)
    out_root = Path("/home/darryld/documents/EEG/preprocessed/bipolaire/0_raw_bip") if args.out_root == None else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    patients = discover_patients(edf_root)
    if not patients:
        raise SystemExit("Aucun fichier *_raw.edf ou *_raw.EDF trouvé")

    cpu = os.cpu_count() or 1
    n_workers = (cpu-1 if args.workers == 0 else args.workers)
    n_workers = min(max(1,n_workers), len(patients))
    print(f"Patients détectés: {[b for b,_ in patients]}")
    print(f"Lancement en parallèle avec {n_workers} workers (overwrite={args.overwrite})")

    manager = mp.Manager()
    skipped = manager.list()

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        maxtasksperchild=1,
        initializer=_init_worker,
        initargs=(str(out_root), args.overwrite, skipped),
    ) as pool:
        for _ in pool.imap_unordered(_worker_wrapper, patients, chunksize=1):
            pass

    # Récapitulatif
    if skipped:
        print("\n=== RÉCAP PATIENTS NON TRAITÉS ===")
        for base, missing in skipped:
            print(f"- {base}: manquant {missing}")
        recap_file = out_root / "patients_skipped.txt"
        with open(recap_file, "w") as f:
            for base, missing in skipped:
                f.write(f"{base}: manquant {missing}\n")
        print(f"\nRécap sauvegardé dans {recap_file}")
    else:
        print("\nTous les patients ont été traités avec succès !")
