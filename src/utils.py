# src/utils.py

import os
import mne


from pathlib import Path

def list_fif_files(directory: str):
    root = Path(directory)
    out = []
    for p in root.rglob("*.fif"):
        name = p.name
        if name.startswith(("._", ".")):
            continue
        out.append(str(p.relative_to(root)))
    return sorted(out)



def get_base_name(file_path, suffix="_eeg_cleaned_raw.fif"):
    """
    Supprime le suffixe standard pour obtenir le nom de base du fichier.

    Args:
        file_path (str): Nom de fichier
        suffix (str): Suffixe à retirer

    Returns:
        str: Nom de base sans suffixe
    """
    return os.path.basename(file_path).replace(suffix, "")


def apply_custom_bipolar_montage(raw):
    """
    Applique un montage bipolaire personnalisé à l'objet raw.
    Ignore les paires si les canaux n'existent pas.
    """
    bipolar_mappings = {
        "EOGD-A1": ("EOGD", "A1"),
        "EOGG-A1": ("EOGG", "A1"),
        "Fp2-C4": ("Fp2", "C4"),
        "C4-O2": ("C4", "O2"),
        "T4-O2": ("T4", "O2"),
        "Cz-Pz": ("Cz", "Pz"),
        "Fp1-C3": ("Fp1", "C3"),
        "C3-O1": ("C3", "O1"),
        "Fp1-T3": ("Fp1", "T3"),
        "T3-O1": ("T3", "O1")
    }

    raw_bip = raw.copy()
    all_new_channels = []

    for new_name, (anode, cathode) in bipolar_mappings.items():
        if anode in raw_bip.ch_names and cathode in raw_bip.ch_names:
            raw_bip = mne.set_bipolar_reference(
                raw_bip, anode, cathode,
                ch_name=new_name,
                drop_refs=False,
                copy=False
            )
            all_new_channels.append(new_name)
            print(f"[OK] Canal bipolaire ajouté : {new_name}")
        else:
            print(f"[SKIP] Canal ignoré : {new_name} (manque {anode} ou {cathode})")

    # Canaux monopolaire à conserver
    keep_channels = all_new_channels + [
        "Menton", "JAMBG", "JAMBD", "RONF", "EMG1", "EMG2", "ECG"
    ]
    keep_channels = [ch for ch in keep_channels if ch in raw_bip.ch_names]

    raw_bip.pick_channels(keep_channels)
    return raw_bip
