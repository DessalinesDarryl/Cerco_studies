import mne

def apply_custom_bipolar_montage(raw):
    """
    Applique un montage bipolaire personnalisé à l'objet raw,
    selon la configuration MYMONTAGE_BIP fournie.
    """
    # Nettoyage des noms de canaux : supprime le préfixe "EEG "
    raw.rename_channels(lambda name: name.replace("EEG ", ""))

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

    # Crée les canaux bipolaires
    bipolars = []
    for new_name, (anode, cathode) in bipolar_mappings.items():
        raw_bip = mne.set_bipolar_reference(raw, anode, cathode,
                                            ch_name=new_name,
                                            drop_refs=False,
                                            copy=True)
        bipolars.append(raw_bip.pick_channels([new_name]))

    # Récupère les canaux monopolaire à conserver
    monopolar_keep = ["Menton", "JAMBG", "JAMBD", "RONF", "EMG1", "EMG2", "ECG"]
    raw_mono = raw.copy().pick_channels(monopolar_keep)

    # Concatène les canaux bipolaires + monopolaire
    raw_final = mne.concatenate_raws(bipolars + [raw_mono])
    return raw_final
