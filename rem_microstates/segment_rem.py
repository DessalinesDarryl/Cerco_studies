def load_annotation_file(txt_path):
    """
    Charge un fichier d’annotations avec structure :
    <start_sec> <time_str> <stage_str> <code>
    Retourne les segments REM (début, fin) en secondes.
    """
    segments = []
    current_start = None

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) >= 3:
            start_sec = float(parts[0])
            stage = parts[2].upper()

            if stage == 'REM':
                if current_start is None:
                    current_start = start_sec
            else:
                if current_start is not None:
                    segments.append((current_start, start_sec))
                    current_start = None

    # Fin du fichier
    if current_start is not None:
        segments.append((current_start, start_sec))

    return segments


def extract_rem_segments(txt_path):
    return load_annotation_file(txt_path)
