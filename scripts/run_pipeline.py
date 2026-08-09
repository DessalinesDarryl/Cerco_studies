#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
run_pipeline.py - Pipeline RSWA en 5 etapes, selectionnables individuellement
=============================================================================

Remplace run_full_pipeline.py par une version DECOUPEE EN ETAPES
INDEPENDANTES, chacune lisant/ecrivant sous results/ (cf. results_io.py).
Cela permet de relancer uniquement une partie du pipeline (ex. relancer
la reduction de dimension sans re-extraire tous les patients).

ETAPES
------
  1) Extraction par patient (parallelisee, ProcessPoolExecutor)
     -> results/cohort/per_patient/<patient_id>.csv
  2) Assemblage de la cohorte (concatene tous les per_patient/*.csv)
     -> results/cohort/cohort_metrics_real.csv
  3) Reduction de dimension (Spearman -> VIF -> KMO/Bartlett -> ACP)
     -> results/dimensionality_reduction/
  4) Score composite de severite (necessite un groupe controle ;
     ajoute severity_score_mean / severity_score_pca a la cohorte)
     -> results/cohort/cohort_metrics_real.csv (mis a jour)
  5) Comparaison statistique multi-groupes (Kruskal-Wallis / Mann-Whitney
     / ROC-vs-controles si applicable)
     -> results/statistics/

Chaque etape verifie la presence des fichiers dont elle depend et
s'arrete avec un message explicite si un prerequis manque -- au lieu de
planter au milieu d'un traitement.

USAGE
-----
    python run_pipeline.py                  # toutes les etapes (1-5)
    python run_pipeline.py --steps 1 2       # extraction + assemblage seuls
    python run_pipeline.py --steps 3         # reduction de dimension seule
    python run_pipeline.py --steps 2 4       # reassemblage + score composite
    python run_pipeline.py --steps 5         # stats seules (relance rapide
                                              # apres avoir change CONTROL_LABEL
                                              # ou CANDIDATE_FEATURES par ex.)

    python run_pipeline.py --steps 1 2 3 4 5 2>&1 | tee logfiles/run_pipeline-full_$(date +%Y%m%d_%H%M%S).txt
=============================================================================
"""

# ============================================================================
# PARAMETRES  <<<  A MODIFIER SELON CONFIGURATION SOUHAITEE
# ============================================================================

RAW_DATA_ROOT = r"c:\dev\raw\data_raw_fif"
# ^ Peut pointer directement vers vos .edf BRUTS (pas besoin d'avoir
#   lance 01_preprocess.py au prealable) OU vers des .fif deja convertis
#   -- les deux formats sont geres automatiquement (cf. discover_patient_recordings
#   ci-dessous et human_data_adapter.py qui choisit le lecteur mne selon
#   l'extension). Le filtrage EMG (preprocessing.py, 10-100 Hz + notch)
#   est fait EN INTERNE par human_data_adapter.py  

HYPNO_ROOT = r"c:\dev\raw\hypnogrammes"          # None si annotations deja dans le .fif
LABELS_TXT = r"c:\dev\Cerco_studies\patients_label.txt"
CHANNEL_NAME = "Menton"

FOLD = 4.0
EPOCH_DURATION_S = 1.0
DURATION_METHOD = "gmm"
TONIC_EPOCH_S = 30
DETECT_ARTIFACTS = True

N_WORKERS = 8

CONTROL_LABEL = "iRBD"   # None si pas de groupe controle dans vos donnees ("control"/None)

CANDIDATE_FEATURES = [
    "AI_atonia_index", "pct_epochs_phasic", "pct_any_RSWA", "pct_tonic",
    "episode_count", "phasic_density_per_min",
    "duration_mean", "duration_median", "duration_sd", "duration_max",
    "interval_mean", "interval_median", "interval_variance", "interval_cv",
    "mean_relative_amplitude", "max_relative_amplitude",
    "short_density_per_min", "medium_density_per_min", "long_density_per_min",
    "phasic_tonic_ratio", "progression_index",
]

VIF_THRESHOLD = 5.0
CORR_THRESHOLD = 0.8
VARIANCE_THRESHOLD = 0.80
IMPUTE_MISSING = True

# ============================================================================
# IMPORTS
# ============================================================================

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from results_io import results_path, save_dataframe


def _get_logger(name: str = "run_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


COHORT_CSV_PARTS = ("cohort", "cohort_metrics_real.csv")
RETAINED_FEATURES_PARTS = ("dimensionality_reduction", "vif_retained_features.txt")


def _cohort_csv_path():
    return results_path(*COHORT_CSV_PARTS)


def _require_file(path: Path, log, step_needed: str) -> bool:
    if not path.exists():
        log.error(
            f"Fichier requis introuvable : {path}\n"
            f"-> Lancez d'abord l'etape {step_needed} (python run_pipeline.py --steps {step_needed})."
        )
        return False
    return True


# ============================================================================
# DECOUVERTE DES FICHIERS / LABELS (identique aux scripts precedents)
# ============================================================================

def discover_patient_recordings(root):
    """
    Decouvre les enregistrements patients sous `root`, de facon
    FLEXIBLE : les deux structures suivantes sont geree, y compris si
    elles COEXISTENT dans le meme RAW_DATA_ROOT (certains patients en
    sous-dossier, d'autres a plat) :

    1) Sous-dossier dedie par patient :
           {root}/{patient_id}/{...}.edf ou .fif
       -> patient_id = nom du sous-dossier direct du fichier.

    2) Fichier directement a la racine, un par patient :
           {root}/{patient_id}_....edf ou .fif   (ex. AN166_raw.edf)
       -> patient_id extrait du nom de fichier (partie avant le premier
       "_"), meme convention que 01_preprocess.py::_process_one_edf.

    La recherche est RECURSIVE (rglob) pour capter les deux cas en un
    seul passage. Si plusieurs fichiers candidats existent pour un
    meme patient (ex. .fif deja pretraite ET .edf brut), le .fif est
    priorise (evite de refiltrer un signal deja pretraite).

    Retourne une liste triee de tuples (patient_id, recording_path).
    """
    root = Path(root)
    patterns = ("*.fif", "*.FIF", "*.edf", "*.EDF")

    all_files = []
    for pat in patterns:
        all_files.extend(f for f in root.rglob(pat) if f.is_file())
    all_files = sorted(set(all_files))

    # .fif priorise sur .edf pour un meme patient
    all_files.sort(key=lambda f: 0 if f.suffix.lower() == ".fif" else 1)

    found = {}
    for f in all_files:
        if f.parent != root:
            # Fichier dans un sous-dossier -> patient_id = nom du sous-dossier
            patient_id = f.parent.name
        else:
            # Fichier a plat a la racine -> patient_id extrait du nom
            patient_id = f.stem.split("_")[0]
        found.setdefault(patient_id, f)

    return sorted(found.items())


def find_hypno_file(patient_id, hypno_root):
    if hypno_root is None:
        return None
    hypno_root = Path(hypno_root)
    candidates = list(hypno_root.glob(f"{patient_id}*hypno*.txt"))
    candidates += list(hypno_root.glob(f"{patient_id}*hypno*.csv"))
    sub = hypno_root / patient_id
    if sub.exists():
        candidates += list(sub.glob("*hypno*.txt"))
        candidates += list(sub.glob("*hypno*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def to_macro_label(raw_label):
    u = str(raw_label).strip().upper()
    if any(k in u for k in ("PARK", "MPI", "AMS", "DCL", "DLB", "PAF")):
        return "Syn"
    if "NARCO" in u:
        return "Narco"
    if "TCSP" in u or "RBDI" in u:
        return "iRBD"
    if "EAI" in u or "ENCEPHALITE" in u:
        return "AI"
    return str(raw_label).strip()


def load_labels(labels_txt):
    lbl = pd.read_csv(labels_txt, header=None, names=["patient_id", "label"])
    lbl["patient_id"] = lbl["patient_id"].astype(str).str.strip()
    lbl["label"] = lbl["label"].astype(str).str.strip().map(to_macro_label)
    return dict(zip(lbl["patient_id"], lbl["label"]))


def _process_one_patient_worker(fif_path_str, hypno_path_str, diagnosis, patient_id):
    """Fonction top-level (picklable) executee dans un processus separe."""
    from main_pipeline import run_subject_pipeline_from_human_recording

    hypno_path = hypno_path_str if hypno_path_str else None
    try:
        metrics = run_subject_pipeline_from_human_recording(
            fif_path_str, channel_name=CHANNEL_NAME, diagnosis=diagnosis,
            subject_id=patient_id, hypno_path=hypno_path, fold=FOLD,
            epoch_duration_s=EPOCH_DURATION_S, duration_method=DURATION_METHOD,
            tonic_epoch_s=TONIC_EPOCH_S, detect_artifacts=DETECT_ARTIFACTS,
        )
        return patient_id, metrics, None
    except Exception as e:
        return patient_id, None, f"{type(e).__name__}: {e}"


# ============================================================================
# ETAPE 1 : EXTRACTION PAR PATIENT (parallelisee)
# ============================================================================

def step1_extract_patients(log):
    raw_root = Path(RAW_DATA_ROOT)
    if not raw_root.is_dir():
        log.error(f"[Etape 1] RAW_DATA_ROOT introuvable : {raw_root}")
        return False

    labels_map = {}
    if LABELS_TXT is not None and Path(LABELS_TXT).exists():
        labels_map = load_labels(LABELS_TXT)
        log.info(f"[Etape 1] {len(labels_map)} labels charges depuis {LABELS_TXT}")
    else:
        log.warning(f"[Etape 1] LABELS_TXT introuvable ({LABELS_TXT}) -> diagnosis='unknown'.")

    recordings = discover_patient_recordings(raw_root)
    if not recordings:
        log.error(f"[Etape 1] Aucun fichier .fif/.edf trouve sous {raw_root} "
                  f"(ni a la racine, ni dans des sous-dossiers).")
        return False
    log.info(f"[Etape 1] {len(recordings)} enregistrements patients trouves sous {raw_root}")

    tasks = []
    for patient_id, fif_path in recordings:
        hypno_path = find_hypno_file(patient_id, HYPNO_ROOT)
        diagnosis = labels_map.get(patient_id, "unknown")
        if diagnosis == "unknown":
            log.warning(f"[Etape 1][{patient_id}] Pas de label trouve dans {LABELS_TXT}.")
        tasks.append((str(fif_path), str(hypno_path) if hypno_path else "", diagnosis, patient_id))

    if not tasks:
        log.error(f"[Etape 1] Aucun patient exploitable trouve sous {raw_root}.")
        return False

    log.info(f"[Etape 1] {len(tasks)} patients a traiter avec {N_WORKERS} worker(s).")

    n_ok, n_err = 0, 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_process_one_patient_worker, *t): t[3] for t in tasks}
        for fut in as_completed(futures):
            patient_id = futures[fut]
            try:
                pid, metrics, err = fut.result()
            except Exception as e:
                log.error(f"[Etape 1][{patient_id}] ERREUR (process) : {e}")
                n_err += 1
                continue

            if err is not None:
                log.error(f"[Etape 1][{pid}] ERREUR : {err}")
                n_err += 1
                continue

            save_dataframe(pd.DataFrame([metrics]), "cohort", "per_patient", f"{pid}.csv")
            log.info(f"[Etape 1][{pid}] OK - AI={metrics['AI_atonia_index']:.3f} "
                     f"({metrics['AI_atonia_index_category']}), "
                     f"episodes={metrics['episode_count']}, "
                     f"artefacts={metrics['pct_epochs_artifact']:.1f}%")
            n_ok += 1

    log.info(f"[Etape 1] Termine : {n_ok} patients OK, {n_err} en erreur.")
    return n_ok > 0


# ============================================================================
# ETAPE 2 : ASSEMBLAGE DE LA COHORTE
# ============================================================================

def step2_assemble_cohort(log):
    per_patient_dir = results_path("cohort", "per_patient", ".keep").parent
    csv_files = sorted(per_patient_dir.glob("*.csv"))
    if not csv_files:
        log.error(
            f"[Etape 2] Aucun fichier sous {per_patient_dir} -> lancez d'abord "
            f"l'etape 1 (python run_pipeline.py --steps 1)."
        )
        return False

    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    out_path = save_dataframe(df, *COHORT_CSV_PARTS)
    log.info(f"[Etape 2] Cohorte assemblee ({len(df)} patients, {len(csv_files)} fichiers) "
             f"-> {out_path}")
    return True


# ============================================================================
# ETAPE 3 : REDUCTION DE DIMENSION
# ============================================================================

def step3_dimensionality_reduction(log):
    from multicollinearity_pca import (
        full_dimensionality_reduction_report,
        save_dimensionality_reduction_report,
    )

    cohort_path = _cohort_csv_path()
    if not _require_file(cohort_path, log, "2"):
        return False

    df = pd.read_csv(cohort_path)
    features = [f for f in CANDIDATE_FEATURES if f in df.columns]
    missing = [f for f in CANDIDATE_FEATURES if f not in df.columns]
    if missing:
        log.warning(f"[Etape 3] Variables candidates absentes (ignorees) : {missing}")

    log.info(f"[Etape 3] Reduction de dimension sur {len(features)} variables...")
    report = full_dimensionality_reduction_report(
        df, features, vif_threshold=VIF_THRESHOLD, corr_threshold=CORR_THRESHOLD,
        variance_threshold=VARIANCE_THRESHOLD, impute_missing=IMPUTE_MISSING,
        verbose=True,
    )
    paths = save_dimensionality_reduction_report(report)
    log.info(f"[Etape 3] Rapport sauvegarde sous {results_path('dimensionality_reduction', '.').parent}")
    log.info(f"[Etape 3] Variables retenues : {report['retained_features']}")
    return True


# ============================================================================
# ETAPE 4 : SCORE COMPOSITE DE SEVERITE
# ============================================================================

def step4_composite_score(log):
    from statistics_pipeline import zscore_against_controls, composite_severity_score

    cohort_path = _cohort_csv_path()
    if not _require_file(cohort_path, log, "2"):
        return False
    retained_path = results_path(*RETAINED_FEATURES_PARTS)
    if not _require_file(retained_path, log, "3"):
        return False

    if CONTROL_LABEL is None:
        log.warning(
            "[Etape 4] CONTROL_LABEL est a None -> pas de groupe controle "
            "defini dans vos donnees. Score composite DESACTIVE -- ce "
            "n'est PAS une erreur, l'etape 5 comparera les groupes "
            "cliniques entre eux sans reference controle."
        )
        return True

    df = pd.read_csv(cohort_path)
    if not (df["diagnosis"] == CONTROL_LABEL).any():
        log.warning(
            f"[Etape 4] Aucun sujet avec diagnosis == '{CONTROL_LABEL}' dans "
            f"{cohort_path} -> score composite DESACTIVE -- ce n'est PAS "
            f"une erreur, l'etape 5 comparera les groupes cliniques entre "
            f"eux sans reference controle."
        )
        return True

    retained = [l.strip() for l in retained_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    retained = [f for f in retained if f in df.columns]

    z_df, feats = zscore_against_controls(
        df, control_label=CONTROL_LABEL, group_col="diagnosis", feature_cols=retained
    )
    if "AI_atonia_index" in feats:
        z_df["AI_atonia_index"] = -z_df["AI_atonia_index"]

    df["severity_score_mean"] = composite_severity_score(z_df, feats, method="mean")
    pca_score, explained_var = composite_severity_score(z_df, feats, method="pca")
    df["severity_score_pca"] = pca_score

    out_path = save_dataframe(df, *COHORT_CSV_PARTS)
    log.info(f"[Etape 4] Score composite ajoute (variance PC1 = {explained_var:.2%}) "
             f"-> {out_path}")
    return True


# ============================================================================
# ETAPE 5 : COMPARAISON STATISTIQUE MULTI-GROUPES
# ============================================================================

def step5_statistics(log):
    from statistics_pipeline import (
        full_group_comparison,
        kruskal_wallis_by_feature,
        pairwise_mannwhitney_bonferroni,
        save_group_comparison,
    )

    cohort_path = _cohort_csv_path()
    if not _require_file(cohort_path, log, "2"):
        return False
    retained_path = results_path(*RETAINED_FEATURES_PARTS)
    if not _require_file(retained_path, log, "3"):
        return False

    df = pd.read_csv(cohort_path)
    retained = [l.strip() for l in retained_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    retained = [f for f in retained if f in df.columns]

    has_scores = "severity_score_mean" in df.columns
    has_control = CONTROL_LABEL is not None and (df["diagnosis"] == CONTROL_LABEL).any()

    if has_control:
        feature_cols = retained + (["severity_score_mean", "severity_score_pca"] if has_scores else [])
        log.info(f"[Etape 5] Groupe controle '{CONTROL_LABEL}' present -> "
                 f"Kruskal-Wallis + Mann-Whitney + ROC-vs-controles.")
        kw_df, mw_df, roc_df = full_group_comparison(
            df, feature_cols=feature_cols, group_col="diagnosis", control_label=CONTROL_LABEL
        )
    else:
        if not has_scores:
            log.warning(
                "[Etape 5] Pas de score composite dans la cohorte (etape 4 non "
                "executee ou pas de controle) -> comparaison sur les variables "
                "individuelles uniquement."
            )
        log.warning(
            f"[Etape 5] Pas de groupe controle -> ROC desactive ; "
            f"Kruskal-Wallis + Mann-Whitney pairwise entre groupes cliniques presents."
        )
        kw_rows, mw_frames = [], []
        for feat in retained:
            stat, p = kruskal_wallis_by_feature(df, feat, group_col="diagnosis")
            kw_rows.append({"feature": feat, "H": stat, "p": p})
            mw_df_f, _ = pairwise_mannwhitney_bonferroni(df, feat, group_col="diagnosis")
            if not mw_df_f.empty:
                mw_df_f["feature"] = feat
                mw_frames.append(mw_df_f)
        kw_df = pd.DataFrame(kw_rows)
        mw_df = pd.concat(mw_frames, ignore_index=True) if mw_frames else pd.DataFrame()
        roc_df = pd.DataFrame()

    paths = save_group_comparison(kw_df, mw_df, roc_df)
    log.info(f"[Etape 5] Comparaison statistique sauvegardee -> {list(paths.values())}")
    return True


# ============================================================================
# POINT D'ENTREE
# ============================================================================

STEP_FUNCS = {
    1: ("Extraction par patient", step1_extract_patients),
    2: ("Assemblage de la cohorte", step2_assemble_cohort),
    3: ("Reduction de dimension", step3_dimensionality_reduction),
    4: ("Score composite de severite", step4_composite_score),
    5: ("Comparaison statistique", step5_statistics),
}


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline RSWA en 5 etapes selectionnables (cf. docstring du fichier)."
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", choices=sorted(STEP_FUNCS.keys()),
        default=sorted(STEP_FUNCS.keys()),
        help="Etapes a executer, dans l'ordre donne. Defaut : toutes (1 2 3 4 5).",
    )
    args = parser.parse_args()

    log = _get_logger()
    log.info(f"Etapes demandees : {args.steps}")

    for step_num in args.steps:
        name, func = STEP_FUNCS[step_num]
        log.info("=" * 70)
        log.info(f"ETAPE {step_num} : {name}")
        log.info("=" * 70)
        ok = func(log)
        if not ok:
            log.error(f"Etape {step_num} ({name}) a echoue ou a ete ignoree -> arret.")
            sys.exit(1)

    log.info("Pipeline termine (etapes demandees executees avec succes).")


if __name__ == "__main__":
    main()