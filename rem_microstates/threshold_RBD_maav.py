#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")  # backend non interactif
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= Paramètres par défaut (alignés sur le script RAW) =================
DEFAULT_WIN_LEN_S   = 0.1           # longueur fenêtre MAAV (s)
DEFAULT_OVERLAP     = 0.5           # recouvrement MAAV
DEFAULT_HP          = 10.0          # passe-haut EMG
DEFAULT_LP          = 100.0         # passe-bas EMG
DEFAULT_NOTCH       = 50.0          # notch secteur
DEFAULT_NOTCH_Q     = 30.0          # facteur Q notch
DEFAULT_EDGE_GUARD  = 0.5           # marge aux bords des RBD (s)
DEFAULT_ONLY_REM    = True          # ne garder que les points en REM
DEFAULT_CLIP_Q      = 99.9          # clipping des scores (anti-outliers, en quantile)

def to_uV(x): return np.asarray(x) * 1e6

def log(msg=""):
    print(msg, flush=True)

# ---------------- Hypnogramme (REM) ----------------
def read_hypnogram_intervals_rem(path: str):
    if path is None or not os.path.isfile(path):
        log(f"[WARN] Hypnogramme manquant : {path}")
        return [], 30.0
    df = pd.read_csv(path, sep=r"\s+", engine="python", header=None, comment="#",
                     names=["t_end_s","hhmmss","stage","extra"], usecols=[0,1,2],
                     dtype={0:float,1:str,2:str}, encoding_errors="ignore")
    df = df.dropna(subset=["t_end_s","stage"]).copy()
    df["stage"] = df["stage"].astype(str).str.strip().str.upper()
    te = df["t_end_s"].to_numpy()
    diffs = np.diff(te) if len(te)>1 else np.array([])
    epoch_len = float(pd.Series(np.round(diffs,1)).mode().iloc[0]) if diffs.size else 30.0
    is_rem = df["stage"].isin(["R","REM"])
    rem_intervals, run_start, prev_end = [], None, None
    for t_end, flag in zip(df["t_end_s"], is_rem):
        t_start = t_end - epoch_len
        if flag:
            if run_start is None: run_start = t_start
            prev_end = t_end
        else:
            if run_start is not None:
                rem_intervals.append((float(run_start), float(prev_end)))
                run_start, prev_end = None, None
    if run_start is not None and prev_end is not None:
        rem_intervals.append((float(run_start), float(prev_end)))
    return rem_intervals, epoch_len

# ---------------- Lecture windows_by_patient.csv ----------------
def load_windows_by_patient_csv(csv_path: str, gp2_dir: str):
    """
    CSV attendu avec au minimum: patient_id, start, end
    Colonnes optionnelles ignorées: duration(s), description, extrait/entier, etc.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")
    if not os.path.isdir(gp2_dir):
        raise FileNotFoundError(f"Dossier gp2 introuvable: {gp2_dir}")

    df = pd.read_csv(csv_path)
    # normalisation noms possibles
    cols = {c.lower(): c for c in df.columns}
    for key in ("patient_id", "patient", "id", "identifiant"):
        if key in cols: 
            pid_col = cols[key]; break
    else:
        raise ValueError("Colonne patient_id/patient/id/identifiant absente du CSV.")
    for key in ("start", "debut", "debut_rbd(s)"):
        if key in cols:
            start_col = cols[key]; break
    else:
        raise ValueError("Colonne start/debut absente du CSV.")
    for key in ("end", "fin", "fin_rbd(s)"):
        if key in cols:
            end_col = cols[key]; break
    else:
        raise ValueError("Colonne end/fin absente du CSV.")

    # nettoyer / caster
    sub = df[[pid_col, start_col, end_col]].copy()
    sub.rename(columns={pid_col:"patient_id", start_col:"start", end_col:"end"}, inplace=True)
    sub["patient_id"] = sub["patient_id"].astype(str).str.strip()
    sub["start"] = pd.to_numeric(sub["start"], errors="coerce")
    sub["end"]   = pd.to_numeric(sub["end"],   errors="coerce")
    sub = sub.dropna(subset=["start","end"])
    sub = sub[sub["end"] > sub["start"]].copy()

    # filtrer sur patients réellement présents sur disque
    patients_gp2 = {name.strip() for name in os.listdir(gp2_dir)
                    if os.path.isdir(os.path.join(gp2_dir, name))}
    sub = sub[sub["patient_id"].isin(patients_gp2)].copy()

    # construire dict
    from collections import defaultdict
    windows_by_patient = defaultdict(list)
    for pid, g in sub.groupby("patient_id"):
        rows = g.sort_values("start").to_dict(orient="records")
        for r in rows:
            windows_by_patient[pid].append({
                "start": float(r["start"]),
                "end":   float(r["end"]),
            })

    log(f"[INFO] Patients listés dans CSV (présents sur le serveur): {len(windows_by_patient)}")
    log(f"[INFO] Total de fenêtres RBD (CSV) : {sum(len(v) for v in windows_by_patient.values())}")
    return windows_by_patient

# ---------------- MAAV & prétraitement EMG ----------------
def compute_maav(x, sfreq, win_len_s, overlap):
    x = np.asarray(x).ravel()
    n = len(x)
    L = int(round(win_len_s*sfreq))
    L = max(L, 1)
    H = int(round(L*(1-overlap)))
    H = max(H, 1)
    starts = np.arange(0, max(n-L+1,0), H, dtype=int)
    mav, tcent = [], []
    for s0 in starts:
        s1 = s0 + L
        if s1 > n: break
        mav.append(np.mean(np.abs(x[s0:s1])))
        tcent.append((s0+s1)/(2*sfreq))
    return np.asarray(mav), np.asarray(tcent)

def preprocess_emg_channel(raw: mne.io.BaseRaw, pick_idx: int,
                           hp, lp, notch_freq, notch_q):
    raw1 = raw.copy().load_data()
    ch_name = raw1.ch_names[pick_idx]
    if raw1.get_channel_types(picks=[pick_idx])[0] != 'emg':
        raw1.set_channel_types({ch_name:'emg'})
    raw_m = raw1.copy().pick(picks=[pick_idx])
    log(f"      [+] Filtrage canal {ch_name} : bandpass {hp}-{lp} Hz, notch {notch_freq} Hz (Q={notch_q})")
    raw_m.filter(l_freq=hp, h_freq=lp, method="iir", picks=[0], verbose=False)
    fs = raw_m.info['sfreq']
    b, a = iirnotch(w0=notch_freq/(fs/2), Q=notch_q)
    data = filtfilt(b, a, raw_m.get_data(), axis=1)
    raw_m._data[:] = data
    return raw_m, ch_name

# ---------------- Helpers temps ----------------
def in_any_interval(t: float, intervals, margin=0.0) -> bool:
    for a,b in intervals:
        if (a+margin) <= t <= (b-margin): return True
    return False

def restrict_to_intervals(tcent, intervals):
    mask = np.zeros_like(tcent, dtype=bool)
    for a,b in intervals: mask |= (tcent >= a) & (tcent <= b)
    return mask

# ---------------- Groupes de canaux ----------------
def channel_group(ch_name: str) -> str:
    n = ch_name.lower()
    if any(k in n for k in ("menton","chin","subment","masseter")): return "chin"
    if "jamb" in n: return "legs"   # JAMBD/JAMBG
    if "emg"  in n: return "arm"    # EMG1/EMG2 (si non mentonniers)
    return "other"

# ---------------- ROC & seuil (Youden) ----------------
def best_threshold_roc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)

    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0 or neg == 0:
        log(f"      [ROC] Labels dégénérés: pos={pos}, neg={neg} → ROC impossible")
        return np.nan, np.nan, np.nan, (np.array([]), np.array([]))

    if not np.isfinite(scores).any():
        log("      [ROC] Scores non finis (NaN/Inf) → ROC impossible")
        return np.nan, np.nan, np.nan, (np.array([]), np.array([]))

    if np.nanstd(scores) == 0.0:
        log("      [ROC] Variance nulle des scores → ROC impossible")
        return np.nan, np.nan, np.nan, (np.array([]), np.array([]))

    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(labels, scores)
    except Exception as e:
        log(f"      [ROC] Exception roc_curve: {e} → ROC impossible")
        return np.nan, np.nan, np.nan, (np.array([]), np.array([]))

    mask = np.isfinite(fpr) & np.isfinite(tpr) & np.isfinite(thr)
    if mask.sum() == 0:
        log("      [ROC] fpr/tpr/thr non finis → ROC impossible")
        return np.nan, np.nan, np.nan, (np.array([]), np.array([]))

    fpr_f = fpr[mask]; tpr_f = tpr[mask]; thr_f = thr[mask]
    j_f = tpr_f - fpr_f
    if j_f.size == 0 or not np.isfinite(j_f).any():
        log("      [ROC] Youden J vide/non fini → ROC impossible")
        return np.nan, np.nan, np.nan, (np.array([]), np.array([]))

    j_max = np.nanmax(j_f)
    cand = np.where(j_f == j_max)[0]
    if cand.size == 0:
        log("      [ROC] Aucun maximum de J détecté → ROC impossible")
        return np.nan, np.nan, np.nan, (np.array([]), np.array([]))
    idx = int(cand[np.nanargmin(fpr_f[cand])])

    return float(thr_f[idx]), float(tpr_f[idx]), float(fpr_f[idx]), (fpr, tpr)

def avg_roc(curves, n_grid=300):
    grid = np.linspace(0,1,n_grid)
    tprs=[]; ws=[]
    for fpr, tpr, w in curves:
        fpr = np.asarray(fpr); tpr = np.asarray(tpr)
        if fpr.size==0 or tpr.size==0: 
            continue
        if fpr[0]>0 or tpr[0]>0:
            fpr = np.r_[0,fpr]; tpr = np.r_[0,tpr]
        if fpr[-1]<1 or tpr[-1]<1:
            fpr = np.r_[fpr,1]; tpr = np.r_[tpr,1]
        tprs.append(np.interp(grid,fpr,tpr)); ws.append(w)
    if not tprs:
        return grid, np.zeros_like(grid), 0.0
    tprs = np.vstack(tprs); ws = np.asarray(ws,float)
    wnorm = ws / ws.sum()
    tpr_mean = (wnorm[:,None]*tprs).sum(axis=0)
    auc_mean = float(np.trapz(tpr_mean, grid))
    return grid, tpr_mean, auc_mean

# ---------------- I/O fichiers patients ----------------
def find_patient_files(pid: str, gp2_dir: str, raw_dir: str):
    fif = sorted(glob.glob(os.path.join(gp2_dir, pid, "*annotated.fif")))
    hyp = sorted(glob.glob(os.path.join(raw_dir, pid, "*hypnoEXP.txt")))
    return (fif[0] if fif else None), (hyp[0] if hyp else None)

# ---------------- Figures ----------------
def save_group_hist(out_dir, group_name, per_group_scores, per_group_labels, df_glob):
    if group_name not in df_glob["group"].values: 
        return None
    scores_uV = np.concatenate(per_group_scores[group_name]) * 1e6
    labels    = np.concatenate(per_group_labels[group_name])

    eps = max(1e-3, np.percentile(scores_uV, 0.1)/10.0)
    lo  = max(eps, np.min(scores_uV[scores_uV>0]) if np.any(scores_uV>0) else eps)
    hi  = np.percentile(scores_uV, 99.5)
    bins = np.logspace(np.log10(lo), np.log10(hi), 70)
    thr_uV = float(df_glob.loc[df_glob["group"].eq(group_name), "thr_uV"].iloc[0])

    fig, ax = plt.subplots(figsize=(7,4.2))
    ax.hist(scores_uV[labels==0], bins=bins, alpha=0.5, density=True, label="Non-RBD (REM)")
    ax.hist(scores_uV[labels==1], bins=bins, alpha=0.5, density=True, label="RBD (REM)")
    ax.axvline(thr_uV, ls="--", lw=2, color="k", label=f"Seuil moyen ≈ {thr_uV:.2f} µV")
    ax.set_xscale("log"); ax.set_xlim(lo, hi)
    ax.set_xlabel("MAAV (µV)"); ax.set_ylabel("densité")
    ax.set_title(f"Histogrammes MAAV • groupe {group_name}")
    ax.grid(which="both", alpha=0.2); ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"hist_maav_{group_name}.png")
    fig.savefig(out_path, dpi=150); plt.close(fig)
    return out_path

# ---------------- Worker parallèle (traitement d'un patient) ----------------
def _process_one_patient(pid, args, windows_by_patient):
    try:
        log(f"[{pid}] ==== DÉBUT patient ====")
        rows_patients = []
        per_group_scores = {"chin": [], "legs": [], "arm": []}
        per_group_labels = {"chin": [], "legs": [], "arm": []}
        per_group_roc_curv = {"chin": [], "legs": [], "arm": []}
        per_group_thr = {"chin": [], "legs": [], "arm": []}

        fif_path, hyp_path = find_patient_files(pid, args.gp2_dir, args.raw_dir)
        if not fif_path:
            log(f"[{pid}] [SKIP] .fif introuvable")
            return (pid, None)

        rbd_intervals = [(float(w['start']), float(w['end'])) for w in windows_by_patient[pid]]
        if len(rbd_intervals) == 0:
            log(f"[{pid}] [SKIP] aucune fenêtre RBD dans le CSV")
            return (pid, None)

        rem_intervals, _ = read_hypnogram_intervals_rem(hyp_path) if args.only_rem else ([], 30.0)
        if args.only_rem and not rem_intervals:
            log(f"[{pid}] [WARN] Aucun intervalle REM trouvé (analyse sans REM si --only-rem=False)")

        log(f"[{pid}] Lecture RAW: {fif_path}")
        raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
        emg_picks = mne.pick_types(raw.info, emg=True)
        log(f"[{pid}] Canaux EMG détectés: {len(emg_picks)} -> {[raw.ch_names[i] for i in emg_picks]}")

        pat_scores = {"chin":[], "legs":[], "arm":[]}
        pat_labels = {"chin":[], "legs":[], "arm":[]}

        for p in emg_picks:
            raw_emg, ch_name = preprocess_emg_channel(
                raw, p, args.hp, args.lp, args.notch, args.notch_q
            )
            g = channel_group(ch_name)
            log(f"[{pid}]   -> Canal {ch_name} mappé en groupe '{g}'")
            if g not in ("chin","legs","arm"):
                log(f"[{pid}]   [IGN] Groupe '{g}' non utilisé")
                continue

            x = raw_emg.get_data()[0]
            sf = raw_emg.info["sfreq"]
            maav, tcent_rel = compute_maav(x, sf, args.win_len, args.overlap)
            tcent = tcent_rel + raw.first_time

            # clipping optionnel
            if args.clip_quantile is not None and args.clip_quantile > 0:
                cap = np.nanpercentile(maav, args.clip_quantile)
                if np.isfinite(cap) and cap > 0:
                    maav = np.clip(maav, 0.0, cap)

            mask_rem = restrict_to_intervals(tcent, rem_intervals) if (args.only_rem and rem_intervals) else np.ones_like(tcent, bool)
            labels = np.array([1 if in_any_interval(t, rbd_intervals, margin=args.edge_guard) else 0 for t in tcent])
            keep = mask_rem & np.isfinite(maav)
            n_keep = int(keep.sum())
            n_pos = int(labels[keep].sum())
            n_neg = int(n_keep - n_pos)
            if n_keep == 0:
                log(f"[{pid}]   [SKIP canal] 0 fenêtre après masque (REM/validité)")
                continue
            log(f"[{pid}]   Fenêtres gardées: {n_keep} (pos={n_pos}, neg={n_neg})")

            pat_scores[g].append(maav[keep])
            pat_labels[g].append(labels[keep])

        # ROC/Seuils par groupe (patient)
        for g in ("chin","legs","arm"):
            if len(pat_scores[g]) == 0:
                log(f"[{pid}] [SKIP {g}] Aucune fenêtre disponible (après masques/filtres)")
                continue

            scores = np.concatenate(pat_scores[g])
            labels = np.concatenate(pat_labels[g])

            n_pos = int((labels == 1).sum())
            n_neg = int((labels == 0).sum())
            var_scores = float(np.nanvar(scores))
            if n_pos == 0 or n_neg == 0:
                log(f"[{pid}] [SKIP {g}] labels dégénérés: pos={n_pos}, neg={n_neg} (ROC impossible)")
                continue
            if not np.isfinite(scores).any():
                log(f"[{pid}] [SKIP {g}] scores MAAV NaN/Inf détectés")
                continue
            if np.nanstd(scores) == 0.0:
                log(f"[{pid}] [SKIP {g}] variance nulle des scores (var={var_scores:.3e})")
                continue

            log(f"[{pid}] [ROC {g}] Calcul ROC (N={len(scores)}, pos={n_pos}, neg={n_neg})")
            thr_v, tpr, fpr, (fpr_c, tpr_c) = best_threshold_roc(scores, labels)
            if not np.isfinite(thr_v):
                log(f"[{pid}] [SKIP {g}] seuil ROC non exploitable (NaN/Inf)")
                continue

            weight = float(len(scores))
            log(f"[{pid}] [OK {g}] seuil={to_uV(thr_v):.2f} µV | TPR={tpr:.2f} | FPR={fpr:.2f} | fenêtres={int(weight)}")
            per_group_scores[g].append(scores)
            per_group_labels[g].append(labels)
            per_group_roc_curv[g].append((fpr_c, tpr_c, weight))
            per_group_thr[g].append((thr_v, weight))
            rows_patients.append({
                "patient_id": pid, "group": g,
                "thr_volts": thr_v, "thr_uV": float(to_uV(thr_v)),
                "TPR_at_thr": tpr, "FPR_at_thr": fpr,
                "n_windows": int(weight),
                "n_rbd_windows": int((labels==1).sum())
            })

        log(f"[{pid}] ==== FIN patient ====")
        result = {
            "rows_patients": rows_patients,
            "per_group_scores": per_group_scores,
            "per_group_labels": per_group_labels,
            "per_group_roc_curv": per_group_roc_curv,
            "per_group_thr": per_group_thr,
        }
        return (pid, result)
    except Exception as e:
        log(f"[{pid}] [ERROR] Exception: {e}")
        return (pid, None)

# ==================== MAIN ====================
def main():
    ap = argparse.ArgumentParser(description="Seuils MAAV pour RBD (multi-patients) avec paramètres alignés sur le script RAW.")
    ap.add_argument("--windows-csv", default="/home/darryld/Cerco_studies/data/windows_by_patient.csv", help="Chemin vers windows_by_patient.csv")
    ap.add_argument("--gp2-dir", default="/home/darryld/documents/EEG/preprocessed/bipolaire/1_noArtefacts/gp2/", help="Dossier gp2 contenant les .fif annotés")
    ap.add_argument("--raw-dir", default="/home/darryld/documents/EEG/raw/", help="Dossier raw contenant les hypnogrammes *hypnoEXP.txt")
    ap.add_argument("--out-dir", default="/home/darryld/Cerco_studies/data/out_thresholds_maav", help="Dossier de sortie (CSV/figures)")
    ap.add_argument("--win-len", type=float, default=DEFAULT_WIN_LEN_S)
    ap.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    ap.add_argument("--hp", type=float, default=DEFAULT_HP)
    ap.add_argument("--lp", type=float, default=DEFAULT_LP)
    ap.add_argument("--notch", type=float, default=DEFAULT_NOTCH)
    ap.add_argument("--notch-q", type=float, default=DEFAULT_NOTCH_Q)
    ap.add_argument("--edge-guard", type=float, default=DEFAULT_EDGE_GUARD)
    ap.add_argument("--only-rem", action="store_true", default=DEFAULT_ONLY_REM)
    ap.add_argument("--clip-quantile", type=float, default=DEFAULT_CLIP_Q,
                    help="Quantile de clipping des MAAV (anti-outliers), ex: 99.9. Mettre <=0 pour désactiver.")
    ap.add_argument("--n-jobs", type=int, default=10, help="Nombre de processus parallèles")
    args = ap.parse_args()

    # Récap paramètres
    log("== PARAMÈTRES ==")
    log(f"  windows_csv : {args.windows_csv}")
    log(f"  gp2_dir     : {args.gp2_dir}")
    log(f"  raw_dir     : {args.raw_dir}")
    log(f"  out_dir     : {args.out_dir}")
    log(f"  filtres     : HP={args.hp} Hz | LP={args.lp} Hz | Notch={args.notch} Hz (Q={args.notch_q})")
    log(f"  MAAV        : win_len={args.win_len}s | overlap={args.overlap}")
    log(f"  only_rem    : {args.only_rem} | edge_guard={args.edge_guard}s | clip_q={args.clip_quantile}")
    log(f"  n_jobs      : {args.n_jobs}")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Charger windows_by_patient.csv
    log("== Lecture windows_by_patient.csv ==")
    windows_by_patient = load_windows_by_patient_csv(args.windows_csv, args.gp2_dir)

    # 2) Liste patients (croisement CSV × disque)
    patient_ids = sorted([pid for pid in windows_by_patient.keys()
                          if os.path.isdir(os.path.join(args.gp2_dir, pid))])
    log(f"\n=== Début traitement multi-patients ===")
    log(f"Patients à traiter : {len(patient_ids)}")
    log("="*52)

    # Agrégats globaux
    per_group_scores   = {g: [] for g in ("chin","legs","arm")}
    per_group_labels   = {g: [] for g in ("chin","legs","arm")}
    per_group_roc_curv = {g: [] for g in ("chin","legs","arm")}  # (fpr,tpr,weight)
    per_group_thr      = {g: [] for g in ("chin","legs","arm")}  # (thr_volts, weight)
    rows_patients = []

    # --------- Parallélisation sur les patients ---------
    log("[POOL] Soumission des jobs patients…")
    futures = []
    with ProcessPoolExecutor(max_workers=max(1, args.n_jobs)) as ex:
        for pid in patient_ids:
            futures.append(ex.submit(_process_one_patient, pid, args, windows_by_patient))
        for fut in as_completed(futures):
            pid, out = fut.result()
            if out is None:
                log(f"[skip] Patient {pid} (aucune donnée exploitable ou erreur)")
                continue
            log(f"[AGG] Agrégation résultats patient {pid}")
            rows_patients.extend(out["rows_patients"])
            for g in ("chin","legs","arm"):
                per_group_scores[g].extend(out["per_group_scores"][g])
                per_group_labels[g].extend(out["per_group_labels"][g])
                per_group_roc_curv[g].extend(out["per_group_roc_curv"][g])
                per_group_thr[g].extend(out["per_group_thr"][g])

    # 3) Tableaux de sortie
    df_pat = pd.DataFrame(rows_patients).sort_values(["group","patient_id"])
    if df_pat.empty:
        log("\n[ERREUR] Aucun résultat : vérifie chemins .fif / hypno et ton CSV.")
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)
    pat_csv = os.path.join(args.out_dir, "thresholds_by_patient_maav.csv")
    df_pat.to_csv(pat_csv, index=False)
    log(f"\n[OK] Seuils par patient (MAAV) → {pat_csv}")

    # Moyennes pondérées par groupe
    rows_glob=[]
    for g in ("chin","legs","arm"):
        vals = [(t,w) for (t,w) in per_group_thr[g] if np.isfinite(t)]
        if not vals:
            log(f"[WARN] Aucun seuil exploitable pour le groupe {g}")
            continue
        thrs = np.array([t for (t,w) in vals], float)
        ws   = np.array([w for (t,w) in vals], float)
        thr_mean = float(np.sum(thrs*ws) / np.sum(ws))
        rows_glob.append({
            "group": g,
            "thr_volts": thr_mean,
            "thr_uV": float(to_uV(thr_mean)),
            "n_patients": len(ws),
            "total_windows": int(ws.sum())
        })
    df_glob = pd.DataFrame(rows_glob).sort_values("group")
    glob_csv = os.path.join(args.out_dir, "thresholds_group_weighted_maav.csv")
    df_glob.to_csv(glob_csv, index=False)
    log(f"[OK] Seuils moyens pondérés (MAAV) → {glob_csv}")
    log("\n=== RÉSUMÉ ===")
    log(df_glob.to_string(index=False))

    # 4) Figures (Histogrammes + ROC moyenne)
    for g in ("chin","legs","arm"):
        if g in df_glob["group"].values:
            log(f"[FIG] Histogramme MAAV : groupe {g}")
            _ = save_group_hist(args.out_dir, g, per_group_scores, per_group_labels, df_glob)

    for g in ("chin","legs","arm"):
        if len(per_group_roc_curv[g])==0: 
            log(f"[FIG] ROC moyenne : groupe {g} (aucune courbe à agréger)")
            continue
        grid, tpr_mean, auc_mean = avg_roc(per_group_roc_curv[g])
        fig, ax = plt.subplots(figsize=(7,4.2))
        ax.plot(grid, tpr_mean, lw=2, label=f"ROC moyenne (AUC = {auc_mean:.2f})")
        ax.plot([0,1],[0,1],"--", alpha=0.5)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC moyenne • groupe {g} (MAAV)")
        ax.legend(loc="lower right"); ax.grid(alpha=0.2)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"roc_mean_maav_{g}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log(f"[FIG] ROC moyenne groupe {g} (AUC={auc_mean:.2f}) → {out_path}")

    log("\nTerminé.")

if __name__ == "__main__":
    main()
