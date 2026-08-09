import numpy as np

from main_pipeline import run_subject_pipeline_from_amplitudes, build_cohort_dataframe
from multicollinearity_pca import full_dimensionality_reduction_report
from statistics_pipeline import zscore_against_controls, composite_severity_score


def simulate_subject_amplitudes(n_epochs, n_activations, background=0.5,
                                 amp_range=(2, 6), dur_range=(1, 6), seed=0):
    rng = np.random.RandomState(seed)
    amplitudes = background + 0.1 * rng.rand(n_epochs)
    if n_activations > 0:
        onsets = sorted(
            rng.choice(np.arange(10, n_epochs - 15), size=n_activations, replace=False)
        )
        for onset in onsets:
            dur = rng.randint(dur_range[0], dur_range[1])
            amp = rng.uniform(*amp_range)
            amplitudes[onset: onset + dur] = amp
    return amplitudes


GROUPS = {
    "control": dict(n_subj=15, n_act_range=(2, 8), amp_range=(1.5, 3)),
    "narcolepsy": dict(n_subj=15, n_act_range=(15, 30), amp_range=(2, 5)),
    "iRBD": dict(n_subj=20, n_act_range=(30, 60), amp_range=(3, 7)),
    "synucleinopathy": dict(n_subj=15, n_act_range=(40, 80), amp_range=(4, 9)),
    "autoimmune_encephalitis": dict(n_subj=12, n_act_range=(20, 90), amp_range=(3, 10)),
}

subject_metrics = []
seed = 0
for diagnosis, params in GROUPS.items():
    for i in range(params["n_subj"]):
        seed += 1
        n_act = np.random.RandomState(seed).randint(*params["n_act_range"])
        amps = simulate_subject_amplitudes(
            n_epochs=1200, n_activations=n_act,
            amp_range=params["amp_range"], seed=seed,
        )
        m = run_subject_pipeline_from_amplitudes(
            amps, diagnosis=diagnosis, subject_id=f"{diagnosis}-{i:03d}",
            fold=4.0, duration_method="gmm",
        )
        subject_metrics.append(m)

df = build_cohort_dataframe(subject_metrics)
print(f"Cohorte simulee : {len(df)} sujets, {df['diagnosis'].nunique()} groupes\n")

# Ensemble de variables candidates pour le score composite (post retrait
# des variables #4 et #9 jugees redondantes)
candidate_features = [
    "AI_atonia_index", "pct_epochs_phasic", "pct_any_RSWA", "pct_tonic",
    "episode_count", "phasic_density_per_min",
    "duration_mean", "duration_median", "duration_sd", "duration_max",
    "interval_mean", "interval_median", "interval_variance", "interval_cv",
    "mean_relative_amplitude", "max_relative_amplitude",
    "short_density_per_min", "medium_density_per_min", "long_density_per_min",
    "phasic_tonic_ratio", "progression_index",
]
candidate_features = [c for c in candidate_features if c in df.columns]

print("=" * 70)
print("ETAPE 1-4 : rapport complet Spearman / VIF / KMO-Bartlett / ACP")
print("=" * 70)
report = full_dimensionality_reduction_report(
    df, candidate_features, vif_threshold=5.0, corr_threshold=0.8,
    variance_threshold=0.80, impute_missing=True, verbose=True,
)

print("\n=== Variables retenues apres reduction VIF ===")
print(report["retained_features"])

print("\n=== Variables retirees (VIF au moment du retrait) ===")
print(report["vif_reduction"]["removed_features"])

pca = report["pca"]
print(f"\n=== ACP : {pca['n_components_selected']} composantes retenues "
      f"(Kaiser={pca['n_components_kaiser']}, "
      f"seuil variance 80%={pca['n_components_variance_threshold']}) ===")
print("Variance expliquee cumulee :", np.round(pca["cumulative_variance"][:5], 3))

print("\n=== Loadings (contribution des variables a chaque composante) ===")
print(pca["loadings"].round(3))

# ---- score composite final construit sur les variables non redondantes --
retained = report["retained_features"]
z_df, feats = zscore_against_controls(df, control_label="control", feature_cols=retained)
if "AI_atonia_index" in feats:
    z_df["AI_atonia_index"] = -z_df["AI_atonia_index"]  # inversion (AI haut = moins severe)

df["severity_score_mean_reduced"] = composite_severity_score(z_df, feats, method="mean")
df["severity_score_PC1"] = pca["scores"]["PC1"].reindex(df.index)

print("\n=== Score composite (post-reduction) par groupe ===")
print(
    df.groupby("diagnosis")[["severity_score_mean_reduced", "severity_score_PC1"]]
    .mean()
    .round(3)
)
