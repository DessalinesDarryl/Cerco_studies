import numpy as np

from main_pipeline import run_subject_pipeline_from_amplitudes, build_cohort_dataframe
from statistics_pipeline import (
    zscore_against_controls,
    composite_severity_score,
    full_group_comparison,
)


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


# ---- simulation de 4 groupes cliniques + controles ----------------------
GROUPS = {
    "control": dict(n_subj=10, n_act_range=(2, 8), amp_range=(1.5, 3)),
    "narcolepsy": dict(n_subj=10, n_act_range=(15, 30), amp_range=(2, 5)),
    "iRBD": dict(n_subj=15, n_act_range=(30, 60), amp_range=(3, 7)),
    "synucleinopathy": dict(n_subj=10, n_act_range=(40, 80), amp_range=(4, 9)),
    "autoimmune_encephalitis": dict(n_subj=8, n_act_range=(20, 90), amp_range=(3, 10)),
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
print("=== Cohorte simulee ===")
print(df.groupby("diagnosis")[["AI_atonia_index", "pct_epochs_phasic",
                                 "phasic_density_per_min", "episode_count"]].mean().round(3))

# ---- score composite de severite -----------------------------------------
feature_cols = [
    "AI_atonia_index", "pct_epochs_phasic", "pct_any_RSWA",
    "episode_count", "phasic_density_per_min", "duration_mean",
    "interval_median", "interval_variance", "interval_cv",
    "mean_relative_amplitude", "long_density_per_min",
    "pct_tonic", "phasic_tonic_ratio",
]
z_df, feats = zscore_against_controls(df, control_label="control", feature_cols=feature_cols)
# AI_atonia_index : plus haut = MOINS severe -> on l'inverse pour que
# "positif = plus severe" reste coherent sur toutes les variables du score
z_df["AI_atonia_index"] = -z_df["AI_atonia_index"]

mean_score = composite_severity_score(z_df, feats, method="mean")
pca_score, explained_var = composite_severity_score(z_df, feats, method="pca")
df["severity_score_mean"] = mean_score
df["severity_score_pca"] = pca_score

print("\n=== Score composite de severite par groupe ===")
print(df.groupby("diagnosis")[["severity_score_mean", "severity_score_pca"]].mean().round(3))
print(f"\nVariance expliquee par la 1ere composante PCA : {explained_var:.2%}")

# ---- comparaison statistique multi-groupes --------------------------------
kw_df, mw_df, roc_df = full_group_comparison(
    df, feature_cols=feature_cols + ["severity_score_mean"],
    group_col="diagnosis", control_label="control",
)
print("\n=== Kruskal-Wallis (top 5 par p-value) ===")
print(kw_df.sort_values("p").head(5).round(4))

print("\n=== ROC vs controles pour severity_score_mean ===")
print(roc_df[roc_df["feature"] == "severity_score_mean"].round(3))
