# Pipeline RSWA — échelle de sévérité multi-étiologique

Pipeline Python pour la quantification automatisée du REM Sleep Without
Atonia (RSWA) sur signal EMG (submentalis ± membres), destiné à une étude
exploratoire comparant plusieurs étiologies (iRBD, narcolepsie,
synucléinopathies, encéphalites auto-immunes) à des contrôles.

## Structure des fichiers

| Fichier | Rôle |
|---|---|
| `preprocessing.py` | Filtrage (10–100 Hz + notch 50 Hz), redressement, calcul d'amplitude par mini-époque |
| `activation_detection.py` | Fond de bruit local (minimum glissant, Ferri 2010), détection des activations, durées, intervalles |
| `threshold_estimation.py` | Bornes short/medium/long **data-driven** (GMM sur log-durée, repli tertiles) |
| `severity_metrics.py` | Calcul des métriques de sévérité par sujet (classiques + proposées) |
| `statistics_pipeline.py` | Score composite (z-score/PCA), Kruskal-Wallis, Mann-Whitney Bonferroni, ROC bootstrap |
| `multicollinearity_pca.py` | **Spearman (FDR) → VIF itératif → KMO/Bartlett → ACP**, en amont du score composite |
| `main_pipeline.py` | Orchestration : signal brut ou amplitudes → dict de métriques par sujet |
| `test_synthetic.py`, `test_cohort.py`, `test_pca_vif.py` | Scripts de démonstration |

## Utilisation rapide

```python
from main_pipeline import run_subject_pipeline_from_raw, build_cohort_dataframe
from multicollinearity_pca import full_dimensionality_reduction_report
from statistics_pipeline import zscore_against_controls, composite_severity_score, full_group_comparison

# 1) Par sujet
metrics = run_subject_pipeline_from_raw(
    raw_signal=submentalis_array, fs=256,
    diagnosis='iRBD', subject_id='sub-001', fold=4.0, duration_method='gmm',
)
df = build_cohort_dataframe([metrics_sujet1, metrics_sujet2, ...])

# 2) Reduction de dimension (Spearman -> VIF -> KMO/Bartlett -> ACP)
candidate_features = ['AI_atonia_index', 'pct_epochs_phasic', 'pct_any_RSWA', 'pct_tonic',
                       'episode_count', 'phasic_density_per_min', 'duration_mean',
                       'duration_median', 'duration_sd', 'duration_max',
                       'interval_mean', 'interval_median', 'interval_variance', 'interval_cv',
                       'mean_relative_amplitude', 'max_relative_amplitude',
                       'short_density_per_min', 'medium_density_per_min', 'long_density_per_min',
                       'phasic_tonic_ratio', 'progression_index']

report = full_dimensionality_reduction_report(
    df, candidate_features, vif_threshold=5.0, corr_threshold=0.8,
    variance_threshold=0.80, impute_missing=True,   # True si sujets a tres peu d'activations
)
retained = report['retained_features']    # variables non redondantes -> score composite
pca_scores = report['pca']['scores']      # alternative : score = PC1 directement

# 3) Score composite sur les variables retenues (non redondantes)
z_df, feats = zscore_against_controls(df, control_label='control', feature_cols=retained)
z_df['AI_atonia_index'] = -z_df.get('AI_atonia_index', 0)  # inversion si presente (AI haut = moins severe)
df['severity_score'] = composite_severity_score(z_df, feats, method='mean')

# 4) Comparaison statistique multi-groupes
kw_df, mw_df, roc_df = full_group_comparison(
    df, feature_cols=retained + ['severity_score'],
    group_col='diagnosis', control_label='control')
```

## Variables retirées pour redondance

Deux variables du tableau initial ont été retirées de la sortie par défaut du
pipeline (`compute_all_metrics`) car quasi-colinéaires avec d'autres
métriques déjà présentes :

| Variable retirée | Redondante avec | Raison |
|---|---|---|
| `activation_density_per_hour` (ev/heure REM) | `phasic_density_per_min`, `episode_count` | Simple changement d'unité temporelle de la même quantité |
| `pct_rem_time_phasic` (% REM pondéré durée) | `pct_epochs_phasic` (% mini-époques) | Mesurent la même charge phasique sous deux angles quasi-équivalents (rho > 0.95 en pratique) |

Les fonctions restent disponibles dans `severity_metrics.py` (`activation_density_per_hour()`,
`pct_rem_time_with_activation()`) pour un usage ponctuel, mais ne sont plus
incluses dans le bundle par défaut.

## Méthodologie de réduction de dimension (`multicollinearity_pca.py`)

Avant de construire le score composite, la démarche recommandée (standards
de statistique multivariée, indépendants de la littérature RBD) :

1. **Corrélations de Spearman** entre toutes les paires de variables candidates,
   avec correction **FDR de Benjamini-Hochberg** (le nombre de tests croît en
   p(p-1)/2 ; Bonferroni serait trop conservateur ici, contrairement aux
   comparaisons de groupes où Bonferroni est conservé par cohérence avec
   Khalil et al. 2013).
2. **VIF (Variance Inflation Factor) en réduction itérative** : retrait de la
   variable au VIF le plus élevé tant qu'une variable dépasse le seuil
   (5 = modéré, 10 = sévère ; O'Brien 2007).
3. **KMO + test de sphéricité de Bartlett** sur les variables retenues, pour
   vérifier statistiquement que l'ACP est justifiée sur ce sous-ensemble
   (et non l'appliquer par défaut sans vérification).
4. **ACP** avec sélection du nombre de composantes par règle de Kaiser
   (valeurs propres > 1) et seuil de variance cumulée (80% par défaut).

```python
from multicollinearity_pca import full_dimensionality_reduction_report

report = full_dimensionality_reduction_report(df, candidate_features)
report['spearman_rho']          # matrice de corrélations
report['redundant_pairs']       # paires |rho| >= 0.8
report['vif_reduction']         # variables retenues/retirées + historique VIF
report['kmo']                   # KMO global + par variable
report['bartlett']              # chi2, dof, p-value
report['pca']                   # eigenvalues, loadings, scores, n composantes
```

**Note sur `impute_missing`** : le VIF et l'ACP exigent des cas complets
(listwise deletion). Si certains sujets (typiquement les contrôles, avec très
peu d'événements RSWA) ont des statistiques d'intervalle non définies
(< 2 intervalles disponibles → `interval_variance`/`interval_cv` = NaN),
activer `impute_missing=True` (imputation par médiane) évite une perte de
puissance disproportionnée. À documenter comme limite méthodologique dans
votre manuscrit si utilisée — ce n'est pas une recommandation pour
l'analyse statistique finale elle-même, seulement pour la phase de
sélection de variables.

## Tableau complet des variables — formule et référence bibliographique

| # | Variable (clé dans le code) | Formule | Référence complète | Statut |
|---|---|---|---|---|
| 1 | `pct_epochs_phasic` — % Phasique | (mini-époques activées phasiques / total mini-époques REM) × 100 | Ferri R, et al. J Sleep Res 2008;17:89–100 · Frauscher B, et al. Sleep 2012;35:835–847 · Khalil A, et al. J Clin Sleep Med 2013;9(10):1039–1048 · McCarter SJ, et al. Sleep Med 2017;33:23–29 | Validée |
| 2 | `pct_tonic` — % Tonique | (époques 30s toniques / total époques REM) × 100 | Frauscher B, et al. Sleep 2012;35:835–847 · Khalil A, et al. J Clin Sleep Med 2013;9(10):1039–1048 (époques 20s) · McCarter SJ, et al. Sleep Med 2017;33:23–29 | Validée |
| 3 | `pct_any_RSWA` — % RSWA global ("any") | (mini-époques phasique OU tonique / total) × 100 | Frauscher B, et al. Sleep 2012;35:835–847 (AUC 0.990 mentalis seul, 0.998 combiné mentalis+FDS) | Validée |
| 4 | `duration_mean` — Durée moyenne d'activation | moyenne des durées de burst (s) | McCarter SJ, et al. Sleep Med 2017;33:23–29 (cutoffs SM 0.66s / AT 0.71s) | Validée |
| 5 | `interval_median`, `interval_variance` — Distribution des intervalles | médiane + variance des intervalles onset-to-onset (s) | Ferri R, et al. J Sleep Res 2008;17:89–100 (Fig. 5) | Validée |
| 6 | `AI_atonia_index` — Atonia Index | %(amp≤1)/[100−%(1<amp≤2)], corrigé du bruit | Ferri R, et al. J Sleep Res 2008;17:89–100 · Ferri R, et al. Sleep Med 2010;11:947–949 | Validée |
| 7 | `episode_count` — Nombre d'épisodes RSWA | count(activations) | Ferri R, et al. J Sleep Res 2008;17:89–100 · McCarter SJ, et al. Sleep Med 2017;33:23–29 | Validée |
| 8 | `duration_median/sd/max/total` — Stats de durée | dérivées de la distribution des durées de burst | Extension de McCarter SJ, et al. Sleep Med 2017;33:23–29 et Ferri R, et al. J Sleep Res 2008;17:89–100 (Fig. 4) | Validée/étendue |
| 9 | `interval_cv` — Coefficient de variation des intervalles | SD(intervalles)/moyenne(intervalles) | **Proposée** — fondée sur la structure de bruit 1/f monomodale observée par Ferri R, et al. J Sleep Res 2008;17:89–100 (Discussion) | Proposée |
| 10 | `mean/max_relative_amplitude` — Intensité relative des bursts | amplitude burst / amplitude fond local | **Proposée** — variable continue dérivée du seuil binaire "amplitude > N × fond" de Khalil A, et al. J Clin Sleep Med 2013;9(10):1039–1048 et McCarter SJ, et al. Sleep Med 2017;33:23–29 | Proposée |
| 11 | `short/medium/long_*` — Densités par sous-catégorie de durée | comptage/densité/% par catégorie, bornes GMM sur log-durée | **Proposée** — l'absence de bimodalité démontrée par Ferri R, et al. J Sleep Res 2008;17:89–100 justifie le rejet d'un seuil fixe a priori | Proposée |
| 12 | `phasic_tonic_ratio` — Ratio phasique/tonique | %phasique / %tonique | **Proposée** — combinaison de McCarter SJ, et al. Sleep Med 2017;33:23–29 et Frauscher B, et al. Sleep 2012;35:835–847 | Proposée |
| 13 | `bilateral_asymmetry_index` — Asymétrie bilatérale | (densité_droite − densité_gauche)/(densité_droite + densité_gauche) | **Proposée** — s'appuie sur le montage bilatéral d'Iranzo A, et al. Sleep Med 2011;12:284–288 et Frauscher B, et al. Sleep 2012;35:835–847 | Proposée |
| 14 | `progression_index` — Progression intra-nuit | densité (dernier tiers REM) / densité (premier tiers REM) | **Proposée** — inspirée de la variabilité nuit-à-nuit citée dans Khalil A, et al. J Clin Sleep Med 2013;9(10):1039–1048 (réf. Ferri R, et al. J Clin Sleep Med 2013;9:253–258 ; Cygan F, et al. J Clin Sleep Med 2010;6:551–555) | Proposée |

~~`activation_density_per_hour`~~ et ~~`pct_rem_time_phasic`~~ retirées (redondance, voir section dédiée ci-dessus).

## Références complètes

**Littérature RBD/RSWA :**
- Ferri R, Manconi M, Plazzi G, Bruni O, Vandi S, Montagna P, Ferini-Strambi L, Zucconi M. A quantitative statistical analysis of the submentalis muscle EMG amplitude during sleep in normal controls and patients with REM sleep behavior disorder. *J Sleep Res* 2008;17:89–100.
- Ferri R, Rundo F, Manconi M, Plazzi G, Bruni O, Oldani A, Ferini-Strambi L, Zucconi M. Improved computation of the atonia index in normal controls and patients with REM sleep behavior disorder. *Sleep Med* 2010;11:947–949.
- Frauscher B, Iranzo A, Gaig C, et al. Normative EMG values during REM sleep for the diagnosis of REM sleep behavior disorder. *Sleep* 2012;35(6):835–847.
- Iranzo A, Frauscher B, Santos H, et al. Usefulness of the SINBAR electromyographic montage to detect the motor and vocal manifestations occurring in REM sleep behavior disorder. *Sleep Med* 2011;12:284–288.
- Khalil A, Wright MA, Walker MC, Eriksson SH. Loss of rapid eye movement sleep atonia in patients with REM sleep behavioral disorder, narcolepsy, and isolated loss of REM atonia. *J Clin Sleep Med* 2013;9(10):1039–1048.
- McCarter SJ, St Louis EK, Sandness DJ, et al. Diagnostic REM sleep muscle activity thresholds in patients with idiopathic REM sleep behavior disorder with and without obstructive sleep apnea. *Sleep Med* 2017;33:23–29.

**Statistique multivariée (méthodologie générale, indépendante du domaine RBD) :**
- Spearman C. The proof and measurement of association between two things. *Am J Psychol* 1904;15:72–101.
- Bartlett MS. Tests of significance in factor analysis. *Br J Psychol* 1950;3:77–85.
- Kaiser HF. An index of factorial simplicity. *Psychometrika* 1974;39:31–36.
- Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J R Stat Soc Series B* 1995;57:289–300.
- O'Brien RM. A caution regarding rules of thumb for variance inflation factors. *Qual Quant* 2007;41:673–690.

## Limites et points de vigilance

- Les bornes GMM (`estimate_duration_categories`) sont estimées **par sujet** dans `main_pipeline.py` : pour comparer les groupes, recalculer au niveau cohorte (ou par groupe étiologique) et appliquer uniformément.
- Aucune donnée de la littérature ne couvre les encéphalites auto-immunes : traiter tout résultat sur ce groupe comme hypothèse-génératrice, pas confirmatoire.
- `pct_any_RSWA` combine deux échelles temporelles (mini-époques 1s phasique, époques 30s tonique projetées sur la résolution 1s) — comportement attendu (Frauscher et al. 2012), à garder en tête pour l'interprétation.
- Le VIF/KMO exigent des cas complets ; `impute_missing=True` est un palliatif pragmatique, pas une solution statistique neutre — vérifier la proportion de valeurs imputées avant de s'y fier.
- Un KMO bas (<0.5) signale que l'ACP n'est pas adaptée au jeu de variables retenu : dans ce cas, privilégier le score composite par moyenne de z-scores plutôt que le score PCA (PC1).
- Le module ROC nécessite plusieurs sujets par groupe pour un bootstrap stable ; les IC95% seront larges avec de petits effectifs (attendu en exploratoire).
