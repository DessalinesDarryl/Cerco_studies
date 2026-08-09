import numpy as np
from activation_detection import detect_activations, compute_intervals, local_background
from threshold_estimation import estimate_duration_categories
from severity_metrics import compute_all_metrics, tonic_epoch_flags

np.random.seed(1)
n_epochs = 1200  # 20 min REM, epoques de 1s
background_level = 0.5
amplitudes = background_level + 0.1 * np.random.rand(n_epochs)

onsets = sorted(np.random.choice(np.arange(10, n_epochs - 15), size=60, replace=False))
for onset in onsets:
    dur = np.random.randint(1, 8)
    amp = np.random.uniform(3, 10)
    amplitudes[onset:onset + dur] = amp

activations = detect_activations(amplitudes, epoch_duration_s=1.0, fold=4.0)
intervals = compute_intervals(activations)
print("Nb activations detectees:", len(activations))

durations = [a["duration_s"] for a in activations]
thresholds = estimate_duration_categories(durations, method="gmm")
print("Seuils data-driven:", thresholds)

bg = local_background(amplitudes)
tonic_flags, epochs_per_tonic = tonic_epoch_flags(amplitudes, epoch_duration_s=1.0)
tonic_pct = 100 * np.mean(tonic_flags) if len(tonic_flags) > 0 else np.nan
metrics = compute_all_metrics(
    amplitudes_1s=amplitudes,
    activations=activations,
    intervals=intervals,
    total_rem_time_s=n_epochs * 1.0,
    n_bound=thresholds["n"],
    k_bound=thresholds["k"],
    background_mean=float(np.mean(bg)),
    tonic_pct=tonic_pct,
    tonic_flags=tonic_flags,
    epochs_per_tonic=epochs_per_tonic,
)
for k, v in metrics.items():
    print(k, ":", v)
