import numpy as np
import matplotlib.pyplot as plt


def plot_hypnodensity(probas, y_pred, y_true=None, class_names=None, save_path=None):
    """
    Plot hypnodensity (per-class probabilities), predicted and optionally true hypnogram.

    Parameters
    ----------
    probas : ndarray
        Softmax probability matrix of shape (N, 5), one row per epoch.
    y_pred : ndarray
        Predicted class labels (N,) from model (integers 0–4).
    y_true : ndarray or None
        Ground truth class labels (N,) if available. Default is None.
    class_names : list of str or None
        List of class names (e.g., ['W', 'N1', 'N2', 'N3', 'REM']). Default uses integers.
    save_path : str or None
        If given, the figure will be saved at this path instead of shown.
    """
    num_epochs = probas.shape[0]
    time_axis = np.arange(num_epochs)

    # Si l'ordre souhaité n'est pas précisé, on impose REM en haut
    if class_names is None:
        class_names = ['REM', 'N3', 'N2', 'N1', 'W']

    class_indices = {name: i for i, name in enumerate(class_names)}

    # Remap probas selon nouvel ordre
    probas_ordered = np.array([probas[:, i] for i in [class_names.index(c) for c in ['W', 'N1', 'N2', 'N3', 'REM']]])
    probas_ordered = probas_ordered[[4, 3, 2, 1, 0]].T  # Pour affichage : REM en haut

    fig, axes = plt.subplots(3 if y_true is not None else 2, 1, figsize=(14, 7), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1, 1] if y_true is not None else [2, 1]})

    # Top: hypnodensity
    ax_hd = axes[0]
    for i, label in enumerate(class_names):
        ax_hd.plot(time_axis, probas_ordered[:, i], label=label)
    ax_hd.set_title("Hypnodensity (per-class probability)")
    ax_hd.set_ylabel("Probability")
    ax_hd.set_ylim(0, 1)
    ax_hd.legend(loc='upper right')

    # Middle: predicted hypnogram
    ax_pred = axes[1]
    ax_pred.step(time_axis, y_pred, where='mid', color='black')
    ax_pred.set_title("Predicted Hypnogram")
    ax_pred.set_ylabel("Stage")
    ax_pred.set_yticks([1, 3, 2, 4, 5])
    ax_pred.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'W'])

    # Bottom: true hypnogram
    if y_true is not None:
        ax_true = axes[2]
        ax_true.step(time_axis, y_true, where='mid', color='blue')
        ax_true.set_title("True Hypnogram (Manual Scoring)")
        ax_true.set_ylabel("Stage")
        ax_true.set_yticks([1, 3, 2, 4, 5])
        ax_true.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'W'])

    plt.xlabel("Epoch Number (30s each)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


