from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, class_labels, class_names, save_path=None):
    """
    Affiche les performances du modèle : classification report + matrice de confusion.

    Paramètres
    ----------
    y_true : ndarray
        Labels vrais (par exemple ceux issus du fichier .txt, remappés).
    y_pred : ndarray
        Labels prédits par le modèle (après remapping).
    class_labels : list of int
        Liste des classes entières utilisées (ex: [1, 3, 2, 4, 5]).
    class_names : list of str
        Noms des classes dans l’ordre correspondant à class_labels.
    save_path : str or None
        Si précisé, sauvegarde la figure de la matrice de confusion.
    """
    print("Classification report :")
    print(classification_report(y_true, y_pred, labels=class_labels, target_names=class_names, zero_division=0))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=class_labels,
        display_labels=class_names,
        normalize='true',
        cmap='Blues'
    )
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
