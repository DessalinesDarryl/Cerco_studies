import torch


def predict(model, X):
    """
    Predict sleep stages using a trained ResNet model.

    Parameters
    ----------
    model : nn.Module
        Trained sleep stage classifier.
    X : ndarray
        Input tensor of shape (N, C, 1, T).

    Returns
    -------
    y_pred : ndarray
        Predicted sleep stage labels (N,).
    probas : ndarray
        Softmax probability matrix of shape (N, 5).
    """
    model.eval()
    device = next(model.parameters()).device
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probas = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = probas.argmax(axis=1)

    return y_pred, probas
