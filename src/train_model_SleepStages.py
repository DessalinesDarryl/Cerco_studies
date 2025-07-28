import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from resnet_model import ResNetSleep
from sklearn.model_selection import GroupShuffleSplit


def train_model(X_train, y_train, epochs=10, batch_size=16, lr=1e-3):
    """
    Train the ResNetSleep model on preprocessed PSG data.

    Parameters
    ----------
    X_train : ndarray
        Input tensor of shape (N, C, 1, T).
    y_train : ndarray
        Target labels of shape (N,).
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    lr : float
        Learning rate for the optimizer.

    Returns
    -------
    model : nn.Module
        Trained PyTorch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ResNetSleep().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    return model


def train_test_split_by_patient(X_all, y_all, patient_ids, test_size=0.2, random_state=42):
    """
    Réalise un split train/test en garantissant qu’un patient ne figure que dans un split.

    Paramètres
    ----------
    X_all : ndarray (N, C, 1, T)
    y_all : ndarray (N,)
    patient_ids : list or ndarray
        ID patient pour chaque époque (ex: ['MN143', 'MN143', ..., 'MN144', ...])
    test_size : float
    random_state : int

    Retour
    ------
    X_train, X_test, y_train, y_test : ndarray
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X_all, y_all, groups=patient_ids))
    return X_all[train_idx], X_all[test_idx], y_all[train_idx], y_all[test_idx]
