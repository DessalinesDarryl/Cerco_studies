# src/training/lit_module.py
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.training.utils import compute_metrics
class SimpleModule(nn.Module):
    def __init__(self, model, lr=1e-3):
        super().__init__(); self.model=model; self.loss=nn.CrossEntropyLoss(); self.lr=lr
    def step(self, batch):
        x,y = batch; logits = self.model(x); loss = self.loss(logits,y); probs = logits.softmax(1)
        return loss, probs
    def fit(self, train_dl, val_dl, epochs=10, device="cpu"):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        best = {"bac": -1, "state": None}
        for _ in range(epochs):
            self.train()
            for b in train_dl:
                x,y = [t.to(device) for t in b]
                loss,_ = self.step((x,y)); opt.zero_grad(); loss.backward(); opt.step()
            self.eval()
            with torch.no_grad():
                y_true=[]; y_prob=[]
                for b in val_dl:
                    x,y = [t.to(device) for t in b]
                    _,p = self.step((x,y)); y_true.append(y.cpu()); y_prob.append(p.cpu())
            import torch as T
            y_true = T.cat(y_true); y_prob = T.cat(y_prob)
            metrics = compute_metrics(y_true.numpy(), y_prob.numpy())
            if metrics["bac"]>best["bac"]: best={"bac":metrics["bac"], "state":self.state_dict()}
        if best["state"] is not None: self.load_state_dict(best["state"])
    def predict_proba(self, dl, device="cpu"):
        self.eval(); out=[]; import torch as T
        with torch.no_grad():
            for x,_ in dl:
                x=x.to(device); p=self.model(x).softmax(1).cpu(); out.append(p)
        return T.cat(out)
def make_dataloaders(X_train,y_train,X_val,y_val,batch=64):
    td = TensorDataset(X_train,y_train); vd = TensorDataset(X_val,y_val)
    return DataLoader(td,batch_size=batch,shuffle=True), DataLoader(vd,batch_size=batch)
