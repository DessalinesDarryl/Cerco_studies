from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_model(name):
    name = name.lower()
    
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight='balanced',
            random_state=42
        )
    
    elif name == "svm":
        return SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    
    elif name == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
    
    else:
        raise ValueError(f"Modèle inconnu: {name}")
