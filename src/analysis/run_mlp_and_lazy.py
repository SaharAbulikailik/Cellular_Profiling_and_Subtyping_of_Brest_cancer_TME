#!/usr/bin/env python3
import argparse, os, json, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ===== LazyPredict baseline stays exactly the same =====
def try_lazypredict(Xtr_df, Xte_df, ytr, yte):
    try:
        from lazypredict.Supervised import LazyClassifier
        lc = LazyClassifier(verbose=0, ignore_warnings=True, random_state=42)
        models, _ = lc.fit(Xtr_df, Xte_df, ytr, yte)
        models = models.reset_index().rename(columns={"index":"Model"})
        used_real = True
        out = models
    except Exception:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

        models = [
            ("LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))),
            ("LinearSVC",          make_pipeline(StandardScaler(), LinearSVC(random_state=42))),
            ("RandomForest",       RandomForestClassifier(n_estimators=120, random_state=42)),
            ("KNN",                make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11))),
            ("GaussianNB",         GaussianNB()),
            ("DecisionTree",       DecisionTreeClassifier(random_state=42)),
        ]
        rows = []
        Xtr, Xte = Xtr_df.values, Xte_df.values
        for name, est in models:
            est.fit(Xtr, ytr)
            preds = est.predict(Xte)
            acc = accuracy_score(yte, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(yte, preds, average="binary", zero_division=0)
            auc = float("nan")
            if hasattr(est, "predict_proba"):
                try: auc = roc_auc_score(yte, est.predict_proba(Xte)[:,1])
                except Exception: pass
            elif hasattr(est, "decision_function"):
                try:
                    s = est.decision_function(Xte)
                    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
                    auc = roc_auc_score(yte, s)
                except Exception: pass
            rows.append([name, acc, prec, rec, f1, auc])
        used_real = False
        out = pd.DataFrame(rows, columns=["Model","Accuracy","Precision","Recall","F1","AUC"]).sort_values("Accuracy", ascending=False)

    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(3)
    return out, used_real

# ===== Data utilities (unchanged) =====
def find_col(df, needle):
    for c in df.columns:
        if c.lower() == needle.lower(): return c
    for c in df.columns:
        if needle.lower() in c.lower(): return c
    return None

def load_and_prepare(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training file not found: {path}")
    df = pd.read_excel(path)

    want = ["area","pleomorphism","elongation","mean_intensity_DAPI","total_intensity_DAPI","TARGET"]
    colmap = {}
    for nm in want:
        col = find_col(df, nm)
        if col is None and nm == "pleomorphism":
            col = find_col(df, "solidity")
        if col is None:
            raise ValueError(f"Missing required column '{nm}' (or 'solidity' for pleomorphism). "
                             f"Found: {list(df.columns)[:12]}...")
        colmap[nm] = col

    X_df = df[[colmap["area"], colmap["pleomorphism"], colmap["elongation"],
               colmap["mean_intensity_DAPI"], colmap["total_intensity_DAPI"]]].apply(pd.to_numeric, errors="coerce")
    y_sr = pd.to_numeric(df[colmap["TARGET"]], errors="coerce")
    valid = (~X_df.isna().any(axis=1)) & (~y_sr.isna())
    X_df = X_df.loc[valid].copy()
    y_sr = y_sr.loc[valid].astype(int).copy()

    uniq = np.sort(y_sr.unique())
    if not set(uniq.tolist()).issubset({0,1}):
        mapping = {uniq[0]:0, uniq[-1]:1}
        y_sr = y_sr.map(mapping).astype(int)

    return X_df, y_sr

# ===== PyTorch MLP exactly as specified =====
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=2, p_drop=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        x = self.fc1(x); x = self.bn1(x); x = self.relu(x); x = self.dropout(x)
        x = self.fc2(x); x = self.bn2(x); x = self.relu(x); x = self.dropout(x)
        x = self.fc3(x)
        return x

def train_eval_torch_mlp(X_df, y_sr, max_epochs=150, batch_size=256, lr=1e-3, weight_decay=1e-4, p_drop=0.5):
    # Split
    Xtr_df, Xte_df, ytr, yte = train_test_split(
        X_df, y_sr, test_size=0.2, random_state=42, stratify=y_sr
    )

    # Scale (fit on train, apply to test)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_df.values)
    Xte = scaler.transform(Xte_df.values)

    # Tensors & loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr.values, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte.values, dtype=torch.long)

    ds_tr = TensorDataset(Xtr_t, ytr_t)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    # Model, loss, optimizer (Adam + CrossEntropy)
    model = SimpleNN(input_dim=Xtr.shape[1], hidden_dim1=64, hidden_dim2=32, output_dim=2, p_drop=p_drop).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train
    model.train()
    for epoch in range(max_epochs):
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = np.argmax(logits.cpu().numpy(), axis=1)

    acc = accuracy_score(yte, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(yte, probs)
    except Exception:
        auc = float("nan")

    metrics = {
        "Model": "PyTorch-MLP(64,32)+BN+Dropout",
        "Accuracy": round(float(acc), 3),
        "Precision": round(float(prec), 3),
        "Recall": round(float(rec), 3),
        "F1": round(float(f1), 3),
        "AUC": None if np.isnan(auc) else round(float(auc), 3),
        "N_test": int(len(yte))
    }
    splits = (Xtr_df, Xte_df, ytr, yte)
    return metrics, model, scaler, splits

# ===== Main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/analysis/final_traindata.xlsx")
    ap.add_argument("--max-epochs", type=int, default=150, help="Training epochs for the PyTorch MLP")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    # Save artifacts (compatible with your inference stub)
    ap.add_argument("--out-pth", default="./simple_nn_model.pth", help="PyTorch state_dict file for SimpleNN")
    ap.add_argument("--out-meta", default="./lymphocyte_mlp_meta.pkl", help="joblib bundle with scaler + feature names")
    args = ap.parse_args()

    # Load data
    X_df, y_sr = load_and_prepare(args.data)

    # Train/eval EXACT MLP you specified
    mlp_metrics, torch_model, scaler, splits = train_eval_torch_mlp(
        X_df, y_sr, max_epochs=args.max_epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, p_drop=args.dropout
    )
    Xtr_df, Xte_df, ytr, yte = splits

    # LazyPredict baseline (unchanged)
    lazy_df, used_real = try_lazypredict(Xtr_df, Xte_df, ytr, yte)

    # Print metrics
    print("=== PyTorch MLP Test Metrics (3 d.p.) ===")
    print(json.dumps(mlp_metrics, indent=2))
    print("\n=== {} Baseline (3 d.p.) ===".format("LazyPredict" if used_real else "LazyPredict-style (fallback)"))
    print(lazy_df.to_string(index=False))

    # Save artifacts
    # 1) PyTorch model weights (for your inference code with SimpleNN)
    torch.save(torch_model.state_dict(), args.out_pth)
    # 2) Scaler + feature metadata (for preprocessing at inference)
    joblib.dump({"scaler": scaler, "feature_names": list(X_df.columns)}, args.out_meta)

    print(f"\n[Saved] PyTorch MLP state_dict -> {args.out_pth}")
    print(f"[Saved] Meta (scaler, feature_names) -> {args.out_meta}")

if __name__ == "__main__":
    main()
