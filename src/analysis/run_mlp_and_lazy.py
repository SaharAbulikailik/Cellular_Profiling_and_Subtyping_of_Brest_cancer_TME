#!/usr/bin/env python3
import argparse, os, json, numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.neural_network import MLPClassifier

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

def train_eval_mlp(X_df, y_sr, max_iter=150):
    Xtr_df, Xte_df, ytr, yte = train_test_split(
        X_df, y_sr, test_size=0.2, random_state=42, stratify=y_sr
    )
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_df.values)
    Xte = scaler.transform(Xte_df.values)

    clf = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                        alpha=1e-4, learning_rate_init=1e-3,
                        max_iter=max_iter, random_state=42)
    clf.fit(Xtr, ytr)

    probs = clf.predict_proba(Xte)[:,1]
    preds = clf.predict(Xte)

    acc = accuracy_score(yte, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(yte, probs)
    except Exception:
        auc = float("nan")

    metrics = {
        "Model": "MLPClassifier(64,32)",
        "Accuracy": round(float(acc), 3),
        "Precision": round(float(prec), 3),
        "Recall": round(float(rec), 3),
        "F1": round(float(f1), 3),
        "AUC": None if np.isnan(auc) else round(float(auc), 3),
        "N_test": int(len(yte))
    }
    splits = (Xtr_df, Xte_df, ytr, yte)
    return metrics, clf, scaler, splits

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

    # round numeric cols to 3 decimals for printing
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(3)
    return out, used_real

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/analysis/final_traindata.xlsx")
    ap.add_argument("--out-model", default="./lymphocyte_mlp.pkl")
    ap.add_argument("--max-iter", type=int, default=150)
    args = ap.parse_args()

    X_df, y_sr = load_and_prepare(args.data)
    mlp_metrics, clf, scaler, splits = train_eval_mlp(X_df, y_sr, max_iter=args.max_iter)
    Xtr_df, Xte_df, ytr, yte = splits
    lazy_df, used_real = try_lazypredict(Xtr_df, Xte_df, ytr, yte)

    print("=== MLP Test Metrics (3 d.p.) ===")
    print(json.dumps(mlp_metrics, indent=2))

    print("\n=== {} Baseline (3 d.p.) ===".format("LazyPredict" if used_real else "LazyPredict-style (fallback)"))
    print(lazy_df.to_string(index=False))

    joblib.dump({"model": clf, "scaler": scaler, "feature_names": list(X_df.columns)}, args.out_model)
    print(f"\n[Saved] MLP model -> {args.out_model}")

if __name__ == "__main__":
    main()
