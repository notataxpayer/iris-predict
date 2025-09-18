# iris_classification.py
# ------------------------------------------------------------
# Pakai dataset iris.csv
# Pake kolomnya:
#   sepal.length, sepal.width, petal.length, petal.width, variety
# ------------------------------------------------------------

import argparse
import sys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os, joblib
import mlflow
import mlflow.sklearn

# Kolom yg dipakai
FEATURE_COLS = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
# Targetnya
TARGET_COL = "variety"

# Fun load dataset iris.csv
def load_dataset(csv_path: str):
    # Baca csvnya
    df = pd.read_csv(csv_path)
    # Validasi kolom fitur n target
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom yg hilang: {missing}\n"
                         f"List kolom yang ada: {FEATURE_COLS + [TARGET_COL]}")
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values  # Labelnya ada Setosa/Versicolor/Virginica
    return df, X, y

# Fun train split sama eval
def train_and_evaluate(X, y, test_size=0.2, random_state=42, artifacts_dir="artifacts", experiment_name="iris-predict"):
    """
    Train + evaluate — MLflow always digunakan (start_run dibuat di sini).
    """
    # Split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Pipeline sederhana: scaling + logistic regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    # MLflow setart
    mlflow.set_experiment(experiment_name)
    run_ctx = mlflow.start_run()

    try:
        model.fit(X_train, y_train)
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(model, os.path.join(artifacts_dir, "iris_model.joblib"))
        print(f"[OK] Model tersimpan di {os.path.join(artifacts_dir, 'iris_model.joblib')}")

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"[OK] Akurasi test: {acc:.4f}")
        print("\nClassification report:\n", classification_report(y_test, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        # hasil reportnyoh
        metrics_dir = "metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = os.path.join(metrics_dir, "classification_report.json")
        with open(report_path, "w") as fh:
            json.dump({"accuracy": acc, "f1_macro": f1, "report": report}, fh, indent=2)    

        preds_df = pd.DataFrame({"y_true": list(y_test), "y_pred": list(y_pred)})
        preds_path = os.path.join(artifacts_dir, "predictions.csv")
        preds_df.to_csv(preds_path, index=False)

        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(artifacts_dir, "confusion_matrix.csv")
        pd.DataFrame(cm).to_csv(cm_path, index=False)

        # log params: ambil bbrp params classifier n split info
        try:
            clf = model.named_steps.get("clf")
            clf_params = clf.get_params() if hasattr(clf, "get_params") else {}
            keys_to_log = ["C", "max_iter", "penalty", "solver", "multi_class", "random_state"]
            params_to_log = {k: clf_params[k] for k in keys_to_log if k in clf_params}
            params_to_log.update({"test_size": test_size, "random_state": random_state})
            mlflow.log_params(params_to_log)
        except Exception as e:
            print(f"[WARN] Failed to log params to MLflow: {e}", file=sys.stderr)

        # log metrics
        try:
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("f1_macro", float(f1))
        except Exception as e:
            print(f"[WARN] Failed to log metrics to MLflow: {e}", file=sys.stderr)

        # log artifacts
        try:
            mlflow.log_artifact(report_path)
            mlflow.log_artifact(preds_path)
            mlflow.log_artifact(cm_path)
        except Exception as e:
            print(f"[WARN] Failed to log artifacts to MLflow: {e}", file=sys.stderr)

        # log sklearn pipeline sbg model
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
        except Exception as e:
            print(f"[WARN] Failed to log sklearn model to MLflow: {e}", file=sys.stderr)

        return model
    finally:
        mlflow.end_run()

def predict_one(model, features_csv: str | None, features_cli: str | None):
    """" 
    - Jika features_cli diisi: pakai 4 angka dipisah koma.
    - Kalau tidak, ambil 5 baris pertama dari CSV sebagai demo.
    """
    if features_cli:
        vals = [float(x) for x in features_cli.split(",")]
        if len(vals) != 4:
            raise SystemExit("Butuh 4 nilai fitur dipisah koma: sepal.len,sepal.wid,petal.len,petal.wid")
        X = np.array([vals], dtype=float)
        preds = model.predict(X).tolist()
        print("Prediksi (1 sampel):", preds[0])
        return

    if not features_csv:
        raise SystemExit("Jika tidak pakai --features, berikan --csv untuk contoh prediksi dari file.")
    _, X, _ = load_dataset(features_csv)
    X_demo = X[:5]
    preds = model.predict(X_demo).tolist()
    print("Prediksi 5 baris pertama dari CSV:", preds)

def predict_from_csv(model, csv_path: str, out_path: str | None = None):
    df_in = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS if c not in df_in.columns]
    if missing :
        raise ValueError(
            f"Kolom yg hilang: {missing}\n"
            f"List kolom yang ada: {df_in.columns.tolist()}"
        )
    x = df_in[FEATURE_COLS].values
    preds = model.predict(x)

    df_out = df_in.copy()
    if TARGET_COL in df_out.columns:
        df_out.rename(columns={TARGET_COL: f"{TARGET_COL}_true"}, inplace=True)
    df_out["prediction"] = preds

    if out_path:
        df_out.to_csv(out_path, index=False)
        print(f"[OK] Hasil prediksi disimpan di: {out_path}")
    else:
        print("[INFO] Contoh 5 baris hasil prediksi:")
        print(df_out.head())
    return df_out

def main():
    p = argparse.ArgumentParser(description="Iris classification")
    p.add_argument("--csv", type=str, default="iris.csv", help="Path nya utk iris.csv (defaultnya: iris.csv)")
    p.add_argument("--predict", dest="features", type=str, default=None,
                   help='Fiturnya "sepal.len,sepal.wid,petal.len,petal.wid"')
    p.add_argument("--predict-file", type=str, default=None,
                   help="Path CSV untuk PREDIKSI (batch). Kolom fitur harus sama.")
    p.add_argument("--out-pred", type=str, default=None,
                   help="Path CSV output hasil prediksi (opsional)")
    p.add_argument("--model-path", default="artifacts/model.joblib")
 
    p.add_argument("--mlflow-experiment", type=str, default="iris-predict", help="MLflow experiment name (default: iris-predict)")
    args = p.parse_args()

    # 1) load & tampilkan head
    try:
        df, X, y = load_dataset(args.csv)
    except FileNotFoundError:
        print(f"File '{args.csv}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)
    print(df.head())

    # 2) training + evaluasi cepat (MLflow always on)
    model = train_and_evaluate(X, y, artifacts_dir=os.path.dirname(args.model_path) or "artifacts", experiment_name=args.mlflow_experiment)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    joblib.dump(model, args.model_path)

    # 3) opsional: prediksi
    if args.predict_file:
        predict_from_csv(model, csv_path=args.predict_file, out_path=args.out_pred)
    else:
        predict_one(model, features_csv=args.csv, features_cli=args.features)

if __name__ == "__main__":
    main()
