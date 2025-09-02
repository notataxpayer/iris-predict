# ------------------------------------------------------------
# Pakai dataset iris.csv
# Pake kolomnya:
#   sepal.length, sepal.width, petal.length, petal.width, variety
# ------------------------------------------------------------

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os, joblib

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
def train_and_evaluate(X, y, test_size=0.2, random_state=42):
    # Split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Pipeline sederhana: scaling + logistic regression
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    model.fit(X_train, y_train)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/iris_model.joblib")
    print("[OK] Model tersimpan di artifacts/iris_model.joblib")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[OK] Akurasi test: {acc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return model


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
    args = p.parse_args()

    # 1) load & tampilkan head
    try:
        df, X, y = load_dataset(args.csv)
    except FileNotFoundError:
        print(f"File '{args.csv}' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)
    print(df.head())

    # 2) training + evaluasi cepat
    model = train_and_evaluate(X, y)

    # 3) opsional: prediksi
    if args.predict_file:
        predict_from_csv(model, csv_path=args.predict_file, out_path=args.out_pred)
    else:
        predict_one(model, features_csv=args.csv, features_cli=args.features)

if __name__ == "__main__":
    main()
