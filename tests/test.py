import os
import pandas as pd
import pandas.api.types as ptypes

REQ = ["sepal.length", "sepal.width", "petal.length", "petal.width", "variety"]
FEATURES = REQ[:-1] 

def test_csv_extists():
    assert os.path.exists("iris.csv")

def test_scheme_and_types():
    df = pd.read_csv("iris.csv")
    missing = [c for c in REQ if c not in df.columns]
    assert not missing, f"kolom hilang: {missing}. Harusnya ada {REQ}"
    for col in FEATURES:
        assert ptypes.is_numeric_dtype(df[col]), f"kolom {col} harusnya numeric, data sekarang berupa: {df[col].dtype}"
    assert not df[FEATURES].isna().any().any(), f"ada nilai NaN di kolom fitur {col}"
    assert df["variety"].notna().all(), "ada nilai NaN di kolom target 'variety'"
    
def test_import_script():
    __import__("iris_classification")

