import pandas as pd

pd.set_option("mode.chained_assignment", None)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def predict_stock_daily(symbol: str):
    path_name = "https://github.com/reckoning-machines/datasets/blob/main/betas_table_zip.zip?raw=true"
    df = pd.read_csv(path_name, compression="zip")

    df = df.loc[df["date"] > "2019-01-01"]
    df["target"] = np.where(df["one_day_alpha"] > 0, 1, 0)
    df.columns
    featureset_columns = [
        "hyg_return",
        "tlt_return",
        "vb_return",
        "vtv_return",
        "vug_return",
        "rut_return",
        "spx_return",
        "DGS10_return",
        "DGS2_return",
        "DTB3_return",
        "DFF_return",
        "T10Y2Y_return",
        "T5YIE_return",
        "BAMLH0A0HYM2_return",
        "DEXUSEU_return",
        "KCFSI_return",
        "DRTSCILM_return",
        "RSXFS_return",
        "MARTSMPCSM44000USS_return",
        "H8B1058NCBCMG_return",
        "DCOILWTICO_return",
        "VXVCLS_return",
        "H8B1247NCBCMG_return",
        "SP500_return",
        "GASREGW_return",
        "CSUSHPINSA_return",
        "UNEMPLOY_return",
    ]

    dataset = df.loc[df["symbol"] == symbol]
    dataset["target"] = dataset["target"].shift(-1)
    # dataset[['date','target']]
    prediction_record = dataset[-1:]
    dataset = dataset.iloc[:-1]
    y = dataset["target"]
    X = dataset[featureset_columns]
    X = X.astype(float)
    X = X.diff(periods=1, axis=0)
    X = X.dropna()
    y = y.iloc[1:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=10
    )

    classifier = LogisticRegression(random_state=10)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    y_pred = classifier.predict_proba(prediction_record[featureset_columns])

    return {
        "accuracy": test_acc,
        "ticker": symbol,
        "prediction date": prediction_record["date"],
        "outperformance probability": round(y_pred[:, 0][0], 2),
    }
