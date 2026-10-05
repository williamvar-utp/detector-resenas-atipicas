"""Entrena el modelo del Nivel 1 con TODO el conjunto de entrenamiento y lo exporta a modelo/*.json.

Uso (desde la carpeta del repositorio, con salidas/ de las Partes 2 y 6 disponible):
    pip install -r requirements-entrenamiento.txt
    python entrenar_y_exportar.py --train salidas/train.csv.gz --oof salidas/parte6_oof.csv

Genera en modelo/: modelo_nivel1.json, calibracion.json, mapa_stems.json e info.json.
La app NO necesita scikit-learn: solo estos JSON y resenas_modelo.py.
"""
import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import resenas_modelo as rm

SEED = 42
FEATS = ["texto", "anio", "grupo_cod"]
TFIDF = dict(lowercase=False, sublinear_tf=True, ngram_range=(1, 1), min_df=20, max_features=50_000)  # Parte 3


class LimpiadorTexto(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({"texto": X["texto"].values,
                             "texto_limpio": [rm.limpiar(t) for t in X["texto"].values],
                             "anio": X["anio"].values})


def rasgos_estructurales(d):
    # Se calculan con Python (re), igual que en la app. No con pandas .str: en pandas >= 3 el motor de regex
    # reconoce \w solo en ASCII y el conteo de palabras cambiaria en textos con acentos o ñ.
    return np.array([rm.rasgos(t, a) for t, a in zip(d["texto"].fillna("").astype(str), d["anio"])], dtype=float)


def pipeline_nivel1():
    pre = ColumnTransformer([
        ("tfidf", TfidfVectorizer(**TFIDF), "texto_limpio"),
        ("num", Pipeline([("f", FunctionTransformer(rasgos_estructurales)),
                          ("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), ["texto", "anio"]),
    ])
    clf = LogisticRegression(C=10, class_weight="balanced", solver="liblinear", max_iter=300)   # Parte 3
    return Pipeline([("limpieza", LimpiadorTexto()), ("pre", pre), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="salidas/train.csv.gz")
    ap.add_argument("--oof", default="salidas/parte6_oof.csv")
    ap.add_argument("--salida", default="modelo")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.salida)
    out.mkdir(exist_ok=True)

    train = pd.read_csv(a.train, parse_dates=["date"])
    train["anio"] = train["date"].dt.year
    train["grupo_cod"] = 0
    print(f"entrenamiento: {len(train):,} filas, prevalencia {train['y'].mean():.4f}")

    pipe = pipeline_nivel1().fit(train[FEATS], train["y"].values)
    tf = pipe.named_steps["pre"].named_transformers_["tfidf"]
    sc = pipe.named_steps["pre"].named_transformers_["num"].named_steps["sc"]
    clf = pipe.named_steps["clf"]
    terminos = [str(t) for t in tf.get_feature_names_out()]
    coef = clf.coef_.ravel()
    datos = {"terminos": terminos, "idf": tf.idf_.tolist(), "coef_terminos": coef[:len(terminos)].tolist(),
             "coef_num": coef[len(terminos):].tolist(), "num_media": sc.mean_.tolist(),
             "num_escala": sc.scale_.tolist(), "intercepto": float(clf.intercept_[0])}
    ruta_modelo = out / "modelo_nivel1.json"
    ruta_modelo.write_text(json.dumps(datos), encoding="utf-8")
    print(f"modelo exportado: {len(terminos):,} terminos -> {ruta_modelo}")

    # --- verificacion: la inferencia en Python puro coincide con el pipeline de scikit-learn
    modelo = rm.ModeloResenas(datos)
    muestra = train.sample(min(1000, len(train)), random_state=SEED)
    esperado = pipe.decision_function(muestra[FEATS])
    obtenido = np.array([modelo.puntuar(t, an)["score"] for t, an in zip(muestra["texto"], muestra["anio"])])
    dif = float(np.max(np.abs(esperado - obtenido)))
    print(f"verificacion de equivalencia (1000 filas): diferencia maxima = {dif:.2e}")
    if dif > 1e-6:
        sys.exit("ERROR: la inferencia exportada no coincide con el pipeline original")

    # --- calibracion con las predicciones fuera de muestra de la Parte 6 (nunca con el test)
    oof = pd.read_csv(a.oof)
    s, y = oof["s_clasico"].values, oof["y"].values.astype(float)
    bordes = np.quantile(s, np.linspace(0.05, 0.95, 19))
    bins = np.searchsorted(bordes, s, side="right")
    nb = np.bincount(bins, minlength=20)
    tasa = np.bincount(bins, weights=y, minlength=20) / np.maximum(nb, 1)
    rg = np.random.default_rng(SEED)
    gidx = list(oof.groupby("grupo").indices.values())
    tasas = []
    for _ in range(a.boot):                       # bootstrap por grupos (las filas de un anuncio no son independientes)
        idx = np.concatenate([gidx[i] for i in rg.integers(0, len(gidx), len(gidx))])
        n_ = np.bincount(bins[idx], minlength=20)
        tasas.append(np.bincount(bins[idx], weights=y[idx], minlength=20) / np.maximum(n_, 1))
    lo, hi = np.percentile(tasas, [2.5, 97.5], axis=0)
    calib = {"bordes": bordes.tolist(), "tasa": tasa.tolist(), "ic_inf": lo.tolist(), "ic_sup": hi.tolist(),
             "n": nb.tolist(), "cuantiles": np.quantile(s, np.linspace(0, 1, 101)).tolist(),
             "umbral_top5": float(np.quantile(s, 0.95)), "prevalencia": float(y.mean()), "n_total": int(len(s))}
    (out / "calibracion.json").write_text(json.dumps(calib), encoding="utf-8")
    print("calibracion (tasa de positivos por tramo de puntaje, 20 tramos):", np.round(tasa, 3).tolist())

    # --- mapa raiz -> palabra original mas frecuente (solo para mostrar terminos legibles)
    mapa = {}
    for t in train["texto"].sample(min(30_000, len(train)), random_state=SEED):
        for ini, fin, raiz in rm.tokens_resaltables(t):
            mapa.setdefault(raiz, Counter())[str(t)[ini:fin].lower()] += 1
    vocab = set(terminos)
    mapa = {r: c.most_common(1)[0][0] for r, c in mapa.items() if r in vocab}
    (out / "mapa_stems.json").write_text(json.dumps(mapa, ensure_ascii=False), encoding="utf-8")

    info = {"filas_entrenamiento": int(len(train)), "terminos": len(terminos), "dif_equivalencia": dif,
            "sha256_modelo": hashlib.sha256(ruta_modelo.read_bytes()).hexdigest(),
            "python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__}
    (out / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("listo:", json.dumps(info))


if __name__ == "__main__":
    main()
