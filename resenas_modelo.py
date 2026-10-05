"""Modelo del Nivel 1 (TF-IDF + regresion logistica) para la aplicacion web.

La inferencia esta escrita en Python puro: NO depende de scikit-learn, asi que no hay
problemas de versiones al desplegar. Los parametros del modelo se leen de modelo/*.json,
generados por entrenar_y_exportar.py.
"""
import bisect
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from nltk.stem.snowball import SnowballStemmer

# ---------------------------------------------------------------------------------
# Limpieza de texto (IDENTICA a la usada al entrenar; Parte 3 del proyecto)
# ---------------------------------------------------------------------------------
ES_STOP = set("""de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este sí porque esta
entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante todos uno les ni contra otros ese eso ante
ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho quienes nada muchos cual poco ella estar
estas algunas algo nosotros mi mis tú te ti tu tus ellas nosotras vosotros vosotras os mío mía míos mías tuyo tuya suyo suya
nuestro nuestra nuestros nuestras es son fue era eran ha han he hemos ser soy eres somos está están estaba estaban estoy
estamos estuvo tiene tienen tengo tenemos había habían hubo sea fueron será sido muy mucha muchas todas toda tan así aquí allí
ahí""".split())
EN_STOP = set("""i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it
its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has
had having do does did doing a an the and but if or because as until while of at by for with about against between into through
during before after above below to from up down in out on off over under again further then once here there when where why how
all any both each few more most other some such no nor not only own same so than too very s t can will just don should now""".split())
ES_ONLY, EN_ONLY = ES_STOP - EN_STOP, EN_STOP - ES_STOP
STOP = ES_STOP | EN_STOP
STEMMERS = {"es": SnowballStemmer("spanish"), "en": SnowballStemmer("english")}
URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")
NUM_RE = re.compile(r"\d+")
TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

@lru_cache(maxsize=None)
def _stem(lang, w):
    return STEMMERS[lang].stem(w)

def detectar_idioma(tokens):
    es = sum(w in ES_ONLY for w in tokens)
    en = sum(w in EN_ONLY for w in tokens)
    return "es" if es > en else ("en" if en > es else "otro")

def limpiar(t):
    t = URL_RE.sub(" urltoken ", str(t).lower())
    t = NUM_RE.sub(" numtoken ", t)
    toks = TOKEN_RE.findall(t)
    lang = detectar_idioma(toks)
    toks = [w for w in toks if w not in STOP and len(w) > 1]
    if lang != "otro":
        toks = [_stem(lang, w) for w in toks]
    return " ".join(toks)


# ---------------------------------------------------------------------------------
# Rasgos, puntaje y explicacion
# ---------------------------------------------------------------------------------
NUM_NOMBRES = ["longitud (log)", "prop. exclamaciones", "prop. mayusculas", "palabras (log)", "anio"]
TOKEN_TFIDF = re.compile(r"(?u)\b\w\w+\b")            # mismo patron que TfidfVectorizer por defecto
URL_RE_I = re.compile(URL_RE.pattern, re.IGNORECASE)


def rasgos(texto, anio):
    """Las 5 variables estructurales (antes de estandarizar)."""
    s = "" if texto is None else str(texto)
    n = float(len(s))
    return [math.log1p(n),
            s.count("!") / (n + 1),
            len(re.findall(r"[A-Z]", s)) / (n + 1),
            math.log1p(len(re.findall(r"\w+", s))),
            float(anio)]


def idioma_del_texto(texto):
    t = NUM_RE.sub(" numtoken ", URL_RE.sub(" urltoken ", str(texto).lower()))
    return detectar_idioma(TOKEN_RE.findall(t))


def tokens_resaltables(texto):
    """Lista de (inicio, fin, raiz) de las palabras del texto original que el modelo procesa."""
    s = str(texto)
    enmascarado = URL_RE_I.sub(lambda m: " " * len(m.group()), s)
    lang = idioma_del_texto(s)
    salida = []
    for m in TOKEN_RE.finditer(enmascarado):
        w = m.group().lower()
        if w in STOP or len(w) <= 1:
            continue
        salida.append((m.start(), m.end(), _stem(lang, w) if lang != "otro" else w))
    return salida


class ModeloResenas:
    def __init__(self, d):
        self.terminos = d["terminos"]
        self.indice = {t: i for i, t in enumerate(self.terminos)}
        self.idf = d["idf"]
        self.coef_t = d["coef_terminos"]
        self.coef_n = d["coef_num"]
        self.media = d["num_media"]
        self.escala = d["num_escala"]
        self.intercepto = d["intercepto"]

    def puntuar(self, texto, anio):
        """Puntaje (logit) y descomposicion exacta: intercepto + terminos + variables estructurales."""
        limpio = limpiar(texto)
        cuentas = Counter(t for t in TOKEN_TFIDF.findall(limpio) if t in self.indice)
        idxs = [self.indice[t] for t in cuentas]
        v = [(1.0 + math.log(c)) * self.idf[i] for c, i in zip(cuentas.values(), idxs)]   # tf sublineal * idf
        norma = math.sqrt(sum(x * x for x in v))
        if norma > 0:
            v = [x / norma for x in v]
        contrib_terminos = [(self.terminos[i], x, self.coef_t[i] * x) for i, x in zip(idxs, v)]
        r = rasgos(texto, anio)
        z = [(r[k] - self.media[k]) / self.escala[k] for k in range(len(r))]
        contrib_num = [(NUM_NOMBRES[k], z[k], self.coef_n[k] * z[k]) for k in range(len(r))]
        score = self.intercepto + sum(c for _, _, c in contrib_terminos) + sum(c for _, _, c in contrib_num)
        return {"score": score, "prob_cruda": 1.0 / (1.0 + math.exp(-score)), "intercepto": self.intercepto,
                "terminos": contrib_terminos, "num": contrib_num, "n_terminos_reconocidos": len(idxs)}


def probabilidad_calibrada(score, calib):
    """Probabilidad empirica de positivo (con IC 95%) para el tramo de puntaje, y percentil del puntaje."""
    k = bisect.bisect_right(calib["bordes"], score)
    percentil = max(0, min(100, bisect.bisect_right(calib["cuantiles"], score) - 1))
    return {"p": calib["tasa"][k], "ic_inf": calib["ic_inf"][k], "ic_sup": calib["ic_sup"][k],
            "n_tramo": calib["n"][k], "percentil": percentil, "marcar": score >= calib["umbral_top5"]}


def cargar_modelo(carpeta):
    carpeta = Path(carpeta)
    modelo = ModeloResenas(json.loads((carpeta / "modelo_nivel1.json").read_text(encoding="utf-8")))
    calib = json.loads((carpeta / "calibracion.json").read_text(encoding="utf-8"))
    mapa = json.loads((carpeta / "mapa_stems.json").read_text(encoding="utf-8"))
    return modelo, calib, mapa
